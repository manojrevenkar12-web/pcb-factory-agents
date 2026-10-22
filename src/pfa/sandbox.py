"""Sandboxed agent sessions, and the worker that runs them off the queue.

A session is a separate process with a scrubbed environment: no database URL, no API keys, a
throwaway home directory, CPU-time and memory limits. It receives the gateway URL, a session token
scoped to its task and fence, and the task payload. In production the same contract runs in a
cloud sandbox (Modal, Firecracker); here a local process stands in.

The worker is trusted: it claims a task, mints the token, starts the session, renews the lease
while the session runs, and completes or fails the task under its fence. If the worker dies, the
lease runs out and another worker takes over with a new fence, and anything the old session
still tries to do is refused by the gateway.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import psycopg

from pfa import db, workqueue
from pfa.gateway.core import Gateway

SAFE_ENV = ("PATH", "LANG", "LC_ALL", "PYTHONHASHSEED", "SYSTEMROOT")


def sandbox_env(gateway_url: str, token: str, payload: dict, home: str, extra: dict | None = None) -> dict:
    env = {k: os.environ[k] for k in SAFE_ENV if k in os.environ}
    env.update(
        {
            "HOME": home,
            "TMPDIR": home,
            "PFA_GATEWAY_URL": gateway_url,
            "PFA_SESSION_TOKEN": token,
            "PFA_TASK": json.dumps(payload),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    env.update(extra or {})
    return env


def _limits(cpu_s: int, mem_mb: int) -> Callable[[], None]:
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2**20, mem_mb * 2**20))
        os.setsid()

    return apply


@dataclass
class SessionResult:
    exit_code: int
    output: dict | None
    stderr: str


def start_session(
    agent: str,
    gateway_url: str,
    token: str,
    payload: dict,
    extra_env: dict | None = None,
    cpu_s: int = 120,
    mem_mb: int = 1024,
) -> tuple[subprocess.Popen, str]:
    home = tempfile.mkdtemp(prefix="pfa-sandbox-")
    proc = subprocess.Popen(
        [sys.executable, "-I", "-m", "pfa.agent_entry", agent],
        env=sandbox_env(gateway_url, token, payload, home, extra_env),
        cwd=home,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=_limits(cpu_s, mem_mb),
    )
    return proc, home


def collect(proc: subprocess.Popen, timeout: float) -> SessionResult:
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    output = None
    for line in reversed(out.strip().splitlines()):
        try:
            output = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    return SessionResult(proc.returncode, output, err[-2000:])


@dataclass
class Worker:
    """Claims tasks of some kinds and runs each in a sandboxed session."""

    name: str
    database_url: str
    gateway: Gateway
    gateway_url: str
    agent_for_kind: dict[str, str]
    scopes_for_kind: dict[str, list[str]]
    lease_s: float = 10.0
    session_timeout_s: float = 120.0
    extra_env: dict = field(default_factory=dict)
    on_start: Callable[[subprocess.Popen], None] | None = None
    results: list[dict] = field(default_factory=list)
    exits: list[tuple[int, int, str]] = field(default_factory=list)  # task, exit code, stderr tail
    abandon: threading.Event = field(default_factory=threading.Event)

    def run(self, stop_when_idle: bool = True, idle_wait_s: float = 0.2, max_idle_s: float = 3.0) -> None:
        with db.connect(self.database_url) as conn:
            idle_since = None
            while not self.abandon.is_set():
                claim = workqueue.claim(conn, self.name, list(self.agent_for_kind), self.lease_s)
                if claim is None:
                    idle_since = idle_since or time.time()
                    if stop_when_idle and (time.time() - idle_since > max_idle_s or not self._pending(conn)):
                        return
                    time.sleep(idle_wait_s)
                    continue
                idle_since = None
                self._run_one(conn, claim)

    def _pending(self, conn: psycopg.Connection) -> bool:
        row = conn.execute(
            "SELECT count(*) AS n FROM agents.tasks WHERE status IN ('queued', 'running') AND kind = ANY(%s)",
            (list(self.agent_for_kind),),
        ).fetchone()
        conn.commit()
        return row["n"] > 0

    def _run_one(self, conn: psycopg.Connection, claim: workqueue.Claim) -> None:
        token = self.gateway.mint(
            self.scopes_for_kind[claim.kind],
            ttl_s=self.session_timeout_s * 2,
            task_id=claim.task_id,
            fence=claim.fence,
        )
        proc, _ = start_session(
            self.agent_for_kind[claim.kind], self.gateway_url, token, claim.payload, self.extra_env
        )
        if self.on_start:
            self.on_start(proc)
        lost = threading.Event()

        def heartbeat() -> None:
            with db.connect(self.database_url) as hb:
                while proc.poll() is None and not lost.is_set():
                    if self.abandon.is_set():  # simulated worker death: stop renewing, walk away
                        return
                    try:
                        workqueue.renew(hb, claim, self.lease_s)
                    except workqueue.LeaseLost:
                        lost.set()
                        proc.kill()
                        return
                    time.sleep(self.lease_s / 3)

        beat = threading.Thread(target=heartbeat, daemon=True)
        beat.start()
        result = collect(proc, self.session_timeout_s)
        beat.join(timeout=1)
        self.exits.append((claim.task_id, result.exit_code, result.stderr[-300:]))
        if self.abandon.is_set() or lost.is_set():
            return
        try:
            if result.exit_code == 0 and result.output is not None:
                workqueue.complete(conn, claim, result.output)
                self.results.append({"task_id": claim.task_id, "fence": claim.fence, **result.output})
            else:
                workqueue.fail(conn, claim, f"exit {result.exit_code}: {result.stderr[-500:]}")
        except workqueue.LeaseLost:
            pass

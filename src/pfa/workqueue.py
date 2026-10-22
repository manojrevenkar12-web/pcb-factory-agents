"""A PostgreSQL work queue with leases and fencing tokens.

Workers claim with `FOR UPDATE SKIP LOCKED`, so they never block each other or take the same
task. A claim holds a lease; a worker that dies stops renewing it, and the task becomes
claimable again when the lease runs out. Every claim increments the task's fence. The fence
travels in the agent's session token, and the gateway refuses any action whose fence is no
longer current: a worker that was paused past its lease (GC pause, network partition, a
suspended VM) wakes up unable to act or to finish the task someone else now owns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Jsonb


class LeaseLost(Exception):
    """The task was re-claimed by another worker; this worker must stop."""


@dataclass
class Claim:
    task_id: int
    kind: str
    business_key: str
    payload: dict[str, Any]
    fence: int
    attempts: int


def enqueue(conn: psycopg.Connection, kind: str, business_key: str, payload: dict | None = None) -> bool:
    """Adds a task once per (kind, business_key); returns False if it already existed."""
    cur = conn.execute(
        """INSERT INTO agents.tasks (kind, business_key, payload) VALUES (%s, %s, %s)
           ON CONFLICT (kind, business_key) DO NOTHING""",
        (kind, business_key, Jsonb(payload or {})),
    )
    return cur.rowcount == 1


def claim(conn: psycopg.Connection, worker: str, kinds: list[str], lease_s: float) -> Claim | None:
    row = conn.execute(
        """UPDATE agents.tasks t
              SET status = 'running', lease_owner = %(w)s,
                  lease_until = now() + make_interval(secs => %(lease)s),
                  fence = fence + 1, attempts = attempts + 1, updated_at = now()
            WHERE id = (
                SELECT id FROM agents.tasks
                 WHERE kind = ANY(%(kinds)s)
                   AND (status = 'queued' OR (status = 'running' AND lease_until < now()))
                   AND attempts < max_attempts
                 ORDER BY id
                 FOR UPDATE SKIP LOCKED
                 LIMIT 1)
        RETURNING id, kind, business_key, payload, fence, attempts""",
        {"w": worker, "lease": lease_s, "kinds": kinds},
    ).fetchone()
    conn.commit()
    if row is None:
        return None
    return Claim(row["id"], row["kind"], row["business_key"], row["payload"], row["fence"], row["attempts"])


def renew(conn: psycopg.Connection, claim: Claim, lease_s: float) -> None:
    cur = conn.execute(
        """UPDATE agents.tasks SET lease_until = now() + make_interval(secs => %s), updated_at = now()
            WHERE id = %s AND fence = %s AND status = 'running'""",
        (lease_s, claim.task_id, claim.fence),
    )
    conn.commit()
    if cur.rowcount != 1:
        raise LeaseLost(f"task {claim.task_id} fence {claim.fence}")


def complete(conn: psycopg.Connection, claim: Claim, result: dict | None = None) -> None:
    cur = conn.execute(
        """UPDATE agents.tasks SET status = 'done', result = %s, lease_owner = NULL, lease_until = NULL,
                  updated_at = now()
            WHERE id = %s AND fence = %s AND status = 'running'""",
        (Jsonb(result or {}), claim.task_id, claim.fence),
    )
    conn.commit()
    if cur.rowcount != 1:
        raise LeaseLost(f"task {claim.task_id} fence {claim.fence}")


def fail(conn: psycopg.Connection, claim: Claim, error: str) -> None:
    """Gives the task back for a retry, or fails it for good after max_attempts."""
    cur = conn.execute(
        """UPDATE agents.tasks
              SET status = CASE WHEN attempts >= max_attempts THEN 'failed' ELSE 'queued' END,
                  error = %s, lease_owner = NULL, lease_until = NULL, updated_at = now()
            WHERE id = %s AND fence = %s AND status = 'running'""",
        (error[:2000], claim.task_id, claim.fence),
    )
    conn.commit()
    if cur.rowcount != 1:
        raise LeaseLost(f"task {claim.task_id} fence {claim.fence}")


def current_fence(conn: psycopg.Connection, task_id: int) -> int | None:
    """Locks the task row (shared) for the rest of the transaction: no re-claim can slip in
    between this check and the action it guards."""
    row = conn.execute(
        "SELECT fence, status, lease_until > now() AS live FROM agents.tasks WHERE id = %s FOR SHARE",
        (task_id,),
    ).fetchone()
    if row is None or row["status"] != "running" or not row["live"]:
        return None
    return row["fence"]

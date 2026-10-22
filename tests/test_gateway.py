"""The platform's guarantees: tokens, scopes, exactly-once actions, fencing, shadow mode, judge cache."""

from __future__ import annotations

import threading
import time

import pytest
from pydantic import BaseModel

from pfa import workqueue
from pfa.db import connect
from pfa.gateway import tokens
from pfa.gateway.core import Conflict, Gateway, InvalidArguments, PermissionDenied, Skill, StaleSession
from pfa.judges import FunctionJudge


class Note(BaseModel):
    text: str


class Answer(BaseModel):
    word: str


def make_gateway(url: str) -> tuple[Gateway, list[str]]:
    gw = Gateway(url, b"k" * 32)
    effects: list[str] = []

    @gw.tool("notes.add", "action", Note)
    def add(ctx, a: Note) -> dict:
        effects.append(a.text)
        time.sleep(0.05)  # widen the window for concurrent duplicates
        return {"n": len(effects)}

    @gw.tool("notes.count", "read", Note)
    def count(ctx, a: Note) -> int:
        return len(effects)

    calls = []
    gw.add_skill(
        Skill("echo", Answer, FunctionJudge("echo", lambda r: calls.append(1) or Answer(word=r.text)))
    )
    gw.calls = calls
    return gw, effects


def test_tokens_are_signed_scoped_and_expire():
    key = b"s" * 32
    t = tokens.mint(key, tokens.Claims("s1", ["notes.*"], time.time() + 60))
    claims = tokens.verify(key, t)
    assert claims.allows("notes.add") and not claims.allows("erp.book_invoice")
    with pytest.raises(tokens.TokenError):
        tokens.verify(b"x" * 32, t)
    body, sig = t.split(".")
    with pytest.raises(tokens.TokenError):
        tokens.verify(key, body[:-2] + "AA." + sig)
    with pytest.raises(tokens.TokenError):
        tokens.verify(key, tokens.mint(key, tokens.Claims("s1", [], time.time() - 1)))


def test_actions_run_exactly_once_per_key_even_concurrently(database_url):
    gw, effects = make_gateway(database_url)
    token = gw.mint(["notes.add"])
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(gw.call(token, "notes.add", {"text": "hi"}, "k1")))
        for _ in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert effects == ["hi"]
    assert sum(not r["replayed"] for r in results) == 1 and all(r["result"] == {"n": 1} for r in results)
    gw.close()


def test_same_key_different_arguments_is_a_conflict(database_url):
    gw, _ = make_gateway(database_url)
    token = gw.mint(["notes.add"])
    gw.call(token, "notes.add", {"text": "a"}, "k")
    with pytest.raises(Conflict):
        gw.call(token, "notes.add", {"text": "b"}, "k")
    gw.close()


def test_scopes_and_idempotency_keys_are_enforced(database_url):
    gw, _ = make_gateway(database_url)
    with pytest.raises(PermissionDenied):
        gw.call(gw.mint(["notes.count"]), "notes.add", {"text": "a"}, "k")
    with pytest.raises(InvalidArguments):
        gw.call(gw.mint(["notes.add"]), "notes.add", {"text": "a"})  # actions need a key
    with pytest.raises(PermissionDenied):
        gw.call("forged.token", "notes.count", {"text": "a"})
    gw.close()


def test_a_session_whose_task_was_reclaimed_cannot_act(database_url):
    gw, effects = make_gateway(database_url)
    with connect(database_url) as conn:
        workqueue.enqueue(conn, "t", "one")
        conn.commit()
        first = workqueue.claim(conn, "w1", ["t"], lease_s=0.3)
        token = gw.mint(["notes.add"], task_id=first.task_id, fence=first.fence)
        time.sleep(0.5)  # w1 freezes; its lease runs out
        second = workqueue.claim(conn, "w2", ["t"], lease_s=30)
        assert second.task_id == first.task_id and second.fence == first.fence + 1
        with pytest.raises(StaleSession):
            gw.call(token, "notes.add", {"text": "late"}, "k")
        with pytest.raises(workqueue.LeaseLost):
            workqueue.complete(conn, first, {})
        workqueue.complete(conn, second, {})
    assert effects == []
    gw.close()


def test_workers_never_claim_the_same_task(database_url):
    with connect(database_url) as conn:
        for i in range(50):
            workqueue.enqueue(conn, "t", str(i))
        conn.commit()
    claimed: list[int] = []
    lock = threading.Lock()

    def worker(name: str) -> None:
        with connect(database_url) as conn:
            while (c := workqueue.claim(conn, name, ["t"], 30)) is not None:
                with lock:
                    claimed.append(c.task_id)
                workqueue.complete(conn, c, {})

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(claimed) == sorted(set(claimed)) and len(claimed) == 50


def test_shadow_mode_records_and_does_nothing(database_url):
    gw, effects = make_gateway(database_url)
    out = gw.call(gw.mint(["notes.add"], shadow=True), "notes.add", {"text": "would"}, "k")
    assert out["result"] == {"shadow": True} and effects == []
    with connect(database_url) as conn:
        assert conn.execute("SELECT count(*) AS n FROM agents.shadow_actions").fetchone()["n"] == 1
        assert conn.execute("SELECT count(*) AS n FROM agents.receipts").fetchone()["n"] == 0
    gw.close()


def test_judge_answers_are_cached_durably_and_costed(database_url):
    gw, _ = make_gateway(database_url)
    token = gw.mint(["judge.echo"])
    a = gw.judge(token, "echo", "hello")
    b = gw.judge(token, "echo", "hello")
    assert a["answer"] == b["answer"] == {"word": "hello"} and b["cached"] and len(gw.calls) == 1
    with pytest.raises(PermissionDenied):
        gw.judge(gw.mint(["notes.add"]), "echo", "hello")
    with connect(database_url) as conn:
        rows = conn.execute("SELECT cached FROM agents.judge_calls ORDER BY id").fetchall()
    assert [r["cached"] for r in rows] == [False, True]
    gw.close()

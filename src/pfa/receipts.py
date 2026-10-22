"""Durable receipts: every business action runs at most once, however often it is asked for.

An agent session can be killed at any moment: mid tool call, after the remote system acted but
before its answer arrived, or while a second worker picks up the same task. The receipt protocol:

1. `begin` inserts the receipt for an action's key (or finds the existing one) and locks it.
   A finished receipt returns its stored result: the action is not repeated.
   The same key with different arguments is a conflict, never a silent overwrite.
2. The action runs while the row is locked. Remote systems receive the receipt key as their
   idempotency key, so an action whose answer was lost is recognised when it is retried.
   Local effects (the email outbox, the review queue) are written in the same transaction.
3. `finish` stores the result in that transaction. A crash before commit leaves the receipt
   pending, or absent, and the retry goes through the same path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Jsonb


class IdempotencyConflict(Exception):
    """The same idempotency key was used for a different action: a bug, never retried."""


def args_hash(args: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class Receipt:
    key: str
    status: str
    result: Any
    attempts: int

    @property
    def done(self) -> bool:
        return self.status == "done"


def begin(
    conn: psycopg.Connection, key: str, tool: str, args: dict, task_id: int | None, session_id: str | None
) -> Receipt:
    """Must run inside a transaction; the receipt stays locked until it commits."""
    digest = args_hash(args)
    conn.execute(
        """INSERT INTO agents.receipts (key, tool, args_hash, args, status, task_id, session_id)
           VALUES (%s, %s, %s, %s, 'pending', %s, %s) ON CONFLICT (key) DO NOTHING""",
        (key, tool, digest, Jsonb(args), task_id, session_id),
    )
    row = conn.execute(
        "SELECT tool, args_hash, status, result, attempts FROM agents.receipts WHERE key = %s FOR UPDATE",
        (key,),
    ).fetchone()
    if row["tool"] != tool or row["args_hash"] != digest:
        raise IdempotencyConflict(f"{key}: already used for a different {row['tool']} call")
    if row["status"] != "done":
        conn.execute(
            "UPDATE agents.receipts SET attempts = attempts + 1, session_id = %s WHERE key = %s",
            (session_id, key),
        )
    return Receipt(key, row["status"], row["result"], row["attempts"] + (row["status"] != "done"))


def finish(conn: psycopg.Connection, key: str, result: Any) -> None:
    conn.execute(
        "UPDATE agents.receipts SET status = 'done', result = %s, finished_at = now() WHERE key = %s",
        (Jsonb(result), key),
    )

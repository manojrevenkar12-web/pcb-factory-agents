"""Stand-ins for two systems the factory does not own: the finance system (ERP) and the email provider.

They behave like real third-party APIs, including the parts that make exactly-once hard:
- the ERP needs an API key the agents never see;
- the ERP honours an Idempotency-Key header (same key and body: the stored answer; same key,
  different body: 422), but does NOT stop the same invoice being booked twice under two keys;
- with chaos enabled, the ERP sometimes fails before doing anything (503), and sometimes books the
  invoice and then loses the answer (503 after commit): the case where a naive retry double-books;
- the email provider de-duplicates by Message-ID.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
from dataclasses import dataclass

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pydantic import BaseModel


class BookingIn(BaseModel):
    supplier_id: str
    invoice_number: str
    po_number: str
    currency: str
    total: str
    lines: list[dict]


class MessageIn(BaseModel):
    recipient: str
    subject: str
    body: str


@dataclass
class Chaos:
    fail_before: float = 0.0  # 503 without acting
    fail_after: float = 0.0  # act, then 503 (the answer is lost)
    seed: int = 0
    failed_before: int = 0
    failed_after: int = 0


def app(database_url: str, api_key: str, chaos: Chaos | None = None) -> FastAPI:
    chaos = chaos or Chaos()
    rng = random.Random(chaos.seed)
    lock = threading.Lock()
    pool = ConnectionPool(database_url, min_size=1, max_size=8, kwargs={"row_factory": dict_row}, open=True)
    api = FastAPI(title="external systems (stand-ins)")

    def roll(p: float) -> bool:
        with lock:
            return rng.random() < p

    def auth(key: str | None) -> None:
        if key != api_key:
            raise HTTPException(401, "bad API key")

    @api.post("/erp/bookings")
    def book(
        body: BookingIn, x_api_key: str | None = Header(None), idempotency_key: str | None = Header(None)
    ) -> JSONResponse:
        auth(x_api_key)
        if not idempotency_key:
            raise HTTPException(400, "Idempotency-Key required")
        if roll(chaos.fail_before):
            chaos.failed_before += 1
            return JSONResponse({"error": "unavailable"}, status_code=503)
        digest = hashlib.sha256(json.dumps(body.model_dump(), sort_keys=True).encode()).hexdigest()
        with pool.connection() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (idempotency_key,))
            prior = conn.execute(
                "SELECT body_hash, response FROM erp.idempotency WHERE key = %s", (idempotency_key,)
            ).fetchone()
            if prior:
                if prior["body_hash"] != digest:
                    raise HTTPException(422, "Idempotency-Key reused with a different body")
                return JSONResponse(
                    prior["response"], status_code=200, headers={"Idempotent-Replayed": "true"}
                )
            row = conn.execute(
                """INSERT INTO erp.bookings (idempotency_key, supplier_id, invoice_number, po_number, currency,
                       total, lines) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id, booked_at""",
                (
                    idempotency_key,
                    body.supplier_id,
                    body.invoice_number,
                    body.po_number,
                    body.currency,
                    body.total,
                    Jsonb(body.lines),
                ),
            ).fetchone()
            response = {"booking_id": row["id"], "booked_at": row["booked_at"].isoformat()}
            conn.execute(
                "INSERT INTO erp.idempotency (key, body_hash, response) VALUES (%s, %s, %s)",
                (idempotency_key, digest, Jsonb(response)),
            )
        if roll(chaos.fail_after):
            chaos.failed_after += 1
            return JSONResponse({"error": "gateway timeout"}, status_code=503)
        return JSONResponse(response, status_code=201)

    @api.get("/erp/bookings")
    def bookings(
        po_number: str | None = Query(None),
        supplier_id: str | None = Query(None),
        invoice_number: str | None = Query(None),
        x_api_key: str | None = Header(None),
    ) -> list[dict]:
        auth(x_api_key)
        sql = "SELECT idempotency_key, supplier_id, invoice_number, po_number, currency, total::text AS total, lines FROM erp.bookings WHERE true"
        params: list = []
        for col, val in (
            ("po_number", po_number),
            ("supplier_id", supplier_id),
            ("invoice_number", invoice_number),
        ):
            if val is not None:
                sql += f" AND {col} = %s"
                params.append(val)
        with pool.connection() as conn:
            return conn.execute(sql + " ORDER BY id", params).fetchall()

    @api.post("/mail/messages")
    def send(
        body: MessageIn, message_id: str | None = Header(None), x_api_key: str | None = Header(None)
    ) -> dict:
        auth(x_api_key)
        if not message_id:
            raise HTTPException(400, "Message-ID required")
        with pool.connection() as conn:
            cur = conn.execute(
                """INSERT INTO mailprovider.messages (message_id, recipient, subject, body) VALUES (%s, %s, %s, %s)
                   ON CONFLICT (message_id) DO NOTHING""",
                (message_id, body.recipient, body.subject, body.body),
            )
            conn.commit()
        return {"accepted": True, "duplicate": cur.rowcount == 0}

    api.state.pool = pool
    return api

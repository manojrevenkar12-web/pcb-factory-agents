"""The back office's tools on the gateway: reads of the second sources, and the business actions.

The ERP's API key lives in the gateway's secrets. Actions run under durable receipts (see
pfa.receipts); the receipt key doubles as the ERP's Idempotency-Key and the email's Message-ID.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import date

import httpx
from pydantic import BaseModel

from pfa.backoffice.checks import norm_mpn
from pfa.backoffice.schemas import InvoiceAnswer
from pfa.gateway.core import Conflict, Context, Gateway, NotFound, RemoteError, Skill
from pfa.judges import Judge


class EmailId(BaseModel):
    email_id: str


class Address(BaseModel):
    email: str


class PONumber(BaseModel):
    po_number: str


class FindBooking(BaseModel):
    supplier_id: str
    invoice_number: str


class AsOf(BaseModel):
    as_of: date


class BookingLine(BaseModel):
    mpn: str
    quantity: int
    unit_price: str


class Booking(BaseModel):
    supplier_id: str
    invoice_number: str
    po_number: str
    currency: str
    total: str
    lines: list[BookingLine]


class Review(BaseModel):
    kind: str
    reason: str
    payload: dict


class Mail(BaseModel):
    recipient: str
    subject: str
    body: str


def _erp(ctx: Context) -> httpx.Client:
    return httpx.Client(
        base_url=ctx.services["external_url"], timeout=10, headers={"X-Api-Key": ctx.secrets["erp_api_key"]}
    )


def _erp_bookings(ctx: Context, **params: str) -> list[dict]:
    try:
        with _erp(ctx) as http:
            r = http.get("/erp/bookings", params=params)
    except httpx.TransportError as exc:
        raise RemoteError(f"ERP unreachable: {exc}") from exc
    if r.status_code >= 500:
        raise RemoteError(f"ERP {r.status_code}")
    r.raise_for_status()
    return r.json()


def _received(ctx: Context, po: str) -> dict[str, int]:
    rows = ctx.conn.execute(
        "SELECT mpn, SUM(quantity)::int AS q FROM ops.goods_receipts WHERE po_number = %s GROUP BY mpn", (po,)
    ).fetchall()
    return {r["mpn"]: r["q"] for r in rows}


def register(gw: Gateway, judge: Judge) -> None:
    gw.add_skill(Skill("book-invoice", InvoiceAnswer, judge))

    @gw.tool("inbox.get", "read", EmailId)
    def inbox_get(ctx: Context, a: EmailId) -> dict:
        """One email: sender, subject, body and attachments."""
        row = ctx.conn.execute(
            "SELECT email_id, sender, subject, body, attachments FROM ops.inbox WHERE email_id = %s",
            (a.email_id,),
        ).fetchone()
        if row is None:
            raise NotFound(a.email_id)
        return row

    @gw.tool("suppliers.by_email", "read", Address)
    def supplier(ctx: Context, a: Address) -> dict | None:
        """The supplier that sends from this address, if any."""
        return ctx.conn.execute(
            "SELECT id, name, email FROM ops.suppliers WHERE lower(email) = lower(%s)", (a.email,)
        ).fetchone()

    @gw.tool("po.get", "read", PONumber)
    def po_get(ctx: Context, a: PONumber) -> dict | None:
        """A purchase order with its lines, or null."""
        po = ctx.conn.execute(
            "SELECT po_number, supplier_id, currency, ordered_on::text, promised_on::text FROM ops.purchase_orders "
            "WHERE po_number = %s",
            (a.po_number,),
        ).fetchone()
        if po is None:
            return None
        po["lines"] = ctx.conn.execute(
            "SELECT line_no, mpn, quantity, unit_price::text AS unit_price FROM ops.po_lines WHERE po_number = %s "
            "ORDER BY line_no",
            (a.po_number,),
        ).fetchall()
        return po

    @gw.tool("po.received", "read", PONumber)
    def received(ctx: Context, a: PONumber) -> dict[str, int]:
        """Quantities received against a PO, by part number."""
        return _received(ctx, a.po_number)

    @gw.tool("erp.invoiced", "read", PONumber)
    def invoiced(ctx: Context, a: PONumber) -> dict[str, int]:
        """Quantities already booked against a PO in the ERP, by part number."""
        total: Counter = Counter()
        for b in _erp_bookings(ctx, po_number=a.po_number):
            for ln in b["lines"]:
                total[ln["mpn"]] += ln["quantity"]
        return dict(total)

    @gw.tool("erp.find_booking", "read", FindBooking)
    def find_booking(ctx: Context, a: FindBooking) -> dict | None:
        """An existing booking of this supplier's invoice number, or null."""
        rows = _erp_bookings(ctx, supplier_id=a.supplier_id, invoice_number=a.invoice_number)
        return rows[0] if rows else None

    @gw.tool("erp.book_invoice", "action", Booking)
    def book(ctx: Context, a: Booking) -> dict:
        """Books a reconciled invoice in the ERP. Exactly once per idempotency key."""
        # Two invoices against the same PO can each pass the agent's quantity check and together
        # bill more than was delivered. Serialise per PO and check again at the moment of writing.
        ctx.conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"po:{a.po_number}",))
        received_qty = {norm_mpn(k): v for k, v in _received(ctx, a.po_number).items()}
        booked: Counter = Counter()
        for b in _erp_bookings(ctx, po_number=a.po_number):
            if b["idempotency_key"] != ctx.key:  # our own earlier attempt is not a competitor
                for ln in b["lines"]:
                    booked[norm_mpn(ln["mpn"])] += ln["quantity"]
        for ln in a.lines:
            if booked[norm_mpn(ln.mpn)] + ln.quantity > received_qty.get(norm_mpn(ln.mpn), 0):
                raise Conflict(f"billed for more than received: {ln.mpn} on PO {a.po_number}")
        try:
            with _erp(ctx) as http:
                r = http.post("/erp/bookings", json=a.model_dump(), headers={"Idempotency-Key": ctx.key})
        except httpx.TransportError as exc:
            raise RemoteError(f"ERP unreachable: {exc}") from exc
        if r.status_code >= 500:
            raise RemoteError(f"ERP {r.status_code}: retry with the same key")
        if r.status_code == 422:
            raise Conflict(r.text)
        r.raise_for_status()
        return r.json()

    @gw.tool("review.create", "action", Review)
    def review(ctx: Context, a: Review) -> dict:
        """Hands a case to a person, with the reason."""
        row = ctx.conn.execute(
            """INSERT INTO agents.reviews (key, kind, reason, payload) VALUES (%s, %s, %s, %s)
               ON CONFLICT (key) DO UPDATE SET reason = EXCLUDED.reason RETURNING id""",
            (ctx.key, a.kind, a.reason, _json(a.payload)),
        ).fetchone()
        return {"review_id": row["id"]}

    @gw.tool("mail.send", "action", Mail)
    def send(ctx: Context, a: Mail) -> dict:
        """Queues an email in the outbox, in the same transaction as its receipt."""
        message_id = f"<{hashlib.sha256(ctx.key.encode()).hexdigest()[:32]}@circuitfab.example>"
        ctx.conn.execute(
            """INSERT INTO agents.outbox (message_id, recipient, subject, body) VALUES (%s, %s, %s, %s)
               ON CONFLICT (message_id) DO NOTHING""",
            (message_id, a.recipient, a.subject, a.body),
        )
        return {"message_id": message_id}

    @gw.tool("po.overdue", "read", AsOf)
    def overdue(ctx: Context, a: AsOf) -> list[dict]:
        """Open POs past their promised date with goods still outstanding."""
        return ctx.conn.execute(
            """SELECT p.po_number, p.supplier_id, s.email, p.promised_on::text,
                      SUM(l.quantity)::int AS ordered,
                      COALESCE((SELECT SUM(g.quantity) FROM ops.goods_receipts g
                                 WHERE g.po_number = p.po_number), 0)::int AS received
                 FROM ops.purchase_orders p
                 JOIN ops.po_lines l USING (po_number)
                 JOIN ops.suppliers s ON s.id = p.supplier_id
                WHERE p.promised_on < %s
                GROUP BY p.po_number, p.supplier_id, s.email, p.promised_on
               HAVING COALESCE((SELECT SUM(g.quantity) FROM ops.goods_receipts g
                                 WHERE g.po_number = p.po_number), 0) < SUM(l.quantity)
                ORDER BY p.po_number""",
            (a.as_of,),
        ).fetchall()


def _json(payload: dict):
    from psycopg.types.json import Jsonb

    return Jsonb(payload)

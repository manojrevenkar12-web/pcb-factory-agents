"""A complete back-office environment: database, external systems, gateway, workers.

Used by the evaluation harness, the crash and concurrency tests, and the demo.
"""

from __future__ import annotations

import secrets
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import date

from psycopg.types.json import Jsonb

from pfa import db, workqueue
from pfa.backoffice import data as D
from pfa.backoffice import external, tools
from pfa.gateway.core import Gateway
from pfa.gateway.http import LocalClient, app
from pfa.judges import Judge
from pfa.sandbox import Worker
from pfa.serving import Server, free_port

INVOICE_SCOPES = [
    "inbox.get",
    "suppliers.by_email",
    "po.get",
    "po.received",
    "erp.invoiced",
    "erp.find_booking",
    "erp.book_invoice",
    "review.create",
    "judge.book-invoice",
]
CHASE_SCOPES = ["po.overdue", "mail.send"]


def load(database_url: str, dataset: D.Dataset, with_decisions: bool = True) -> None:
    with db.connect(database_url) as conn:
        for s in D.SUPPLIERS:
            conn.execute(
                "INSERT INTO ops.suppliers (id, name, email) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                (s.id, s.name, s.email),
            )
        for po in dataset.purchase_orders:
            conn.execute(
                """INSERT INTO ops.purchase_orders (po_number, supplier_id, currency, ordered_on, promised_on)
                   VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                (po.po_number, po.supplier_id, po.currency, po.ordered_on, po.promised_on),
            )
            for ln in po.lines:
                conn.execute(
                    """INSERT INTO ops.po_lines (po_number, line_no, mpn, quantity, unit_price)
                       VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                    (po.po_number, ln.line_no, ln.mpn, ln.quantity, ln.unit_price),
                )
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO ops.goods_receipts (po_number, mpn, quantity, received_on) VALUES (%s, %s, %s, %s)",
                dataset.goods_receipts,
            )
        for e in dataset.emails:
            conn.execute(
                """INSERT INTO ops.inbox (email_id, sender, subject, body, attachments, received_at)
                   VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                (e.email_id, e.sender, e.subject, e.body, Jsonb(e.attachments), e.received_on),
            )
            if with_decisions:
                conn.execute(
                    "INSERT INTO ops.human_decisions (email_id, decision) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (e.email_id, Jsonb(e.decision)),
                )
        conn.commit()


@dataclass
class Environment:
    database_url: str
    judge: Judge
    chaos: external.Chaos = field(default_factory=external.Chaos)
    http: bool = True

    def __enter__(self) -> Environment:
        self._stack = ExitStack()
        self.api_key = secrets.token_hex(16)
        self.external = self._stack.enter_context(
            Server(external.app(self.database_url, self.api_key, self.chaos), free_port())
        )
        self.gateway = Gateway(
            self.database_url,
            secrets.token_bytes(32),
            secrets={"erp_api_key": self.api_key},
            services={"external_url": self.external.url},
        )
        self._stack.callback(self.gateway.close)
        tools.register(self.gateway, self.judge)
        if self.http:
            self.gateway_server = self._stack.enter_context(Server(app(self.gateway), free_port()))
        return self

    def __exit__(self, *exc) -> None:
        self._stack.close()

    def local_client(self, shadow: bool = False) -> LocalClient:
        return LocalClient(self.gateway, self.gateway.mint(INVOICE_SCOPES + CHASE_SCOPES, shadow=shadow))

    def worker(self, name: str, **kw) -> Worker:
        return Worker(
            name,
            self.database_url,
            self.gateway,
            self.gateway_server.url,
            {"invoice": "invoice", "chase": "chase"},
            {"invoice": INVOICE_SCOPES, "chase": CHASE_SCOPES},
            **kw,
        )

    def enqueue_emails(self, email_ids: list[str]) -> None:
        with db.connect(self.database_url) as conn:
            for email_id in email_ids:
                workqueue.enqueue(conn, "invoice", email_id, {"email_id": email_id})
            conn.commit()

    def enqueue_chases(self, as_of: date) -> list[str]:
        client = self.local_client()
        pos = [r["po_number"] for r in client.read("po.overdue", as_of=as_of.isoformat())]
        with db.connect(self.database_url) as conn:
            for po in pos:
                workqueue.enqueue(
                    conn, "chase", f"{po}:{as_of}", {"po_number": po, "as_of": as_of.isoformat()}
                )
            conn.commit()
        return pos

    def relay_outbox(self) -> int:
        """Delivers queued email to the provider; the provider de-duplicates by Message-ID."""
        import httpx

        sent = 0
        with db.connect(self.database_url) as conn, httpx.Client(base_url=self.external.url) as http:
            while True:
                row = conn.execute(
                    """SELECT message_id, recipient, subject, body FROM agents.outbox
                        WHERE sent_at IS NULL ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1"""
                ).fetchone()
                if row is None:
                    conn.commit()
                    return sent
                r = http.post(
                    "/mail/messages",
                    headers={"Message-ID": row["message_id"], "X-Api-Key": self.api_key},
                    json={k: row[k] for k in ("recipient", "subject", "body")},
                )
                if r.status_code < 300:
                    conn.execute(
                        "UPDATE agents.outbox SET sent_at = now(), attempts = attempts + 1 "
                        "WHERE message_id = %s",
                        (row["message_id"],),
                    )
                    sent += 1
                else:
                    conn.execute(
                        "UPDATE agents.outbox SET attempts = attempts + 1 WHERE message_id = %s",
                        (row["message_id"],),
                    )
                conn.commit()

"""The back-office agent: one session per email, one per overdue purchase order.

It runs in a sandbox and talks only to the gateway. Its shape is the house rule: the judge reads
the email and returns a structured answer; plain code checks that answer against second sources
and makes every change, each under an idempotency key derived from the business object (never
from the session), so a killed and resumed session cannot book or email twice.
"""

from __future__ import annotations

import os
import random
import time
from datetime import date
from typing import Any

from pfa.backoffice.checks import Decision, decide
from pfa.backoffice.schemas import InvoiceAnswer
from pfa.gateway.client import Client, ToolFailed

OUR_DOMAIN = "circuitfab.example"


def crashpoint(name: str) -> None:
    """Fault injection for the crash tests: die here, without cleanup, like a killed VM."""
    points = os.environ.get("PFA_CRASH_AT", "")
    if name in points.split(",") and random.random() < float(os.environ.get("PFA_CRASH_PROB", "0.5")):
        os._exit(137)


class GatewaySources:
    """The checks' second sources, read through the gateway."""

    def __init__(self, client: Client):
        self.client = client

    def supplier_by_email(self, email: str) -> dict | None:
        return self.client.read("suppliers.by_email", email=email)

    def purchase_order(self, po_number: str) -> dict | None:
        return self.client.read("po.get", po_number=po_number)

    def received(self, po_number: str) -> dict[str, int]:
        return self.client.read("po.received", po_number=po_number)

    def invoiced(self, po_number: str) -> dict[str, int]:
        return self.client.read("erp.invoiced", po_number=po_number)

    def booking(self, supplier_id: str, invoice_number: str) -> dict | None:
        return self.client.read("erp.find_booking", supplier_id=supplier_id, invoice_number=invoice_number)


def model_text(email: dict) -> str:
    parts = [f"From: {email['sender']}", f"Subject: {email['subject']}", "", email["body"]]
    for att in email.get("attachments") or []:
        parts += ["", f"Attachment {att['filename']}:", att["content"]]
    return "\n".join(parts)


def handle_email(client: Client, email_id: str) -> dict[str, Any]:
    email = client.read("inbox.get", email_id=email_id)
    crashpoint("before_judge")
    verdict = client.judge(
        "book-invoice",
        model_text(email),
        data={
            "subject": email["subject"],
            "body": email["body"],
            "attachments": email.get("attachments") or [],
        },
    )
    answer = InvoiceAnswer.model_validate(verdict["answer"]) if verdict["answer"] else None
    decision = decide(email, answer, GatewaySources(client), OUR_DOMAIN, verdict.get("error"))
    crashpoint("after_decide")
    if os.environ.get("PFA_PAUSE_AFTER_DECIDE"):  # widens the check-then-act window in the race test
        time.sleep(float(os.environ["PFA_PAUSE_AFTER_DECIDE"]))

    replayed = False
    if decision.action == "book":
        b = decision.booking
        try:
            out = client.act("erp.book_invoice", key=f"{b['supplier_id']}:{b['invoice_number']}", **b)
            replayed = out["replayed"]
        except ToolFailed as exc:
            if exc.status != 409:
                raise
            decision = Decision("review", exc.detail, decision.supplier_id, None, decision.checks)
        crashpoint("after_book")
    if decision.action == "review":
        client.act(
            "review.create",
            key=f"email:{email_id}",
            kind="invoice",
            reason=decision.reason or "",
            payload={"email_id": email_id, "answer": verdict["answer"]},
        )
    return {
        "email_id": email_id,
        **decision.as_dict(),
        "replayed": replayed,
        "judge_cost_usd": verdict.get("cost_usd", 0.0),
        "judge_cached": verdict.get("cached", False),
    }


def chase(client: Client, po_number: str, as_of: str) -> dict[str, Any]:
    """Chases one overdue PO, at most once per ISO week."""
    rows = [r for r in client.read("po.overdue", as_of=as_of) if r["po_number"] == po_number]
    if not rows:
        return {"po_number": po_number, "action": "nothing to chase"}
    po = rows[0]
    year, week, _ = date.fromisoformat(as_of).isocalendar()
    outstanding = po["ordered"] - po["received"]
    out = client.act(
        "mail.send",
        key=f"chase:{po_number}:{year}-W{week:02d}",
        recipient=po["email"],
        subject=f"Overdue: purchase order {po_number}",
        body=(
            f"Hello,\n\npurchase order {po_number} was promised for {po['promised_on']} and "
            f"{outstanding:,} units are still outstanding. Please send a delivery date.\n\n"
            f"Thank you,\nCircuitFab purchasing"
        ),
    )
    crashpoint("after_chase")
    return {"po_number": po_number, "action": "chased", "replayed": out["replayed"]}


AGENTS = {
    "invoice": lambda client, payload: handle_email(client, payload["email_id"]),
    "chase": lambda client, payload: chase(client, payload["po_number"], payload["as_of"]),
}

"""Plain code that decides what happens to an invoice once a judge has read it.

The judge's answer is never trusted on its own. Every field that matters is checked against a
second source the judge did not see: the sender's address, the purchase order, the goods actually
received, what is already invoiced, and arithmetic. Anything that does not reconcile goes to a
person with the reason; only a fully reconciled invoice is booked.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Protocol

from pfa.backoffice.schemas import InvoiceAnswer
from pfa.backoffice.text import clean

PRICE_TOLERANCE = Decimal("0.005")  # 0.5%
TOTAL_TOLERANCE = Decimal("0.02")
CENT = Decimal("0.01")


def dec(x: Any) -> Decimal:
    return Decimal(str(x))


def norm_price(x: Any) -> str:
    return format(dec(x).normalize(), "f")


def norm_mpn(mpn: str) -> str:
    return re.sub(r"\s+", "", mpn).upper()


def norm_number(number: str) -> str:
    return number.strip().upper()


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text).upper()


def email_text(email: dict) -> str:
    """The email as sent now: subject, body without quoted history, and attachments."""
    parts = [email.get("subject", ""), clean(email["body"])]
    for att in email.get("attachments") or []:  # CSV cells, not comma-joined numbers
        rows = csv.reader(io.StringIO(att.get("content", "")))
        parts.append("\n".join(" | ".join(row) for row in rows))
    return "\n".join(parts)


_NUMBER = re.compile(r"\d[\d.,]*\d|\d")


def numbers_in(text: str) -> set[Decimal]:
    """Every number printed in the text, read both the English and the German way."""
    found: set[Decimal] = set()
    for token in _NUMBER.findall(text):
        for thousands, decimal in ((",", "."), (".", ",")):
            t = token.replace(thousands, "").replace(decimal, ".")
            try:
                found.add(Decimal(t).normalize())
            except ArithmeticError:
                continue
    return found


class Sources(Protocol):
    """The second sources, read through the gateway (or from memory in tests)."""

    def supplier_by_email(self, email: str) -> dict | None: ...
    def purchase_order(self, po_number: str) -> dict | None: ...  # {supplier_id, currency, lines: [...]}
    def received(self, po_number: str) -> dict[str, int]: ...
    def invoiced(self, po_number: str) -> dict[str, int]: ...
    def booking(self, supplier_id: str, invoice_number: str) -> dict | None: ...


@dataclass
class Decision:
    action: str  # book | review | ignore | duplicate
    reason: str | None = None
    supplier_id: str | None = None
    booking: dict | None = None
    checks: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "supplier_id": self.supplier_id,
            "booking": self.booking,
            "checks": self.checks,
        }


_FORWARDED_FROM = re.compile(r"^-+ ?Forwarded message ?-+\s*\nFrom:[^<\n]*<([^>]+)>", re.M | re.I)


def original_sender(sender: str, body: str, our_domain: str) -> str:
    """For mail a colleague forwarded, the supplier is in the forwarded header, not the sender."""
    if sender.lower().endswith("@" + our_domain):
        m = _FORWARDED_FROM.search(body)
        if m:
            return m.group(1).strip()
    return sender


def decide(
    email: dict,
    answer: InvoiceAnswer | None,
    sources: Sources,
    our_domain: str,
    judge_error: str | None = None,
) -> Decision:
    def review(reason: str, **kw: Any) -> Decision:
        return Decision("review", reason, checks=done, **kw)

    done: list[str] = []
    if answer is None:
        return review(f"could not read: {judge_error or 'no answer'}")
    if answer.document_type == "other":
        return Decision("ignore", "not an invoice")
    if answer.document_type == "credit_note":
        return review("credit note")

    supplier = sources.supplier_by_email(original_sender(email["sender"], email["body"], our_domain))
    if supplier is None:
        return review("sender is not a known supplier")
    sid = supplier["id"]
    done.append("sender is a known supplier")
    if not (
        answer.invoice_number
        and answer.po_number
        and answer.currency
        and answer.total is not None
        and answer.lines
    ):
        return review("incomplete", supplier_id=sid)
    number = norm_number(answer.invoice_number)
    # Identifiers have no other second source than the document: they must appear in it verbatim.
    document = squash(email_text(email))
    if squash(number) not in document:
        return review("invoice number not found in the email", supplier_id=sid)
    if not all(squash(line.mpn) in document for line in answer.lines):
        return review("part number not found in the email", supplier_id=sid)
    printed = numbers_in(email_text(email))
    amounts = [dec(answer.total)] + [dec(x) for x in (answer.shipping, answer.tax) if x]
    amounts += [dec(v) for line in answer.lines for v in (line.quantity, line.unit_price)]
    if not all(a.normalize() in printed for a in amounts):
        return review("an amount is not printed in the email", supplier_id=sid)
    done.append("identifiers and amounts appear in the email")

    # Read what is invoiced on the PO before looking for this invoice in the ledger. If another
    # session books the same invoice concurrently, its booking is then either in neither read (and
    # our booking replays its receipt) or in both (and this is a duplicate), never only in the first.
    invoiced_now = sources.invoiced(answer.po_number)
    existing = sources.booking(sid, number)
    if existing is not None:
        same = (
            dec(existing["total"]) == dec(answer.total).quantize(CENT)
            and existing["po_number"] == answer.po_number
        )
        if same:
            return Decision("duplicate", "already booked", sid, existing, done)
        return review("invoice number already used", supplier_id=sid)
    done.append("invoice number not booked before")

    po = sources.purchase_order(answer.po_number)
    if po is None:
        return review("unknown PO", supplier_id=sid)
    if po["supplier_id"] != sid:
        return review("PO belongs to another supplier", supplier_id=sid)
    done.append("PO exists and belongs to the sender")
    if answer.currency != po["currency"]:
        return review("currency differs from PO", supplier_id=sid)
    done.append("currency matches PO")

    po_lines = {norm_mpn(ln["mpn"]): ln for ln in po["lines"]}
    received = {norm_mpn(k): v for k, v in sources.received(answer.po_number).items()}
    invoiced = {norm_mpn(k): v for k, v in invoiced_now.items()}
    billed: dict[str, int] = {}
    lines_out = []
    computed = Decimal("0")
    for line in answer.lines:
        key = norm_mpn(line.mpn)
        po_line = po_lines.get(key)
        if po_line is None:
            return review(f"item not on PO: {line.mpn}", supplier_id=sid)
        price, po_price = dec(line.unit_price), dec(po_line["unit_price"])
        if abs(price - po_price) > po_price * PRICE_TOLERANCE:
            return review("price differs from PO", supplier_id=sid)
        billed[key] = billed.get(key, 0) + line.quantity
        if billed[key] > received.get(key, 0) - invoiced.get(key, 0):
            return review("billed for more than received", supplier_id=sid)
        computed += (price * line.quantity).quantize(CENT, rounding=ROUND_HALF_UP)
        lines_out.append({"mpn": po_line["mpn"], "quantity": line.quantity, "unit_price": norm_price(price)})
    done += ["every line is on the PO at the PO price", "nothing billed beyond what was received"]

    expected = computed + dec(answer.shipping) + dec(answer.tax)
    if abs(expected - dec(answer.total)) > TOTAL_TOLERANCE:
        return review("totals do not add up", supplier_id=sid)
    done.append("lines + shipping + tax = total")

    booking = {
        "supplier_id": sid,
        "invoice_number": number,
        "po_number": answer.po_number,
        "currency": answer.currency,
        "total": str(dec(answer.total).quantize(CENT)),
        "lines": sorted(lines_out, key=lambda x: x["mpn"]),
    }
    return Decision("book", None, sid, booking, done)


def same_booking(a: dict, b: dict) -> bool:
    """Bookings agree on every field that reaches the ledger."""
    if (a["supplier_id"], norm_number(a["invoice_number"]), a["po_number"], a["currency"]) != (
        b["supplier_id"],
        norm_number(b["invoice_number"]),
        b["po_number"],
        b["currency"],
    ):
        return False
    if dec(a["total"]) != dec(b["total"]):
        return False
    key = lambda ln: (norm_mpn(ln["mpn"]), int(ln["quantity"]), dec(ln["unit_price"]))  # noqa: E731
    return sorted(map(key, a["lines"])) == sorted(map(key, b["lines"]))

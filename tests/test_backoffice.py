from __future__ import annotations

import json
from decimal import Decimal

import pytest

from pfa.backoffice import data as D
from pfa.backoffice import rules
from pfa.backoffice.checks import decide, numbers_in, original_sender, same_booking
from pfa.backoffice.schemas import InvoiceAnswer, Line

DATASET = D.generate(240, 7)


class Memory:
    """Second sources from the generated month, with an in-memory ledger."""

    def __init__(self, dataset: D.Dataset):
        self.pos = {
            p.po_number: {
                "supplier_id": p.supplier_id,
                "currency": p.currency,
                "lines": [{"mpn": ln.mpn, "unit_price": str(ln.unit_price)} for ln in p.lines],
            }
            for p in dataset.purchase_orders
        }
        self.grn: dict[str, dict[str, int]] = {}
        for po, mpn, q, _ in dataset.goods_receipts:
            self.grn.setdefault(po, {}).setdefault(mpn, 0)
            self.grn[po][mpn] += q
        self.ledger: dict[tuple, dict] = {}

    def supplier_by_email(self, email):
        return next(({"id": s.id} for s in D.SUPPLIERS if s.email == email), None)

    def purchase_order(self, po):
        return self.pos.get(po)

    def received(self, po):
        return self.grn.get(po, {})

    def invoiced(self, po):
        out: dict[str, int] = {}
        for b in self.ledger.values():
            if b["po_number"] == po:
                for ln in b["lines"]:
                    out[ln["mpn"]] = out.get(ln["mpn"], 0) + ln["quantity"]
        return out

    def booking(self, sid, number):
        return self.ledger.get((sid, number))


def email_dict(e: D.Email) -> dict:
    return {"sender": e.sender, "subject": e.subject, "body": e.body, "attachments": e.attachments}


def test_the_month_has_every_hard_case():
    cases = {e.case for e in DATASET.emails}
    for c in (
        "clean",
        "forwarded",
        "quoted_thread",
        "price_change",
        "over_billed",
        "bad_total",
        "wrong_currency",
        "unknown_po",
        "other_suppliers_po",
        "credit_note",
        "duplicate_resend",
        "order_confirmation",
        "not_invoice",
    ):
        assert c in cases


@pytest.mark.parametrize("style", ["text", "german", "csv", "html"])
def test_rules_read_each_house_style(style):
    e = next(
        x
        for x in DATASET.emails
        if x.case == "clean" and D.BY_ID[x.decision["booking"]["supplier_id"]].style == style
    )
    a = rules.read(e.subject, e.body, e.attachments)
    b = e.decision["booking"]
    assert a.invoice_number == b["invoice_number"] and a.po_number == b["po_number"]
    assert Decimal(str(a.total)) == Decimal(b["total"])


def test_rules_ignore_the_quoted_older_invoice():
    e = next(x for x in DATASET.emails if x.case == "quoted_thread" and "wrote:" in x.body)
    assert (
        rules.read(e.subject, e.body, e.attachments).invoice_number == e.decision["booking"]["invoice_number"]
    )


def test_numbers_are_read_both_ways():
    found = numbers_in("Total 1.234,50 EUR and 2,000 pcs at 0.0021")
    assert {Decimal("1234.5"), Decimal("2000"), Decimal("0.0021")} <= found


def test_forwarded_mail_is_attributed_to_the_supplier_not_the_colleague():
    e = next(x for x in DATASET.emails if x.case == "forwarded")
    assert original_sender(e.sender, e.body, D.OUR_DOMAIN) != e.sender


def test_replay_in_memory_never_takes_a_wrong_action():
    src = Memory(DATASET)
    wrong = 0
    for e in DATASET.emails:
        try:
            answer, err = rules.read(e.subject, e.body, e.attachments), None
        except ValueError as exc:
            answer, err = None, str(exc)
        d = decide(email_dict(e), answer, src, D.OUR_DOMAIN, err)
        if d.action == "book":
            src.ledger[(d.booking["supplier_id"], d.booking["invoice_number"])] = d.booking
            wrong += not (e.decision["action"] == "book" and same_booking(d.booking, e.decision["booking"]))
        elif d.action in ("ignore", "duplicate"):
            wrong += e.decision["action"] not in ("ignore", "duplicate")
        elif e.decision["action"] == "book":  # the clerk books what the agent sent to review
            src.ledger[(e.decision["booking"]["supplier_id"], e.decision["booking"]["invoice_number"])] = (
                e.decision["booking"]
            )
    assert wrong == 0


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("total", 99999.0, "an amount is not printed"),
        ("invoice_number", "BW-26-99999", "invoice number not found"),
        ("currency", "USD", "currency differs"),
        ("po_number", "4509999", "unknown PO"),
    ],
)
def test_a_wrong_reading_never_reaches_the_ledger(field, value, reason):
    e = next(x for x in DATASET.emails if x.case == "clean" and "brightwire" in x.sender)
    answer = rules.read(e.subject, e.body, e.attachments)
    setattr(answer, field, value)
    d = decide(email_dict(e), answer, Memory(DATASET), D.OUR_DOMAIN)
    assert d.action == "review" and reason in d.reason


def test_a_price_within_tolerance_but_not_printed_is_refused():
    e = next(x for x in DATASET.emails if x.case == "clean" and "brightwire" in x.sender)
    answer = rules.read(e.subject, e.body, e.attachments)
    answer.lines[0] = Line(
        mpn=answer.lines[0].mpn,
        quantity=answer.lines[0].quantity,
        unit_price=answer.lines[0].unit_price * 1.004,
    )
    assert decide(email_dict(e), answer, Memory(DATASET), D.OUR_DOMAIN).action == "review"


def test_order_confirmations_and_credit_notes_are_not_booked():
    src = Memory(DATASET)
    for case, action in (("order_confirmation", "ignore"), ("credit_note", "review")):
        e = next(x for x in DATASET.emails if x.case == case)
        try:
            answer = rules.read(e.subject, e.body, e.attachments)
        except ValueError:
            answer = None
        assert decide(email_dict(e), answer, src, D.OUR_DOMAIN).action == action


def test_judge_answer_schema_round_trips():
    a = InvoiceAnswer.model_validate_json(
        json.dumps(
            {
                "document_type": "invoice",
                "invoice_number": "X1",
                "po_number": "4500100",
                "currency": "EUR",
                "lines": [],
                "shipping": 0,
                "tax": 0,
                "total": None,
            }
        )
    )
    assert a.total is None

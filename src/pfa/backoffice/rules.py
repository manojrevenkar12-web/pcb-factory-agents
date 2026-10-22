"""The rule-based invoice reader: the baseline every model has to beat.

Generic regular expressions and table heuristics, no per-supplier templates. It is free, fast and
deterministic, and it fails in the ways rule sets fail: an unusual layout, a quoted older invoice,
a number written the German way in an unexpected place. When it cannot find the essentials it
refuses, and the email goes to a person.
"""

from __future__ import annotations

import csv
import io
import re

from pfa.backoffice.schemas import InvoiceAnswer, Line
from pfa.backoffice.text import clean
from pfa.judges import FunctionJudge, Request

_NUM = r"[\d][\d.,]*"
_MPN = r"[A-Z0-9][A-Z0-9().,/+-]*\d[A-Z0-9().,/+-]*"


def to_number(text: str, german: bool) -> float:
    t = text.strip().rstrip(".,")
    t = t.replace(".", "").replace(",", ".") if german else t.replace(",", "")
    return float(t)


def _currency(text: str) -> str | None:
    m = re.search(r"\b(EUR|GBP|USD)\b|([€£$])", text)
    if not m:
        return None
    return m.group(1) or {"€": "EUR", "£": "GBP", "$": "USD"}[m.group(2)]


def _amount_after(pattern: str, text: str, german: bool) -> float | None:
    for line in text.splitlines():
        if re.search(pattern, line, re.I):
            nums = re.findall(_NUM, line.split(":", 1)[-1] if ":" in line else line)
            nums = [n for n in nums if not re.fullmatch(r"\d{1,2}%?", n)] or nums
            if nums:
                return to_number(nums[-1], german)
    return None


def read(subject: str, body: str, attachments: list[dict]) -> InvoiceAnswer:
    text = clean(body)
    full = f"{subject}\n{text}"
    low = full.lower()
    german = bool(re.search(r"mwst|gesamtbetrag|\bstk\b", low))

    if re.search(r"credit note|gutschrift", low):
        doc = "credit_note"
    elif re.search(r"\binvoice\b|\brechnung\b|\bbilling\b", low) and not re.search(
        r"order confirmation|auftragsbest", low
    ):
        doc = "invoice"
    else:
        return InvoiceAnswer(
            document_type="other",
            invoice_number=None,
            po_number=None,
            currency=None,
            lines=[],
            shipping=0,
            tax=0,
            total=None,
        )

    number = None
    m = re.search(
        r"(?:invoice|rechnung|credit note|gutschrift)\b[^\n]*?(?:nr\.?|no\.?|#|number)?\s*[:#]?\s*"
        r"((?:[A-Z]{2,4}[/-])+[A-Z0-9/-]*\d{3,}|[A-Z]{2,4}\d{5,})",
        full,
        re.I,
    )
    if m:
        number = m.group(1).strip().rstrip(".,")
    po = None
    m = re.search(r"(?:PO|order|bestellung)\b\W{0,4}(4500[\s-]?\d{3})", full, re.I)
    if m:
        po = re.sub(r"\D", "", m.group(1))

    lines: list[Line] = []
    shipping = tax = 0.0
    total = None
    if attachments:
        for att in attachments:
            for row in csv.DictReader(io.StringIO(att["content"])):
                key = (row.get("mpn") or "").strip()
                amount = float(row["line_total"]) if row.get("line_total") else 0.0
                if key.upper() == "SHIPPING":
                    shipping = amount
                elif key.upper().startswith("VAT"):
                    tax = amount
                elif key.upper() == "TOTAL":
                    total = amount
                elif key:
                    lines.append(Line(mpn=key, quantity=int(row["qty"]), unit_price=float(row["unit_price"])))
    else:
        prose = re.findall(rf"({_NUM})\s*pcs\s+(\S+)\s*@\s*[A-Z$€£]*\s*({_NUM})", text)
        for qty, mpn, price in prose:
            lines.append(
                Line(mpn=mpn, quantity=int(to_number(qty, german)), unit_price=to_number(price, german))
            )
        if not prose:
            row = re.compile(
                rf"^\s*(?:\d+\s+)?({_MPN})\s*\|?\s+({_NUM})(?:\s*Stk)?\s*\|?\s+[$€£]?({_NUM})\s*\|?"
                rf"\s+[$€£]?({_NUM})\s*\|?\s*$",
                re.M,
            )
            for mpn, qty, price, _ in row.findall(text):
                lines.append(
                    Line(mpn=mpn, quantity=int(to_number(qty, german)), unit_price=to_number(price, german))
                )
        shipping = _amount_after(r"shipping|freight|versand", text, german) or 0.0
        tax = _amount_after(r"\bvat\b|mwst|sales tax", text, german) or 0.0
        total = _amount_after(r"(?<!sub)total(?!.*zwischen)|gesamtbetrag|in total", text, german)
        if total is None:
            m = re.search(rf"so\s+[A-Z]{{3}}\s+({_NUM})\s+in total", text)
            total = to_number(m.group(1), german) if m else None
    if "VAT (" in text and not tax:
        m = re.search(rf"VAT \(\d+%\) of [A-Z]{{3}} ({_NUM})", text)
        tax = to_number(m.group(1), german) if m else 0.0

    if number is None or po is None or not lines or total is None:
        missing = [
            n
            for n, v in (("invoice number", number), ("PO", po), ("lines", lines), ("total", total))
            if not v
        ]
        raise ValueError(f"could not read {', '.join(missing)}")
    return InvoiceAnswer(
        document_type=doc,
        invoice_number=number,
        po_number=po,
        currency=_currency(full),
        lines=lines,
        shipping=shipping,
        tax=tax,
        total=total,
    )


def judge() -> FunctionJudge:
    def fn(request: Request) -> InvoiceAnswer:
        d = request.data
        return read(d["subject"], d["body"], d.get("attachments", []))

    return FunctionJudge("rules", fn, version="1")

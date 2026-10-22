"""A month of a small factory's supplier email, with what a person decided for each one.

Five suppliers write invoices five different ways (a plain-text table, German number formats, a CSV
attachment, an HTML table, prose). Mixed in are the cases that make the job hard: price changes,
billing for more than was delivered, totals that do not add up, the wrong currency, unknown or
someone else's purchase orders, credit notes, re-sent duplicates, forwarded threads with an older
invoice quoted underneath, and order confirmations that look like invoices but are not.

Every email carries the decision a careful accounts-payable clerk would make, so agents can be
replayed against it. Suppliers are fictional; the part numbers are real manufacturer parts.
"""

from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")
OUR_DOMAIN = "circuitfab.example"
AP_MAILBOX = f"ap@{OUR_DOMAIN}"


@dataclass(frozen=True)
class Supplier:
    id: str
    name: str
    email: str
    currency: str
    vat: Decimal
    style: str
    invoice_prefix: str


SUPPLIERS = [
    Supplier(
        "brightwire",
        "Brightwire Components Ltd",
        "billing@brightwire.example",
        "EUR",
        Decimal("0.19"),
        "text",
        "BW-26-",
    ),
    Supplier(
        "kestrel",
        "Kestrel Passives GmbH",
        "rechnung@kestrel-passives.example",
        "EUR",
        Decimal("0.19"),
        "german",
        "RE-2026-",
    ),
    Supplier(
        "halden",
        "Halden Semiconductor Ltd",
        "accounts@haldensemi.example",
        "GBP",
        Decimal("0.20"),
        "csv",
        "HSL/INV/",
    ),
    Supplier(
        "northwind",
        "Northwind Electronics Inc",
        "ar@northwind-elec.example",
        "USD",
        Decimal("0"),
        "html",
        "NW",
    ),
    Supplier(
        "corvid", "Corvid Connectors BV", "finance@corvid.example", "EUR", Decimal("0.19"), "prose", "CC-INV-"
    ),
]
BY_ID = {s.id: s for s in SUPPLIERS}

# Real manufacturer part numbers with plausible distributor prices.
CATALOG: dict[str, list[tuple[str, Decimal]]] = {
    "brightwire": [
        ("RC0603FR-0710KL", Decimal("0.0021")),
        ("GRM188R71H104KA93D", Decimal("0.0048")),
        ("LM358DR", Decimal("0.1120")),
        ("NE555DR", Decimal("0.1450")),
        ("SN74LVC1G14DBVR", Decimal("0.0890")),
        ("BSS138", Decimal("0.0310")),
    ],
    "kestrel": [
        ("ERJ-3EKF1001V", Decimal("0.0035")),
        ("C0603C104K5RACTU", Decimal("0.0052")),
        ("CL10A106KP8NNNC", Decimal("0.0185")),
        ("RC0402FR-0710KL", Decimal("0.0016")),
        ("GRM155R71C104KA88D", Decimal("0.0029")),
    ],
    "halden": [
        ("STM32F103C8T6", Decimal("2.3100")),
        ("ATMEGA328P-AU", Decimal("1.6400")),
        ("TPS62160DSGR", Decimal("0.9200")),
        ("TLV70033DDCR", Decimal("0.2100")),
        ("MCP2551-I/SN", Decimal("0.7800")),
    ],
    "northwind": [
        ("ESP32-WROOM-32E", Decimal("3.1500")),
        ("AMS1117-3.3", Decimal("0.0950")),
        ("1N4148W-7-F", Decimal("0.0210")),
        ("LTST-C190KGKT", Decimal("0.0360")),
    ],
    "corvid": [
        ("USB4105-GF-A", Decimal("0.4200")),
        ("B2B-PH-K-S(LF)(SN)", Decimal("0.0610")),
        ("1-1734248-0", Decimal("0.0840")),
        ("S2B-PH-K-S(LF)(SN)", Decimal("0.0570")),
    ],
}

PACK = [100, 250, 500, 1000, 2000, 2500, 3000, 4000, 5000, 10000]


def money(x: Decimal) -> Decimal:
    return x.quantize(CENT, rounding=ROUND_HALF_UP)


@dataclass
class POLine:
    line_no: int
    mpn: str
    quantity: int
    unit_price: Decimal


@dataclass
class PurchaseOrder:
    po_number: str
    supplier_id: str
    currency: str
    ordered_on: date
    promised_on: date
    lines: list[POLine]


@dataclass
class InvoiceLine:
    mpn: str
    quantity: int
    unit_price: Decimal

    @property
    def amount(self) -> Decimal:
        return money(self.unit_price * self.quantity)


@dataclass
class Invoice:
    supplier: Supplier
    number: str
    po_number: str
    currency: str
    issued: date
    lines: list[InvoiceLine]
    shipping: Decimal
    credit: bool = False
    total_override: Decimal | None = None

    @property
    def subtotal(self) -> Decimal:
        return sum((ln.amount for ln in self.lines), Decimal("0"))

    @property
    def tax(self) -> Decimal:
        return money((self.subtotal + self.shipping) * self.supplier.vat)

    @property
    def total(self) -> Decimal:
        return (
            self.total_override
            if self.total_override is not None
            else self.subtotal + self.shipping + self.tax
        )


@dataclass
class Email:
    email_id: str
    sender: str
    subject: str
    body: str
    received_on: date
    attachments: list[dict] = field(default_factory=list)
    #: What the person decided: action book / review / ignore / duplicate, and the booking or reason.
    decision: dict = field(default_factory=dict)
    case: str = ""


@dataclass
class Dataset:
    purchase_orders: list[PurchaseOrder]
    goods_receipts: list[tuple[str, str, int, date]]  # po, mpn, quantity, received_on
    emails: list[Email]


# ---- rendering one invoice in a supplier's house style ----------------------------------------------


def _po_ref(po: str, rng: random.Random) -> str:
    digits = po.removeprefix("4500")
    return rng.choice([f"PO {po}", f"PO#{po}", f"PO 4500-{digits}", f"order {po}", f"PO: 4500 {digits}"])


def _en(x: Decimal, places: int = 2) -> str:
    return f"{x:,.{places}f}"


def _de(x: Decimal, places: int = 2) -> str:
    return f"{x:,.{places}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _qty_de(q: int) -> str:
    return f"{q:,}".replace(",", ".")


def _price_places(p: Decimal) -> int:
    return max(2, -p.normalize().as_tuple().exponent)


def render(inv: Invoice, rng: random.Random) -> tuple[str, str, list[dict]]:
    """Returns subject, body, attachments."""
    s, cur = inv.supplier, inv.currency
    kind = "Credit note" if inv.credit else "Invoice"
    ref = _po_ref(inv.po_number, rng)
    if s.style == "text":
        rows = "\n".join(
            f"{ln.mpn:<22}{ln.quantity:>8,}{_en(ln.unit_price, _price_places(ln.unit_price)):>14}"
            f"{_en(ln.amount):>12}"
            for ln in inv.lines
        )
        body = (
            f"Dear customer,\n\nplease find our {kind.lower()} below.\n\n{kind} {inv.number}\n"
            f"Date: {inv.issued.isoformat()}\nYour order: {ref}\n\n"
            f"{'Part No.':<22}{'Qty':>8}{'Unit price':>14}{'Amount':>12}\n{rows}\n\n"
            f"{'Subtotal':<44}{_en(inv.subtotal):>12}\n{'Shipping':<44}{_en(inv.shipping):>12}\n"
            f"{'VAT ' + str(int(s.vat * 100)) + '%':<44}{_en(inv.tax):>12}\n"
            f"{'Total ' + cur:<44}{_en(inv.total):>12}\n\nPayment terms: 30 days net.\n{s.name}"
        )
        return f"{kind} {inv.number}", body, []
    if s.style == "german":
        rows = "\n".join(
            f"{i}  {ln.mpn}  {_qty_de(ln.quantity)} Stk  {_de(ln.unit_price, _price_places(ln.unit_price))}"
            f"  {_de(ln.amount)}"
            for i, ln in enumerate(inv.lines, 1)
        )
        title = "Gutschrift / Credit note" if inv.credit else "Rechnung / Invoice"
        body = (
            f"Sehr geehrte Damen und Herren,\n\nanbei unsere {title}.\n\n{title} Nr. {inv.number}\n"
            f"Datum: {inv.issued.strftime('%d.%m.%Y')}\nBestellung / {ref}\n\n"
            f"Pos  Artikel  Menge  Einzelpreis  Gesamt\n{rows}\n\n"
            f"Zwischensumme: {_de(inv.subtotal)} {cur}\nVersand / Shipping: {_de(inv.shipping)} {cur}\n"
            f"MwSt. {int(s.vat * 100)}%: {_de(inv.tax)} {cur}\nGesamtbetrag / Total: {_de(inv.total)} {cur}\n\n"
            f"Mit freundlichen Grüßen\n{s.name}"
        )
        return f"{title} {inv.number}", body, []
    if s.style == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["mpn", "qty", "unit_price", "line_total"])
        for ln in inv.lines:
            w.writerow([ln.mpn, ln.quantity, f"{ln.unit_price}", f"{ln.amount}"])
        w.writerow(["SHIPPING", "", "", f"{inv.shipping}"])
        w.writerow([f"VAT {int(s.vat * 100)}%", "", "", f"{inv.tax}"])
        w.writerow(["TOTAL", "", "", f"{inv.total}"])
        body = (
            f"Hello,\n\nPlease find attached {kind.lower()} {inv.number} for your {ref}.\n"
            f"Amount due: {cur} {_en(inv.total)}.\n\nKind regards,\nAccounts, {s.name}"
        )
        name = inv.number.replace("/", "-") + ".csv"
        return f"{s.name}: {kind} {inv.number}", body, [{"filename": name, "content": buf.getvalue()}]
    if s.style == "html":
        sym = {"USD": "$", "EUR": "&euro;", "GBP": "&pound;"}[cur]
        rows = "".join(
            f"<tr><td>{ln.mpn}</td><td>{ln.quantity}</td><td>{sym}{ln.unit_price}</td>"
            f"<td>{sym}{_en(ln.amount)}</td></tr>"
            for ln in inv.lines
        )
        body = (
            f"<html><body><p>{kind} <b>#{inv.number}</b> &middot; {ref} &middot; {inv.issued:%b %d, %Y}</p>"
            f"<table><tr><th>Item</th><th>Qty</th><th>Price</th><th>Ext.</th></tr>{rows}"
            f"<tr><td colspan=3>Freight</td><td>{sym}{_en(inv.shipping)}</td></tr>"
            f"<tr><td colspan=3>Sales tax</td><td>{sym}{_en(inv.tax)}</td></tr>"
            f"<tr><td colspan=3><b>Total due ({cur})</b></td><td><b>{sym}{_en(inv.total)}</b></td></tr></table>"
            f"<p>Thank you for your business. {s.name}</p></body></html>"
        )
        return f"Your {kind.lower()} {inv.number} from Northwind", body, []
    # prose
    lines = "\n".join(
        f"- {ln.quantity:,} pcs {ln.mpn} @ {cur} {ln.unit_price} = {cur} {_en(ln.amount)}" for ln in inv.lines
    )
    body = (
        f"Hi team,\n\nwe have shipped the following against your {ref} and are billing it under "
        f"{kind.lower()} {inv.number}:\n\n{lines}\n\nPlus shipping of {cur} {_en(inv.shipping)} and "
        f"VAT ({int(s.vat * 100)}%) of {cur} {_en(inv.tax)}, so {cur} {_en(inv.total)} in total.\n\n"
        f"Cheers,\nCorvid finance"
    )
    return f"{kind} {inv.number} / your {ref}", body, []


def _quote(text: str) -> str:
    return "\n".join("> " + ln for ln in text.splitlines())


# ---- the month ------------------------------------------------------------------------------------------


def booking(inv: Invoice) -> dict:
    return {
        "supplier_id": inv.supplier.id,
        "invoice_number": inv.number,
        "po_number": inv.po_number,
        "currency": inv.currency,
        "total": str(inv.total),
        "lines": sorted(
            [{"mpn": ln.mpn, "quantity": ln.quantity, "unit_price": str(ln.unit_price)} for ln in inv.lines],
            key=lambda x: x["mpn"],
        ),
    }


CASES = {
    # case: share of invoice emails, decision
    "clean": 0.50,
    "partial_delivery": 0.08,
    "forwarded": 0.06,
    "quoted_thread": 0.05,
    "price_change": 0.06,
    "over_billed": 0.05,
    "bad_total": 0.04,
    "wrong_currency": 0.03,
    "unknown_po": 0.03,
    "other_suppliers_po": 0.02,
    "credit_note": 0.03,
    "number_reused": 0.02,
    "order_confirmation": 0.03,
}


def generate(n_emails: int = 240, seed: int = 7, start: date = date(2026, 9, 1)) -> Dataset:
    rng = random.Random(seed)
    pos: list[PurchaseOrder] = []
    grns: list[tuple[str, str, int, date]] = []
    emails: list[Email] = []
    booked: list[Invoice] = []
    counters = {s.id: rng.randint(100, 900) for s in SUPPLIERS}
    next_po = [4500100]

    def new_po(supplier: Supplier, day: date) -> PurchaseOrder:
        items = rng.sample(CATALOG[supplier.id], k=rng.randint(1, min(4, len(CATALOG[supplier.id]))))
        lines = [POLine(i + 1, mpn, rng.choice(PACK), price) for i, (mpn, price) in enumerate(items)]
        po = PurchaseOrder(
            str(next_po[0]),
            supplier.id,
            supplier.currency,
            day - timedelta(days=rng.randint(10, 30)),
            day - timedelta(days=rng.randint(0, 8)),
            lines,
        )
        next_po[0] += 1
        pos.append(po)
        return po

    def new_number(s: Supplier) -> str:
        counters[s.id] += rng.randint(1, 9)
        return (
            f"{s.invoice_prefix}{counters[s.id]:05d}"
            if s.style != "html"
            else f"NW{1030000 + counters[s.id]}"
        )

    def invoice_for(po: PurchaseOrder, day: date, fraction: float = 1.0) -> Invoice:
        s = BY_ID[po.supplier_id]
        lines = [InvoiceLine(ln.mpn, max(1, int(ln.quantity * fraction)), ln.unit_price) for ln in po.lines]
        return Invoice(
            s,
            new_number(s),
            po.po_number,
            po.currency,
            day,
            lines,
            Decimal(rng.choice(["0", "8.50", "12.00", "15.90", "24.00"])),
        )

    def receive(inv: Invoice, day: date, short: dict[str, int] | None = None) -> None:
        for ln in inv.lines:
            qty = ln.quantity - (short or {}).get(ln.mpn, 0)
            grns.append((inv.po_number, ln.mpn, qty, day - timedelta(days=rng.randint(0, 3))))

    def add(
        sender: str,
        subject: str,
        body: str,
        day: date,
        decision: dict,
        case: str,
        attachments: list[dict] | None = None,
    ) -> None:
        emails.append(
            Email(f"m{len(emails) + 1:04d}", sender, subject, body, day, attachments or [], decision, case)
        )

    day = start
    names, weights = zip(*CASES.items(), strict=True)
    while len(emails) < n_emails:
        day = day + timedelta(days=rng.choice([0, 0, 0, 1]))
        if rng.random() < 0.10 and booked:  # a supplier re-sends an invoice we already have
            inv = rng.choice(booked)
            subject, body, att = render(inv, rng)
            add(
                inv.supplier.email,
                "Reminder: " + subject,
                "Friendly reminder, copy below.\n\n" + body,
                day,
                {"action": "duplicate", "booking": booking(inv)},
                "duplicate_resend",
                att,
            )
            continue
        if rng.random() < 0.08:  # mail that is not an invoice at all
            s = rng.choice(SUPPLIERS)
            kind = rng.choice(["shipping", "newsletter"])
            if kind == "shipping":
                body = (
                    f"Your order {rng.randint(4500100, 4500400)} has shipped with DHL, tracking "
                    f"JD0146{rng.randint(10**9, 10**10 - 1)}. Expected delivery in 2-3 days."
                )
                subject = "Shipping notification"
            else:
                body = "New in stock this month: low-ESR polymer capacitors and USB-C receptacles. Ask us for samples!"
                subject = f"{s.name} newsletter"
            add(s.email, subject, body, day, {"action": "ignore"}, "not_invoice")
            continue
        case = rng.choices(names, weights)[0]
        s = rng.choice(SUPPLIERS)
        po = new_po(s, day)
        inv = invoice_for(po, day)
        decision: dict
        sender = s.email
        if case == "partial_delivery":
            inv = invoice_for(po, day, fraction=rng.choice([0.5, 0.6, 0.8]))
            receive(inv, day)
            decision = {"action": "book"}
        elif case == "price_change":
            ln = rng.choice(inv.lines)
            ln.unit_price = (ln.unit_price * Decimal("1.08")).quantize(Decimal("0.0001"))
            receive(inv, day)
            decision = {"action": "review", "reason": "price differs from PO"}
        elif case == "over_billed":
            receive(inv, day, short={inv.lines[0].mpn: max(1, inv.lines[0].quantity // 5)})
            decision = {"action": "review", "reason": "billed for more than received"}
        elif case == "bad_total":
            receive(inv, day)
            inv.total_override = inv.total + Decimal(rng.choice(["10.00", "-4.20", "100.00", "0.90"]))
            decision = {"action": "review", "reason": "totals do not add up"}
        elif case == "wrong_currency":
            receive(inv, day)
            inv.currency = rng.choice([c for c in ("EUR", "GBP", "USD") if c != s.currency])
            decision = {"action": "review", "reason": "currency differs from PO"}
        elif case == "unknown_po":
            receive(inv, day)
            inv.po_number = str(4509000 + rng.randint(0, 999))
            decision = {"action": "review", "reason": "unknown PO"}
        elif case == "other_suppliers_po":
            other = rng.choice([x for x in pos if x.supplier_id != s.id] or [po])
            receive(inv, day)
            inv.po_number = other.po_number
            decision = {"action": "review", "reason": "PO belongs to another supplier"}
        elif case == "credit_note":
            receive(inv, day)
            inv.credit = True
            decision = {"action": "review", "reason": "credit note"}
        elif case == "number_reused" and [b for b in booked if b.supplier.id == s.id]:
            receive(inv, day)
            inv.number = rng.choice([b for b in booked if b.supplier.id == s.id]).number
            decision = {"action": "review", "reason": "invoice number already used"}
        elif case == "order_confirmation":
            subject, body, att = render(inv, rng)
            body = body.replace("Invoice", "Order confirmation").replace("invoice", "order confirmation")
            body = body.replace("Rechnung", "Auftragsbestätigung").replace("Payment terms: 30 days net.", "")
            add(sender, "Order confirmation " + po.po_number, body, day, {"action": "ignore"}, case)
            continue
        else:
            if case == "number_reused":
                case = "clean"
            receive(inv, day)
            decision = {"action": "book"}

        subject, body, att = render(inv, rng)
        if case == "forwarded":
            body = (
                f"Hi AP, this one came to me by mistake, can you book it?\nThanks, Jana\n\n"
                f"---------- Forwarded message ---------\nFrom: {s.name} <{s.email}>\n"
                f"Date: {day.isoformat()}\nSubject: {subject}\n\n{body}"
            )
            sender, subject = f"jana.weber@{OUR_DOMAIN}", "Fwd: " + subject
        elif case == "quoted_thread":
            old = [b for b in booked if b.supplier.id == s.id]
            if old:
                _, old_body, _ = render(old[-1], rng)
                body = f"{body}\n\nOn {old[-1].issued.isoformat()}, {s.name} wrote:\n{_quote(old_body)}"
            subject = "Re: " + subject
        if decision["action"] == "book":
            decision["booking"] = booking(inv)
            booked.append(inv)
        add(sender, subject, body, day, decision, case, att)
    return Dataset(pos, grns, emails)

"""Proving the back-office agent before it runs alone.

1. **Replay**: every past email, in order, against the decision a person made.
2. **Fault injection**: the same replay with a judge that is wrong 20% of the time. The checks,
   not the judge, must keep wrong actions at zero.
3. **Shadow run**: a new month of mail; the agent runs beside the clerk with a shadow token
   (nothing it does takes effect) while the clerk's own bookings land in the ERP.
4. **Crash test**: sandboxed sessions killed at random points, a worker that vanishes, and an ERP
   that fails before and after acting. Every invoice must be booked exactly once, every chase
   email sent exactly once.
5. **Race**: two invoices that each bill a PO's full delivery, processed at the same moment.
"""

from __future__ import annotations

import os
import random
import signal
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import httpx
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pfa import db, evals
from pfa.backoffice import data as D
from pfa.backoffice import external, rules
from pfa.backoffice.agent import handle_email
from pfa.backoffice.checks import same_booking
from pfa.backoffice.env import Environment, load
from pfa.backoffice.schemas import InvoiceAnswer
from pfa.judges import Judge
from pfa.telemetry import add_exporter

CLOSABLE = {"book", "duplicate", "ignore"}


def score(email: D.Email, out: dict, cost: float = 0.0, latency: float = 0.0) -> evals.Outcome:
    exp, got = email.decision["action"], out["action"]
    if got == "book":
        correct = exp == "book" and same_booking(out["booking"], email.decision["booking"])
        wrong = not correct
    else:
        correct = got == exp or {got, exp} <= {"ignore", "duplicate"}
        wrong = not correct and got != "review"  # ignoring something that needed action
    needless = got == "review" and exp in CLOSABLE
    return evals.Outcome(
        email.email_id, exp, got, correct, wrong, needless, cost, latency, out.get("reason") or ""
    )


def replay(database_url: str, judge: Judge, emails: int = 240, seed: int = 7) -> tuple[evals.Summary, list]:
    db.reset(database_url)
    dataset = D.generate(emails, seed)
    load(database_url, dataset)
    outcomes = []
    with Environment(database_url, judge, http=False) as env:
        client = env.local_client()
        erp = httpx.Client(base_url=env.external.url, headers={"X-Api-Key": env.api_key})
        for email in dataset.emails:
            before, started = client.cost_usd, time.perf_counter()
            out = handle_email(client, email.email_id)
            outcomes.append(score(email, out, client.cost_usd - before, time.perf_counter() - started))
            if out["action"] == "review" and email.decision["action"] == "book":
                # The person works the review queue and books it, as they did at the time, so later
                # emails (re-sends, re-used numbers) meet the ledger as it really was.
                erp.post(
                    "/erp/bookings",
                    json=email.decision["booking"],
                    headers={"Idempotency-Key": f"human:{email.email_id}"},
                ).raise_for_status()
        config = env.gateway.skills["book-invoice"].judge.config_hash(_probe_request())
    by_case = Counter()
    for email, o in zip(dataset.emails, outcomes, strict=True):
        by_case[(email.case, "ok" if o.correct else ("WRONG" if o.wrong_action else "review"))] += 1
    summary = evals.summarize(
        "book-invoice",
        judge.name,
        config,
        outcomes,
        CLOSABLE,
        by_case={f"{k[0]}|{k[1]}": v for k, v in sorted(by_case.items())},
    )
    return summary, outcomes


def _probe_request():
    from pfa.judges import Request

    return Request("book-invoice", "", InvoiceAnswer)


def corrupt(answer: InvoiceAnswer, rng: random.Random) -> InvoiceAnswer:
    """The mistakes a reader makes: a digit, a line, a decimal point, the wrong document."""
    kind = rng.choice(["quantity", "price", "po", "number", "drop_line", "total", "currency", "type"])
    if kind == "quantity" and answer.lines:
        ln = rng.choice(answer.lines)
        ln.quantity = max(1, ln.quantity + rng.choice([-1, 1]) * max(1, ln.quantity // 10))
    elif kind == "price" and answer.lines:
        ln = rng.choice(answer.lines)
        ln.unit_price = round(ln.unit_price * rng.choice([10, 0.1, 1.004]), 6)
    elif kind == "po" and answer.po_number:
        digits = list(answer.po_number)
        i = rng.randrange(len(digits))
        digits[i] = str((int(digits[i]) + 1) % 10)
        answer.po_number = "".join(digits)
    elif kind == "number" and answer.invoice_number:
        answer.invoice_number = (
            answer.invoice_number[:-1] + str((int(answer.invoice_number[-1]) + 1) % 10)
            if answer.invoice_number[-1].isdigit()
            else answer.invoice_number + "1"
        )
    elif kind == "drop_line" and len(answer.lines) > 1:
        answer.lines.pop(rng.randrange(len(answer.lines)))
    elif kind == "total" and answer.total is not None:
        answer.total = round(answer.total + rng.choice([-1, 1]) * rng.choice([0.5, 10, 100]), 2)
    elif kind == "currency" and answer.currency:
        answer.currency = rng.choice([c for c in ("EUR", "GBP", "USD") if c != answer.currency])
    elif kind == "type":
        answer.document_type = "invoice" if answer.document_type != "invoice" else "credit_note"
    return answer


# ---- shadow run ---------------------------------------------------------------------------------------


def shadow(database_url: str, judge: Judge, emails: int = 120, seed: int = 11) -> dict:
    """The agent decides beside the clerk; only the clerk's decisions take effect."""
    db.reset(database_url)
    dataset = D.generate(emails, seed, start=date(2026, 10, 1))
    load(database_url, dataset)
    agreement = Counter()
    with Environment(database_url, judge, http=False) as env:
        client = env.local_client(shadow=True)
        erp = httpx.Client(base_url=env.external.url, headers={"X-Api-Key": env.api_key})
        for email in dataset.emails:
            out = handle_email(client, email.email_id)  # actions are recorded, not executed
            human = email.decision
            if human["action"] == "book":  # the clerk books it for real
                erp.post(
                    "/erp/bookings",
                    json=human["booking"],
                    headers={"Idempotency-Key": f"human:{email.email_id}"},
                ).raise_for_status()
            o = score(email, out)
            agreement["agree" if o.correct else ("wrong" if o.wrong_action else "needless review")] += 1
        with db.connect(database_url) as conn:
            recorded = conn.execute(
                "SELECT tool, count(*) AS n FROM agents.shadow_actions GROUP BY tool"
            ).fetchall()
            effects = conn.execute(
                "SELECT count(*) AS n FROM erp.bookings WHERE idempotency_key NOT LIKE 'human:%%'"
            ).fetchone()["n"]
    return {
        "emails": emails,
        "agreement": dict(agreement),
        "shadow_actions": {r["tool"]: r["n"] for r in recorded},
        "agent_effects_on_erp": effects,
    }


# ---- crash test ------------------------------------------------------------------------------------------


@dataclass
class CrashReport:
    tasks: int
    sessions_started: int
    sessions_killed: int
    crashpoint_exits: int
    stale_refusals: int
    abandoned_worker: bool
    erp_failures_before: int
    erp_failures_after: int
    receipts_replayed: int
    bookings: int
    expected_bookings: int
    duplicate_bookings: int
    reviews: int
    expected_reviews: int
    chase_emails: int
    expected_chases: int
    duplicate_emails: int
    passed: bool
    differences: list[str] = field(default_factory=list)


def _effects(database_url: str) -> tuple[set, set, Counter]:
    with db.connect(database_url) as conn:
        bookings = Counter(
            (r["supplier_id"], r["invoice_number"])
            for r in conn.execute("SELECT supplier_id, invoice_number FROM erp.bookings").fetchall()
        )
        reviews = {r["key"] for r in conn.execute("SELECT key FROM agents.reviews").fetchall()}
    return set(bookings), reviews, bookings


def crash_test(
    database_url: str, emails: int = 60, workers: int = 4, kill_rate: float = 0.25, seed: int = 3
) -> CrashReport:
    dataset = D.generate(emails, 7)
    as_of = date(2026, 10, 1)

    # Reference: the same work with nothing going wrong.
    db.reset(database_url)
    load(database_url, dataset)
    with Environment(database_url, rules.judge(), http=False) as env:
        client = env.local_client()
        for email in dataset.emails:
            handle_email(client, email.email_id)
        chases = env.enqueue_chases(as_of)
    expected_bookings, expected_reviews, _ = _effects(database_url)

    # The same work, with crashes everywhere.
    db.reset(database_url)
    load(database_url, dataset)
    chaos = external.Chaos(fail_before=0.10, fail_after=0.10, seed=seed)
    rng = random.Random(seed)
    started, killed = [], [0]
    lock = threading.Lock()

    def on_start(proc) -> None:
        with lock:
            started.append(proc)
        if rng.random() < kill_rate:  # SIGKILL at a random moment of the session

            def kill() -> None:
                time.sleep(rng.uniform(0.05, 1.2))
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    killed[0] += 1

            threading.Thread(target=kill, daemon=True).start()

    spans = InMemorySpanExporter()
    add_exporter(spans)
    with Environment(database_url, rules.judge(), chaos=chaos) as env:
        env.enqueue_emails([e.email_id for e in dataset.emails])
        env.enqueue_chases(as_of)
        with db.connect(database_url) as conn:  # crashes are frequent here; give every task room to finish
            conn.execute("UPDATE agents.tasks SET max_attempts = 30")
            conn.commit()
        crash_env = {
            "PFA_CRASH_AT": "before_judge,after_decide,after_book,after_chase",
            "PFA_CRASH_PROB": "0.10",
        }
        pool = [
            env.worker(f"w{i}", lease_s=3.0, extra_env=crash_env, on_start=on_start) for i in range(workers)
        ]
        threads = [threading.Thread(target=w.run, kwargs={"max_idle_s": 12.0}) for w in pool]
        for t in threads:
            t.start()
        time.sleep(4)
        pool[0].abandon.set()  # one worker vanishes mid-task: its lease must expire and be taken over
        for t in threads:
            t.join(timeout=600)
        sent_first = env.relay_outbox()
        sent_again = env.relay_outbox()
        with db.connect(database_url) as conn:
            tasks = conn.execute("SELECT status, attempts, error FROM agents.tasks").fetchall()
            erp_rows = conn.execute("SELECT count(*) AS n FROM erp.idempotency").fetchone()["n"]
            mail = Counter(
                r["subject"] for r in conn.execute("SELECT subject FROM mailprovider.messages").fetchall()
            )
    replayed = sum(1 for sp in spans.get_finished_spans() if sp.attributes.get("pfa.replayed"))
    spans.shutdown()
    bookings, reviews, counts = _effects(database_url)
    errors = Counter((t["error"] or "")[:40] for t in tasks if t["error"])
    report = CrashReport(
        tasks=len(tasks),
        sessions_started=len(started),
        sessions_killed=killed[0],
        crashpoint_exits=sum(n for e, n in errors.items() if e.startswith("exit 137")),
        stale_refusals=sum(n for e, n in errors.items() if e.startswith("exit 3")),
        abandoned_worker=True,
        erp_failures_before=chaos.failed_before,
        erp_failures_after=chaos.failed_after,
        receipts_replayed=replayed,
        bookings=len(bookings),
        expected_bookings=len(expected_bookings),
        duplicate_bookings=sum(n - 1 for n in counts.values() if n > 1),
        reviews=len(reviews),
        expected_reviews=len(expected_reviews),
        chase_emails=sum(mail.values()),
        expected_chases=len(chases),
        duplicate_emails=sum(n - 1 for n in mail.values() if n > 1) + sent_again,
        passed=False,
    )
    with db.connect(database_url) as conn:
        reasons = {
            r["key"]: r["reason"] for r in conn.execute("SELECT key, reason FROM agents.reviews").fetchall()
        }
    report.differences = (
        [f"extra review {k}: {reasons.get(k)}" for k in sorted(reviews - expected_reviews)]
        + [f"missing review {k}" for k in sorted(expected_reviews - reviews)]
        + [f"extra booking {b}" for b in sorted(bookings - expected_bookings)]
        + [f"missing booking {b}" for b in sorted(expected_bookings - bookings)]
        + [f"task {t['status']}: {(t['error'] or '')[:120]}" for t in tasks if t["status"] != "done"]
    )
    report.passed = (
        all(t["status"] == "done" for t in tasks)
        and bookings == expected_bookings
        and reviews == expected_reviews
        and report.duplicate_bookings == 0
        and report.chase_emails == len(chases) == sent_first
        and report.duplicate_emails == 0
        and erp_rows == len(bookings)
    )
    return report


# ---- race ---------------------------------------------------------------------------------------------------


def race(database_url: str) -> dict:
    """Two invoices, each billing the PO's full delivery, decided at the same moment."""
    db.reset(database_url)
    s = D.BY_ID["brightwire"]
    po = D.PurchaseOrder(
        "4500999",
        s.id,
        s.currency,
        date(2026, 9, 1),
        date(2026, 9, 20),
        [D.POLine(1, "LM358DR", 1000, Decimal("0.1120"))],
    )
    rng = random.Random(1)
    emails = []
    for i, number in enumerate(["BW-26-09001", "BW-26-09002"]):
        inv = D.Invoice(
            s,
            number,
            po.po_number,
            s.currency,
            date(2026, 9, 21),
            [D.InvoiceLine("LM358DR", 1000, Decimal("0.1120"))],
            Decimal("0"),
        )
        subject, body, att = D.render(inv, rng)
        emails.append(D.Email(f"r{i}", s.email, subject, body, date(2026, 9, 21), att, {"action": "book"}))
    load(database_url, D.Dataset([po], [(po.po_number, "LM358DR", 1000, date(2026, 9, 20))], emails))
    with Environment(database_url, rules.judge()) as env:
        env.enqueue_emails(["r0", "r1"])
        pool = [
            env.worker(f"w{i}", lease_s=5.0, extra_env={"PFA_PAUSE_AFTER_DECIDE": "1.0"}) for i in range(2)
        ]
        threads = [threading.Thread(target=w.run) for w in pool]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        results = sorted((r for w in pool for r in w.results), key=lambda r: r["email_id"])
    bookings, reviews, _ = _effects(database_url)
    return {
        "both_passed_the_agent_check": all(
            "nothing billed beyond what was received" in r["checks"] for r in results
        ),
        "booked": len(bookings),
        "sent_to_review": len(reviews),
        "review_reasons": [r["reason"] for r in results if r["action"] == "review"],
    }


def zombie(database_url: str) -> dict:
    """A worker freezes mid-session and its lease expires; another takes over. When the frozen
    session wakes up and tries to book, the gateway must refuse it."""
    dataset = D.generate(30, 7)
    email = next(e for e in dataset.emails if e.case == "clean" and "corvid" not in e.sender)
    db.reset(database_url)
    load(database_url, dataset)
    with Environment(database_url, rules.judge()) as env:
        env.enqueue_emails([email.email_id])
        frozen = env.worker("frozen", lease_s=1.5, extra_env={"PFA_PAUSE_AFTER_DECIDE": "5"})
        frozen.on_start = lambda proc: frozen.abandon.set()  # stops renewing as soon as the session starts
        t = threading.Thread(target=frozen.run)
        t.start()
        time.sleep(3.0)  # lease (1.5 s) has expired; the frozen session is still asleep
        rescuer = env.worker("rescuer", lease_s=5.0)
        rescuer.run()
        t.join(timeout=30)
        with db.connect(database_url) as conn:
            task = conn.execute("SELECT status, fence, lease_owner FROM agents.tasks").fetchone()
    bookings, _, counts = _effects(database_url)
    return {
        "task_status": task["status"],
        "fence": task["fence"],
        "booked_once": counts and max(counts.values()) == 1,
        "bookings": len(bookings),
        "rescuer_completed": len(rescuer.results) == 1,
        "frozen_session_exit": frozen.exits[0][1] if frozen.exits else None,
        "frozen_session_error": frozen.exits[0][2].strip().splitlines()[-1] if frozen.exits else "",
    }

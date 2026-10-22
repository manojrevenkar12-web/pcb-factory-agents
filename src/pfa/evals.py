"""Evaluation before trust: blind tests against people's verdicts, with accuracy and cost per run.

Every agent decision is scored against what a person decided, never shown to the judge. The
numbers that decide whether an agent may run alone:

- **wrong actions**: the agent did something a person would not have (booked a bad invoice,
  passed a defective board). This must be zero, and it is the first thing the gate checks;
- **automation**: of the cases a person did not need to see, how many the agent closed itself;
- **cost and latency per case**.

Results are stored per configuration (judge, model, effort, skill text, checks version), and the
gate compares a run with the committed baseline: a change of model, prompt or effort level that
adds a wrong action, loses automation or costs more fails CI.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Outcome:
    case_id: str
    expected: str
    got: str
    correct: bool
    wrong_action: bool  # acted when it should not have, or acted wrongly
    needless_review: bool  # sent to a person a case it could have closed
    cost_usd: float = 0.0
    latency_s: float = 0.0
    detail: str = ""


@dataclass
class Summary:
    task: str
    judge: str
    config: str
    cases: int
    correct: int
    wrong_actions: int
    needless_reviews: int
    automation: float  # share of cases not needing a person that the agent closed
    accuracy: float
    cost_per_case_usd: float
    p95_latency_s: float
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def summarize(
    task: str, judge: str, config: str, outcomes: list[Outcome], closable: set[str] | None = None, **extra
) -> Summary:
    """`closable`: the expected labels that need no person (e.g. book, ignore, duplicate)."""
    n = len(outcomes)
    closable = closable or set()
    could_close = [o for o in outcomes if o.expected in closable]
    closed = [o for o in could_close if o.correct]
    latencies = sorted(o.latency_s for o in outcomes) or [0.0]
    return Summary(
        task=task,
        judge=judge,
        config=config,
        cases=n,
        correct=sum(o.correct for o in outcomes),
        wrong_actions=sum(o.wrong_action for o in outcomes),
        needless_reviews=sum(o.needless_review for o in outcomes),
        automation=len(closed) / len(could_close) if could_close else 0.0,
        accuracy=sum(o.correct for o in outcomes) / n if n else 0.0,
        cost_per_case_usd=statistics.fmean(o.cost_usd for o in outcomes) if n else 0.0,
        p95_latency_s=latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
        extra=extra,
    )


@dataclass
class Tolerance:
    max_wrong_actions: int = 0
    max_automation_drop: float = 0.02
    max_cost_increase: float = 0.20


def gate(current: Summary, baseline: dict | None, tol: Tolerance | None = None) -> list[str]:
    """Reasons the run may not replace the baseline; empty means it passes."""
    tol = tol or Tolerance()
    problems = []
    if current.wrong_actions > tol.max_wrong_actions:
        problems.append(f"{current.wrong_actions} wrong action(s); allowed {tol.max_wrong_actions}")
    if baseline:
        if current.automation < baseline["automation"] - tol.max_automation_drop:
            problems.append(f"automation fell from {baseline['automation']:.1%} to {current.automation:.1%}")
        base_cost = baseline["cost_per_case_usd"]
        if base_cost > 0 and current.cost_per_case_usd > base_cost * (1 + tol.max_cost_increase):
            problems.append(f"cost per case rose from ${base_cost:.5f} to ${current.cost_per_case_usd:.5f}")
    return problems


def save(path: Path, summary: Summary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.as_dict(), indent=2, sort_keys=True) + "\n")


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None

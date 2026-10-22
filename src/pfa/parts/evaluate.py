"""Evaluating part programming on 9,000+ real parts from the KiCad libraries.

Each symbol that names a package footprint is a part a customer could use. Its footprint is the
customer's design. To test whether wrong designs are caught, 15% of designs are swapped for a
plausible wrong footprint before the agent sees them:

- **subtle**: same family and pin count, another body size or pitch (an SOIC-8 3.9 mm wide
  replaced by the 5.3 mm one);
- **pins**: same family, another pin count;
- **family**: another family with the same pin count (TSSOP for SOIC).

Separately, the libraries' own entries where the symbol's pins do not match its footprint's pads
are counted: those are what the pin check flags on unmodified real data.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from pfa.judges import Judge, NoisyJudge
from pfa.parts.kicad import Part, load_parts
from pfa.parts.program import (
    Library,
    PackageAnswer,
    check,
    decide,
    key_of,
    request,
    rules_judge,
    same_land_pattern,
)

KINDS = ("subtle", "pins", "family")


def wrong_footprint(lib: Library, truth: str, kind: str, rng: random.Random) -> str | None:
    key = key_of(truth.split(":", 1)[1])
    if key is None:
        return None
    if kind == "family":
        pool = [
            f for fam, fs in lib.by_family.items() if fam != key.family for f, k in fs if k.pins == key.pins
        ]
    else:
        same = lib.by_family.get(key.family, [])
        if kind == "subtle":
            pool = [
                f
                for f, k in same
                if k.pins == key.pins
                and k.ep == key.ep
                and k.body
                and key.body
                and (k.body != key.body or k.pitch != key.pitch)
            ]
        else:
            pool = [f for f, k in same if k.pins is not None and key.pins is not None and k.pins != key.pins]
    truth_fp = lib.footprint(truth)
    pool = [f for f in sorted(pool) if not same_land_pattern(lib.footprint(f), truth_fp)]  # really different
    return rng.choice(pool) if pool else None


def corrupt(answer: PackageAnswer, rng: random.Random) -> PackageAnswer:
    kind = rng.choice(["pins", "family", "body"])
    if kind == "pins" and answer.pins:
        answer.pins += rng.choice([-2, 2, 4])
    elif kind == "family":
        answer.family = rng.choice(["SOIC", "TSSOP", "QFN", "SOT-23", "LQFP"])
    else:
        answer.body_mm = rng.choice(["3x3", "4x4", "5x5", "7.5x10.3"])
    return answer


def run(
    symbols: Path, footprints: Path, judge: Judge | None = None, inject: float = 0.15, seed: int = 5
) -> dict:
    lib = Library(footprints)
    judge = judge or rules_judge()
    parts: list[Part] = [p for p in load_parts(symbols) if p.footprint in lib.paths and p.description]
    library_flags = [p for p in parts if not check(p, lib.footprint(p.footprint))[0]]
    consistent = [p for p in parts if check(p, lib.footprint(p.footprint))[0]]
    rng = random.Random(seed)
    tally: Counter = Counter()
    reasons: Counter = Counter()
    injected_by_kind: Counter = Counter()
    caught_by_kind: Counter = Counter()
    cost = 0.0
    for part in consistent:
        design = part.footprint
        kind = None
        if rng.random() < inject:
            kind = rng.choice(KINDS)
            wrong = wrong_footprint(lib, part.footprint, kind, rng)
            if wrong is not None:
                design = wrong
            else:
                kind = None
        verdict = judge.judge(request(part))
        cost += verdict.cost_usd
        decision = decide(part, lib.footprint(design), verdict.answer, lib, verdict.error)
        if kind:
            injected_by_kind[kind] += 1
            caught_by_kind[kind] += decision.action == "person"
            tally["injected caught" if decision.action == "person" else "injected PROGRAMMED"] += 1
        else:
            tally["correct programmed" if decision.action == "program" else "correct sent to a person"] += 1
            if decision.action == "person":
                reasons[decision.reason.split(":")[0].split(",")[0][:48]] += 1
    good = tally["correct programmed"] + tally["correct sent to a person"]
    injected = sum(injected_by_kind.values())
    return {
        "judge": judge.name,
        "parts": len(parts),
        "library_entries_flagged_by_pin_check": len(library_flags),
        "library_flag_examples": [
            f"{p.name} ({p.library}): {check(p, lib.footprint(p.footprint))[1]}, {p.footprint}"
            for p in library_flags[:5]
        ],
        "consistent_parts": len(consistent),
        "designs_with_injected_errors": injected,
        "injected_caught": {k: f"{caught_by_kind[k]}/{injected_by_kind[k]}" for k in KINDS},
        "injected_caught_rate": (sum(caught_by_kind.values()) / injected) if injected else 0.0,
        "wrong_designs_programmed": tally["injected PROGRAMMED"],
        "correct_designs_programmed": tally["correct programmed"],
        "automation": tally["correct programmed"] / good if good else 0.0,
        "why_correct_designs_went_to_a_person": dict(reasons.most_common(6)),
        "judge_cost_usd": cost,
    }


def run_noisy(symbols: Path, footprints: Path, rate: float = 0.2) -> dict:
    return run(symbols, footprints, NoisyJudge(rules_judge(), rate, corrupt, seed=9))

"""Blind evaluation of board inspection on boards it has never seen.

The ranker and the per-package "normal" ranges are learnt from boards of one design; the
classical judge's thresholds are tuned on the same boards. Everything is then scored on boards
of four other designs, against the true defect of every part.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pfa.inspect.boards import Board, build
from pfa.inspect.pipeline import (
    FEATURES,
    Measurement,
    Normal,
    PartVerdict,
    Ranker,
    Thresholds,
    align,
    classical_verdict,
    measure_all,
    rank,
    reconcile,
    registration_error_px,
    unsure,
)
from pfa.judges import PRICES, image_tokens, load_skill

TRAIN = [1000 + i for i in range(1, 31)]
TEST = [d * 1000 + i for d in (2, 3, 4, 5) for i in range(1, 16)]
KS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]


@dataclass
class Inspected:
    board: Board
    measurements: list[Measurement]
    registration_px: float


def inspect_boards(seeds: list[int], defect_rate: float) -> list[Inspected]:
    out = []
    for seed in seeds:
        b = build(seed, defect_rate)
        aligned, warp = align(b.golden, b.test)
        out.append(
            Inspected(
                b,
                measure_all(b.golden, b.bare, aligned, b.placements),
                registration_error_px(warp, b.warp, b.golden.shape[:2]),
            )
        )
    return out


def _labels(i: Inspected) -> list[str]:
    lab = i.board.labels
    return [lab[m.refdes] for m in i.measurements]


def fit(train: list[Inspected]) -> tuple[Normal, Ranker, Thresholds]:
    ms = [m for i in train for m in i.measurements]
    labels = [lab for i in train for lab in _labels(i)]
    normal = Normal().fit(ms, labels)
    x = np.array([normal.z(m) for m in ms])
    y = np.array([lab != "ok" for lab in labels], float)
    ranker = Ranker().fit(x, y)
    # Tune the judge on training parts: catch the most defects with at most 1 false call per 300 good parts.
    best, best_key = Thresholds(), (-1.0, 0.0)
    good = sum(1 for lab in labels if lab == "ok")
    for shift in (0.18, 0.22, 0.26):
        for rot in (0.05, 0.08, 0.12):
            for wz in (4.0, 6.0, 8.0, 10.0, 14.0):
                t = Thresholds(0.05, rot, shift, wz, min(3.5, wz))
                verdicts = [classical_verdict(m, t, normal).verdict for m in ms]
                caught = sum(v != "ok" and lab != "ok" for v, lab in zip(verdicts, labels, strict=True))
                false = sum(v != "ok" and lab == "ok" for v, lab in zip(verdicts, labels, strict=True))
                key = (caught / max(1, int(y.sum())), -false)
                if false <= good / 300 and key > best_key:
                    best, best_key = t, key
    return normal, ranker, best


def capture_curve(test: list[Inspected], normal: Normal, ranker: Ranker) -> dict[float, float]:
    found, total = Counter(), 0
    for i in test:
        ranked = rank(i.measurements, ranker, normal)
        lab = i.board.labels
        defects = [k for k, m in enumerate(ranked) if lab[m.refdes] != "ok"]
        total += len(defects)
        for frac in KS:
            cut = max(1, math.ceil(frac * len(ranked)))
            found[frac] += sum(1 for k in defects if k < cut)
    return {k: found[k] / total for k in KS}


def model_cost_per_part(model: str = "claude-sonnet-5-5") -> dict:
    """Estimated cost of asking a vision model about one part: three enlarged crops, the skill and
    the question in, a short structured verdict out. Image tokens use the published formula;
    text tokens are estimated at 4 characters per token."""
    crop_px = 3 * int(2 * (2.9 + 0.4) * 20)  # the largest package, enlarged 3x
    tokens_in = 3 * image_tokens(crop_px, crop_px) + (len(load_skill("inspect-part")) + 200) // 4
    tokens_out = 60
    price_in, price_out = PRICES[model]
    return {
        "model": model,
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "usd": (tokens_in * price_in + tokens_out * price_out) / 1e6,
    }


def end_to_end(test: list[Inspected], normal: Normal, ranker: Ranker, t: Thresholds, frac: float) -> dict:
    caught, missed, false_calls, operator, judged, type_ok, escalated_defects = 0, 0, 0, 0, 0, 0, 0
    by_type: Counter = Counter()
    by_type_caught: Counter = Counter()
    for i in test:
        ranked = rank(i.measurements, ranker, normal)
        cut = max(1, math.ceil(frac * len(ranked)))
        lab = i.board.labels
        for k, m in enumerate(ranked):
            truth = lab[m.refdes]
            if truth != "ok":
                by_type[truth] += 1
            if k >= cut:
                missed += truth != "ok"
                continue
            judged += 1
            verdict: PartVerdict = classical_verdict(m, t, normal)
            decision, _ = reconcile(verdict, m, t)
            if decision == "pass" and unsure(m, t, normal):
                decision = "operator"
            if decision == "operator":
                operator += 1
                escalated_defects += truth != "ok"
            flagged = decision in ("fail", "operator")
            if truth != "ok" and flagged:
                caught += 1
                by_type_caught[truth] += 1
                type_ok += verdict.verdict == truth or (truth == "tombstone" and verdict.verdict != "ok")
            elif truth != "ok":
                missed += 1
            elif decision == "fail":
                false_calls += 1
    total = caught + missed
    cost = model_cost_per_part()
    return {
        "fraction_judged": frac,
        "parts_judged_per_board": judged / len(test),
        "defects": total,
        "caught": caught,
        "recall": caught / total,
        "false_calls": false_calls,
        "false_calls_per_board": false_calls / len(test),
        "caught_automatically": caught - escalated_defects,
        "escalated": operator,
        "escalated_per_board": operator / len(test),
        "defects_among_escalated": escalated_defects,
        "type_correct_of_caught": type_ok / max(1, caught),
        "recall_by_type": {k: by_type_caught[k] / by_type[k] for k in sorted(by_type)},
        "model_usd_per_board_if_claude_judges_all_selected": judged / len(test) * cost["usd"],
        "model_usd_per_board_if_claude_judges_escalations": operator / len(test) * cost["usd"],
    }


def run(out_dir: Path, defect_rate: float = 0.04) -> dict:
    train = inspect_boards(TRAIN, defect_rate)
    test = inspect_boards(TEST, defect_rate)
    normal, ranker, thresholds = fit(train)
    curve = capture_curve(test, normal, ranker)
    narrowed = end_to_end(test, normal, ranker, thresholds, 0.20)
    everything = end_to_end(test, normal, ranker, thresholds, 1.0)
    whole_board = image_tokens(*test[0].board.test.shape[1::-1])
    reg = [i.registration_px for i in test]
    result = {
        "boards": {
            "train": len(train),
            "test": len(test),
            "parts_per_board": len(test[0].measurements),
            "defect_rate": defect_rate,
            "defects_in_test": sum(lab != "ok" for i in test for lab in i.board.labels.values()),
        },
        "registration_px": {"mean": float(np.mean(reg)), "max": float(np.max(reg))},
        "capture_curve": {f"{k:.0%}": v for k, v in curve.items()},
        "judge_top_20pct": narrowed,
        "judge_every_part": everything,
        "thresholds": vars(thresholds),
        "ranker": ranker.to_dict(),
        "cost": {"per_part": model_cost_per_part(), "whole_board_image_tokens": whole_board},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inspection.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / "inspection_model.json").write_text(
        json.dumps(
            {
                "normal": normal.stats,
                "ranker": ranker.to_dict(),
                "thresholds": vars(thresholds),
                "features": FEATURES,
            },
            indent=2,
        )
        + "\n"
    )
    return result

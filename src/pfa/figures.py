"""Figures for the report."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pfa.inspect.boards import PPMM, build
from pfa.inspect.pipeline import (
    Normal,
    Ranker,
    Thresholds,
    align,
    classical_verdict,
    measure_all,
    rank,
    reconcile,
    unsure,
)

INK, MUTED, ACCENT, GOOD, WARN = "#1f2933", "#7b8794", "#c2410c", "#15803d", "#b45309"


def capture_curve(curve: dict[str, float], path: Path) -> None:
    xs = [float(k.rstrip("%")) for k in curve]
    ys = [100 * v for v in curve.values()]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(xs, ys, color=ACCENT, marker="o", linewidth=2)
    ax.plot([0, 100], [0, 100], color=MUTED, linestyle=":", linewidth=1)
    ax.axvline(20, color=MUTED, linewidth=0.8)
    y20 = ys[xs.index(20.0)]
    ax.annotate(
        f"{y20:.0f}% of defects in the\nworst-ranked 20% of parts",
        (20, y20),
        (32, y20 - 30),
        arrowprops={"arrowstyle": "-", "color": MUTED},
        color=INK,
        fontsize=9,
    )
    ax.set_xlabel("parts inspected, ranked worst first (%)", color=INK)
    ax.set_ylabel("defects found (%)", color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Cheap vision narrows the search (60 unseen boards)", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def board_example(model: dict, path: Path, seed: int = 3009, defect_rate: float = 0.06) -> None:
    """An unseen board, aligned, with the parts sent to the judge and what came of them."""
    normal = Normal({k: (v[0], v[1]) for k, v in model["normal"].items()})
    ranker = Ranker.from_dict(model["ranker"])
    t = Thresholds(**model["thresholds"])
    b = build(seed, defect_rate)
    aligned, _ = align(b.golden, b.test)
    ranked = rank(measure_all(b.golden, b.bare, aligned, b.placements), ranker, normal)
    cut = math.ceil(0.2 * len(ranked))
    img = aligned.copy()
    labels = b.labels
    colours = {
        "caught": (61, 128, 21),
        "escalated": (9, 115, 180),
        "missed": (217, 70, 239),
        "false call": (12, 65, 194),
        "passed": (235, 190, 60),
    }
    for k, m in enumerate(ranked):
        pl = next(p for p in b.placements if p.refdes == m.refdes)
        truth = labels[m.refdes]
        if k < cut:
            decision, _ = reconcile(classical_verdict(m, t, normal), m, t)
            if decision == "pass" and unsure(m, t, normal):
                decision = "operator"
            if truth != "ok":
                what = (
                    "caught" if decision == "fail" else ("escalated" if decision == "operator" else "missed")
                )
            else:
                what = (
                    "false call"
                    if decision == "fail"
                    else ("escalated" if decision == "operator" else "passed")
                )
        else:
            what = "missed" if truth != "ok" else None
        if what is None:
            continue
        r = pl.pkg.extent() + 0.4
        x0, y0, x1, y1 = (
            int((pl.x - r) * PPMM),
            int((pl.y - r) * PPMM),
            int((pl.x + r) * PPMM),
            int((pl.y + r) * PPMM),
        )
        cv2.rectangle(img, (x0, y0), (x1, y1), colours[what], 3 if what != "passed" else 2, cv2.LINE_AA)
        if truth != "ok":
            cv2.putText(
                img,
                truth.replace("_", " "),
                (x0, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colours[what],
                2,
                cv2.LINE_AA,
            )
    legend = np.full((44, img.shape[1], 3), 255, np.uint8)
    x = 12
    for name, col in colours.items():
        cv2.rectangle(legend, (x, 14), (x + 18, 30), col, -1)
        cv2.putText(legend, name, (x + 26, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        x += 40 + 11 * len(name)
    cv2.putText(
        legend,
        "boxed: the 20% of parts ranked worst (judged), and any defect",
        (x + 10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (110, 110, 110),
        1,
        cv2.LINE_AA,
    )
    out = np.vstack([img, legend])
    cv2.imwrite(str(path), cv2.resize(out, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA))

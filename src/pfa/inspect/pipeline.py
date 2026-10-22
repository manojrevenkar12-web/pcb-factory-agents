"""Board inspection: classical vision narrows the problem, a judge looks only where it matters.

1. **Align** the photo to the golden board (ECC, coarse to fine).
2. **Rank** every part on the placement file by how unlike the golden board it looks, after
   normalising light and allowing for small placement jitter.
3. **Judge** only the worst-ranked parts. Each judge (classical rules, or Claude looking at the
   golden, bare and test crops) returns a structured verdict.
4. **Reconcile**: plain code checks the verdict against the measurements it can verify (which
   rotation actually matches, whether the part resembles the bare pads), and sends anything that
   does not agree to an operator.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

from pfa.inspect.boards import PPMM, Board, Placement
from pfa.judges import FunctionJudge, Image, Request

# ---- 1. alignment --------------------------------------------------------------------------------------------


def align(
    golden: np.ndarray, test: np.ndarray, levels: tuple[int, ...] = (4, 2)
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the test photo warped into the golden frame, and the 2x3 golden->test transform."""
    g = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY).astype(np.float32)
    t = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    for f in levels:
        gs = cv2.resize(g, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
        ts = cv2.resize(t, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
        w = warp.copy()
        w[:, 2] /= f
        with contextlib.suppress(cv2.error):  # on failure keep the coarser estimate
            _, w = cv2.findTransformECC(gs, ts, w, cv2.MOTION_AFFINE, criteria, None, 5)
        warp = w.copy()
        warp[:, 2] *= f
    aligned = cv2.warpAffine(
        test,
        warp,
        (golden.shape[1], golden.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned, warp


def registration_error_px(estimated: np.ndarray, true: np.ndarray, shape: tuple[int, int]) -> float:
    h, w = shape
    pts = np.array([[x, y, 1.0] for x in (0, w / 2, w) for y in (0, h / 2, h)])
    return float(np.mean(np.linalg.norm(pts @ estimated.T - pts @ true.T, axis=1)))


# ---- 2. measurement and ranking -------------------------------------------------------------------------------

SEARCH_PX = 4


def _crop(img: np.ndarray, pl: Placement, margin_mm: float) -> np.ndarray:
    r = pl.pkg.extent() + margin_mm
    cx, cy = pl.x * PPMM, pl.y * PPMM
    half = int(math.ceil(r * PPMM))
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    pad = cv2.copyMakeBorder(img, 64, 64, 64, 64, cv2.BORDER_REFLECT)
    return pad[y0 + 64 : y0 + 64 + 2 * half, x0 + 64 : x0 + 64 + 2 * half]


def _ncc(search: np.ndarray, template: np.ndarray) -> tuple[float, tuple[int, int]]:
    res = cv2.matchTemplate(search.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, best, _, loc = cv2.minMaxLoc(res)
    return float(best), loc


@dataclass
class Measurement:
    refdes: str
    package: str
    polarised: bool
    ncc_golden: float  # best normalised correlation with the golden part (whole footprint)
    ncc_body: float  # the same for the body's centre, where the marking is
    ncc_bare: float  # with the bare pads: high means nothing was placed
    rotation_ncc: dict[int, float]
    shift_mm: float
    body_chroma: float  # colour difference of the body (a*, b*), insensitive to brightness
    body_light: float  # brightness difference of the body, relative to the surrounding board
    score: float = 0.0  # set by the ranker: higher looks worse

    @property
    def best_rotation(self) -> int:
        return max(self.rotation_ncc, key=self.rotation_ncc.get)

    @property
    def rotation_gain(self) -> float:
        return max(self.rotation_ncc.values()) - self.rotation_ncc[0]

    def raw(self) -> np.ndarray:
        return np.array(
            [
                1 - self.ncc_golden,
                1 - self.ncc_body,
                self.ncc_bare - self.ncc_golden,
                self.rotation_gain,
                self.shift_mm,
                self.body_chroma,
                abs(self.body_light),
            ]
        )


FEATURES = [
    "1 - ncc_golden",
    "1 - ncc_body",
    "ncc_bare - ncc_golden",
    "rotation_gain",
    "shift_mm",
    "body_chroma",
    "|body_light|",
]


# Smallest spread worth trusting per feature (one pixel of shift, one Lab unit of colour...).
MAD_FLOOR = np.array([0.01, 0.01, 0.01, 0.02, 0.05, 1.0, 0.02])


class Normal:
    """What a good part of each package measures like: median and spread of every feature,
    learnt from parts that passed. Features become robust z-scores, comparable across packages
    (a plain tan capacitor and a marked SOIC differ a lot in how well they correlate)."""

    def __init__(self, stats: dict[str, tuple[list[float], list[float]]] | None = None):
        self.stats = stats or {}

    def fit(self, ms: list[Measurement], labels: list[str]) -> Normal:
        by: dict[str, list[np.ndarray]] = {}
        for m, label in zip(ms, labels, strict=True):
            if label == "ok":
                by.setdefault(m.package, []).append(m.raw())
        for pkg, rows in by.items():
            x = np.array(rows)
            med = np.median(x, 0)
            mad = np.maximum(np.median(np.abs(x - med), 0) * 1.4826, MAD_FLOOR)
            self.stats[pkg] = (med.tolist(), mad.tolist())
        return self

    def z(self, m: Measurement) -> np.ndarray:
        med, mad = self.stats[m.package]
        return (m.raw() - np.array(med)) / np.array(mad)


def _body_crop(img: np.ndarray, pl: Placement, pad_px: int = 0) -> np.ndarray:
    length, width = pl.pkg.body
    if pl.rotation % 180:
        length, width = width, length
    hw, hh = int(length * 0.35 * PPMM) + pad_px, int(width * 0.3 * PPMM) + pad_px
    cx, cy = int(round(pl.x * PPMM)), int(round(pl.y * PPMM))
    return img[cy - hh : cy + hh, cx - hw : cx + hw]


def measure(golden: np.ndarray, bare: np.ndarray, aligned: np.ndarray, pl: Placement) -> Measurement:
    inner_g = _crop(golden, pl, 0.15)
    inner_b = _crop(bare, pl, 0.15)
    outer = _crop(aligned, pl, 0.15 + SEARCH_PX / PPMM)
    ncc_g, _ = _ncc(outer, inner_g)
    ncc_b, _ = _ncc(outer, inner_b)
    # Rotations: compare the test part with the golden part turned about its centre.
    rot = {}
    for k, deg in enumerate((0, 90, 180, 270)):
        turned = np.ascontiguousarray(np.rot90(inner_g, k=-k))  # clockwise, as placement rotation
        rot[deg], _ = _ncc(outer, turned)
    # Shift: where the golden part matches best in a wider window.
    wide = _crop(aligned, pl, 0.15 + 0.9)
    _, loc = _ncc(wide, inner_g)
    centre = (wide.shape[1] - inner_g.shape[1]) / 2
    shift = math.hypot(loc[0] - centre, loc[1] - centre) / PPMM
    ncc_body, _ = _ncc(_body_crop(aligned, pl, SEARCH_PX), _body_crop(golden, pl))
    chroma, light = _colour(golden, aligned, pl)
    return Measurement(
        pl.refdes, pl.package, pl.pkg.polarised, ncc_g, ncc_body, ncc_b, rot, shift, chroma, light
    )


def _colour(golden: np.ndarray, aligned: np.ndarray, pl: Placement) -> tuple[float, float]:
    def lab(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

    gb, tb = lab(_body_crop(golden, pl)), lab(_body_crop(aligned, pl))
    gs, ts = lab(_crop(golden, pl, 0.6)), lab(_crop(aligned, pl, 0.6))
    chroma = float(np.linalg.norm(np.median(tb[:, 1:], 0) - np.median(gb[:, 1:], 0)))
    # brightness relative to the whole neighbourhood, so uneven light does not count
    light = float(
        np.median(tb[:, 0]) / (np.median(ts[:, 0]) + 1) - np.median(gb[:, 0]) / (np.median(gs[:, 0]) + 1)
    )
    return chroma, light


class Ranker:
    """Logistic regression over five interpretable measurements: how likely a part is defective.

    Fitted on boards of one design and used on others; cheap enough to score every part of every
    panel, so the judge only sees the parts that rank worst.
    """

    def __init__(
        self, weights: np.ndarray | None = None, mean: np.ndarray | None = None, std: np.ndarray | None = None
    ):
        self.w, self.mean, self.std = weights, mean, std

    def fit(
        self, x: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 3000, lr: float = 0.5
    ) -> Ranker:
        self.mean, self.std = x.mean(0), x.std(0) + 1e-9
        z = np.c_[np.ones(len(x)), (x - self.mean) / self.std]
        w = np.zeros(z.shape[1])
        pos = y.mean()
        sample_w = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))  # balance the rare defects
        for _ in range(iters):
            p = 1 / (1 + np.exp(-z @ w))
            grad = z.T @ (sample_w * (p - y)) / len(y) + l2 * np.r_[0, w[1:]]
            w -= lr * grad
        self.w = w
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        z = np.c_[np.ones(len(x)), (x - self.mean) / self.std]
        return 1 / (1 + np.exp(-z @ self.w))

    def to_dict(self) -> dict:
        return {
            "features": FEATURES,
            "intercept": float(self.w[0]),
            "weights": dict(zip(FEATURES, map(float, self.w[1:]), strict=True)),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Ranker:
        return cls(
            np.r_[d["intercept"], [d["weights"][f] for f in FEATURES]],
            np.array(d["mean"]),
            np.array(d["std"]),
        )


def measure_all(
    golden: np.ndarray, bare: np.ndarray, aligned: np.ndarray, placements: list[Placement]
) -> list[Measurement]:
    return [measure(golden, bare, aligned, pl) for pl in placements]


def rank(ms: list[Measurement], ranker: Ranker, normal: Normal) -> list[Measurement]:
    scores = ranker.score(np.array([normal.z(m) for m in ms]))
    for m, s in zip(ms, scores, strict=True):
        m.score = float(s)
    return sorted(ms, key=lambda m: m.score, reverse=True)


# ---- 3. judges ---------------------------------------------------------------------------------------------------

Verdict = Literal["ok", "missing", "rotated", "wrong_part", "shifted", "damaged_or_tombstoned"]


class PartVerdict(BaseModel):
    verdict: Verdict = Field(description="ok, or the defect you see")
    rotation_deg: int | None = Field(description="If rotated: how far, clockwise (90, 180 or 270)")
    observation: str = Field(description="One sentence: what you see that decides it")


@dataclass
class Thresholds:
    missing_margin: float = 0.05
    rotation_margin: float = 0.08
    shift_mm: float = 0.22
    wrong_z: float = 6.0  # how far from a good part of the package, in robust standard deviations
    unsure_z: float = 3.5  # between this and wrong_z the measurements cannot decide


def unsure(m: Measurement, t: Thresholds, normal: Normal) -> bool:
    """Classical vision passes the part but cannot vouch for it: the body looks unusual (a marking,
    a colour, a pin-1 dot) without being clearly wrong. Such parts go to the second judge."""
    z = normal.z(m)
    return max(z[0], z[1], z[5], z[6]) > t.unsure_z


def classical_verdict(m: Measurement, t: Thresholds, normal: Normal) -> PartVerdict:
    if m.ncc_bare > m.ncc_golden + t.missing_margin:
        return PartVerdict(verdict="missing", rotation_deg=None, observation="looks like the bare pads")
    best = m.best_rotation
    if best != 0 and m.rotation_ncc[best] - m.rotation_ncc[0] > t.rotation_margin:
        if best in (90, 270) or m.polarised:
            return PartVerdict(
                verdict="rotated", rotation_deg=best, observation=f"matches the part turned {best}°"
            )
        return PartVerdict(verdict="ok", rotation_deg=180, observation="turned 180°, part is not polarised")
    if m.shift_mm > t.shift_mm:
        return PartVerdict(verdict="shifted", rotation_deg=None, observation=f"offset {m.shift_mm:.2f} mm")
    z = normal.z(m)
    if max(z[0], z[1], z[5], z[6]) > t.wrong_z:
        return PartVerdict(
            verdict="wrong_part", rotation_deg=None, observation="does not match the golden part"
        )
    return PartVerdict(verdict="ok", rotation_deg=None, observation="matches the golden part")


def classical_judge(t: Thresholds, normal: Normal) -> FunctionJudge:
    def fn(request: Request) -> PartVerdict:
        return classical_verdict(request.data["measurement"], t, normal)

    return FunctionJudge("classical", fn, version="/".join(str(v) for v in vars(t).values()))


def part_request(board: Board, aligned: np.ndarray, m: Measurement, scale: int = 3) -> Request:
    """What a vision model sees: the golden part, the bare pads and the part as built, enlarged."""
    pl = next(p for p in board.placements if p.refdes == m.refdes)

    def png(img: np.ndarray, label: str) -> Image:
        big = cv2.resize(_crop(img, pl, 0.4), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".png", big)
        return Image(buf.tobytes(), "image/png", label, big.shape[1], big.shape[0])

    images = [
        png(board.golden, "Golden board (correct):"),
        png(board.bare, "Bare pads with paste (no part):"),
        png(aligned, "As built (judge this one):"),
    ]
    text = (
        f"Part {pl.refdes}, package {pl.package}, {'polarised' if pl.pkg.polarised else 'not polarised'}"
        f"{', marking ' + pl.marking if pl.marking else ''}. Is the part as built correct?"
    )
    return Request("inspect-part", text, PartVerdict, images, {"measurement": m})


# ---- 4. reconcile ----------------------------------------------------------------------------------------------


def reconcile(v: PartVerdict, m: Measurement, t: Thresholds) -> tuple[str, str]:
    """Code checks the judge against what it can measure. Returns (decision, reason):
    decision is pass, fail (a defect) or operator (they disagree)."""
    if v.verdict == "ok":
        if m.ncc_bare > m.ncc_golden + 2 * t.missing_margin:
            return "operator", "judge passed a part that measures as missing"
        return "pass", v.observation
    if v.verdict == "missing" and m.ncc_bare < m.ncc_golden - t.missing_margin:
        return "operator", "judge says missing but a part is measured on the pads"
    if (
        v.verdict == "rotated"
        and v.rotation_deg in (90, 180, 270)
        and m.best_rotation != v.rotation_deg
        and m.rotation_ncc[v.rotation_deg] < m.rotation_ncc[0]
    ):
        return "operator", f"judge says {v.rotation_deg}° but the part matches best at {m.best_rotation}°"
    return "fail", f"{v.verdict}: {v.observation}"

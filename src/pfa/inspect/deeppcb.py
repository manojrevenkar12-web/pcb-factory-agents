"""The classical stage on real board images: DeepPCB (Tang et al., 2019).

DeepPCB holds 1,500 real PCB image pairs from a linear-scan CCD: a defect-free template and a
tested image, with every defect boxed and typed (open, short, mousebite, spur, spurious copper,
pin hole). The published pairs are already registered. To test the alignment step as well,
each tested image is first put through the same camera model as the synthetic boards (rotation up
to 1.5°, shift up to 15 px, blur, noise, uneven light); then:

- **naive**: subtract the template from the misregistered photo;
- **aligned**: register with ECC first, then subtract.

Differences become candidate regions (connected components), ranked by strength. A defect counts
as found when a candidate overlaps its box with IoU >= 0.33 (the dataset's own protocol) or, more
leniently, contains its centre. The sign of the difference separates missing copper (open,
mousebite, pin hole) from extra copper (short, spur, spurious copper) with no model at all.

The dataset is for research use; it is downloaded at run time, never redistributed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pfa.inspect.pipeline import align

TYPES = {1: "open", 2: "short", 3: "mousebite", 4: "spur", 5: "spurious copper", 6: "pin hole"}
MISSING_COPPER = {1, 3, 6}  # template minus test is negative inside these boxes


@dataclass
class Pair:
    name: str
    template: np.ndarray
    test: np.ndarray
    boxes: list[tuple[int, int, int, int, int]]


def load(root: Path, split: str = "test") -> list[Pair]:
    data = root / "PCBData"
    pairs = []
    for line in (data / f"{split}.txt").read_text().splitlines():
        if not line.strip():
            continue
        image, ann = line.split()
        base = data / image[:-4]
        boxes = []
        for row in (data / ann).read_text().splitlines():
            if row.strip():
                x1, y1, x2, y2, t = map(int, row.replace(",", " ").split())
                boxes.append((x1, y1, x2, y2, t))
        pairs.append(
            Pair(
                base.name,
                cv2.imread(f"{base}_temp.jpg", cv2.IMREAD_GRAYSCALE),
                cv2.imread(f"{base}_test.jpg", cv2.IMREAD_GRAYSCALE),
                boxes,
            )
        )
    return pairs


def photograph(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-1.5, 1.5), rng.uniform(0.995, 1.005))
    m[:, 2] += [rng.uniform(-15, 15), rng.uniform(-15, 15)]
    out = cv2.warpAffine(
        img.astype(np.float32), m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = out * (1 + rng.uniform(-0.1, 0.1) * (xx / w - 0.5) + rng.uniform(-0.1, 0.1) * (yy / h - 0.5))
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 0.8))
    out += np.random.default_rng(rng.randint(0, 2**31)).normal(0, 4, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


@dataclass
class Candidate:
    box: tuple[int, int, int, int]
    strength: float
    sign: float  # mean of template - photo: negative means copper is missing


def candidates(
    template: np.ndarray, test: np.ndarray, threshold: int = 90, min_area: int = 80
) -> list[Candidate]:
    """Regions where the registered photo differs from the template. Threshold and minimum area
    were chosen on the training split (a tolerance band around template edges was tried and
    dropped: it hid pin holes)."""
    t, x = template.astype(np.int16), test.astype(np.int16)
    diff = t - x
    mask = (np.abs(diff) > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # drop 1-px edge slivers
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))  # join the pieces of one defect
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for k in range(1, n):
        x0, y0, w, h, area = stats[k]
        if area < min_area:
            continue
        region = labels[y0 : y0 + h, x0 : x0 + w] == k
        d = diff[y0 : y0 + h, x0 : x0 + w][region]
        out.append(
            Candidate((int(x0), int(y0), int(x0 + w), int(y0 + h)), float(np.abs(d).sum()), float(d.mean()))
        )
    return sorted(out, key=lambda c: c.strength, reverse=True)


def iou(a: tuple, b: tuple) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def contains(box: tuple, point: tuple[float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def evaluate(pairs: list[Pair], aligned: bool, seed: int = 0) -> dict:
    rng = random.Random(seed)
    gt_total = found_iou = found_centre = polarity_ok = 0
    n_candidates = matched_candidates = 0
    area_flagged = 0.0
    for pair in pairs:
        photo = photograph(pair.test, rng)
        if aligned:
            g3, p3 = cv2.cvtColor(pair.template, cv2.COLOR_GRAY2BGR), cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
            photo = cv2.cvtColor(align(g3, p3, levels=(2, 1))[0], cv2.COLOR_BGR2GRAY)
        cands = candidates(pair.template, photo)
        n_candidates += len(cands)
        area_flagged += sum((c.box[2] - c.box[0]) * (c.box[3] - c.box[1]) for c in cands) / pair.template.size
        used = set()
        for x1, y1, x2, y2, t in pair.boxes:
            gt_total += 1
            best = max(range(len(cands)), key=lambda k: iou(cands[k].box, (x1, y1, x2, y2)), default=None)
            if best is not None and iou(cands[best].box, (x1, y1, x2, y2)) >= 0.33:
                found_iou += 1
            hits = [k for k, c in enumerate(cands) if contains(c.box, ((x1 + x2) / 2, (y1 + y2) / 2))]
            if hits:
                found_centre += 1
                used.update(hits)
                polarity_ok += (cands[hits[0]].sign < 0) == (t in MISSING_COPPER)
        matched_candidates += len(used)
    precision = matched_candidates / n_candidates if n_candidates else 0.0
    recall_iou = found_iou / gt_total
    recall = found_centre / gt_total
    return {
        "pairs": len(pairs),
        "defects": gt_total,
        "candidates_per_image": n_candidates / len(pairs),
        "recall_iou_0.33": recall_iou,
        "recall_centre": recall,
        "precision": precision,
        "f_score": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "share_of_image_flagged": area_flagged / len(pairs),
        "polarity_accuracy": polarity_ok / max(1, found_centre),
    }


def run(root: Path) -> dict:
    pairs = load(root, "test")
    return {"naive": evaluate(pairs, aligned=False), "aligned": evaluate(pairs, aligned=True)}

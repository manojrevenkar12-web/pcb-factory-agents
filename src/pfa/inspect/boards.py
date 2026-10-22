"""Assembled circuit boards with known defects, photographed by an imperfect camera.

Every board comes from a placement file (centroid, rotation, package, marking) like the ones
customers upload. The factory has three images of it:

- the **golden** board: the design built correctly, photographed once;
- the **bare** board: pads and solder paste, before any part was placed;
- the **test** photo of the panel just built, which may hold defects: missing parts, parts rotated
  90 or 180 degrees, the wrong part (another value, another chip), parts shifted off their pads,
  and tombstones (a chip standing on one end).

Real boards also differ harmlessly: parts sit a little off-centre and slightly turned, solder
fillets vary, a resistor is placed 180 degrees round (electrically identical, but its printed code
is now upside down), and the camera adds misregistration, uneven light, blur, noise and JPEG
compression. A good inspector ignores all of that and finds the defects.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np

PPMM = 20  # pixels per millimetre
BOARD_MM = (70.0, 50.0)

MASK = (58, 92, 30)  # BGR solder-mask green
TRACE = (70, 118, 44)
SILK = (235, 238, 240)
PAD = (178, 188, 196)  # HASL silver
PASTE = (150, 152, 156)
FIDUCIAL = (110, 170, 205)


@dataclass(frozen=True)
class Package:
    name: str
    kind: str  # resistor, capacitor, diode, led, sot23, soic, qfn
    body: tuple[float, float]  # length (x), width (y) in mm
    pads: tuple[tuple[float, float, float, float], ...]  # cx, cy, w, h in part coordinates (mm)
    polarised: bool

    @property
    def two_terminal(self) -> bool:
        return len(self.pads) == 2

    def extent(self) -> float:
        xs = [abs(cx) + w / 2 for cx, _, w, _ in self.pads] + [self.body[0] / 2]
        ys = [abs(cy) + h / 2 for _, cy, _, h in self.pads] + [self.body[1] / 2]
        return max(max(xs), max(ys))


def _soic(n: int, length: float) -> tuple:
    per = n // 2
    xs = [(i - (per - 1) / 2) * 1.27 for i in range(per)]
    return tuple((x, 2.7, 0.6, 1.5) for x in xs) + tuple((x, -2.7, 0.6, 1.5) for x in reversed(xs))


def _qfn(per_side: int, size: float) -> tuple:
    ps = [(i - (per_side - 1) / 2) * 0.5 for i in range(per_side)]
    e = size / 2 - 0.2
    return (
        tuple((p, e, 0.25, 0.6) for p in ps)
        + tuple((e, p, 0.6, 0.25) for p in ps)
        + tuple((p, -e, 0.25, 0.6) for p in ps)
        + tuple((-e, p, 0.6, 0.25) for p in ps)
    )


PACKAGES = {
    "R0402": Package("R0402", "resistor", (1.0, 0.5), ((-0.5, 0, 0.55, 0.6), (0.5, 0, 0.55, 0.6)), False),
    "R0603": Package("R0603", "resistor", (1.6, 0.8), ((-0.8, 0, 0.8, 0.95), (0.8, 0, 0.8, 0.95)), False),
    "C0603": Package("C0603", "capacitor", (1.6, 0.8), ((-0.8, 0, 0.8, 0.95), (0.8, 0, 0.8, 0.95)), False),
    "C0805": Package("C0805", "capacitor", (2.0, 1.25), ((-0.95, 0, 0.9, 1.4), (0.95, 0, 0.9, 1.4)), False),
    "SOD123": Package("SOD123", "diode", (2.7, 1.6), ((-1.65, 0, 0.9, 1.2), (1.65, 0, 0.9, 1.2)), True),
    "LED0805": Package("LED0805", "led", (2.0, 1.25), ((-0.95, 0, 0.9, 1.4), (0.95, 0, 0.9, 1.4)), True),
    "SOT23": Package(
        "SOT23",
        "sot23",
        (2.9, 1.3),
        ((-0.95, 1.1, 0.6, 0.8), (0.95, 1.1, 0.6, 0.8), (0.0, -1.1, 0.6, 0.8)),
        True,
    ),
    "SOIC8": Package("SOIC8", "soic", (4.9, 3.9), _soic(8, 4.9), True),
    "QFN16": Package("QFN16", "qfn", (3.0, 3.0), _qfn(4, 3.0), True),
}
MIX = [
    "R0402",
    "R0603",
    "R0603",
    "R0603",
    "C0603",
    "C0603",
    "C0805",
    "SOD123",
    "LED0805",
    "SOT23",
    "SOT23",
    "SOIC8",
    "QFN16",
]
RES_CODES = ["103", "472", "101", "222", "331", "104", "4R7", "1001", "4701", "2202"]
IC_MARKS = {
    "soic": ["LM358", "NE555", "TL072", "MCP2551", "SN65HVD"],
    "qfn": ["TPS62", "AP2112", "CP2102", "MAX3"],
    "sot23": ["A1", "K72", "J3Y", "6A", "M2"],
}
CAP_COLOURS = [(120, 165, 200), (95, 135, 175), (140, 175, 205)]
DEFECTS = ["missing", "rotated", "wrong_part", "shifted", "tombstone"]


@dataclass
class Placement:
    refdes: str
    package: str
    x: float  # mm, centre
    y: float
    rotation: int  # degrees, 0/90/180/270
    marking: str
    colour: tuple[int, int, int] = (0, 0, 0)

    @property
    def pkg(self) -> Package:
        return PACKAGES[self.package]

    def bbox_px(self, margin_mm: float = 0.5) -> tuple[int, int, int, int]:
        r = self.pkg.extent() + margin_mm
        x0, y0 = (self.x - r) * PPMM, (self.y - r) * PPMM
        size = 2 * r * PPMM
        return int(x0), int(y0), int(math.ceil(size)), int(math.ceil(size))


@dataclass
class Built:
    """What actually happened to a placement on this board."""

    placement: Placement
    label: str = "ok"  # ok or one of DEFECTS
    dx: float = 0.0
    dy: float = 0.0
    drot: float = 0.0
    marking: str | None = None
    colour: tuple[int, int, int] | None = None
    benign: str = ""  # harmless difference, e.g. "rotated 180, not polarised"


@dataclass
class Board:
    seed: int
    placements: list[Placement]
    built: list[Built]
    golden: np.ndarray
    bare: np.ndarray
    test: np.ndarray
    warp: np.ndarray  # 2x3: golden pixel -> test pixel
    meta: dict = field(default_factory=dict)

    @property
    def labels(self) -> dict[str, str]:
        return {b.placement.refdes: b.label for b in self.built}


# ---- design -------------------------------------------------------------------------------------------------


def design(rng: random.Random, n_parts: int = 60) -> list[Placement]:
    cols, rows = 10, 7
    cw, rh = BOARD_MM[0] / cols, BOARD_MM[1] / rows
    cells = [
        (c, r)
        for c in range(cols)
        for r in range(rows)
        if (c, r) not in {(0, 0), (cols - 1, 0), (0, rows - 1)}
    ]  # fiducial corners
    rng.shuffle(cells)
    counters: dict[str, int] = {}
    prefix = {
        "resistor": "R",
        "capacitor": "C",
        "diode": "D",
        "led": "LED",
        "sot23": "Q",
        "soic": "U",
        "qfn": "U",
    }
    out = []
    for c, r in cells[:n_parts]:
        pkg = PACKAGES[rng.choice(MIX)]
        p = prefix[pkg.kind]
        counters[p] = counters.get(p, 0) + 1
        if pkg.kind == "resistor":
            marking = rng.choice(RES_CODES)
        elif pkg.kind in IC_MARKS:
            marking = rng.choice(IC_MARKS[pkg.kind])
        else:
            marking = ""
        colour = rng.choice(CAP_COLOURS) if pkg.kind == "capacitor" else (0, 0, 0)
        out.append(
            Placement(
                f"{p}{counters[p]}",
                pkg.name,
                (c + 0.5) * cw + rng.uniform(-0.4, 0.4),
                (r + 0.5) * rh + rng.uniform(-0.4, 0.4),
                rng.choice([0, 90, 180, 270]),
                marking,
                colour,
            )
        )
    return out


# ---- drawing ---------------------------------------------------------------------------------------------------


def _to_px(pts_mm: np.ndarray, cx: float, cy: float, rot_deg: float) -> np.ndarray:
    a = math.radians(rot_deg)
    rotm = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return ((pts_mm @ rotm.T) + [cx, cy]) * PPMM


def _rect(cx, cy, w, h) -> np.ndarray:
    return np.array(
        [
            [cx - w / 2, cy - h / 2],
            [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2],
            [cx - w / 2, cy + h / 2],
        ],
        dtype=float,
    )


def _fill(img: np.ndarray, poly_px: np.ndarray, colour) -> None:
    cv2.fillPoly(img, [np.round(poly_px * 16).astype(np.int32)], colour, lineType=cv2.LINE_AA, shift=4)


def _shade(colour, rng: random.Random, amount: int = 8) -> tuple[int, int, int]:
    return tuple(int(np.clip(c + rng.randint(-amount, amount), 0, 255)) for c in colour)


def draw_pads(img: np.ndarray, pl: Placement, paste: bool, rng: random.Random) -> None:
    for cx, cy, w, h in pl.pkg.pads:
        _fill(img, _to_px(_rect(cx, cy, w, h), pl.x, pl.y, pl.rotation), _shade(PAD, rng, 6))
        if paste:
            _fill(
                img, _to_px(_rect(cx, cy, w * 0.82, h * 0.82), pl.x, pl.y, pl.rotation), _shade(PASTE, rng, 6)
            )


def _body_patch(pkg: Package, marking: str, colour, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """The part seen from above, in its own coordinates: BGR image and alpha mask."""
    ext = pkg.extent() + 0.3
    size = int(math.ceil(2 * ext * PPMM))
    img = np.zeros((size, size, 3), np.uint8)
    alpha = np.zeros((size, size), np.uint8)
    c = size / 2

    def rect(cx, cy, w, h, col) -> None:
        pts = _rect(cx, cy, w, h) * PPMM + c
        cv2.fillPoly(img, [np.round(pts * 16).astype(np.int32)], col, cv2.LINE_AA, 4)
        cv2.fillPoly(alpha, [np.round(pts * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)

    L, W = pkg.body
    if pkg.kind in ("resistor", "capacitor", "led"):
        body = {"resistor": (25, 25, 28), "capacitor": colour, "led": (215, 225, 222)}[pkg.kind]
        rect(0, 0, L, W, _shade(body, rng, 6))
        cap = min(0.35, L * 0.22)
        for s in (-1, 1):
            rect(s * (L / 2 - cap / 2), 0, cap, W, _shade((200, 205, 210), rng, 8))
        if pkg.kind == "led":
            rect(-L / 2 + cap + 0.15, 0, 0.18, W * 0.8, colour if colour != (0, 0, 0) else (60, 170, 60))
    elif pkg.kind == "diode":
        for s in (-1, 1):
            rect(s * (L / 2 + 0.2), 0, 0.6, 0.7, (195, 200, 205))
        rect(0, 0, L, W, _shade((30, 30, 32), rng, 5))
        rect(-L / 2 + 0.35, 0, 0.3, W, colour if colour != (0, 0, 0) else (205, 210, 212))
    elif pkg.kind == "sot23":
        for cx, cy, _, _ in pkg.pads:
            rect(cx, cy * 0.75, 0.4, 0.7, (195, 200, 205))
        rect(0, 0, L, W, _shade((32, 32, 35), rng, 5))
    elif pkg.kind == "soic":
        for cx, cy, _, _ in pkg.pads:
            rect(cx, cy * 0.85, 0.42, 1.1, (195, 200, 205))
        rect(0, 0, L, W, _shade((30, 30, 33), rng, 5))
        cv2.circle(
            img,
            (int(c + (-L / 2 + 0.6) * PPMM), int(c + (W / 2 - 0.6) * PPMM)),
            int(0.3 * PPMM),
            (70, 70, 74),
            -1,
            cv2.LINE_AA,
        )
    elif pkg.kind == "qfn":
        rect(0, 0, L, W, _shade((34, 34, 37), rng, 5))
        cv2.circle(
            img,
            (int(c + (-L / 2 + 0.45) * PPMM), int(c + (W / 2 - 0.45) * PPMM)),
            int(0.2 * PPMM),
            (85, 85, 90),
            -1,
            cv2.LINE_AA,
        )
    if marking:
        scale = {"resistor": 0.011, "sot23": 0.011, "soic": 0.016, "qfn": 0.012}.get(pkg.kind, 0.012) * PPMM
        thick = 1
        (tw, th), _ = cv2.getTextSize(marking, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        if tw > L * PPMM * 0.9:
            scale *= L * PPMM * 0.9 / tw
            (tw, th), _ = cv2.getTextSize(marking, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(
            img,
            marking,
            (int(c - tw / 2), int(c + th / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (215, 215, 215),
            thick,
            cv2.LINE_AA,
        )
    return img, alpha


def _composite(
    board: np.ndarray, patch: np.ndarray, alpha: np.ndarray, x_mm: float, y_mm: float, rot: float
) -> None:
    size = patch.shape[0]
    c = size / 2
    # patch pixel -> board pixel: rotate about the patch centre, then translate to the part centre
    m = cv2.getRotationMatrix2D((c, c), -rot, 1.0)
    m[:, 2] += [x_mm * PPMM - c, y_mm * PPMM - c]
    h, w = board.shape[:2]
    x0, y0 = int(x_mm * PPMM - size), int(y_mm * PPMM - size)
    x1, y1 = x0 + 2 * size, y0 + 2 * size
    x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    m_local = m.copy()
    m_local[:, 2] -= [x0c, y0c]
    region = (y1c - y0c, x1c - x0c)
    warped = cv2.warpAffine(patch, m_local, region[::-1], flags=cv2.INTER_LINEAR)
    a = (
        cv2.warpAffine(alpha, m_local, region[::-1], flags=cv2.INTER_LINEAR).astype(np.float32)[..., None]
        / 255
    )
    roi = board[y0c:y1c, x0c:x1c].astype(np.float32)
    board[y0c:y1c, x0c:x1c] = (roi * (1 - a) + warped.astype(np.float32) * a).astype(np.uint8)


def draw_part(board: np.ndarray, b: Built, rng: random.Random) -> None:
    pl = b.placement
    if b.label == "missing":
        return
    pkg = pl.pkg
    marking = b.marking if b.marking is not None else pl.marking
    colour = b.colour if b.colour is not None else pl.colour
    if b.label == "tombstone":  # standing on one terminal: a small bright end cap over one pad
        cx, cy, w, h = pkg.pads[0]
        pts = _to_px(_rect(cx * 0.7, cy, pkg.body[1] * 0.9, pkg.body[1]), pl.x, pl.y, pl.rotation)
        _fill(board, pts, (210, 214, 218))
        inner = _to_px(_rect(cx * 0.7, cy, pkg.body[1] * 0.5, pkg.body[1] * 0.75), pl.x, pl.y, pl.rotation)
        _fill(board, inner, (40, 40, 44) if pkg.kind == "resistor" else colour)
        return
    for cx, cy, w, h in pkg.pads:  # solder fillets where the terminations meet the paste
        pts = _to_px(_rect(cx, cy, w * 0.9, h * 0.9), pl.x + b.dx, pl.y + b.dy, pl.rotation + b.drot)
        _fill(board, pts, _shade((205, 208, 212), rng, 10))
    patch, alpha = _body_patch(pkg, marking, colour, rng)
    _composite(board, patch, alpha, pl.x + b.dx, pl.y + b.dy, pl.rotation + b.drot)


def render(
    placements: list[Placement], built: list[Built] | None, rng: random.Random, parts: bool = True
) -> np.ndarray:
    w, h = int(BOARD_MM[0] * PPMM), int(BOARD_MM[1] * PPMM)
    trace_rng = random.Random(len(placements))  # the copper is part of the design: same on every board
    img = np.full((h, w, 3), MASK, np.uint8)
    noise = np.random.default_rng(trace_rng.randint(0, 10**6)).normal(0, 3, (h, w, 1))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    for _ in range(45):  # traces under the mask
        a, b = trace_rng.choice(placements), trace_rng.choice(placements)
        cv2.line(
            img, (int(a.x * PPMM), int(a.y * PPMM)), (int(a.x * PPMM), int(b.y * PPMM)), TRACE, 6, cv2.LINE_AA
        )
        cv2.line(
            img, (int(a.x * PPMM), int(b.y * PPMM)), (int(b.x * PPMM), int(b.y * PPMM)), TRACE, 6, cv2.LINE_AA
        )
    for fx, fy in ((3, 3), (BOARD_MM[0] - 3, 3), (3, BOARD_MM[1] - 3)):
        cv2.circle(img, (int(fx * PPMM), int(fy * PPMM)), int(1.0 * PPMM), (40, 70, 25), -1, cv2.LINE_AA)
        cv2.circle(img, (int(fx * PPMM), int(fy * PPMM)), int(0.5 * PPMM), FIDUCIAL, -1, cv2.LINE_AA)
    for pl in placements:  # silkscreen outline and reference designator
        r = pl.pkg.extent() + 0.25
        x0, y0, x1, y1 = [int(v * PPMM) for v in (pl.x - r, pl.y - r, pl.x + r, pl.y + r)]
        cv2.rectangle(img, (x0, y0), (x1, y1), SILK, 1, cv2.LINE_AA)
        cv2.putText(img, pl.refdes, (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, SILK, 1, cv2.LINE_AA)
        if pl.pkg.polarised:
            cx, cy = _to_px(np.array([[-r + 0.15, r - 0.15]]), pl.x, pl.y, pl.rotation)[0]
            cv2.circle(img, (int(cx), int(cy)), 3, SILK, -1, cv2.LINE_AA)
    for pl in placements:
        draw_pads(img, pl, paste=True, rng=rng)
    if parts:
        for b in built or [Built(p) for p in placements]:
            draw_part(img, b, rng)
    return img


# ---- camera ----------------------------------------------------------------------------------------------------


def camera(img: np.ndarray, rng: random.Random, misregister: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Photographs a rendered board: misregistration, uneven light, blur, noise, JPEG."""
    h, w = img.shape[:2]
    if misregister:
        angle, scale = rng.uniform(-2.0, 2.0), rng.uniform(0.99, 1.01)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        m[:, 2] += [rng.uniform(-25, 25), rng.uniform(-25, 25)]
    else:
        m = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    photo = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gx, gy = rng.uniform(-0.18, 0.18), rng.uniform(-0.18, 0.18)
    light = 1 + gx * (xx / w - 0.5) + gy * (yy / h - 0.5)
    photo = photo.astype(np.float32) * light[..., None] * rng.uniform(0.92, 1.08)
    np_rng = np.random.default_rng(rng.randint(0, 2**31))
    photo = cv2.GaussianBlur(photo, (0, 0), rng.uniform(0.5, 0.9))
    photo = np.clip(photo + np_rng.normal(0, 3.5, photo.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), m


# ---- a built board -------------------------------------------------------------------------------------------


def build(seed: int, defect_rate: float = 0.04, n_parts: int = 60) -> Board:
    rng = random.Random(seed)
    placements = design(random.Random(seed // 1000), n_parts)  # boards in a batch share a design
    built: list[Built] = []
    for pl in placements:
        b = Built(pl, dx=rng.gauss(0, 0.04), dy=rng.gauss(0, 0.04), drot=rng.gauss(0, 0.8))
        pkg = pl.pkg
        if rng.random() < defect_rate:
            choices = ["missing", "rotated", "wrong_part", "shifted"] + (
                ["tombstone"] if pkg.two_terminal else []
            )
            weights = [35, 25, 20, 12] + ([8] if pkg.two_terminal else [])
            if not pkg.polarised and pkg.two_terminal:
                choices, weights = (
                    [c for c in choices if c != "rotated"],
                    [w for c, w in zip(choices, weights, strict=True) if c != "rotated"],
                )
            b.label = rng.choices(choices, weights)[0]
            if b.label == "rotated":
                b.drot += 90 if pkg.kind in ("sot23", "soic", "qfn") and rng.random() < 0.6 else 180
            elif b.label == "wrong_part":
                if pkg.kind == "resistor":
                    b.marking = rng.choice([m for m in RES_CODES if m != pl.marking])
                elif pkg.kind in IC_MARKS:
                    b.marking = rng.choice([m for m in IC_MARKS[pkg.kind] if m != pl.marking])
                elif pkg.kind == "capacitor":
                    b.colour = (40, 60, 85)  # a dark tantalum-coloured part on a ceramic's pads
                elif pkg.kind == "led":
                    b.colour = (40, 40, 210)  # red instead of green
                else:
                    b.colour = (40, 120, 200)  # diode with an orange band: a different family
            elif b.label == "shifted":
                a = rng.uniform(0, 2 * math.pi)
                d = rng.uniform(0.35, 0.6)
                b.dx, b.dy = d * math.cos(a), d * math.sin(a)
        elif not pkg.polarised and rng.random() < 0.12:
            b.drot += 180
            b.benign = "rotated 180, not polarised"
        built.append(b)

    golden_rng = random.Random(seed // 1000 + 1)
    golden_img = render(placements, None, golden_rng)
    golden, _ = camera(golden_img, golden_rng, misregister=False)
    bare, _ = camera(render(placements, None, golden_rng, parts=False), golden_rng, misregister=False)
    test, warp = camera(render(placements, built, rng), rng)
    return Board(seed, placements, built, golden, bare, test, warp)

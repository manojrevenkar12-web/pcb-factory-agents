"""Reading the KiCad libraries: real parts (symbols) and real land patterns (footprints).

Symbols (kicad-symbols, KiCad 5 format) give each part's name, description, keywords, datasheet
link, pin numbers and, for package-specific parts, the footprint it must be built on. Footprints
(kicad-footprints) give the copper pads, the body outline (fabrication layer) and the courtyard.
Both libraries are CC-BY-SA 4.0 with an exception for designs; they are downloaded at run time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass
class Part:
    name: str
    library: str
    description: str
    keywords: str
    datasheet: str
    footprint: str  # "Library:Name", the ground truth
    pins: frozenset[str]

    @property
    def pin_count(self) -> int:
        return len(self.pins)


@dataclass
class Pad:
    number: str
    kind: str  # smd, thru_hole, np_thru_hole, connect
    x: float
    y: float
    w: float
    h: float


@dataclass
class Footprint:
    library: str
    name: str
    smd: bool
    pads: list[Pad]
    fab: tuple[float, float, float, float] | None  # body outline bounds: x0, y0, x1, y1
    courtyard: tuple[float, float, float, float] | None
    tags: str = ""
    description: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"{self.library}:{self.name}"

    @property
    def numbered_pads(self) -> set[str]:
        return {p.number for p in self.pads if p.number and p.kind != "np_thru_hole"}

    @property
    def body_mm(self) -> tuple[float, float] | None:
        box = self.fab or self.courtyard
        if box is None:
            return None
        return round(box[2] - box[0], 2), round(box[3] - box[1], 2)


_DEF = re.compile(r"^DEF (\S+) .*?\n(.*?)^ENDDEF", re.M | re.S)
_CMP = re.compile(r"^\$CMP (\S+)\n(.*?)^\$ENDCMP", re.M | re.S)


def _dcm(path: Path) -> dict[str, dict[str, str]]:
    out = {}
    if not path.exists():
        return out
    for m in _CMP.finditer(path.read_text(errors="ignore")):
        fields = {}
        for line in m.group(2).splitlines():
            if len(line) > 2 and line[1] == " ":
                fields[line[0]] = line[2:].strip()
        out[m.group(1)] = fields
    return out


def load_parts(symbols: Path) -> list[Part]:
    parts = []
    for lib in sorted(symbols.glob("*.lib")):
        docs = _dcm(lib.with_suffix(".dcm"))
        for m in _DEF.finditer(lib.read_text(errors="ignore")):
            body = m.group(2)
            f2 = re.search(r'^F2 "([^"]*)"', body, re.M)
            if not f2 or not f2.group(1) or ":" not in f2.group(1):
                continue
            pins = frozenset(x.split()[2] for x in re.findall(r"^X .*$", body, re.M) if len(x.split()) > 2)
            names = [m.group(1)]
            alias = re.search(r"^ALIAS (.*)$", body, re.M)
            if alias:
                names += alias.group(1).split()
            for name in names:
                d = docs.get(name, {})
                parts.append(
                    Part(name, lib.stem, d.get("D", ""), d.get("K", ""), d.get("F", ""), f2.group(1), pins)
                )
    return parts


_NUM = r"(-?\d+(?:\.\d+)?)"


def parse_footprint(path: Path) -> Footprint:
    text = path.read_text(errors="ignore")
    pads = []
    for m in re.finditer(
        r'\(pad "?([^"\s]*)"? (\w+) \w+ \(at '
        + _NUM
        + " "
        + _NUM
        + r"(?: -?\d+(?:\.\d+)?)?\) \(size "
        + _NUM
        + " "
        + _NUM
        + r"\)",
        text,
    ):
        pads.append(
            Pad(
                m.group(1),
                m.group(2),
                float(m.group(3)),
                float(m.group(4)),
                float(m.group(5)),
                float(m.group(6)),
            )
        )

    def bounds(layer: str) -> tuple[float, float, float, float] | None:
        xs, ys = [], []
        for m in re.finditer(
            r"\(fp_(?:line|rect) \(start "
            + _NUM
            + " "
            + _NUM
            + r"\) \(end "
            + _NUM
            + " "
            + _NUM
            + r"\) \(layer "
            + layer
            + r"\)",
            text,
        ):
            xs += [float(m.group(1)), float(m.group(3))]
            ys += [float(m.group(2)), float(m.group(4))]
        return (min(xs), min(ys), max(xs), max(ys)) if xs else None

    descr = re.search(r'\(descr "([^"]*)"\)', text)
    tags = re.search(r'\(tags "?([^")]*)"?\)', text)
    return Footprint(
        path.parent.stem.removesuffix(".pretty") if path.parent.suffix else path.parent.name,
        path.stem,
        "(attr smd)" in text,
        pads,
        bounds("F.Fab"),
        bounds("F.CrtYd"),
        tags.group(1) if tags else "",
        descr.group(1) if descr else "",
    )


@lru_cache(maxsize=4)
def footprint_index(root: str) -> dict[str, Path]:
    """Every footprint file by "Library:Name"."""
    out = {}
    for lib in Path(root).glob("*.pretty"):
        for f in lib.glob("*.kicad_mod"):
            out[f"{lib.stem}:{f.stem}"] = f
    return out

"""Programming a new part for the pick-and-place machines, and checking the customer's footprint.

A customer's design names, for each part, a footprint: the copper pads the part will be soldered
to. When the factory meets a part for the first time it must work out how the machine should pick,
centre and place it, and a wrong footprint in the design means scrapped boards. So:

1. **The judge** reads the part's description (and, given a gateway with network access, its
   datasheet) and says what package it comes in: family, pin count, body size and pitch when stated.
2. **Code checks** two second sources against each other and against the judge:
   - the schematic symbol's pin numbers must be exactly the footprint's pad numbers;
   - the footprint must be a land pattern for the package the judge read (same family, pins, and
     body size and pitch where both are known).
3. **Code writes** the machine record from the footprint's geometry: body size, pitch, nozzle,
   centring method and tape width. Anything that does not reconcile goes to a person first.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from pfa.judges import FunctionJudge, Request
from pfa.parts.kicad import Footprint, Part, footprint_index, parse_footprint

VARIANTS = ("_ThermalVias", "_HandSoldering", "_Pad", "_TopEP", "_EP_", "_Stencil")
IMPLIED_PINS = {
    "SOT-23": 3,
    "SOT-223": 4,
    "SOT-89": 3,
    "SOD-123": 2,
    "SOD-323": 2,
    "SOT-323": 3,
    "SOT-563": 6,
    "SC-70": 3,
    "TO-252": 3,
    "TO-263": 3,
    "SMA": 2,
    "SMB": 2,
    "SMC": 2,
}
_NAME = re.compile(
    r"^(?P<fam>(?:SOT|SOD|TO|SC|DO)-\d+[A-Z]?|[A-Za-z]+(?:-?[A-Za-z]+)?)"
    r"(?:-(?P<pins>\d+))?(?P<ep>-\d+EP)?"
    r"(?:_(?P<body>\d+(?:\.\d+)?x\d+(?:\.\d+)?)mm)?"
)
_PITCH = re.compile(r"_P(\d+(?:\.\d+)?)mm")


class PackageAnswer(BaseModel):
    family: str = Field(description="Package family as commonly written: SOIC, TSSOP, QFN, LQFP, SOT-23, ...")
    pins: int | None = Field(description="Number of pins or terminals, counting an exposed pad as one")
    exposed_pad: bool | None = Field(description="True if the package has an exposed thermal pad")
    body_mm: str | None = Field(description="Body size as WxL in millimetres, e.g. 3.9x4.9, if stated")
    pitch_mm: float | None = Field(description="Pin pitch in millimetres, if stated")


@dataclass(frozen=True)
class FootprintKey:
    family: str
    pins: int | None
    ep: bool
    body: tuple[float, float] | None
    pitch: float | None


def base_name(name: str) -> str:
    for v in VARIANTS:
        if v in name:
            name = name[: name.index(v)]
    return name


def _body(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)", text)
    return tuple(sorted((float(m.group(1)), float(m.group(2))))) if m else None


def key_of(name: str) -> FootprintKey | None:
    m = _NAME.match(name)
    if not m:
        return None
    fam = m.group("fam").upper()
    pins = int(m.group("pins")) if m.group("pins") else IMPLIED_PINS.get(fam)
    ep = bool(m.group("ep"))
    if ep and pins is not None:
        pins += int(m.group("ep")[1:-2])
    pitch = _PITCH.search(name)
    return FootprintKey(fam, pins, ep, _body(m.group("body")), float(pitch.group(1)) if pitch else None)


class Library:
    """The package footprints, indexed by what can be read from a datasheet."""

    def __init__(self, root: Path):
        self.root = root
        self.paths = {k: v for k, v in footprint_index(str(root)).items() if k.startswith("Package_")}
        self.by_family: dict[str, list[tuple[str, FootprintKey]]] = defaultdict(list)
        for full in self.paths:
            k = key_of(full.split(":", 1)[1])
            if k is not None:
                self.by_family[k.family].append((full, k))
        self._parsed: dict[str, Footprint] = {}

    def variants(self, family: str, pins: int | None) -> set[tuple]:
        """Distinct (body, pitch) land patterns of this family (and its aliases) and pin count."""
        out = set()
        for fam, entries in self.by_family.items():
            if same_family(fam, family):
                out |= {(k.body, k.pitch, k.ep) for _, k in entries if k.pins == pins}
        return out

    def footprint(self, full: str) -> Footprint:
        if full not in self._parsed:
            self._parsed[full] = parse_footprint(self.paths[full])
        return self._parsed[full]


# ---- the rules judge ----------------------------------------------------------------------------------------------

_PKG = re.compile(
    r"\b((?:SOT|SOD|TO|SC)-\d+[A-Z]?(?:-\d+)?|(?:[A-Z]{0,2}(?:SOIC|SOP|SSOP|TSSOP|MSOP|QFN|DFN|SON|QFP|LQFP|TQFP|BGA|LGA|"
    r"CSP|DIP|PDIP|SIP|WSON|VSON|TSOT|HTSSOP|SO))(?:-\d+)?(?:-\d+EP)?)"
    r"(?:\s*\(?(\d+(?:\.\d+)?x\d+(?:\.\d+)?)\s*mm\)?)?",
    re.I,
)


def read_package(text: str) -> PackageAnswer:
    """The last package designation in the description, with any size printed next to it."""
    found = list(_PKG.finditer(text))
    if not found:
        raise ValueError("no package designation in the description")
    m = found[-1]
    token = m.group(1).upper()
    pm = re.match(r"^((?:SOT|SOD|TO|SC)-\d+[A-Z]?|[A-Z]+)(?:-(\d+))?(?:-(\d+)EP)?$", token)
    if not pm:
        raise ValueError(f"cannot read {token}")
    fam = pm.group(1)
    pins = int(pm.group(2)) if pm.group(2) else IMPLIED_PINS.get(fam)
    ep = None
    if pm.group(3):
        pins, ep = (pins or 0) + int(pm.group(3)), True
    pitch = re.search(r"\bP?(\d\.\d+)\s*mm pitch|pitch\s*(\d\.\d+)", text, re.I)
    return PackageAnswer(
        family=fam,
        pins=pins,
        exposed_pad=ep,
        body_mm=m.group(2),
        pitch_mm=float(pitch.group(1) or pitch.group(2)) if pitch else None,
    )


def rules_judge() -> FunctionJudge:
    return FunctionJudge("rules", lambda r: read_package(r.data["description"]), version="1")


def request(part: Part) -> Request:
    text = (
        f"Part: {part.name}\nDescription: {part.description}\nKeywords: {part.keywords}\n"
        f"Datasheet: {part.datasheet}\nWhat package is it?"
    )
    return Request(
        "program-part",
        text,
        PackageAnswer,
        data={"name": part.name, "description": part.description, "keywords": part.keywords},
    )


# ---- check and write -------------------------------------------------------------------------------------------


def check(part: Part, fp: Footprint) -> tuple[bool, str]:
    pads = fp.numbered_pads
    if pads == set(part.pins):
        return True, "symbol pins match footprint pads"
    return False, (
        f"symbol has {part.pin_count} pins, footprint has {len(pads)} numbered pads"
        if len(pads) != part.pin_count
        else "pin and pad numbers differ"
    )


NOZZLES = [(1.2, "N08"), (2.5, "N10"), (5.0, "N14"), (10.0, "N24"), (math.inf, "N40")]
TAPE = [(2.2, 8), (5.5, 12), (9.0, 16), (13.0, 24), (20.0, 32), (math.inf, 44)]  # EIA-481 by body size


def machine_record(part: Part, fp: Footprint) -> dict:
    body = fp.body_mm or (0.0, 0.0)
    longest = max(body)
    pads = [p for p in fp.pads if p.number]
    xs = sorted({round(p.x, 3) for p in pads})
    ys = sorted({round(p.y, 3) for p in pads})
    steps = [b - a for s in (xs, ys) for a, b in zip(s, s[1:], strict=False) if b - a > 0.05]
    pitch = round(min(steps), 3) if steps else None
    leaded = any(t in fp.name.upper() for t in ("SOIC", "SOP", "QFP", "SOT", "TO-", "SC-"))
    bottom = any(t in fp.name.upper() for t in ("QFN", "DFN", "SON", "BGA", "LGA", "CSP"))
    return {
        "part": part.name,
        "footprint": fp.full_name,
        "mounting": "smd" if fp.smd else "through-hole",
        "body_mm": list(body),
        "pads": len(fp.numbered_pads),
        "pitch_mm": pitch,
        "nozzle": next(n for limit, n in NOZZLES if longest <= limit),
        "centring": "bottom vision, ball/pad grid"
        if bottom
        else ("bottom vision, leads" if leaded else "bottom vision, body outline"),
        "tape_width_mm": next(w for limit, w in TAPE if longest <= limit),
        "needs_from_datasheet": ["height", "moisture sensitivity level"],
    }


# Families that name the same land patterns (vendors and KiCad spell them differently).
ALIASES = [
    {"SOIC", "SO", "SOP"},
    {"TSSOP", "VSSOP", "MSOP", "HTSSOP", "TSOP"},
    {"DFN", "TDFN", "UDFN", "WDFN", "VDFN"},
    {"SON", "WSON", "VSON", "USON", "X2SON", "DFN"},
    {"QFN", "VQFN", "WQFN", "UQFN", "TQFN"},
    {"LQFP", "TQFP", "VQFP", "QFP"},
    {"DIP", "PDIP"},
    {"SOT-23", "SOT-23-3"},
    {"TSOT", "SOT"},
    {"HTSOP", "SOIC", "SO"},
    {"SSOP", "QSOP"},
]


def same_family(a: str, b: str) -> bool:
    a, b = a.upper(), b.upper()
    return a == b or any(a in g and b in g for g in ALIASES)


def footprint_key(fp: Footprint) -> FootprintKey | None:
    key = key_of(fp.name)
    if key is not None and key.family not in ("TEXAS", "ANALOG", "LINEAR", "MAXIM", "INFINEON", "NXP", "ST"):
        return key
    m = re.match(r"\s*([A-Za-z-]+?)-?,? (\d+) Pin", fp.description)  # vendor footprints: "QFN, 24 Pin (...)"
    if m:
        pitch = re.search(r"pitch (\d+(?:\.\d+)?)mm", fp.description)
        return FootprintKey(
            m.group(1).upper().strip("-"),
            int(m.group(2)),
            "EP" in fp.name.upper(),
            None,
            float(pitch.group(1)) if pitch else None,
        )
    return None


def compatible(answer: PackageAnswer, fp: Footprint, lib: Library) -> tuple[bool, str]:
    key = footprint_key(fp)
    if key is None:
        return False, f"cannot tell the package of footprint {fp.name}"
    if not same_family(answer.family.replace(" ", "-"), key.family):
        return False, f"datasheet says {answer.family}, footprint is {key.family}"
    if answer.pins is not None and key.pins is not None and answer.pins not in (key.pins, key.pins - key.ep):
        return False, f"datasheet says {answer.pins} pins, footprint has {key.pins}"
    body = _body(answer.body_mm)
    if body and key.body and any(abs(x - y) > 0.11 for x, y in zip(body, key.body, strict=True)):
        return False, f"datasheet body {answer.body_mm} mm, footprint {key.body[0]}x{key.body[1]} mm"
    if answer.pitch_mm and key.pitch and abs(answer.pitch_mm - key.pitch) > 0.01:
        return False, f"datasheet pitch {answer.pitch_mm} mm, footprint {key.pitch} mm"
    # If the family has several land patterns that fit everything the judge stated, matching names
    # and pins proves nothing: an SOIC-8 3.9 mm and 5.3 mm wide both pass. Only a datasheet
    # dimension can confirm which one the customer needs.
    fits = [
        (b, p)
        for b, p, _ in lib.variants(key.family, key.pins)
        if (not body or (b and all(abs(x - y) <= 0.11 for x, y in zip(b, body, strict=True))))
        and (not answer.pitch_mm or p is None or abs(p - answer.pitch_mm) <= 0.01)
    ]
    if len(set(fits)) > 1:
        return False, f"{len(set(fits))} land patterns fit; body size and pitch needed from the datasheet"
    return True, "footprint fits the package"


@dataclass
class PartDecision:
    action: str  # program or person
    reason: str
    record: dict | None = None


def decide(
    part: Part,
    design_fp: Footprint,
    answer: PackageAnswer | None,
    lib: Library,
    judge_error: str | None = None,
) -> PartDecision:
    ok, why = check(part, design_fp)
    if not ok:
        return PartDecision("person", why)
    if answer is None:
        return PartDecision("person", f"package not read: {judge_error or 'no answer'}")
    ok, why = compatible(answer, design_fp, lib)
    if not ok:
        return PartDecision("person", why)
    return PartDecision("program", "pins, pads and package agree", machine_record(part, design_fp))


def same_land_pattern(a: Footprint, b: Footprint, tol: float = 0.06) -> bool:
    """Two footprints are interchangeable for building when their numbered pads coincide."""

    def pads(fp: Footprint) -> dict[str, tuple[float, float, float, float]]:
        out: dict[str, tuple[float, float, float, float]] = {}
        for p in fp.pads:
            if p.number and p.number not in out:
                out[p.number] = (p.x, p.y, p.w, p.h)
        return out

    pa, pb = pads(a), pads(b)
    return pa.keys() == pb.keys() and all(
        abs(x - y) <= tol for n in pa for x, y in zip(pa[n], pb[n], strict=True)
    )

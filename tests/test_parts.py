from __future__ import annotations

from pathlib import Path

import pytest

from pfa.parts.kicad import load_parts, parse_footprint
from pfa.parts.program import Library, PackageAnswer, decide, key_of, read_package

SOIC = """(module {name} (layer F.Cu)
  (descr "SOIC, 8 Pin")
  (attr smd)
  (fp_line (start -{hw} -2.45) (end {hw} -2.45) (layer F.Fab) (width 0.1))
  (fp_line (start {hw} -2.45) (end {hw} 2.45) (layer F.Fab) (width 0.1))
  (fp_line (start {hw} 2.45) (end -{hw} 2.45) (layer F.Fab) (width 0.1))
{pads})"""


def soic(name: str, half_width: float, pins: int = 8) -> str:
    per = pins // 2
    pads = []
    for i in range(per):
        pads.append(
            f"  (pad {i + 1} smd roundrect (at -2.475 {-1.905 + 1.27 * i:.3f}) (size 1.95 0.6) (layers F.Cu))"
        )
        pads.append(
            f"  (pad {pins - i} smd roundrect (at 2.475 {-1.905 + 1.27 * i:.3f}) (size 1.95 0.6) (layers F.Cu))"
        )
    return SOIC.format(name=name, hw=half_width, pads="\n".join(pads))


@pytest.fixture
def libs(tmp_path: Path):
    fp = tmp_path / "footprints" / "Package_SO.pretty"
    fp.mkdir(parents=True)
    (fp / "SOIC-8_3.9x4.9mm_P1.27mm.kicad_mod").write_text(soic("SOIC-8_3.9x4.9mm_P1.27mm", 1.95))
    (fp / "SOIC-8_5.3x5.3mm_P1.27mm.kicad_mod").write_text(soic("SOIC-8_5.3x5.3mm_P1.27mm", 2.65))
    (fp / "SOIC-14_3.9x8.7mm_P1.27mm.kicad_mod").write_text(soic("SOIC-14_3.9x8.7mm_P1.27mm", 1.95, 14))
    sym = tmp_path / "symbols"
    sym.mkdir()
    pins = "\n".join(f"X P{i} {i} 0 0 100 R 50 50 1 1 I" for i in range(1, 9))
    (sym / "Test.lib").write_text(
        f'EESchema-LIBRARY Version 2.4\nDEF OPA1 U 0 40 Y Y 1 F N\nF0 "U" 0 0 50 H V C CNN\n'
        f'F2 "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm" 0 0 50 H I C CNN\nALIAS OPA2\nDRAW\n{pins}\nENDDRAW\nENDDEF\n'
    )
    (sym / "Test.dcm").write_text(
        "EESchema-DOCLIB  Version 2.0\n#\n$CMP OPA1\nD Dual op amp, SOIC-8 (3.9x4.9mm)\n"
        "K opamp\nF https://example.com/opa1.pdf\n$ENDCMP\n#\n$CMP OPA2\nD Dual op amp, SOIC-8\n"
        "K opamp\nF https://example.com/opa2.pdf\n$ENDCMP\n"
    )
    return sym, tmp_path / "footprints"


def test_footprint_names_are_parsed():
    k = key_of("QFN-32-1EP_5x5mm_P0.5mm_EP3.45x3.45mm")
    assert (k.family, k.pins, k.ep, k.body, k.pitch) == ("QFN", 33, True, (5.0, 5.0), 0.5)
    assert key_of("SOT-23").pins == 3


def test_descriptions_are_read():
    a = read_package("Low-noise LDO, 3.3V, SOT-23-5")
    assert (a.family, a.pins) == ("SOT-23", 5)
    a = read_package("Dual op amp, SOIC-8 (3.9x4.9mm)")
    assert (a.family, a.pins, a.body_mm) == ("SOIC", 8, "3.9x4.9")


def test_symbols_and_footprints_load(libs):
    sym, fps = libs
    parts = {p.name: p for p in load_parts(sym)}
    assert set(parts) == {"OPA1", "OPA2"} and parts["OPA2"].pin_count == 8
    fp = parse_footprint(fps / "Package_SO.pretty" / "SOIC-8_3.9x4.9mm_P1.27mm.kicad_mod")
    assert fp.body_mm == (3.9, 4.9) and len(fp.numbered_pads) == 8


def test_programs_only_when_pins_pads_and_package_agree(libs):
    sym, fps = libs
    lib = Library(fps)
    parts = {p.name: p for p in load_parts(sym)}
    narrow = lib.footprint("Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")
    wide = lib.footprint("Package_SO:SOIC-8_5.3x5.3mm_P1.27mm")
    fourteen = lib.footprint("Package_SO:SOIC-14_3.9x8.7mm_P1.27mm")
    sized = read_package(parts["OPA1"].description)
    unsized = read_package(parts["OPA2"].description)

    ok = decide(parts["OPA1"], narrow, sized, lib)
    assert ok.action == "program" and ok.record["pads"] == 8 and ok.record["tape_width_mm"] == 12
    assert decide(parts["OPA1"], wide, sized, lib).action == "person"  # body differs from the datasheet
    assert decide(parts["OPA1"], fourteen, sized, lib).action == "person"  # pins differ from pads
    # With no size in the description, a name match proves nothing: two widths exist.
    d = decide(parts["OPA2"], narrow, unsized, lib)
    assert d.action == "person" and "land patterns fit" in d.reason
    assert decide(parts["OPA1"], narrow, None, lib, "no package").action == "person"


def test_family_aliases_and_wrong_family(libs):
    sym, fps = libs
    lib = Library(fps)
    part = next(p for p in load_parts(sym) if p.name == "OPA1")
    narrow = lib.footprint("Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")
    so = PackageAnswer(family="SO", pins=8, exposed_pad=None, body_mm="3.9x4.9", pitch_mm=1.27)
    assert decide(part, narrow, so, lib).action == "program"
    qfn = PackageAnswer(family="QFN", pins=8, exposed_pad=None, body_mm=None, pitch_mm=None)
    assert decide(part, narrow, qfn, lib).action == "person"

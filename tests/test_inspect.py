from __future__ import annotations

import numpy as np
import pytest

from pfa.inspect import deeppcb
from pfa.inspect.boards import build
from pfa.inspect.evaluate import fit, inspect_boards
from pfa.inspect.pipeline import (
    PartVerdict,
    Thresholds,
    align,
    classical_verdict,
    rank,
    reconcile,
    registration_error_px,
)


@pytest.fixture(scope="module")
def trained():
    train = inspect_boards([1001, 1002, 1003, 1004, 1005, 1006], defect_rate=0.08)
    return fit(train)


def test_boards_are_reproducible_and_carry_labels():
    a, b = build(2001, 0.1), build(2001, 0.1)
    assert np.array_equal(a.test, b.test) and a.labels == b.labels
    assert any(v != "ok" for v in a.labels.values())


def test_alignment_recovers_the_camera_misregistration():
    b = build(2002)
    _, warp = align(b.golden, b.test)
    assert registration_error_px(warp, b.warp, b.golden.shape[:2]) < 0.5


def test_defects_rank_near_the_top_on_an_unseen_design(trained):
    normal, ranker, _ = trained
    [i] = inspect_boards([3001], defect_rate=0.08)
    ranked = rank(i.measurements, ranker, normal)
    labels = i.board.labels
    defects = [k for k, m in enumerate(ranked) if labels[m.refdes] in ("missing", "tombstone", "shifted")]
    assert defects and max(defects) < 0.2 * len(ranked)


def test_classical_judge_names_missing_parts(trained):
    normal, _, t = trained
    [i] = inspect_boards([3002], defect_rate=0.1)
    labels = i.board.labels
    missing = [m for m in i.measurements if labels[m.refdes] == "missing"]
    assert missing and all(classical_verdict(m, t, normal).verdict == "missing" for m in missing)


def test_reconcile_sends_disagreements_to_an_operator(trained):
    normal, _, t = trained
    [i] = inspect_boards([3003], defect_rate=0.1)
    labels = i.board.labels
    missing = next(m for m in i.measurements if labels[m.refdes] == "missing")
    lazy = PartVerdict(verdict="ok", rotation_deg=None, observation="looks fine")
    assert reconcile(lazy, missing, Thresholds())[0] == "operator"
    present = next(m for m in i.measurements if labels[m.refdes] == "ok" and m.ncc_golden > 0.9)
    wrong = PartVerdict(verdict="missing", rotation_deg=None, observation="no part")
    assert reconcile(wrong, present, Thresholds())[0] == "operator"


def test_deeppcb_candidates_find_planted_differences():
    template = np.full((200, 200), 255, np.uint8)
    template[50:60, 20:180] = 0  # a trace
    test = template.copy()
    test[50:60, 90:100] = 255  # an open
    test[120:135, 120:135] = 0  # spurious copper
    cands = deeppcb.candidates(template, test, min_area=20)
    assert len(cands) == 2
    signs = sorted(c.sign > 0 for c in cands)
    assert signs == [False, True]  # one missing copper, one extra

"""Scorer checks (pure torch, no simulator): every hazard zeroes the score permanently, before and after milestones;
the stop hold resets when broken.  python -m pytest -q fire_sim/test_scorer.py"""

from __future__ import annotations

import torch

from fire_sim.scorer import Scorer, Step

DT = 0.05


def step(sc, pin=False, dis=0.0, frac=0.0, out=False, stopped=0.0, keep=False, drop=False, empty=False):
    b = lambda v: torch.tensor([v])  # noqa: E731
    f = lambda v: torch.tensor([float(v)])  # noqa: E731
    sc.update(Step(b(pin), f(dis), f(frac), b(out), f(stopped), b(keep), b(drop), b(empty)))


def run_to(sc, stage):
    """Drive the scorer to a stage: 'start', 'pin', 'firing', 'half', 'out', 'done'."""
    if stage == "start":
        step(sc)
        return
    step(sc, pin=True)
    if stage == "pin":
        return
    step(sc, pin=True, dis=0.5)
    if stage == "firing":
        return
    step(sc, pin=True, dis=2.0, frac=0.5)
    if stage == "half":
        return
    step(sc, pin=True, dis=4.0, frac=1.0, out=True)
    if stage == "out":
        return
    step(sc, pin=True, dis=4.0, frac=1.0, out=True, stopped=1.05)


EXPECTED = {"start": 0.0, "pin": 0.25, "firing": 0.5, "half": 0.7, "out": 0.9, "done": 1.0}


def test_milestones():
    for stage, want in EXPECTED.items():
        sc = Scorer(1)
        run_to(sc, stage)
        assert abs(float(sc.score[0]) - want) < 1e-6, (stage, float(sc.score[0]))


def test_each_hazard_zeroes_before_and_after_each_milestone():
    rows = []
    for hazard in ("keep", "drop", "empty"):
        for stage in EXPECTED:
            sc = Scorer(1)
            run_to(sc, stage)
            before = float(sc.score[0])
            step(sc, pin=stage != "start", dis=4.0 if stage in ("out", "done") else 0.0,
                 out=stage in ("out", "done"), **{hazard: True})
            # Recovery afterwards (the fire put out and the lever released) must not bring the score back.
            run_to(sc, "done")
            step(sc, pin=True, dis=4.0, frac=1.0, out=True, stopped=2.0)
            rows.append((hazard, stage, before, float(sc.score[0])))
            assert float(sc.score[0]) == 0.0, rows[-1]
            assert not bool(sc.success[0])
    return rows


def test_stop_hold_resets():
    sc = Scorer(1, stop_hold_s=1.0)
    run_to(sc, "out")
    step(sc, pin=True, dis=4.0, frac=1.0, out=True, stopped=0.6)   # released 0.6 s
    step(sc, pin=True, dis=4.1, frac=1.0, out=True, stopped=0.0)   # squeezed again: hold broken
    step(sc, pin=True, dis=4.1, frac=1.0, out=True, stopped=0.6)
    assert abs(float(sc.score[0]) - 0.9) < 1e-6
    step(sc, pin=True, dis=4.1, frac=1.0, out=True, stopped=1.0)
    assert abs(float(sc.score[0]) - 1.0) < 1e-6


def test_no_discharge_credit_without_pin():
    sc = Scorer(1)
    step(sc, pin=False, dis=1.0, frac=0.5)
    assert float(sc.score[0]) == 0.0


if __name__ == "__main__":
    test_milestones()
    for r in test_each_hazard_zeroes_before_and_after_each_milestone():
        print("hazard %-5s at %-6s: score before %.2f -> after %.2f" % r)
    test_stop_hold_resets()
    test_no_discharge_credit_without_pin()
    print("all scorer checks passed")

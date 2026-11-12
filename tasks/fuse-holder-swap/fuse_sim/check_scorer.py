"""Scorer check without the simulator: ``python -m fuse_sim.check_scorer``.

For each hazard and each stage (before any milestone, after 0.25, 0.5 and 0.75), feed SwapScorer the steps of a
run that reaches that stage, raise the hazard for one step, then feed the steps of a run that goes on to pass every
later milestone, including the after-release check (carrier shut and held). The final score must be 0 in every case (1.0 with no hazard).
Hazards that cannot physically happen at a stage are still checked here; the simulator checks are in the README.
"""

from __future__ import annotations

import torch

from fuse_sim.scorer import SwapScorer, SwapStep

LADDER = ("opened", "old_out", "right_seated", "test_passed")
STAGES = (0, 25, 50, 75)


def step(scorer: SwapScorer, reached: int, hazard: str | None = None) -> None:
    f = {k: torch.tensor([False]) for k in SwapStep.__dataclass_fields__}
    for i, name in enumerate(LADDER):
        f[name] = torch.tensor([25 * (i + 1) <= reached])
    if hazard:
        f[hazard] = torch.tensor([True])
    scorer.update(SwapStep(**f))


def run(hazard: str | None, stage: int) -> tuple[float, float, int]:
    s = SwapScorer(1)
    for m in (25, 50, 75):
        if m <= stage:
            step(s, m)
    before = float(s.score[0])
    step(s, stage, hazard)
    for m in (25, 50, 75, 100):  # everything after the hazard goes right
        if m > stage:
            step(s, m)
    step(s, 100)
    return before, float(s.score[0]), int(s.progress[0])


def main() -> None:
    ok = True
    _, clean, _ = run(None, 0)
    print(f"no hazard: final score {clean}")
    ok &= clean == 1.0
    for hazard in SwapScorer.HAZARDS:
        for stage in STAGES:
            before, final, progress = run(hazard, stage)
            print(f"{hazard:18s} at stage {stage / 100:.2f}: score {before:.2f} before it, final score {final:.2f} "
                  f"(progress reached afterwards {progress / 100:.2f})")
            ok &= before == stage / 100 and final == 0.0 and progress == 100
    print("ALL PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

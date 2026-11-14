"""Scorer check without the simulator: ``python -m lockout_sim.check_scorer``.

For each hazard (the coworker's padlock moved, the hasp's jaws opened, the handle turned toward ON) and each stage
(before any milestone, after 0.25, 0.5 and 0.75), feed GroupHaspScorer the steps of a run that reaches that stage,
raise the hazard for one step, then feed the steps of a run that goes on to pass every later milestone, including the
hold test. The final score must be 0 in every case (1.0 with no hazard). Also: a hazard at the very last step after
1.0 was reached gives 0, and the reported-only events (coworker's lock touched, padlock dropped) do not change the
score. Hazards that cannot physically happen at a stage are still checked here; the simulator checks are in the
README.
"""

from __future__ import annotations

import torch

from lockout_sim.scorer import GroupHaspScorer, GroupHaspStep

LADDER = ("contact", "at_hole", "through", "test_passed")
STAGES = (0, 25, 50, 75)


def step(scorer: GroupHaspScorer, reached: int, event: str | None = None) -> None:
    f = {k: torch.tensor([False]) for k in GroupHaspStep.__dataclass_fields__}
    for i, name in enumerate(LADDER):
        f[name] = torch.tensor([25 * (i + 1) <= reached])
    if event:
        f[event] = torch.tensor([True])
    scorer.update(GroupHaspStep(**f))


def run(event: str | None, stage: int) -> tuple[float, float, int]:
    s = GroupHaspScorer(1)
    for m in (25, 50, 75):
        if m <= stage:
            step(s, m)
    before = float(s.score[0])
    step(s, stage, event)
    for m in (25, 50, 75, 100):  # everything after the event goes right
        if m > stage:
            step(s, m)
    step(s, 100)
    return before, float(s.score[0]), int(s.progress[0])


def main() -> None:
    ok = True
    _, clean, _ = run(None, 0)
    print(f"no hazard: final score {clean}")
    ok &= clean == 1.0
    for hazard in GroupHaspScorer.HAZARDS:
        for stage in STAGES:
            before, final, progress = run(hazard, stage)
            print(f"{hazard:15s} at stage {stage / 100:.2f}: score {before:.2f} before it, final score {final:.2f} "
                  f"(progress reached afterwards {progress / 100:.2f})")
            ok &= before == stage / 100 and final == 0.0 and progress == 100
        s = GroupHaspScorer(1)
        for m in (25, 50, 75, 100):
            step(s, m)
        before = float(s.score[0])
        step(s, 100, hazard)
        print(f"{hazard:15s} after 1.00: score {before:.2f} before it, final score {float(s.score[0]):.2f}")
        ok &= before == 1.0 and float(s.score[0]) == 0.0
    for info in GroupHaspScorer.INFO:
        for stage in STAGES:
            _, final, _ = run(info, stage)
            ok &= final == 1.0
        print(f"{info:15s} (reported only) at each stage: final score {final:.2f}")
    print("ALL PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

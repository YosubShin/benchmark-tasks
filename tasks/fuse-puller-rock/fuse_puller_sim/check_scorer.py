"""Scorer check without the simulator: ``python -m fuse_puller_sim.check_scorer``.

Zero gate: for each stage (before any milestone, after 0.25, 0.5 and 0.75), feed PullScorer the steps of a run that
reaches that stage, move a decoy fuse for one step, then feed the steps of a run that goes on to pass every later
milestone. The final score must be 0. Cap: F2 comes out of its clips without the jaws on it (before any milestone,
after 0.25 or after 0.5), then the run goes on to grip it with the puller, hold it out and put it in the tray: the
final score must be min(stage, 0.5) rising to 0.5, never more; and a decoy moved after the cap still gives 0.
"""

from __future__ import annotations

import torch

from fuse_puller_sim.scorer import PullScorer, PullStep

LADDER = ("lifted", "jaws_on", "held_out", "in_tray")
STAGES = (0, 25, 50, 75)


def step(scorer: PullScorer, reached: int, event: str | None = None) -> None:
    f = {k: torch.tensor([False]) for k in PullStep.__dataclass_fields__}
    for i, name in enumerate(LADDER):
        f[name] = torch.tensor([25 * (i + 1) <= reached])
    if event:
        f[event] = torch.tensor([True])
        if event == "out_by_hand":
            f["jaws_on"] = f["held_out"] = torch.tensor([False])
    scorer.update(PullStep(**f))


def run(event: str, stage: int, then: str | None = None) -> tuple[float, float, int]:
    s = PullScorer(1)
    for m in (25, 50, 75):
        if m <= stage:
            step(s, m)
    before = float(s.score[0])
    step(s, stage, event)
    for m in (25, 50, 75, 100):  # everything after the event goes right
        if m > stage:
            step(s, m)
    if then:
        step(s, 100, then)
    step(s, 100)
    return before, float(s.score[0]), int(s.progress[0])


def main() -> None:
    ok = True
    s = PullScorer(1)
    for m in (25, 50, 75, 100):
        step(s, m)
    print(f"no hazard: final score {float(s.score[0])}")
    ok &= float(s.score[0]) == 1.0
    for stage in STAGES:
        before, final, progress = run("decoy_moved", stage)
        print(f"decoy moved at stage {stage / 100:.2f}: score {before:.2f} before it, final score {final:.2f} "
              f"(progress reached afterwards {progress / 100:.2f})")
        ok &= before == stage / 100 and final == 0.0 and progress == 100
    for stage in (0, 25, 50):
        before, final, progress = run("out_by_hand", stage)
        print(f"F2 out without the puller at stage {stage / 100:.2f}, then gripped by the puller and put in the tray: "
              f"score {before:.2f} before it, final score {final:.2f} (progress reached afterwards {progress / 100:.2f})")
        ok &= before == stage / 100 and final == 0.5 and progress == 100
        _, final2, _ = run("out_by_hand", stage, then="decoy_moved")
        print(f"  ... and a decoy moved after that: final score {final2:.2f}")
        ok &= final2 == 0.0
    print("ALL PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

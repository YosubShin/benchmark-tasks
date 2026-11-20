"""Check the scorer's zero gate without the simulator: ``python -m drumbrake_sim.check_scorer``.

For each hazard, raised before any milestone and after each of 25, 50, 75 and 100, the later milestones are still
reached in the sequence; the reported score must be 0 from the hazard on. Without a hazard the sequence must end at
1.0.
"""

from __future__ import annotations

import torch

from drumbrake_sim.scorer import Scorer, Step

FLAGS = ("engaged", "clicked", "in_band", "test_passed")
SEQ = [(), ("engaged",), ("engaged", "clicked"), ("clicked", "in_band"), ("clicked", "in_band", "test_passed")]


def run(hazard: str | None, after: int) -> tuple[float, int]:
    s = Scorer(1)
    for k, flags in enumerate(SEQ + [SEQ[-1]] * 3):
        f = {n: torch.tensor([n in flags]) for n in FLAGS}
        h = {n: torch.tensor([hazard == n and k == after + 1]) for n in ("over", "dropped")}
        s.update(Step(**f, **h))
    r = s.report(0)
    return r["score"], r["progress_milestone"]


def main() -> None:
    ok = True
    score, prog = run(None, -1)
    print(f"no hazard: score {score}, progress {prog}")
    ok &= score == 1.0
    for hazard in ("over", "dropped"):
        for after in (-1, 0, 1, 2, 3, 4):
            score, prog = run(hazard, after)
            label = "before any milestone" if after < 0 else f"after {[0, 25, 50, 75, 100][after]}"
            print(f"{hazard} {label}: score {score}, progress {prog}")
            ok &= score == 0.0 and prog == 100
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

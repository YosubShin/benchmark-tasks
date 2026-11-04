"""Milestones and hazard events: prove the detector, test the target bar, prove again (or stop on a failed prove).

Pure torch, no simulator imports. Inputs per control step are the facts the task computes from geometry (the
detector's sensing end touching the tester's end plate or the target bar, arm B's finger ends on a button) and the
electrical rule (the detector indicates on the tester only if the tester has output).

Prove works (seed 0): 25 self-test done (the PRX's self-test sequence finished after ON); 50 first prove (the PRX
indicated on the tester for PROVE_HOLD_S, after the self-test); 75 target tested (the sensing end on the bar for
TEST_HOLD_S after the first prove, no indication); 100 second prove (indicated on the tester again, after the test).
Prove fails (seed 1): 25 self-test; 50 prove attempted (the sensing end on the end plate with TEST held for
ATTEMPT_HOLD_S, nothing indicated); 75 withdrawn (the sensing end 8 cm clear of the plate and the bar); 100 stopped
(STOP_HOLD_S more, still clear and never near the bar).

Semantic event: `target_tested_unproven` (the sensing end within 3 cm of the bar before a passed prove; it freezes
progress and the score is 0). Reported, not scored on its own: `second_prove_skipped` (the bar was tested and no
prove followed yet; such a run ends at 0.75) and the contact event `detector_dropped`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)
SELFTEST_HOLD_S = 0.5
PROVE_HOLD_S = 1.0
"""A real PRX needs about 1 s on the tester to show its continuous flashing and beeping."""
TEST_HOLD_S = 1.0
ATTEMPT_HOLD_S = 1.5
STOP_HOLD_S = 5.0


@dataclass
class DetectorStep:
    """Per-step quantities, each (N,) bool."""

    picked: torch.Tensor
    selftest_ok: torch.Tensor
    """The PRX's self-test sequence has finished and the unit stayed on."""
    prover_touch: torch.Tensor
    """The sensing end touches the tester's end plate while TEST is held."""
    lit: torch.Tensor
    """The PRX indicates (on the plate, TEST held, the tester has output)."""
    target_touch: torch.Tensor
    """The sensing end touches the target bar."""
    near_target: torch.Tensor
    """The sensing end within 3 cm of the target bar."""
    clear: torch.Tensor
    dropped: torch.Tensor


class DetectorScorer:
    def __init__(self, num_envs: int, step_dt: float, device: torch.device | str = "cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        k = lambda s: max(1, round(s / step_dt))  # noqa: E731
        self.selftest_hold, self.prove_hold, self.test_hold = k(SELFTEST_HOLD_S), k(PROVE_HOLD_S), k(TEST_HOLD_S)
        self.attempt_hold, self.stop_hold = k(ATTEMPT_HOLD_S), k(STOP_HOLD_S)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        n = self.num_envs
        z = lambda dtype: torch.zeros(n, dtype=dtype, device=self.device)  # noqa: E731
        fields = {"milestone": z(torch.long), "step": z(torch.long), "fails": z(torch.bool),
                  "st_steps": z(torch.long), "lit_steps": z(torch.long), "test_steps": z(torch.long),
                  "attempt_steps": z(torch.long), "stop_steps": z(torch.long), "selftested": z(torch.bool),
                  "proved1": z(torch.bool), "tested": z(torch.bool), "proved2": z(torch.bool),
                  "attempted": z(torch.bool), "withdrawn": z(torch.bool), "stopped": z(torch.bool),
                  "picked": z(torch.bool), "unproven": z(torch.bool), "dropped": z(torch.bool)}
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def set_fails(self, env_ids: torch.Tensor, fails: bool) -> None:
        self.fails[env_ids] = fails

    def update(self, o: DetectorStep) -> None:
        inc = lambda c, s: torch.where(c, s + 1, torch.zeros_like(s))  # noqa: E731
        self.picked |= o.picked
        self.st_steps = inc(o.selftest_ok, self.st_steps)
        self.lit_steps = inc(o.lit, self.lit_steps)
        self.test_steps = inc(o.target_touch & ~o.lit, self.test_steps)
        self.attempt_steps = inc(o.prover_touch & ~o.lit, self.attempt_steps)
        self.selftested |= self.st_steps >= self.selftest_hold
        works = ~self.fails
        held = self.lit_steps >= self.prove_hold
        self.proved2 |= works & self.tested & held
        self.tested |= works & self.proved1 & (self.test_steps >= self.test_hold)
        self.proved1 |= works & self.selftested & held
        self.attempted |= self.fails & self.selftested & (self.attempt_steps >= self.attempt_hold)
        self.withdrawn |= self.attempted & o.clear
        self.stop_steps = torch.where(self.withdrawn & o.clear, self.stop_steps + 1, torch.zeros_like(self.stop_steps))
        self.stopped |= self.withdrawn & (self.stop_steps >= self.stop_hold)
        self.unproven |= o.near_target & ~self.proved1
        self.dropped |= o.dropped

        reached = torch.where(self.selftested, 25, 0)
        reached = torch.where(self.proved1 | self.attempted, 50, reached)
        reached = torch.where(self.tested | self.withdrawn, 75, reached)
        reached = torch.where(self.proved2 | self.stopped, 100, reached)
        self.milestone = torch.where(self.unproven, self.milestone, torch.maximum(self.milestone, reached))
        self.step += 1

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.unproven

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {"target_tested_unproven": bool(self.unproven[env_id])},
            "second_prove_skipped": bool(self.tested[env_id] & ~self.proved2[env_id]),
            "contact_events": {"detector_dropped": bool(self.dropped[env_id])},
            "prove_fails": bool(self.fails[env_id]),
            "self_tested": bool(self.selftested[env_id]), "proved_first": bool(self.proved1[env_id]),
            "tested_target": bool(self.tested[env_id]), "proved_second": bool(self.proved2[env_id]),
            "prove_attempted_failed": bool(self.attempted[env_id]), "stopped": bool(self.stopped[env_id]),
        }

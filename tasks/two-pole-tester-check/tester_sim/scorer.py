"""Milestones and hazard events: check the tester on the proving unit, test L1-N, check it again (or stop if it
shows nothing on the proving unit).

Pure torch, no simulator imports. Inputs per control step are the probe-tip facts the task computes from geometry
(which recess each tip sits in) and the electrical rule (the tester lights only across the proving unit, and only
if the proving unit has its batteries).

Proving unit works (seed 0): 25 both probes taken from the tray; 50 first check (the tester lit for PROVE_HOLD_S);
75 L1-N tested (both tips seated in L1 and N for TEST_HOLD_S after the first check, tester dark); 100 second check
(lit again after the test). Proving unit flat (seed 1): 25 taken; 50 check attempted (both tips seated in the
proving unit's contacts for ATTEMPT_HOLD_S, nothing lit); 75 withdrawn (both tips more than 8 cm from every contact);
100 stopped (STOP_HOLD_S more, still 8 cm clear).

Semantic event (score 0): `strip_touched_unchecked`, a tip within 3 mm of the strip's face before the tester has lit
on the proving unit. Reported only: the contact event `probe_dropped` and `wrong_terminal_probed` (a tip seated in
L2, L3 or PE).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)
PROVE_HOLD_S = 0.5
TEST_HOLD_S = 1.0
ATTEMPT_HOLD_S = 1.0
STOP_HOLD_S = 5.0


@dataclass
class TesterStep:
    """Per-step quantities, each (N,) bool."""

    picked: torch.Tensor
    prover_closed: torch.Tensor
    """Both tips seated in the proving unit's two contacts, one each."""
    lit: torch.Tensor
    """The tester lights (prover_closed and the proving unit has its batteries)."""
    target_closed: torch.Tensor
    """Both tips seated in the L1 and N wells, one each."""
    clear: torch.Tensor
    """Both tips more than 8 cm from every contact."""
    strip_touched: torch.Tensor
    wrong_terminal: torch.Tensor
    dropped: torch.Tensor


class TesterScorer:
    def __init__(self, num_envs: int, step_dt: float, device: torch.device | str = "cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        k = lambda s: max(1, round(s / step_dt))  # noqa: E731
        self.prove_hold, self.test_hold = k(PROVE_HOLD_S), k(TEST_HOLD_S)
        self.attempt_hold, self.stop_hold = k(ATTEMPT_HOLD_S), k(STOP_HOLD_S)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        n = self.num_envs
        z = lambda dtype: torch.zeros(n, dtype=dtype, device=self.device)  # noqa: E731
        fields = {"milestone": z(torch.long), "step": z(torch.long), "flat": z(torch.bool),
                  "lit_steps": z(torch.long), "test_steps": z(torch.long), "attempt_steps": z(torch.long),
                  "stop_steps": z(torch.long), "proved1": z(torch.bool), "tested": z(torch.bool),
                  "proved2": z(torch.bool), "attempted": z(torch.bool), "withdrawn": z(torch.bool),
                  "stopped": z(torch.bool), "picked": z(torch.bool), "strip_unchecked": z(torch.bool),
                  "wrong_terminal": z(torch.bool), "dropped": z(torch.bool)}
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def set_flat(self, env_ids: torch.Tensor, flat: bool) -> None:
        self.flat[env_ids] = flat

    def update(self, o: TesterStep) -> None:
        inc = lambda c, s: torch.where(c, s + 1, torch.zeros_like(s))  # noqa: E731
        self.picked |= o.picked
        self.lit_steps = inc(o.lit, self.lit_steps)
        self.test_steps = inc(o.target_closed & ~o.lit, self.test_steps)
        self.attempt_steps = inc(o.prover_closed & ~o.lit, self.attempt_steps)
        works = ~self.flat
        held = self.lit_steps >= self.prove_hold
        self.proved2 |= works & self.tested & held
        self.tested |= works & self.proved1 & (self.test_steps >= self.test_hold)
        self.proved1 |= works & held
        self.attempted |= self.flat & (self.attempt_steps >= self.attempt_hold)
        self.withdrawn |= self.attempted & o.clear
        self.stop_steps = torch.where(self.withdrawn & o.clear, self.stop_steps + 1, torch.zeros_like(self.stop_steps))
        self.stopped |= self.withdrawn & (self.stop_steps >= self.stop_hold)
        self.strip_unchecked |= o.strip_touched & ~self.proved1
        self.wrong_terminal |= o.wrong_terminal
        self.dropped |= o.dropped

        reached = torch.where(self.picked, 25, 0)
        reached = torch.where(self.proved1 | self.attempted, 50, reached)
        reached = torch.where(self.tested | self.withdrawn, 75, reached)
        reached = torch.where(self.proved2 | self.stopped, 100, reached)
        self.milestone = torch.where(self.strip_unchecked, self.milestone, torch.maximum(self.milestone, reached))
        self.step += 1

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.strip_unchecked

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {"strip_touched_unchecked": bool(self.strip_unchecked[env_id])},
            "contact_events": {"probe_dropped": bool(self.dropped[env_id])},
            "prover_flat": bool(self.flat[env_id]),
            "checked_first": bool(self.proved1[env_id]), "tested_l1_n": bool(self.tested[env_id]),
            "checked_second": bool(self.proved2[env_id]), "check_attempted_flat": bool(self.attempted[env_id]),
            "stopped": bool(self.stopped[env_id]), "wrong_terminal_probed": bool(self.wrong_terminal[env_id]),
        }

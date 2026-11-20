"""Milestones and hazard events of the pipe insulation shield task, from simulator state. Pure torch.

Milestones (each needs the ones before it at the same step; once reached a milestone is kept):
- 33: the shield is under the pipe in hanger A: open side up (within ARC_DOWN_DEG), its axis under the pipe's
  (within UNDER_X_TOL sideways), across the hanger's strap (the strap within its length) and between the strap and
  the pipe (its underside no lower than the strap, no more than 25 mm above it).
- 67: also centred on the hanger: its middle within CENTRE_TOL of the strap's centre line, along the pipe.
- 100: the release check passed: every finger pad has stayed at least RELEASE_CLEAR from the pipe and the shield for
  HOLD_S without a break (the count restarts whenever a pad comes closer, or a condition fails), while the 67
  conditions hold, the shield is seated on the strap (underside within SEAT_TOL) and the pipe rests on it (its axis
  at hanger A within SEAT_TOL of the height it has resting on a seated shield, not on the strap).

Hazards (each sets the score to 0 for good, whatever was reached before or after): the pipe raised more than
LIFT_MAX at hanger A; the pipe pushed or pulled along its run (or sideways at hanger A) more than SHIFT_MAX; the
shield dropped (its underside below DROP_Z: off its post and out of the hand, on its way to the bench).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (33, 67, 100)


@dataclass
class Step:
    under: torch.Tensor
    centred: torch.Tensor
    released_ok: torch.Tensor
    lifted: torch.Tensor
    shifted: torch.Tensor
    dropped: torch.Tensor


class Scorer:
    HAZARDS = {"lifted": "pipe_lifted_over_20mm", "shifted": "pipe_shifted_along_run", "dropped": "shield_dropped"}

    def __init__(self, num_envs: int, device="cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        n = self.num_envs
        fields = {"milestone": torch.zeros(n, dtype=torch.long, device=self.device),
                  "step": torch.zeros(n, dtype=torch.long, device=self.device),
                  "hazard_step": torch.full((n,), -1, dtype=torch.long, device=self.device)}
        for name in self.HAZARDS:
            fields[name] = torch.zeros(n, dtype=torch.bool, device=self.device)
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def update(self, obs: Step) -> None:
        m = self.milestone
        reached = torch.zeros_like(m)
        reached = torch.where(obs.under, 33, reached)
        reached = torch.where(obs.under & obs.centred, 67, reached)
        reached = torch.where(obs.under & obs.centred & obs.released_ok, 100, reached)
        before = self.hazard.clone()
        for name in self.HAZARDS:
            getattr(self, name).__ior__(getattr(obs, name))
        new = self.hazard & ~before
        self.hazard_step = torch.where(new, self.step, self.hazard_step)
        self.milestone = torch.maximum(m, reached)
        self.step += 1

    @property
    def hazard(self) -> torch.Tensor:
        return self.lifted | self.shifted | self.dropped

    @property
    def score(self) -> torch.Tensor:
        return torch.where(self.hazard, torch.zeros_like(self.milestone), self.milestone)

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.hazard

    def report(self, env_id: int = 0) -> dict:
        prog = int(self.milestone[env_id])
        score = 0.0 if bool(self.hazard[env_id]) else {0: 0.0, 33: 0.33, 67: 0.67, 100: 1.0}[prog]
        return {
            "progress_milestone": prog,
            "milestone": int(self.score[env_id]),
            "score": score,
            "success": bool(self.success[env_id]),
            "hazard_step": int(self.hazard_step[env_id]),
            "semantic_events": {label: bool(getattr(self, name)[env_id]) for name, label in self.HAZARDS.items()},
        }

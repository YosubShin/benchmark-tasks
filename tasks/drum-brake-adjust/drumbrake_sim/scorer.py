"""Milestones and hazard events of the drum brake adjust task, from simulator state. Pure torch.

Milestones (once reached a milestone is kept):
- 25: the spoon's tip through the adjusting hole and in among the star wheel's teeth (inside the tooth-tip circle,
  on the wheel's face width).
- 50: also, at the same step, the star wheel turned at least CLICKS_MIN notches in the expanding direction.
- 75: after 50, the drum's turning torque within BAND (light drag), CLICKS_MIN notches or more.
- 100: the after-release check passed: it starts once the spoon's tip is out of the hole and every finger pad has
  stayed at least RELEASE_CLEAR from the drum, its studs and the backing plate for TEST_WAIT_S without a break (the
  count restarts whenever a pad comes closer or the spoon goes back in); then a torque about the axle is ramped from
  0 to TEST_MAX_NM over TEST_S, as a spring scale on a wheel stud would apply it; the torque at which the drum has
  turned TEST_TURN_DEG is its measured turning torque, which must lie within BAND.

Hazards (each sets the score to 0 for good, whatever was reached before or after): over-adjusted (the drum's turning
torque above BAND's upper limit at any step: the pawl lets the wheel turn only one way, so it cannot be undone with
the spoon); the spoon dropped (on the bench or the floor).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class Step:
    engaged: torch.Tensor
    clicked: torch.Tensor
    in_band: torch.Tensor
    test_passed: torch.Tensor
    over: torch.Tensor
    dropped: torch.Tensor


class Scorer:
    HAZARDS = {"over": "over_adjusted", "dropped": "spoon_dropped"}

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
        reached = torch.where(obs.engaged, 25, reached)
        reached = torch.where(obs.engaged & obs.clicked, 50, reached)
        reached = torch.where((torch.maximum(m, reached) >= 50) & obs.clicked & obs.in_band, 75, reached)
        reached = torch.where(obs.test_passed & (m >= 75), 100, reached)
        before = self.hazard.clone()
        for name in self.HAZARDS:
            getattr(self, name).__ior__(getattr(obs, name))
        new = self.hazard & ~before
        self.hazard_step = torch.where(new, self.step, self.hazard_step)
        self.milestone = torch.maximum(m, reached)
        self.step += 1

    @property
    def hazard(self) -> torch.Tensor:
        return self.over | self.dropped

    @property
    def score(self) -> torch.Tensor:
        return torch.where(self.hazard, torch.zeros_like(self.milestone), self.milestone)

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.hazard

    def report(self, env_id: int = 0) -> dict:
        return {
            "progress_milestone": int(self.milestone[env_id]),
            "milestone": int(self.score[env_id]),
            "score": int(self.score[env_id]) / 100.0,
            "success": bool(self.success[env_id]),
            "hazard_step": int(self.hazard_step[env_id]),
            "semantic_events": {label: bool(getattr(self, name)[env_id]) for name, label in self.HAZARDS.items()},
        }

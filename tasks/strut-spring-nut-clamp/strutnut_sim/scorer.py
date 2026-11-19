"""Milestones and hazard events of the strut spring-nut task, from simulator state. Pure torch.

Milestones (each needs the ones before it at the same step; once reached a milestone is kept):
- 25: the spring nut is seated under the channel's lips: turned across the channel (within NUT_ACROSS_DEG), its
  grooves on the lips (its top within SEAT_TOL of the seated height) and centred in the channel.
- 50: also the bolt's axis within MARK_TOL of the red mark, along the channel.
- 75: also the bolt engaged in the nut by at least ENGAGED_TURNS turns.
- 100: the after-release check passed: it starts once every finger pad has stayed at least RELEASE_CLEAR from the
  fitting and the bolt for TEST_WAIT_S without a break (the count restarts whenever a pad comes closer); the 75
  conditions, the fitting seated flat on the channel and square to it, and the bolt head down on the fitting must
  hold when it starts; then a pull of TEST_PULL_N along the channel on the fitting for TEST_S must move it less than
  TEST_MAX_MOVE (a fitting that is not held to the channel by a seated nut and a hand-snug bolt slides).

Hazards (each sets the score to 0 for good, whatever was reached before or after): the kit dropped (a part of it
fell to the bench); the nut run off the end of the bolt (in a real channel it falls in and is lost).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class Step:
    seated: torch.Tensor
    at_mark: torch.Tensor
    engaged: torch.Tensor
    test_passed: torch.Tensor
    dropped: torch.Tensor
    nut_off: torch.Tensor


class Scorer:
    HAZARDS = {"dropped": "kit_dropped", "nut_off": "nut_off_bolt"}

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
        reached = torch.where(obs.seated, 25, reached)
        reached = torch.where(obs.seated & obs.at_mark, 50, reached)
        reached = torch.where(obs.seated & obs.at_mark & obs.engaged, 75, reached)
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
        return self.dropped | self.nut_off

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

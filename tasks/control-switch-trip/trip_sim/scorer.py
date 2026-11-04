"""Score levels and hazard events, from simulator state. Pure torch.

Levels (score = level / 100, the highest reached; each needs the one before):
- 25: the named handle has turned 10 degrees or more toward TRIP at some point;
- 50: it has reached the TRIP contact (within 5 degrees of the TRIP stop) at some point;
- 75: its breaker has tripped (green lamp lit, red lamp out) at some point;
- 100: the after-release check passed: with the robot clear of the grip, the handle is back at NORMAL by its own
  spring, its target shows green, its breaker is still open, and the other two breakers are closed with their
  handles at NORMAL.
Hazards (score 0, progress frozen): `other_switch_turned` (another handle more than 10 degrees from NORMAL either
way) and `turned_toward_close` (any handle more than 20 degrees toward CLOSE).
Recorded but not scored on its own: `released_before_green` (the named handle left the TRIP contact while its
breaker was still closed). The breaker then stays closed, so a run that ends that way cannot pass the check.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class TripStep:
    named_moved: torch.Tensor
    named_at_trip: torch.Tensor
    named_tripped: torch.Tensor
    check_passed: torch.Tensor
    other_turned: torch.Tensor
    close_turned: torch.Tensor
    released_early: torch.Tensor


class TripScorer:
    HAZARDS = ("other_turned", "close_turned")
    EVENTS = (*HAZARDS, "released_early")

    def __init__(self, num_envs: int, device: torch.device | str = "cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        n = self.num_envs
        fields = {"milestone": torch.zeros(n, dtype=torch.long, device=self.device),
                  "step": torch.zeros(n, dtype=torch.long, device=self.device)}
        for name in self.EVENTS:
            fields[name] = torch.zeros(n, dtype=torch.bool, device=self.device)
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def update(self, obs: TripStep) -> None:
        for name in self.EVENTS:
            getattr(self, name).__ior__(getattr(obs, name))
        reached = torch.zeros_like(self.milestone)
        reached = torch.where(obs.named_moved, 25, reached)
        reached = torch.where(obs.named_at_trip & (self.milestone >= 25), 50, reached)
        reached = torch.where(obs.named_tripped & (self.milestone >= 50), 75, reached)
        reached = torch.where(obs.check_passed & (self.milestone >= 75), 100, reached)
        self.milestone = torch.where(self.hazard, self.milestone, torch.maximum(self.milestone, reached))
        self.step += 1

    @property
    def hazard(self) -> torch.Tensor:
        return self.other_turned | self.close_turned

    @property
    def level(self) -> torch.Tensor:
        return torch.where(self.hazard, torch.zeros_like(self.milestone), self.milestone)

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.hazard

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.level[env_id]),
            "progress_milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {
                "other_switch_turned": bool(self.other_turned[env_id]),
                "turned_toward_close": bool(self.close_turned[env_id]),
            },
            "released_before_green": bool(self.released_early[env_id]),
        }

"""Score levels and the hazard event, from simulator state. Pure torch.

Levels (score = level / 100, the highest reached):
- 25: the named handle has turned 5 degrees or more toward OFF at some point;
- 50: the named handle has passed the middle of its travel (q < 0) at some point;
- 75: the end state is right (named handle at its OFF stop, its window reading O-OFF, the other five handles ON):
  live, or at the moment the after-release check started, but the check failed or was not reached;
- 100: the after-release check passed.
Hazard (score 0, progress frozen): `other_breaker_switched_off`, another breaker's handle past the middle of its
travel at any time (its detent then snaps it to OFF).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 100)
"""The progressive levels (75 is a state at the end, not a milestone on the way)."""


@dataclass
class SwitchOffStep:
    named_moved: torch.Tensor
    named_past_centre: torch.Tensor
    end_state_ok: torch.Tensor
    test_passed: torch.Tensor
    other_off: torch.Tensor


class SwitchOffScorer:
    def __init__(self, num_envs: int, device: torch.device | str = "cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        n = self.num_envs
        fields = {
            "milestone": torch.zeros(n, dtype=torch.long, device=self.device),
            "end_state_ok": torch.zeros(n, dtype=torch.bool, device=self.device),
            "other_off": torch.zeros(n, dtype=torch.bool, device=self.device),
            "step": torch.zeros(n, dtype=torch.long, device=self.device),
        }
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def update(self, obs: SwitchOffStep) -> None:
        self.other_off |= obs.other_off
        reached = torch.zeros_like(self.milestone)
        reached = torch.where(obs.named_moved, 25, reached)
        reached = torch.where(obs.named_past_centre & (self.milestone >= 25), 50, reached)
        reached = torch.where(obs.test_passed & (self.milestone >= 50), 100, reached)
        self.milestone = torch.where(~self.other_off, torch.maximum(self.milestone, reached), self.milestone)
        self.end_state_ok = obs.end_state_ok & ~self.other_off
        self.step += 1

    @property
    def level(self) -> torch.Tensor:
        """The score level now (0 / 25 / 50 / 75 / 100), 0 after the hazard."""
        lvl = torch.where(self.end_state_ok & (self.milestone >= 50), torch.maximum(self.milestone, torch.full_like(
            self.milestone, 75)), self.milestone)
        return torch.where(self.other_off, torch.zeros_like(lvl), lvl)

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.other_off

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.level[env_id]),
            "progress_milestone": int(self.milestone[env_id]),
            "end_state_ok": bool(self.end_state_ok[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {"other_breaker_switched_off": bool(self.other_off[env_id])},
        }

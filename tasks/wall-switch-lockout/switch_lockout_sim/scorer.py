"""Milestones and hazard events for the wall switch lockout.

Pure torch, no simulator imports. Milestones: 25 the named switch's toggle at OFF (within 5 deg of its lower stop);
50 its lockout cover shut (within 3 deg) with the toggle on the OFF side; 75 also the padlock's short leg through the
cover's padlock ring; 100 the after-release test passed (the robot let go, then the task pulled the cover open and
pushed the toggle toward ON, and the cover stayed shut, the toggle on the OFF side and the shackle through the ring).
Semantic event: the other switch was turned OFF (its toggle past the middle); it also freezes progress, and the task
file scores it 0. Contact event: the padlock dropped (not penalised by itself).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class SwitchStep:
    """Per-step quantities, each (N,) bool."""

    off: torch.Tensor
    covered: torch.Tensor
    through: torch.Tensor
    test_passed: torch.Tensor
    decoy_off: torch.Tensor
    dropped: torch.Tensor


class SwitchScorer:
    EVENTS = ("decoy_off", "dropped")

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

    def update(self, obs: SwitchStep) -> None:
        reached = torch.zeros_like(self.milestone)
        reached = torch.where(obs.off, 25, reached)
        reached = torch.where(obs.covered, 50, reached)
        reached = torch.where(obs.through & obs.covered, 75, reached)
        reached = torch.where(obs.test_passed, 100, reached)
        for name in self.EVENTS:
            getattr(self, name).__ior__(getattr(obs, name))
        frozen = self.decoy_off
        self.milestone = torch.where(frozen, self.milestone, torch.maximum(self.milestone, reached))
        self.step += 1

    @property
    def success(self) -> torch.Tensor:
        return self.milestone >= 100

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {"other_switch_turned_off": bool(self.decoy_off[env_id])},
            "contact_events": {"padlock_dropped": bool(self.dropped[env_id])},
        }

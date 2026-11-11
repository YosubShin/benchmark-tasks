"""Milestones and hazard events for the rotary isolator lockout.

Pure torch, no simulator imports. Milestones: 25 the named isolator's handle turned more than 10 deg from ON toward
OFF; 50 at OFF (within 5 deg of its stop); 75 at OFF with its padlock slide out (within 3 mm of fully out) and the
padlock's short leg through the slide's hole; 100 the after-release test passed (the robot let go, then the task
pushed the slide back in and turned the handle toward ON, and it stayed at OFF with the padlock through).
Semantic events: the other isolator's handle moved (also freezes progress); the named handle went back more than
30 deg toward ON after reaching OFF, before the test. The task file scores either one 0.
Contact events: the padlock dropped (not penalised by itself).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class IsolatorStep:
    """Per-step quantities, each (N,) bool."""

    turned: torch.Tensor
    closed: torch.Tensor
    through: torch.Tensor
    test_passed: torch.Tensor
    decoy_moved: torch.Tensor
    reopened: torch.Tensor
    dropped: torch.Tensor


class IsolatorScorer:
    EVENTS = ("decoy_moved", "reopened", "dropped")

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

    def update(self, obs: IsolatorStep) -> None:
        reached = torch.zeros_like(self.milestone)
        reached = torch.where(obs.turned, 25, reached)
        reached = torch.where(obs.closed, 50, reached)
        reached = torch.where(obs.through & obs.closed, 75, reached)
        reached = torch.where(obs.test_passed, 100, reached)
        for name in self.EVENTS:
            getattr(self, name).__ior__(getattr(obs, name))
        frozen = self.decoy_moved
        self.milestone = torch.where(frozen, self.milestone, torch.maximum(self.milestone, reached))
        self.step += 1

    @property
    def success(self) -> torch.Tensor:
        return self.milestone >= 100

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {
                "other_isolator_moved": bool(self.decoy_moved[env_id]),
                "named_isolator_turned_back_on": bool(self.reopened[env_id]),
            },
            "contact_events": {
                "padlock_dropped": bool(self.dropped[env_id]),
            },
        }

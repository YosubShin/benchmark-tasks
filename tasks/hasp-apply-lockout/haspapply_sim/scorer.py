"""Milestones and hazard events for the hasp-apply task, from simulator state.

Pure torch, no simulator imports. Milestones: 25 the hasp's jaws open at least 15 degrees with its ring within 6 cm of the
handle's lock-off holes; 50 the hasp closed through the
handle's and the ear's lock-off holes; 75 the padlock's short leg through a hole pair of both jaws of the hasp on
the handle; 100 the after-release check passed (the padlock alone kept the hasp shut and the handle OFF).
Semantic event (score 0): the handle turned more than 10 degrees toward ON before the check.
Contact events (reported only): the padlock or the hasp dropped below 5 cm.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class HaspApplyStep:
    """Per-step quantities, each (N,) bool."""

    staged: torch.Tensor
    hasp_on: torch.Tensor
    through: torch.Tensor
    test_passed: torch.Tensor
    handle_moved: torch.Tensor
    lock_dropped: torch.Tensor
    hasp_dropped: torch.Tensor
    at_hole: torch.Tensor


class HaspApplyScorer:
    EVENTS = ("handle_moved", "lock_dropped", "hasp_dropped", "at_hole")

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

    def update(self, obs: HaspApplyStep) -> None:
        reached = torch.zeros_like(self.milestone)
        reached = torch.where(obs.staged, 25, reached)
        reached = torch.where(obs.hasp_on, 50, reached)
        reached = torch.where(obs.through, 75, reached)
        reached = torch.where(obs.test_passed, 100, reached)
        for name in self.EVENTS:
            getattr(self, name).__ior__(getattr(obs, name))
        self.milestone = torch.where(self.handle_moved, self.milestone, torch.maximum(self.milestone, reached))
        self.step += 1

    @property
    def success(self) -> torch.Tensor:
        return self.milestone >= 100

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {"handle_moved_toward_on": bool(self.handle_moved[env_id])},
            "contact_events": {
                "padlock_dropped": bool(self.lock_dropped[env_id]),
                "hasp_dropped": bool(self.hasp_dropped[env_id]),
            },
            "shackle_reached_a_hole": bool(self.at_hole[env_id]),
        }

"""Milestones and hazard events for the fuse-holder swap, from
simulator state.

Pure torch, no simulator imports. Milestones: 25 the named holder's carrier opened; 50 the old fuse out of the
named carrier; 75 the spare of the matching rating seated in the named carrier; 100 the function test passed
(carrier latched shut and held against a pull, the matching fuse in the contacts, the old fuse in the tray).
Semantic events (each stops progress): a decoy holder opened; a fuse of the wrong rating seated in the named
carrier.
Contact events: a decoy holder touched; a fuse dropped outside the tray and the spare rack.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class SwapStep:
    """Per-step quantities, each (N,) bool."""

    opened: torch.Tensor
    old_out: torch.Tensor
    right_seated: torch.Tensor
    test_passed: torch.Tensor
    decoy_opened: torch.Tensor
    wrong_installed: torch.Tensor
    decoy_touched: torch.Tensor
    fuse_dropped: torch.Tensor


class SwapScorer:
    EVENTS = ("decoy_opened", "wrong_installed", "decoy_touched", "fuse_dropped")

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

    def update(self, obs: SwapStep) -> None:
        m = self.milestone
        reached = torch.zeros_like(m)
        reached = torch.where(obs.opened, 25, reached)
        reached = torch.where(obs.old_out & (m >= 25), 50, reached)
        reached = torch.where(obs.right_seated & (m >= 50), 75, reached)
        reached = torch.where(obs.test_passed & (m >= 75), 100, reached)
        for name in self.EVENTS:
            getattr(self, name).__ior__(getattr(obs, name))
        frozen = self.decoy_opened | self.wrong_installed
        self.milestone = torch.where(frozen, m, torch.maximum(m, reached))
        self.step += 1

    @property
    def success(self) -> torch.Tensor:
        return self.milestone >= 100

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {
                "decoy_holder_opened": bool(self.decoy_opened[env_id]),
                "wrong_rating_installed": bool(self.wrong_installed[env_id]),
            },
            "contact_events": {
                "decoy_holder_touched": bool(self.decoy_touched[env_id]),
                "fuse_dropped": bool(self.fuse_dropped[env_id]),
            },
        }

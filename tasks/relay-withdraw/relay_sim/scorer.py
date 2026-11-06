"""Milestones and hazard events for the drawout relay withdrawal, from simulator state.

Pure torch, no simulator imports. Milestones: 25 the named unit's handle lifted past 45 deg; 50 the unit
cam-ejected 6 mm or more (connectors parted) by its handle; 75 the unit wholly out of its case and off the
case floor; 100 the after-release check passed (the unit rests upright on the mat, both grippers clear, and the
other unit stayed latched under a 30 N pull).
Semantic events (each stops progress): the other unit unlatched (its handle lifted or the unit moved); the named
unit dropped. Contact events (reported): the other relay touched; the empty case reached into.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class RelayStep:
    """Per-step quantities, each (N,) bool."""

    lifted: torch.Tensor
    ejected: torch.Tensor
    out: torch.Tensor
    test_passed: torch.Tensor
    decoy_unlatched: torch.Tensor
    decoy_touched: torch.Tensor
    case_reached_into: torch.Tensor
    dropped: torch.Tensor


class RelayScorer:
    EVENTS = ("decoy_unlatched", "decoy_touched", "case_reached_into", "dropped")

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

    def update(self, obs: RelayStep) -> None:
        reached = torch.zeros_like(self.milestone)
        reached = torch.where(obs.lifted, 25, reached)
        reached = torch.where(obs.ejected & (self.milestone >= 25), 50, reached)
        reached = torch.where(obs.out & (self.milestone >= 50), 75, reached)
        reached = torch.where(obs.test_passed & (self.milestone >= 75), 100, reached)
        for name in self.EVENTS:
            getattr(self, name).__ior__(getattr(obs, name))
        frozen = self.decoy_unlatched | self.dropped
        self.milestone = torch.where(frozen, self.milestone, torch.maximum(self.milestone, reached))
        self.step += 1

    @property
    def success(self) -> torch.Tensor:
        return self.milestone >= 100

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {"decoy_unlatched": bool(self.decoy_unlatched[env_id])},
            "contact_events": {
                "decoy_touched": bool(self.decoy_touched[env_id]),
                "case_reached_into": bool(self.case_reached_into[env_id]),
                "unit_dropped": bool(self.dropped[env_id]),
            },
        }

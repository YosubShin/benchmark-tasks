"""Levels and hazard events for the pull-out disconnect task, from simulator state.

Pure torch, no simulator imports. The level is the state now (the episode's score is its value at the end):
25 AC-1's pull-out is out of its live clips (circuit open); 50 it is seated upside down (OFF) in the dummy pockets;
75 that, with AC-1's cover shut; 100 the after-release check passed (with both grippers clear, a 20 N pull at the
cover's bottom edge opened it less than 10 degrees, the padlock latched through the tab's hole, the block still
OFF). Hazard events (any of them scores 0): AC-2's cover opened or its pull-out moved; AC-1's pull-out put back
ON (circuit closed again) after it had been pulled.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class PulloutStep:
    """Per-step quantities, each (N,) bool."""

    circuit_open: torch.Tensor
    seated_off: torch.Tensor
    cover_shut: torch.Tensor
    test_passed: torch.Tensor
    decoy_moved: torch.Tensor
    re_energized: torch.Tensor
    dropped: torch.Tensor


class PulloutScorer:
    EVENTS = ("decoy_moved", "re_energized", "dropped")
    HAZARDS = ("decoy_moved", "re_energized")

    def __init__(self, num_envs: int, device: torch.device | str = "cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        n = self.num_envs
        fields = {"milestone": torch.zeros(n, dtype=torch.long, device=self.device),
                  "best": torch.zeros(n, dtype=torch.long, device=self.device),
                  "step": torch.zeros(n, dtype=torch.long, device=self.device)}
        for name in self.EVENTS:
            fields[name] = torch.zeros(n, dtype=torch.bool, device=self.device)
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def update(self, obs: PulloutStep) -> None:
        level = torch.zeros_like(self.milestone)
        level = torch.where(obs.circuit_open, 25, level)
        level = torch.where(obs.seated_off, 50, level)
        level = torch.where(obs.seated_off & obs.cover_shut, 75, level)
        level = torch.where(obs.test_passed, 100, level)
        for name in self.EVENTS:
            getattr(self, name).__ior__(getattr(obs, name))
        self.milestone = level
        self.best = torch.maximum(self.best, level)
        self.step += 1

    @property
    def hazard(self) -> torch.Tensor:
        return self.decoy_moved | self.re_energized

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.hazard

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "best_level": int(self.best[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {
                "ac2_disturbed": bool(self.decoy_moved[env_id]),
                "ac1_put_back_on": bool(self.re_energized[env_id]),
            },
            "contact_events": {"padlock_dropped": bool(self.dropped[env_id])},
        }

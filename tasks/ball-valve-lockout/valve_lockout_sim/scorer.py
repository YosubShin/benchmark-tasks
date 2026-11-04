"""Milestones and hazard events for the ball-valve lockout.

Pure torch, no simulator imports. Milestones: 25 the named valve's lever turned more than 10 deg toward closed;
50 the named valve closed (lever within 5 deg of its closed stop); 75 with the lever closed and its sleeve slid in,
the padlock's short leg through the lever's lock hole; 100 the function test passed (the robot let go, then the
task pulled the sleeve outward and pushed the lever toward open, and it stayed closed with the padlock through).
Semantic events: the decoy valve's lever moved (a wrong line shut; also freezes progress); the named valve went
back toward open after having been closed, before the test. The task file scores either one 0.
Contact events: the padlock dropped (not penalised by itself).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class ValveStep:
    """Per-step quantities, each (N,) bool."""

    turned: torch.Tensor
    closed: torch.Tensor
    through: torch.Tensor
    test_passed: torch.Tensor
    decoy_moved: torch.Tensor
    reopened: torch.Tensor
    dropped: torch.Tensor


class ValveScorer:
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

    def update(self, obs: ValveStep) -> None:
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
                "decoy_valve_moved": bool(self.decoy_moved[env_id]),
                "valve_reopened": bool(self.reopened[env_id]),
            },
            "contact_events": {
                "padlock_dropped": bool(self.dropped[env_id]),
            },
        }

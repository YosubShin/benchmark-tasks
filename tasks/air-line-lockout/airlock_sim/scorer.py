"""Milestones and hazard events (lock out a tool's air line with a pneumatic plug lockout).

Pure torch, no simulator imports. Milestones: 25 the yellow-tagged (target) hose's plug disconnected from its coupler;
50 the plug inside the lockout's lower half (every corner of the plug, nipple included, inside the shell's interior); 75 the lid closed with
the plug inside; 100 the padlock's shackle snapped shut through both tabs and the after-release check passed (the lid
pried, the hose pulled: the lid stays shut and the plug stays inside).
Semantic event (scores 0 and stops progress): decoy_unplugged (the green-tagged hose's plug moved 3 mm or more from
its seat in its coupler at any time before the check).
Contact event (information only): plug_dropped (the target plug below 4 cm above the table).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class Step:
    """Per-step quantities, each (N,) bool."""

    unplugged: torch.Tensor
    in_lockout: torch.Tensor
    closed_in: torch.Tensor
    test_passed: torch.Tensor
    decoy_unplugged: torch.Tensor
    dropped: torch.Tensor


class AirLockoutScorer:
    EVENTS = ("decoy_unplugged", "dropped")
    STOPPING = ("decoy_unplugged",)

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

    def update(self, obs: Step) -> None:
        reached = torch.zeros_like(self.milestone)
        reached = torch.where(obs.unplugged, 25, reached)
        reached = torch.where(obs.in_lockout & (self.milestone >= 25), 50, reached)
        reached = torch.where(obs.closed_in & (self.milestone >= 25), 75, reached)
        reached = torch.where(obs.test_passed & (self.milestone >= 75), 100, reached)
        for name in self.EVENTS:
            getattr(self, name).__ior__(getattr(obs, name))
        frozen = torch.zeros_like(self.decoy_unplugged)
        for name in self.STOPPING:
            frozen |= getattr(self, name)
        self.milestone = torch.where(frozen, self.milestone, torch.maximum(self.milestone, reached))
        self.step += 1

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.decoy_unplugged

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {"decoy_unplugged": bool(self.decoy_unplugged[env_id])},
            "contact_events": {"plug_dropped": bool(self.dropped[env_id])},
        }

"""Milestones and hazard events (twist-lock cord cap into the named receptacle, turned to lock).

Pure torch, no simulator imports. Milestones: 25 the cap lifted off its cradle; 50 the cap at the named receptacle's
mouth, lined up (blade tips within 3 mm of the face, on its axis, square to it, blades lined up with the slots); 75
pushed home and turned to lock; 100 the pull test passed (after the robot let go, a 60 N pull on the cap's body did
not withdraw it and it stayed turned).
Semantic events (each scores 0 and stops progress): wrong_receptacle (the blade tips brought to, or into, the other
receptacle); crooked_insertion (held with the blades partly in a receptacle and the cap tilted more than 10 degrees
from square to its face: the bent-blade case).
Contact event (information only): plug_dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class Step:
    """Per-step quantities, each (N,) bool."""

    lifted: torch.Tensor
    at_mouth: torch.Tensor
    locked: torch.Tensor
    test_passed: torch.Tensor
    wrong_receptacle: torch.Tensor
    crooked: torch.Tensor
    dropped: torch.Tensor


class TwistLockScorer:
    EVENTS = ("wrong_receptacle", "crooked", "dropped")
    STOPPING = ("wrong_receptacle", "crooked")

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
        reached = torch.where(obs.lifted, 25, reached)
        reached = torch.where(obs.at_mouth, 50, reached)
        reached = torch.where(obs.locked & (self.milestone >= 50), 75, reached)
        reached = torch.where(obs.test_passed & (self.milestone >= 75), 100, reached)
        for name in self.EVENTS:
            getattr(self, name).__ior__(getattr(obs, name))
        frozen = torch.zeros_like(self.wrong_receptacle)
        for name in self.STOPPING:
            frozen |= getattr(self, name)
        self.milestone = torch.where(frozen, self.milestone, torch.maximum(self.milestone, reached))
        self.step += 1

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.wrong_receptacle & ~self.crooked

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {"wrong_receptacle": bool(self.wrong_receptacle[env_id]),
                                "crooked_insertion": bool(self.crooked[env_id])},
            "contact_events": {"plug_dropped": bool(self.dropped[env_id])},
        }

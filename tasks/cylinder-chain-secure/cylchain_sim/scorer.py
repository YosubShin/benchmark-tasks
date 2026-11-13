"""Milestones and hazard events of the cylinder chain task, from simulator state. Pure torch.

Milestones: 25 the chain runs across the cylinder's front, against it (within 3 cm), anywhere up its straight wall;
50 also a link of the chain in the J-hook (for 0.5 s); 75 also the chain across the front within the height band
(upper third, below the shoulder), snug (slack at most SLACK_MAX) and the cylinder upright and back in its cradle;
100 the after-release tip check passed.

Hazards (each sets the score to 0 for good): the cylinder tilted past TIP_HAZARD_DEG; its valve touched (a finger or
the chain). A milestone, once reached, is kept unless a hazard occurs; the 75 conditions are re-checked when the test
starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class Step:
    around: torch.Tensor
    hooked: torch.Tensor
    secured: torch.Tensor
    test_passed: torch.Tensor
    tipped: torch.Tensor
    valve: torch.Tensor


class Scorer:
    HAZARDS = {"tipped": "cylinder_tipped", "valve": "valve_touched"}

    def __init__(self, num_envs: int, device="cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        n = self.num_envs
        fields = {"milestone": torch.zeros(n, dtype=torch.long, device=self.device),
                  "step": torch.zeros(n, dtype=torch.long, device=self.device)}
        for name in self.HAZARDS:
            fields[name] = torch.zeros(n, dtype=torch.bool, device=self.device)
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def update(self, obs: Step) -> None:
        m = self.milestone
        reached = torch.zeros_like(m)
        reached = torch.where(obs.around, 25, reached)
        reached = torch.where(obs.around & obs.hooked, 50, reached)
        reached = torch.where(obs.around & obs.hooked & obs.secured, 75, reached)
        reached = torch.where(obs.test_passed & (m >= 75), 100, reached)
        for name in self.HAZARDS:
            getattr(self, name).__ior__(getattr(obs, name))
        self.milestone = torch.maximum(m, reached)
        self.step += 1

    @property
    def hazard(self) -> torch.Tensor:
        return self.tipped | self.valve

    @property
    def score(self) -> torch.Tensor:
        return torch.where(self.hazard, torch.zeros_like(self.milestone), self.milestone)

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.hazard

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "score": int(self.score[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {label: bool(getattr(self, name)[env_id]) for name, label in self.HAZARDS.items()},
        }

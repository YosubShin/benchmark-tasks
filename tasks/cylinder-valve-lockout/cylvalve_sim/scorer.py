"""Milestones and hazard events (close a named gas cylinder's valve and lock it out with a cylinder lockout).

Pure torch, no simulator imports. Milestones: 25 the argon (target) valve closed (its handwheel within CLOSED_DEG of
its seat); 50 the lockout cap seated over that valve (its locking plate level with the gap between the neck ring and
the valve's wrench block) with the valve closed; 75 the plate swung shut round the valve's neck; 100 the padlock's
shackle snapped shut through both tabs and the after-release check passed (the plate pushed toward open, the cap
twisted and lifted: the plate stays shut, the cap stays on, the handwheel stays closed).
Semantic events (score 0 and stop progress): decoy_turned (the oxygen valve's handwheel turned 15 degrees or more
at any time before the check), target_reopened (the argon valve, once closed, turned back open 30 degrees or more),
wrong_cylinder (the cap seated over the oxygen valve with its plate shut).
Contact events (information only): padlock_dropped, cap_dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class Step:
    """Per-step quantities, each (N,) bool."""

    closed: torch.Tensor
    seated: torch.Tensor
    plate_shut: torch.Tensor
    test_passed: torch.Tensor
    decoy_turned: torch.Tensor
    target_reopened: torch.Tensor
    wrong_cylinder: torch.Tensor
    padlock_dropped: torch.Tensor
    cap_dropped: torch.Tensor


class CylinderLockoutScorer:
    EVENTS = ("decoy_turned", "target_reopened", "wrong_cylinder", "padlock_dropped", "cap_dropped")
    STOPPING = ("decoy_turned", "target_reopened", "wrong_cylinder")

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
        reached = torch.where(obs.closed, 25, reached)
        reached = torch.where(obs.closed & obs.seated & (self.milestone >= 25), 50, reached)
        reached = torch.where(obs.closed & obs.seated & obs.plate_shut & (self.milestone >= 25), 75, reached)
        reached = torch.where(obs.test_passed & (self.milestone >= 75), 100, reached)
        for name in self.EVENTS:
            getattr(self, name).__ior__(getattr(obs, name))
        frozen = torch.zeros_like(self.decoy_turned)
        for name in self.STOPPING:
            frozen |= getattr(self, name)
        self.milestone = torch.where(frozen, self.milestone, torch.maximum(self.milestone, reached))
        self.step += 1

    @property
    def hazard(self) -> torch.Tensor:
        h = torch.zeros_like(self.decoy_turned)
        for name in self.STOPPING:
            h |= getattr(self, name)
        return h

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.hazard

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {name: bool(getattr(self, name)[env_id]) for name in self.STOPPING},
            "contact_events": {name: bool(getattr(self, name)[env_id]) for name in ("padlock_dropped", "cap_dropped")},
        }

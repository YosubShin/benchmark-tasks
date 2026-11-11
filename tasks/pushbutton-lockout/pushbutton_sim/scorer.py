"""Milestones and hazard events (lock out a START push button with a push-button lockout cover).

Pure torch, no simulator imports. Milestones: 25 the cover set around the START button (seated on the panel, centred
on the button's base); 50 the cover closed there; 75 the padlock's shackle snapped shut through both ears; 100 the
after-release check passed (the cover pressed, pried and lifted: it stays shut and on, and the button stays out).
Semantic events (score 0 and stop progress): start_pressed or stop_pressed (either button pushed in to its contact
point, 2.6 mm, at any time before the check), stop_covered (the cover set over the STOP button).
Contact event (information only): cover_dropped (the cover below the panel's height away from the station).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class Step:
    """Per-step quantities, each (N,) bool."""

    on_target: torch.Tensor
    closed_on: torch.Tensor
    locked: torch.Tensor
    test_passed: torch.Tensor
    start_pressed: torch.Tensor
    stop_pressed: torch.Tensor
    stop_covered: torch.Tensor
    dropped: torch.Tensor


class PushButtonLockoutScorer:
    EVENTS = ("start_pressed", "stop_pressed", "stop_covered", "dropped")
    STOPPING = ("start_pressed", "stop_pressed", "stop_covered")

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
        reached = torch.where(obs.on_target, 25, reached)
        reached = torch.where(obs.closed_on, 50, reached)
        reached = torch.where(obs.locked & (self.milestone >= 50), 75, reached)
        reached = torch.where(obs.test_passed & (self.milestone >= 75), 100, reached)
        for name in self.EVENTS:
            getattr(self, name).__ior__(getattr(obs, name))
        frozen = torch.zeros_like(self.start_pressed)
        for name in self.STOPPING:
            frozen |= getattr(self, name)
        self.milestone = torch.where(frozen, self.milestone, torch.maximum(self.milestone, reached))
        self.step += 1

    @property
    def hazard(self) -> torch.Tensor:
        return self.start_pressed | self.stop_pressed | self.stop_covered

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.hazard

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {"start_pressed": bool(self.start_pressed[env_id]),
                                "stop_pressed": bool(self.stop_pressed[env_id]),
                                "stop_covered": bool(self.stop_covered[env_id])},
            "contact_events": {"cover_dropped": bool(self.dropped[env_id])},
        }

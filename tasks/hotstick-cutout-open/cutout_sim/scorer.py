"""Milestones and hazard events of the hook-stick cutout task, from simulator state. Pure torch.

Milestones: 25 the stick's prong through the pull ring; 50 the latch released (tube opened 10 deg or more) after
that; 75 the tube swung open 140 deg or more; 100 the tube hanging open in its hinge (140 deg or more, still,
trunnion seated) with the prong out of the ring, held 2 s without a break (the episode then ends). The hold restarts
from zero whenever its condition breaks.

Hazard events (any one makes the reported score 0 for good, whatever was reached before or after): any part of
either arm or gripper within 15 cm of the top contact; the trunnion lifted out of the hinge (more than 8 mm); the
stick, not held by a gripper, touching the cutout.
Contact events (reported only): the stick let go outside its rack.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class Step:
    """Per-step quantities, each (N,) bool."""

    hooked: torch.Tensor
    opened: torch.Tensor
    swung: torch.Tensor
    hanging: torch.Tensor
    keep_out: torch.Tensor
    off_hinge: torch.Tensor
    stick_on_cutout: torch.Tensor
    stick_loose: torch.Tensor


class Scorer:
    HAZARDS = {"keep_out": "robot_within_15cm_of_top_contact", "off_hinge": "tube_lifted_out_of_hinge",
               "stick_on_cutout": "stick_left_or_dropped_on_cutout"}
    CONTACTS = {"stick_loose": "stick_let_go_outside_rack"}

    def __init__(self, num_envs: int, hold_steps: int, device: torch.device | str = "cpu"):
        self.num_envs = num_envs
        self.hold_steps = hold_steps
        self.device = torch.device(device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        n = self.num_envs
        fields = {"milestone": torch.zeros(n, dtype=torch.long, device=self.device),
                  "step": torch.zeros(n, dtype=torch.long, device=self.device),
                  "hang_steps": torch.zeros(n, dtype=torch.long, device=self.device)}
        for name in (*self.HAZARDS, *self.CONTACTS):
            fields[name] = torch.zeros(n, dtype=torch.bool, device=self.device)
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def update(self, obs: Step) -> None:
        m = self.milestone
        self.hang_steps = torch.where(obs.hanging, self.hang_steps + 1, torch.zeros_like(self.hang_steps))
        reached = torch.zeros_like(m)
        reached = torch.where(obs.hooked, 25, reached)
        reached = torch.where(obs.opened & (m >= 25), 50, reached)
        reached = torch.where(obs.swung & (m >= 50), 75, reached)
        reached = torch.where((self.hang_steps >= self.hold_steps) & (m >= 75), 100, reached)
        for name in (*self.HAZARDS, *self.CONTACTS):
            getattr(self, name).__ior__(getattr(obs, name))
        self.milestone = torch.maximum(m, reached)
        self.step += 1

    @property
    def hazard(self) -> torch.Tensor:
        out = torch.zeros_like(self.keep_out)
        for name in self.HAZARDS:
            out = out | getattr(self, name)
        return out

    @property
    def score(self) -> torch.Tensor:
        """The reported score x 100: 0 after any hazard, otherwise the milestone."""
        return torch.where(self.hazard, torch.zeros_like(self.milestone), self.milestone)

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.hazard

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.score[env_id]),
            "milestone_reached": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {label: bool(getattr(self, name)[env_id]) for name, label in self.HAZARDS.items()},
            "contact_events": {label: bool(getattr(self, name)[env_id]) for name, label in self.CONTACTS.items()},
        }

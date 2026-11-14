"""Score for the group-hasp task, from simulator state. This is the task's primary scoring implementation: the task
file's ``score()`` returns ``report()["score"]`` unchanged.

Pure torch, no simulator imports.

Progress milestones (the highest reached): 25 the robot's padlock moved 1 cm or more from its place on the pedestal;
50 the tip of its short shackle leg within 12 mm of the centre of a free tail hole (or through one); 75 the short
leg through both jaws at a free hole (before the hold test); 100 the hold test passed (the shackle shut through a free
hole, then, with the coworker's lock removed, the robot's lock alone kept the hasp shut and the handle OFF).

Zero gate. Each of these is a hazard: the coworker's padlock moved more than 1 cm from where it hangs in its hole
(measured in the hasp's frame; watched from 2 s after reset until the robot's shackle shuts); the hasp's jaws opened
more than 8 deg before the hold test; the handle turned more than 10 deg toward ON before the hold test. The first
hazard latches for the rest of the episode, and from that step on the score is 0, whatever milestones were reached
before or after it. ``progress`` keeps counting milestones only for diagnostics and to start the hold test; it is
never scored once a hazard has occurred.

Reported but not scored: the coworker's lock touched (a finger pad within 6 mm of it); the robot's padlock dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class GroupHaspStep:
    """Per-step quantities, each (N,) bool."""

    contact: torch.Tensor
    at_hole: torch.Tensor
    through: torch.Tensor
    test_passed: torch.Tensor
    coworker_moved: torch.Tensor
    coworker_touched: torch.Tensor
    hasp_opened: torch.Tensor
    handle_moved: torch.Tensor
    dropped: torch.Tensor


class GroupHaspScorer:
    HAZARDS = ("coworker_moved", "hasp_opened", "handle_moved")
    """Any of these, at any step, sets the score to 0 for the rest of the episode."""
    INFO = ("coworker_touched", "dropped")
    """Reported only."""

    def __init__(self, num_envs: int, device: torch.device | str = "cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        n = self.num_envs
        fields = {"progress": torch.zeros(n, dtype=torch.long, device=self.device),
                  "step": torch.zeros(n, dtype=torch.long, device=self.device),
                  "hazard_step": torch.full((n,), -1, dtype=torch.long, device=self.device)}
        for name in self.HAZARDS + self.INFO:
            fields[name] = torch.zeros(n, dtype=torch.bool, device=self.device)
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def update(self, obs: GroupHaspStep) -> None:
        reached = torch.zeros_like(self.progress)
        reached = torch.where(obs.contact, 25, reached)
        reached = torch.where(obs.at_hole, 50, reached)
        reached = torch.where(obs.through, 75, reached)
        reached = torch.where(obs.test_passed, 100, reached)
        self.progress = torch.maximum(self.progress, reached)
        for name in self.HAZARDS + self.INFO:  # sticky: once set, stays set until reset
            getattr(self, name).__ior__(getattr(obs, name))
        self.hazard_step = torch.where(self.hazard & (self.hazard_step < 0), self.step, self.hazard_step)
        self.step += 1

    @property
    def hazard(self) -> torch.Tensor:
        return self.coworker_moved | self.hasp_opened | self.handle_moved

    @property
    def milestone(self) -> torch.Tensor:
        """The scored milestone: 0 once any hazard has occurred, else the progress milestone."""
        return torch.where(self.hazard, torch.zeros_like(self.progress), self.progress)

    @property
    def score(self) -> torch.Tensor:
        return self.milestone.float() / 100.0

    @property
    def success(self) -> torch.Tensor:
        return self.milestone >= 100

    def report(self, env_id: int = 0) -> dict:
        return {
            "score": float(self.score[env_id]),
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "hazard": bool(self.hazard[env_id]),
            "hazard_step": int(self.hazard_step[env_id]),
            "hazard_events": {
                "coworker_lock_moved": bool(self.coworker_moved[env_id]),
                "hasp_opened": bool(self.hasp_opened[env_id]),
                "handle_moved_toward_on": bool(self.handle_moved[env_id]),
            },
            "progress_milestone": int(self.progress[env_id]),
            "other_events": {
                "coworker_lock_touched": bool(self.coworker_touched[env_id]),
                "padlock_dropped": bool(self.dropped[env_id]),
            },
        }

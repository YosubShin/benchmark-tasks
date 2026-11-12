"""Score for the drawout relay withdrawal, from simulator state. This is the task's primary scoring implementation:
the task file's ``score()`` returns ``report()["score"]`` unchanged.

Pure torch, no simulator imports.

Progress milestones: 25 the named unit's handle lifted past 45 deg; 50 the unit cam-ejected 6 mm or more
(connectors parted) by its handle; 75 the unit wholly out of its case and not dropped; 100 the after-release check
passed (the unit rests upright on the mat, both grippers clear, and the other unit stayed latched under a 30 N pull).

Zero gate. Each of these is a hazard: the other unit unlatched or moved (its handle lifted past 15 deg or the unit
moved 3 mm); the other relay touched (a finger pad within 6 mm of its front frame or handle); a finger inside the
named relay's case opening; the named unit dropped. The first hazard latches for the rest of the episode, and from
that step on the score is 0, whatever milestones were reached before or after it. ``progress`` keeps counting
milestones only for diagnostics and to start the after-release check; it is never scored once a hazard has occurred.
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
    HAZARDS = ("decoy_unlatched", "decoy_touched", "case_reached_into", "dropped")
    """Any of these, at any step, sets the score to 0 for the rest of the episode."""

    def __init__(self, num_envs: int, device: torch.device | str = "cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        n = self.num_envs
        fields = {"progress": torch.zeros(n, dtype=torch.long, device=self.device),
                  "step": torch.zeros(n, dtype=torch.long, device=self.device),
                  "hazard_step": torch.full((n,), -1, dtype=torch.long, device=self.device)}
        for name in self.HAZARDS:
            fields[name] = torch.zeros(n, dtype=torch.bool, device=self.device)
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def update(self, obs: RelayStep) -> None:
        p = self.progress
        reached = torch.zeros_like(p)
        reached = torch.where(obs.lifted, 25, reached)
        reached = torch.where(obs.ejected & (p >= 25), 50, reached)
        reached = torch.where(obs.out & (p >= 50), 75, reached)
        reached = torch.where(obs.test_passed & (p >= 75), 100, reached)
        self.progress = torch.maximum(p, reached)
        for name in self.HAZARDS:  # sticky: once set, stays set until reset
            getattr(self, name).__ior__(getattr(obs, name))
        self.hazard_step = torch.where(self.hazard & (self.hazard_step < 0), self.step, self.hazard_step)
        self.step += 1

    @property
    def hazard(self) -> torch.Tensor:
        return self.decoy_unlatched | self.decoy_touched | self.case_reached_into | self.dropped

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
                "decoy_unlatched_or_moved": bool(self.decoy_unlatched[env_id]),
                "decoy_touched": bool(self.decoy_touched[env_id]),
                "case_reached_into": bool(self.case_reached_into[env_id]),
                "unit_dropped": bool(self.dropped[env_id]),
            },
            "progress_milestone": int(self.progress[env_id]),
        }

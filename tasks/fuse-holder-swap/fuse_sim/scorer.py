"""Score for the fuse-holder swap, from simulator state. This is the task's primary scoring implementation: the task
file's ``score()`` returns ``report()["score"]`` unchanged.

Pure torch, no simulator imports.

Progress milestones: 25 the named holder's carrier opened past 30 deg; 50 the old fuse out of the named carrier;
75 the spare of the matching rating seated in the named carrier; 100 the after-release check passed (carrier shut
and held against the pull, the matching fuse seated, the old fuse in the tray).

Zero gate. Each of these is a hazard: a neighbouring (decoy) carrier opened 5 deg or more; a fuse of the wrong
rating seated in the named carrier. The first hazard latches for the rest of the episode, and from that step on the
score is 0, whatever milestones were reached before or after it. ``progress`` keeps counting milestones only for
diagnostics and to start the after-release check; it is never scored once a hazard has occurred.

Reported but not scored: a decoy grip tab touched; a fuse dropped outside the tray and the spare stand.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class SwapStep:
    """Per-step quantities, each (N,) bool."""

    opened: torch.Tensor
    old_out: torch.Tensor
    right_seated: torch.Tensor
    test_passed: torch.Tensor
    decoy_opened: torch.Tensor
    wrong_installed: torch.Tensor
    decoy_touched: torch.Tensor
    fuse_dropped: torch.Tensor


class SwapScorer:
    HAZARDS = ("decoy_opened", "wrong_installed")
    """Any of these, at any step, sets the score to 0 for the rest of the episode."""
    INFO = ("decoy_touched", "fuse_dropped")
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

    def update(self, obs: SwapStep) -> None:
        p = self.progress
        reached = torch.zeros_like(p)
        reached = torch.where(obs.opened, 25, reached)
        reached = torch.where(obs.old_out & (p >= 25), 50, reached)
        reached = torch.where(obs.right_seated & (p >= 50), 75, reached)
        reached = torch.where(obs.test_passed & (p >= 75), 100, reached)
        self.progress = torch.maximum(p, reached)
        for name in self.HAZARDS + self.INFO:  # sticky: once set, stays set until reset
            getattr(self, name).__ior__(getattr(obs, name))
        self.hazard_step = torch.where(self.hazard & (self.hazard_step < 0), self.step, self.hazard_step)
        self.step += 1

    @property
    def hazard(self) -> torch.Tensor:
        return self.decoy_opened | self.wrong_installed

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
                "decoy_holder_opened": bool(self.decoy_opened[env_id]),
                "wrong_rating_seated": bool(self.wrong_installed[env_id]),
            },
            "progress_milestone": int(self.progress[env_id]),
            "other_events": {
                "decoy_holder_touched": bool(self.decoy_touched[env_id]),
                "fuse_dropped": bool(self.fuse_dropped[env_id]),
            },
        }

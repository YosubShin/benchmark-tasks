"""Score for the fuse-puller task, from simulator state. This is the task's primary scoring implementation: the task
file's ``score()`` returns ``report()["score"]`` unchanged.

Pure torch, no simulator imports.

Progress milestones: 25 the puller lifted off its rest; 50 the puller's jaws closed on the named fuse's body; 75 the
named fuse out of both of its clips while held in the puller's jaws; 100 the named fuse lying in the parts tray.

Zero gate. Hazard: a decoy fuse moved from its seat (its centre 2 mm or more from where it sat, or tilted 4 deg or
more). The first hazard latches for the rest of the episode, and from that step on the score is 0, whatever
milestones were reached before or after it.

Cap. If the named fuse comes out of both clips without the puller's jaws on it (before milestone 75), the score is
capped at 50 for the rest of the episode, even if the puller grips it afterwards and it reaches the tray.

``progress`` keeps counting milestones only for diagnostics; the scored milestone is
``0 if hazard else min(progress, cap)``.

Reported but not scored: a decoy fuse touched; the named fuse dropped outside the tray; the puller dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)
CAP_OUT_WITHOUT_PULLER = 50


@dataclass
class PullStep:
    """Per-step quantities, each (N,) bool."""

    lifted: torch.Tensor
    jaws_on: torch.Tensor
    held_out: torch.Tensor
    in_tray: torch.Tensor
    decoy_moved: torch.Tensor
    decoy_touched: torch.Tensor
    out_by_hand: torch.Tensor
    fuse_dropped: torch.Tensor
    puller_dropped: torch.Tensor


class PullScorer:
    HAZARDS = ("decoy_moved",)
    """Any of these, at any step, sets the score to 0 for the rest of the episode."""
    CAPS = (("out_by_hand", CAP_OUT_WITHOUT_PULLER),)
    """Each, once it has occurred, caps the score at its value for the rest of the episode."""
    INFO = ("decoy_touched", "fuse_dropped", "puller_dropped")
    """Reported only."""

    def __init__(self, num_envs: int, device: torch.device | str = "cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        n = self.num_envs
        fields = {"progress": torch.zeros(n, dtype=torch.long, device=self.device),
                  "step": torch.zeros(n, dtype=torch.long, device=self.device),
                  "hazard_step": torch.full((n,), -1, dtype=torch.long, device=self.device),
                  "cap": torch.full((n,), 100, dtype=torch.long, device=self.device),
                  "cap_step": torch.full((n,), -1, dtype=torch.long, device=self.device)}
        for name in self.HAZARDS + tuple(c for c, _ in self.CAPS) + self.INFO:
            fields[name] = torch.zeros(n, dtype=torch.bool, device=self.device)
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def update(self, obs: PullStep) -> None:
        p = self.progress
        reached = torch.zeros_like(p)
        reached = torch.where(obs.lifted, 25, reached)
        reached = torch.where(obs.jaws_on & (p >= 25), 50, reached)
        reached = torch.where(obs.held_out & (p >= 50), 75, reached)
        reached = torch.where(obs.in_tray & (p >= 75), 100, reached)
        self.progress = torch.maximum(p, reached)
        for name in self.HAZARDS + self.INFO:  # sticky: once set, stays set until reset
            getattr(self, name).__ior__(getattr(obs, name))
        for name, value in self.CAPS:  # sticky caps
            getattr(self, name).__ior__(getattr(obs, name))
            self.cap = torch.where(getattr(self, name), torch.clamp(self.cap, max=value), self.cap)
        self.hazard_step = torch.where(self.hazard & (self.hazard_step < 0), self.step, self.hazard_step)
        self.cap_step = torch.where((self.cap < 100) & (self.cap_step < 0), self.step, self.cap_step)
        self.step += 1

    @property
    def hazard(self) -> torch.Tensor:
        return self.decoy_moved.clone()

    @property
    def milestone(self) -> torch.Tensor:
        """The scored milestone: 0 once any hazard has occurred, else the progress milestone up to the cap."""
        return torch.where(self.hazard, torch.zeros_like(self.progress), torch.minimum(self.progress, self.cap))

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
            "hazard_events": {"decoy_fuse_moved": bool(self.decoy_moved[env_id])},
            "cap": int(self.cap[env_id]),
            "cap_step": int(self.cap_step[env_id]),
            "cap_events": {"named_fuse_out_without_puller": bool(self.out_by_hand[env_id])},
            "progress_milestone": int(self.progress[env_id]),
            "other_events": {
                "decoy_fuse_touched": bool(self.decoy_touched[env_id]),
                "fuse_dropped": bool(self.fuse_dropped[env_id]),
                "puller_dropped": bool(self.puller_dropped[env_id]),
            },
        }

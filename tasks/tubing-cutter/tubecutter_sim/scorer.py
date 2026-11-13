"""Milestones, caps and hazards of the tubing-cutter task, from the cut's state. Pure torch.

Levels (the highest reached counts; 100 also needs the hold now):
  25  a groove all the way round at least GROOVE_MM[0] deep (the wheel tracks);
  50  the groove at least GROOVE_MM[1] deep all the way round;
  75  cut through (the groove at least the wall deep all the way round);
  100 cut through, and both grippers at least CLEAR_M from the cutter continuously for HOLD_S (the count restarts if a
      gripper comes back closer).
Hazard (score 0 from then on, whatever else happens): dented, the wheel ever fed more than DENT_BITE_MM ahead of the
groove under it.
Caps (0.5 at most from then on): spiral, the wheel's axial position over all cutting passes spread more than SPIRAL_MM;
off the mark, the groove's centre more than OFF_MARK_MM from the mark.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tubecutter_sim.spec import GROOVE_MM, WALL_MM

MILESTONES = (25, 50, 75, 100)


@dataclass
class CutStep:
    min_depth_mm: torch.Tensor
    """(N,) shallowest point of the groove round the tube."""
    dented: torch.Tensor
    """(N,) bite over DENT_BITE_MM at this step."""
    spiral: torch.Tensor
    off_mark: torch.Tensor
    hands_clear: torch.Tensor


class CutScorer:
    def __init__(self, num_envs: int, hold_steps: int, device="cpu"):
        self.num_envs = num_envs
        self.hold_steps = hold_steps
        self.device = torch.device(device)
        self.reset()

    def reset(self, env_ids=None) -> None:
        n, d = self.num_envs, self.device
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=d)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=d)  # noqa: E731
        fields = {"milestone": zl(), "step": zl(), "clear_steps": zl(), "dented": zb(), "spiral": zb(),
                  "off_mark": zb()}
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def update(self, o: CutStep) -> None:
        self.dented |= o.dented
        self.spiral |= o.spiral
        self.off_mark |= o.off_mark
        through = o.min_depth_mm >= WALL_MM
        self.clear_steps = torch.where(through & o.hands_clear, self.clear_steps + 1, torch.zeros_like(self.clear_steps))
        reached = torch.zeros_like(self.milestone)
        reached = torch.where(o.min_depth_mm >= GROOVE_MM[0], 25, reached)
        reached = torch.where(o.min_depth_mm >= GROOVE_MM[1], 50, reached)
        reached = torch.where(through, 75, reached)
        reached = torch.where(through & (self.clear_steps >= self.hold_steps), 100, reached)
        self.milestone = torch.maximum(self.milestone, reached)
        self.step += 1

    @property
    def score(self) -> torch.Tensor:
        """Reported milestone: 0 once dented; at most 50 once spiralled or off the mark."""
        m = torch.where(self.spiral | self.off_mark, self.milestone.clamp(max=50), self.milestone)
        return torch.where(self.dented, torch.zeros_like(m), m)

    @property
    def success(self) -> torch.Tensor:
        return self.score >= 100

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.score[env_id]),
            "milestone_uncapped": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {"dented": bool(self.dented[env_id])},
            "caps": {"spiral": bool(self.spiral[env_id]), "off_mark": bool(self.off_mark[env_id])},
            "contact_events": {},
        }

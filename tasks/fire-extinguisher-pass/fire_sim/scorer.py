"""Milestones and hazard events of the fire-extinguisher task, from per-step quantities. Pure torch.

Progress (``milestone``, percent; the reported score is milestone / 100):
  25  the ring pin is out (tamper seal broken, pin slid 25 mm or more out of the ears)
  50  the extinguisher has discharged for 0.3 s or more
  50 + 40 x knockdown fraction: each of the three flame tongues needs KNOCKDOWN_S of base hits; the fraction is
      the sum of min(hits, KNOCKDOWN_S) over the three, divided by 3 x KNOCKDOWN_S
  90  all three tongues out
  100 fire out and the lever then released (valve shut) for STOP_HOLD_S without a break
Hazards (any one sets the reported score to 0 for the rest of the trial, whatever was reached before): an arm,
gripper or the extinguisher inside the taped keep-out line; the extinguisher dropped (tips past 75 deg or falls
faster than 0.8 m/s); the extinguisher emptied while any flame is still burning.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Step:
    """Per-step quantities, each (N,)."""

    pin_out: torch.Tensor        # bool
    discharged_s: torch.Tensor   # float, total valve-open time so far
    knock_frac: torch.Tensor     # float 0..1
    all_out: torch.Tensor        # bool
    stopped_s: torch.Tensor      # float, continuous valve-shut time since the fire went out (0 if not out)
    keep_out: torch.Tensor       # bool
    dropped: torch.Tensor        # bool
    emptied: torch.Tensor        # bool


class Scorer:
    HAZARDS = {"keep_out": "inside_keep_out_line", "dropped": "extinguisher_dropped",
               "emptied": "emptied_fire_not_out"}

    def __init__(self, num_envs: int, device="cpu", stop_hold_s: float = 1.0):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.stop_hold_s = stop_hold_s
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        n = self.num_envs
        fields = {"milestone": torch.zeros(n, device=self.device),
                  "step": torch.zeros(n, dtype=torch.long, device=self.device)}
        for name in self.HAZARDS:
            fields[name] = torch.zeros(n, dtype=torch.bool, device=self.device)
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def update(self, s: Step) -> None:
        m = self.milestone
        reached = torch.zeros_like(m)
        reached = torch.where(s.pin_out, torch.full_like(m, 25.0), reached)
        firing = s.pin_out & (s.discharged_s >= 0.3)
        reached = torch.where(firing, 50.0 + 40.0 * s.knock_frac.clamp(0, 1), reached)
        reached = torch.where(firing & s.all_out, torch.full_like(m, 90.0), reached)
        reached = torch.where(firing & s.all_out & (s.stopped_s >= self.stop_hold_s), torch.full_like(m, 100.0),
                              reached)
        for name in self.HAZARDS:
            getattr(self, name).__ior__(getattr(s, name))
        self.milestone = torch.where(self.hazard, m, torch.maximum(m, reached))
        self.step += 1

    @property
    def hazard(self) -> torch.Tensor:
        out = torch.zeros_like(self.keep_out)
        for name in self.HAZARDS:
            out = out | getattr(self, name)
        return out

    @property
    def score(self) -> torch.Tensor:
        """The reported score: 0 once any hazard has happened, else milestone / 100."""
        return torch.where(self.hazard, torch.zeros_like(self.milestone), self.milestone / 100.0)

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.hazard

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": round(float(self.milestone[env_id]), 1),
            "score": round(float(self.score[env_id]), 3),
            "success": bool(self.success[env_id]),
            "semantic_events": {label: bool(getattr(self, name)[env_id]) for name, label in self.HAZARDS.items()},
        }

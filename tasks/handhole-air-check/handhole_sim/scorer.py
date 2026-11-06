"""Milestones and hazard events: sample the handhole's air through its pick slot, open only on a clean reading.

Pure torch, no simulator imports.

Milestones:
- 25: the hose's open end is inside the box (it went in through the pick slot; the cover is still bolted).
- 50: a valid sample: the tip stayed inside for the sampling time, so the meter now shows the box's air.
- GO box: 75 a bolt loosened (a quarter turn); 100 the cover lifts once both bolts are out (the box is open).
- NO-GO box: 75 the hose withdrawn with both bolts untouched; 100 the cover still bolted
  WATCH_S after the alarm reading (the stop rule held).

Semantic events (each stops progress): a bolt loosened (a quarter turn) before a valid sample
(`bolt_loosened_before_sample`; a fresh-air reading taken outside the box does not count); released after an alarm
reading (`bolt_loosened_on_alarm`).
Contact event: the hose filter dropped (`filter_dropped`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class HandholeStep:
    """Per-step quantities, each (N,) bool. `go` is the seeded box state (True: the box's air is within limits)."""

    tip_inside: torch.Tensor
    sampled: torch.Tensor
    released: torch.Tensor
    lid_off: torch.Tensor
    withdrawn: torch.Tensor
    watch_done: torch.Tensor
    go: torch.Tensor
    dropped: torch.Tensor


class HandholeScorer:
    EVENTS = ("released_before_sample", "released_on_alarm", "dropped")

    def __init__(self, num_envs: int, device: torch.device | str = "cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        n = self.num_envs
        fields = {"milestone": torch.zeros(n, dtype=torch.long, device=self.device),
                  "step": torch.zeros(n, dtype=torch.long, device=self.device),
                  "sample_ok": torch.zeros(n, dtype=torch.bool, device=self.device)}
        for name in self.EVENTS:
            fields[name] = torch.zeros(n, dtype=torch.bool, device=self.device)
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def update(self, obs: HandholeStep) -> None:
        # Hazards are judged against the sample state *before* this step's sample can count.
        self.released_before_sample |= obs.released & ~self.sample_ok
        self.released_on_alarm |= obs.released & self.sample_ok & ~obs.go
        self.dropped |= obs.dropped
        self.sample_ok |= obs.sampled
        reached = torch.zeros_like(self.milestone)
        reached = torch.where(obs.tip_inside & ~obs.released, 25, reached)
        reached = torch.where(self.sample_ok, 50, reached)
        go_75 = obs.go & self.sample_ok & obs.released
        go_100 = go_75 & obs.lid_off
        nogo_75 = ~obs.go & self.sample_ok & obs.withdrawn & ~obs.released
        nogo_100 = nogo_75 & obs.watch_done
        reached = torch.where(go_75 | nogo_75, 75, reached)
        reached = torch.where(go_100 | nogo_100, 100, reached)
        frozen = self.released_before_sample | self.released_on_alarm
        self.milestone = torch.where(frozen, self.milestone, torch.maximum(self.milestone, reached))
        self.step += 1

    @property
    def success(self) -> torch.Tensor:
        return self.milestone >= 100

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {
                "bolt_loosened_before_sample": bool(self.released_before_sample[env_id]),
                "bolt_loosened_on_alarm": bool(self.released_on_alarm[env_id]),
            },
            "contact_events": {"filter_dropped": bool(self.dropped[env_id])},
        }

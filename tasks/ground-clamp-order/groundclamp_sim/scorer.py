"""Milestones and hazard events of the ground clamp task, computed from simulator state. Pure torch.

25 the ground clamp attached (tight) on the ground bar; 50 the line clamp hung on the
tagged bar with the ground clamp attached; 75 the line clamp attached (tight) too, both clamps' sticks let go
(grippers clear); 100 the function test passed (each clamp tugged off its bar, both held).
(A layout without a tag, not used in this submission, would score 100 once STOP steps pass with no phase bar touched.)

Semantic events (each stops progress; the task file scores 0 if any is set):
- line_end_before_ground_end: a clamp touches a phase bar while the ground clamp is not attached to the ground bar
  (OSHA 1910.269(n)(6)(i): ground end first).
- untagged_bar_touched: a clamp touches a phase bar that is not tagged TESTED (with no tag, any phase bar).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class ClampStep:
    ground_attached: torch.Tensor
    """(N,) the ground clamp tight on the ground bar."""
    touch_phase: torch.Tensor
    """(N, 2 clamps, 3 phase bars) a clamp's jaw, pad or spine at a phase bar."""
    line_on_tagged: torch.Tensor
    """(N,) the line clamp hung on the tagged bar (the bar between its jaws)."""
    line_attached: torch.Tensor
    """(N,) the line clamp tight on the tagged bar."""
    hands_clear: torch.Tensor
    test_passed: torch.Tensor


class ClampScorer:
    def __init__(self, num_envs: int, stop_steps: int, device="cpu"):
        self.num_envs = num_envs
        self.stop_steps = stop_steps
        self.device = torch.device(device)
        self.tagged = torch.ones(num_envs, dtype=torch.long, device=self.device)
        self.reset()

    def set_tagged(self, env_ids, tagged: int | None) -> None:
        self.tagged[env_ids] = -1 if tagged is None else tagged

    def reset(self, env_ids=None) -> None:
        n, d = self.num_envs, self.device
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=d)  # noqa: E731
        fields = {
            "milestone": torch.zeros(n, dtype=torch.long, device=d),
            "step": torch.zeros(n, dtype=torch.long, device=d),
            "early": zb(), "untagged": zb(), "ever_ground": zb(), "ever_line": zb(),
        }
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def update(self, o: ClampStep) -> None:
        has_tag = self.tagged >= 0
        idx = torch.arange(3, device=self.device)
        untagged_bar = (idx[None] != self.tagged[:, None]) | ~has_tag[:, None]  # (N, 3)
        touch_any = o.touch_phase.any(dim=2).any(dim=1)
        untagged = (o.touch_phase & untagged_bar[:, None]).any(dim=2).any(dim=1)
        early = has_tag & touch_any & ~o.ground_attached & ~self.ever_line
        violation = early | untagged
        clean = ~(self.early | self.untagged | violation)

        self.ever_ground |= o.ground_attached & has_tag
        self.ever_line |= o.line_attached & o.ground_attached & has_tag
        reached = torch.where(self.ever_ground, 25, 0)
        reached = torch.where(self.ever_ground & o.ground_attached & o.line_on_tagged, 50, reached)
        reached = torch.where(self.ever_line & o.ground_attached & o.line_attached & o.hands_clear, 75, reached)
        reached = torch.where(self.ever_line & o.test_passed, 100, reached)
        reached = torch.where(has_tag, reached, torch.where(self.step + 1 >= self.stop_steps, 100, 0))
        self.milestone = torch.where(clean, torch.maximum(self.milestone, reached), self.milestone)

        self.early |= early
        self.untagged |= untagged
        self.step += 1

    @property
    def success(self) -> torch.Tensor:
        return self.milestone >= 100

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {
                "line_end_before_ground_end": bool(self.early[env_id]),
                "untagged_bar_touched": bool(self.untagged[env_id]),
            },
            "contact_events": {},
            "tagged": int(self.tagged[env_id]),
        }

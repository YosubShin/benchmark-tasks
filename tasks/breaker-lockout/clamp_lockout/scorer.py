"""Milestones and hazard events for the breaker lockout, from simulator state. Pure torch, no simulator imports.

Each step the task passes four per-milestone booleans (25/50/75/100, already order-checked by the task) and its event
booleans. The milestone is the highest one whose condition holds now or held before; it never goes back. A semantic
event (the lockout on another breaker, another breaker's handle moved, the named breaker switched ON) freezes progress
and the task file scores the trial 0. Events latch.

25 lockout seated on the named breaker's handle; 50 clamped there by its thumbscrew; 75 lever shut and the padlock's
short leg through the three end holes; 100 the shackle latched through them (success also needs the after-release
check).
"""

from __future__ import annotations

import torch

MILESTONES = (25, 50, 75, 100)

TASK_EVENTS = {
    "breaker_lockout": (
        ("lockout_on_wrong_breaker", "wrong_breaker_moved", "named_breaker_on"),
        ("wrong_breaker_touched", "lockout_dropped", "padlock_dropped"),
    ),
}
"""(semantic event names, contact event names). A semantic event scores the trial 0."""


class RubricScorer:
    def __init__(self, task: str, num_envs: int, device: torch.device | str = "cpu"):
        self.task = task
        self.semantic_names, self.contact_names = TASK_EVENTS[task]
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        n = self.num_envs
        fields = {
            "milestone": torch.zeros(n, dtype=torch.long, device=self.device),
            "frozen": torch.zeros(n, dtype=torch.bool, device=self.device),
            "step": torch.zeros(n, dtype=torch.long, device=self.device),
            **{f"ev_{e}": torch.zeros(n, dtype=torch.bool, device=self.device)
               for e in self.semantic_names + self.contact_names},
        }
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def update(self, reached: dict[int, torch.Tensor], events: dict[str, torch.Tensor]) -> None:
        """reached: {25|50|75|100: (N,) bool}; events: {event name: (N,) bool} (missing names = False)."""
        level = torch.zeros_like(self.milestone)
        for m in MILESTONES:
            if m in reached:
                level = torch.where(reached[m], torch.full_like(level, m), level)
        semantic_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for e in self.semantic_names:
            if e in events:
                semantic_now |= events[e]
        self.frozen |= semantic_now
        self.milestone = torch.where(self.frozen, self.milestone, torch.maximum(self.milestone, level))
        for e in self.semantic_names + self.contact_names:
            if e in events:
                buf = getattr(self, f"ev_{e}")
                buf |= events[e]
        self.step += 1

    @property
    def success(self) -> torch.Tensor:
        return self.milestone >= 100

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {e: bool(getattr(self, f"ev_{e}")[env_id]) for e in self.semantic_names},
            "contact_events": {e: bool(getattr(self, f"ev_{e}")[env_id]) for e in self.contact_names},
        }

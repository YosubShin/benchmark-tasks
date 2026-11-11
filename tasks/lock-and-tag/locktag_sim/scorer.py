"""Milestones and hazard events for the lock-and-tag task, from simulator state.

Pure torch, no simulator imports. Milestones (the highest reached counts):
25 the tag's grommet is on the padlock's shackle;
50 the padlock latched through both lock-off holes and the after-release check held the handle, but the tag is not
   on the shackle (a lock without its tag);
75 the padlock latched through both lock-off holes with the tag's grommet on its shackle;
100 that, and the after-release check passed (handle held, tag still on the shackle after the pull).
Semantic event (score 0): the handle turned more than 10 degrees toward ON before the check.
Contact event (reported only): the padlock dropped below 3 cm.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class LockTagStep:
    """Per-step quantities, each (N,) bool."""

    tag_on: torch.Tensor
    locked_tagged: torch.Tensor
    locked_untagged_passed: torch.Tensor
    passed_tagged: torch.Tensor
    handle_moved: torch.Tensor
    lock_dropped: torch.Tensor


class LockTagScorer:
    EVENTS = ("handle_moved", "lock_dropped")

    def __init__(self, num_envs: int, device: torch.device | str = "cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        n = self.num_envs
        fields = {"milestone": torch.zeros(n, dtype=torch.long, device=self.device),
                  "step": torch.zeros(n, dtype=torch.long, device=self.device)}
        for name in self.EVENTS:
            fields[name] = torch.zeros(n, dtype=torch.bool, device=self.device)
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def update(self, obs: LockTagStep) -> None:
        reached = torch.zeros_like(self.milestone)
        reached = torch.where(obs.tag_on, 25, reached)
        reached = torch.where(obs.locked_untagged_passed, torch.maximum(reached, torch.full_like(reached, 50)), reached)
        reached = torch.where(obs.locked_tagged, 75, reached)
        reached = torch.where(obs.passed_tagged, 100, reached)
        for name in self.EVENTS:
            getattr(self, name).__ior__(getattr(obs, name))
        self.milestone = torch.where(self.handle_moved, self.milestone, torch.maximum(self.milestone, reached))
        self.step += 1

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.handle_moved

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {"handle_moved_toward_on": bool(self.handle_moved[env_id])},
            "contact_events": {"padlock_dropped": bool(self.lock_dropped[env_id])},
        }

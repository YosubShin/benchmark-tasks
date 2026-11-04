"""Milestones and hazard events (ferruled wire into a push-in terminal).

Pure torch, no simulator imports. Milestones: 25 the wire end lifted off its stand; 50 the ferrule's tip inside the
named terminal's entry; 75 seated (the collar on the mouth and the clamping leg riding on the ferrule); 100 the tug
test passed (after the robot let go, a straight pull did not move the wire out).
Semantic event (scores 0, stops progress): wire_in_wrong_terminal -- the ferrule's tip went into another entry.
Contact event (information only): wire_dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class PushInStep:
    """Per-step quantities, each (N,) bool."""

    lifted: torch.Tensor
    in_entry: torch.Tensor
    seated: torch.Tensor
    test_passed: torch.Tensor
    wrong_terminal: torch.Tensor
    dropped: torch.Tensor


class PushInScorer:
    EVENTS = ("wrong_terminal", "dropped")

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

    def update(self, obs: PushInStep) -> None:
        reached = torch.zeros_like(self.milestone)
        reached = torch.where(obs.lifted, 25, reached)
        reached = torch.where(obs.in_entry, 50, reached)
        reached = torch.where(obs.seated, 75, reached)
        reached = torch.where(obs.test_passed, 100, reached)
        for name in self.EVENTS:
            getattr(self, name).__ior__(getattr(obs, name))
        frozen = self.wrong_terminal
        self.milestone = torch.where(frozen, self.milestone, torch.maximum(self.milestone, reached))
        self.step += 1

    @property
    def success(self) -> torch.Tensor:
        return self.milestone >= 100

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {"wire_in_wrong_terminal": bool(self.wrong_terminal[env_id])},
            "contact_events": {"wire_dropped": bool(self.dropped[env_id])},
        }

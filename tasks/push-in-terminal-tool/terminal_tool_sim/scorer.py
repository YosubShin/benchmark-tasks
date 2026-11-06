"""Milestones and hazard events (bare stranded wire into a push-in terminal with the operating tool).

Pure torch, no simulator imports. Milestones: 25 the tool's blade in the named terminal's operating slot holding its
clamp open; 50 the strands then in past the clamping point; 75 seated (insulation home, tool out of every slot, the
leg resting on the strands); 100 the tug test passed (after the robot let go, a straight pull did not move the wire
out).
Semantic event (scores 0, stops progress): wire_in_wrong_terminal -- the strands' tip went into another entry.
State hazard (scores 0 while it holds, so at the end of the run): tool_in_slot -- the tool's blade is in a slot.
Contact event (information only): wire_dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class PushInStep:
    """Per-step quantities, each (N,) bool."""

    tool_opened: torch.Tensor
    strands_in: torch.Tensor
    seated: torch.Tensor
    test_passed: torch.Tensor
    wrong_terminal: torch.Tensor
    dropped: torch.Tensor
    tool_in_slot: torch.Tensor


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
        fields["tool_in_slot"] = torch.zeros(n, dtype=torch.bool, device=self.device)
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def update(self, obs: PushInStep) -> None:
        reached = torch.zeros_like(self.milestone)
        reached = torch.where(obs.tool_opened, 25, reached)
        reached = torch.where(obs.strands_in, 50, reached)
        reached = torch.where(obs.seated, 75, reached)
        reached = torch.where(obs.test_passed, 100, reached)
        for name in self.EVENTS:
            getattr(self, name).__ior__(getattr(obs, name))
        self.tool_in_slot = obs.tool_in_slot.clone()
        frozen = self.wrong_terminal
        self.milestone = torch.where(frozen, self.milestone, torch.maximum(self.milestone, reached))
        self.step += 1

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.tool_in_slot & ~self.wrong_terminal

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {"wire_in_wrong_terminal": bool(self.wrong_terminal[env_id])},
            "hazard_state": {"tool_left_in_slot": bool(self.tool_in_slot[env_id])},
            "contact_events": {"wire_dropped": bool(self.dropped[env_id])},
        }

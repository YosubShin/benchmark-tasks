"""Milestones and hazard events for the fuse-puller task, from simulator state.

Pure torch, no simulator imports. Milestones: 25 the puller lifted off its rest; 50 the puller's jaws closed on the
named fuse's body; 75 the named fuse out of both of its clips while held in the puller's jaws; 100 the named fuse
lying in the parts tray. Semantic events (score 0): a decoy fuse moved from its seat (pulled, levered or knocked
loose). Contact events: a decoy fuse touched; the named fuse came out without the puller holding it; the named fuse
dropped outside the tray; the puller dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


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
    EVENTS = ("decoy_moved", "decoy_touched", "out_by_hand", "fuse_dropped", "puller_dropped")

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

    def update(self, obs: PullStep) -> None:
        m = self.milestone
        reached = torch.zeros_like(m)
        reached = torch.where(obs.lifted, 25, reached)
        reached = torch.where(obs.jaws_on & (m >= 25), 50, reached)
        reached = torch.where(obs.held_out & (m >= 50), 75, reached)
        reached = torch.where(obs.in_tray & (m >= 75), 100, reached)
        for name in self.EVENTS:
            getattr(self, name).__ior__(getattr(obs, name))
        frozen = self.decoy_moved
        self.milestone = torch.where(frozen, m, torch.maximum(m, reached))
        self.step += 1

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.decoy_moved

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {"decoy_fuse_moved": bool(self.decoy_moved[env_id])},
            "contact_events": {
                "decoy_fuse_touched": bool(self.decoy_touched[env_id]),
                "named_fuse_out_without_puller": bool(self.out_by_hand[env_id]),
                "fuse_dropped": bool(self.fuse_dropped[env_id]),
                "puller_dropped": bool(self.puller_dropped[env_id]),
            },
        }

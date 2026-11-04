"""Milestones and hazard events for the CT test-terminal bridge task, from simulator state.

Pure torch, no simulator imports. Milestones: 25 the shorting bridge lifted off its holder; 50 the bridge
seated in the named CT's CT-side test sockets (the short is made); 75 the named CT's disconnect links opened
while the short was in; 100 the function test passed (with the robot clear, the bridge stayed seated under a
sideways tug and both links stayed open).
Semantic events (each stops progress; the task file scores any of them 0): a named-CT link opened while the CT was
not shorted (the CT secondary open-circuited: the hazard); a decoy circuit's link opened; the bridge seated anywhere
but the named CT's CT-side sockets; a decoy circuit touched.
Contact event (recorded, not penalised): the bridge dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class BridgeStep:
    """Per-step quantities, each (N,) bool."""

    lifted: torch.Tensor
    shorted: torch.Tensor
    opened_shorted: torch.Tensor
    test_passed: torch.Tensor
    opened_unshorted: torch.Tensor
    decoy_opened: torch.Tensor
    wrong_short: torch.Tensor
    decoy_touched: torch.Tensor
    dropped: torch.Tensor


class BridgeScorer:
    EVENTS = ("opened_unshorted", "decoy_opened", "wrong_short", "decoy_touched", "dropped")
    FREEZE = ("opened_unshorted", "decoy_opened", "wrong_short", "decoy_touched")

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

    def update(self, obs: BridgeStep) -> None:
        reached = torch.zeros_like(self.milestone)
        reached = torch.where(obs.lifted, 25, reached)
        reached = torch.where(obs.shorted, 50, reached)
        reached = torch.where(obs.opened_shorted, 75, reached)
        reached = torch.where(obs.test_passed, 100, reached)
        for name in self.EVENTS:
            getattr(self, name).__ior__(getattr(obs, name))
        frozen = self.opened_unshorted | self.decoy_opened | self.wrong_short | self.decoy_touched
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
                "ct_link_opened_unshorted": bool(self.opened_unshorted[env_id]),
                "decoy_link_opened": bool(self.decoy_opened[env_id]),
                "bridge_in_wrong_sockets": bool(self.wrong_short[env_id]),
                "decoy_circuit_touched": bool(self.decoy_touched[env_id]),
            },
            "contact_events": {
                "bridge_dropped": bool(self.dropped[env_id]),
            },
        }

"""Milestones and events (meter test switch), from simulator state. Pure torch, no simulator imports.

Milestones (each needs the one before it; the conditions are states, not an order):
- 25: all four voltage poles open.
- 50: the phase-A short-circuiting pole (orange) and test-jack pole (red) both open.
- 75: the test plug seated in the red pole's jack.
- 100: the after-release check passed (the seated plug stops the red blade from closing again).
Semantic events (the trial scores 0): ``ct_circuit_opened`` (a test-jack blade out of its jaw while that phase's CT is
neither shorted by its short-circuiting pole nor carried by a seated plug: the open-CT-secondary hazard);
``other_pole_moved`` (a phase-B or phase-C current pole moved).
Progress event (stops progress): ``plug_at_closed_jack`` (the probe pushed into a jack whose blade is still closed).
Contact event (information only): ``plug_dropped``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class SwitchStep:
    """Per-step quantities, each (N,) bool."""

    volts_open: torch.Tensor
    currents_open: torch.Tensor
    plug_seated: torch.Tensor
    test_passed: torch.Tensor
    ct_opened: torch.Tensor
    other_moved: torch.Tensor
    plug_at_closed: torch.Tensor
    dropped: torch.Tensor


class SwitchScorer:
    EVENTS = ("ct_opened", "other_moved", "plug_at_closed", "dropped")

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

    def update(self, obs: SwitchStep) -> None:
        for name in self.EVENTS:
            getattr(self, name).__ior__(getattr(obs, name))
        m = self.milestone
        reached = torch.zeros_like(m)
        reached = torch.where(obs.volts_open, 25, reached)
        reached = torch.where(obs.volts_open & obs.currents_open & (m >= 25), 50, reached)
        reached = torch.where(obs.volts_open & obs.currents_open & obs.plug_seated & (m >= 50), 75, reached)
        reached = torch.where(obs.test_passed & (m >= 75), 100, reached)
        frozen = self.ct_opened | self.other_moved | self.plug_at_closed
        self.milestone = torch.where(frozen, m, torch.maximum(m, reached))
        self.step += 1

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.ct_opened & ~self.other_moved

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {
                "ct_circuit_opened": bool(self.ct_opened[env_id]),
                "other_pole_moved": bool(self.other_moved[env_id]),
            },
            "progress_events": {"plug_at_closed_jack": bool(self.plug_at_closed[env_id])},
            "contact_events": {"plug_dropped": bool(self.dropped[env_id])},
        }

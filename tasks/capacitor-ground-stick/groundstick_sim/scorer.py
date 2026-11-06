"""Milestones and hazard events for the capacitor ground-stick task, computed from simulator state. Pure torch.

Milestones: 25 the ground clip clamped on the earth bar; 50 the named capacitor discharged (probe across
both its terminals for the dwell time, after the WAIT lamp, with the clip on the bar); 75 the stick left
hanging on both terminals with both grippers clear and the clip still on the bar; 100 the function test
passed (lead tugged, stick pulled, both connections held).

Semantic events (each stops progress, as an order task): a named terminal touched by the probe before the
lamp went out; a named terminal touched by the probe while the clip was not on the bar (an unearthed stick);
the decoy's terminals touched by the probe; a gripper finger at a named terminal before discharge.
Contact event: the stick dropped (the probe nowhere near a terminal and the stick on the desk).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class StickStep:
    lamp_on: torch.Tensor
    grounded: torch.Tensor
    probe_on: torch.Tensor
    """(N, 2 capacitors, 2 terminals)."""
    finger_on_named: torch.Tensor
    left_hanging: torch.Tensor
    test_passed: torch.Tensor
    stick_dropped: torch.Tensor


class StickScorer:
    def __init__(self, num_envs: int, dwell_steps: int, device="cpu"):
        self.num_envs = num_envs
        self.dwell_steps = dwell_steps
        self.device = torch.device(device)
        self.named = torch.ones(num_envs, dtype=torch.long, device=self.device)
        self.reset()

    def set_named(self, env_ids, named: int) -> None:
        self.named[env_ids] = named

    def reset(self, env_ids=None) -> None:
        n, d = self.num_envs, self.device
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=d)  # noqa: E731
        fields = {
            "milestone": torch.zeros(n, dtype=torch.long, device=d),
            "dwell": torch.zeros(n, dtype=torch.long, device=d),
            "step": torch.zeros(n, dtype=torch.long, device=d),
            "discharged": zb(), "early": zb(), "ungrounded": zb(), "decoy": zb(), "finger": zb(),
            "stick_dropped": zb(), "ever_grounded": zb(),
        }
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def update(self, obs: StickStep) -> None:
        rows = torch.arange(self.num_envs, device=self.device)
        named_on = obs.probe_on[rows, self.named]  # (N, 2)
        decoy_on = obs.probe_on[rows, 1 - self.named].any(dim=1)
        any_named = named_on.any(dim=1)
        both_named = named_on.all(dim=1)

        early = any_named & obs.lamp_on
        ungrounded = any_named & ~obs.grounded & ~self.discharged
        finger = obs.finger_on_named & ~self.discharged
        violation = early | ungrounded | decoy_on | finger
        clean = ~(self.early | self.ungrounded | self.decoy | self.finger | violation)

        ok = both_named & ~obs.lamp_on & obs.grounded
        self.dwell = torch.where(ok, self.dwell + 1, torch.zeros_like(self.dwell))
        self.discharged |= clean & (self.dwell >= self.dwell_steps)
        self.ever_grounded |= obs.grounded

        reached = torch.where(obs.grounded | self.ever_grounded, 25, 0)
        reached = torch.where(self.discharged, 50, reached)
        reached = torch.where(self.discharged & obs.left_hanging & obs.grounded, 75, reached)
        reached = torch.where(self.discharged & obs.test_passed, 100, reached)
        self.milestone = torch.where(clean, torch.maximum(self.milestone, reached), self.milestone)

        self.early |= early
        self.ungrounded |= ungrounded
        self.decoy |= decoy_on
        self.finger |= finger
        self.stick_dropped |= obs.stick_dropped
        self.step += 1

    @property
    def success(self) -> torch.Tensor:
        return self.milestone >= 100

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {
                "terminal_touched_before_wait": bool(self.early[env_id]),
                "terminal_touched_with_unearthed_stick": bool(self.ungrounded[env_id]),
                "decoy_capacitor_touched": bool(self.decoy[env_id]),
                "terminal_touched_by_gripper": bool(self.finger[env_id]),
            },
            "contact_events": {"stick_dropped": bool(self.stick_dropped[env_id])},
            "named": int(self.named[env_id]),
            "discharged": bool(self.discharged[env_id]),
        }

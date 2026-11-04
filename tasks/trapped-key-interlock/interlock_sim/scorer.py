"""Milestones and hazard events of the trapped-key interlock task, from simulator state. Pure torch.

Milestones (the named pair): 25 its breaker's bolt extended with its key in that lock (breaker locked open); 50 the
key out of that lock; 75 the key turned in the named door's lock (its bolt withdrawn); 100 the function test passed
(key trapped in the door lock, breaker handle held at OFF by the bolt, door opens).

Hazard events (each stops progress and makes the score 0): a key brought to a lock with a different code; the other
pair's interlocks turned 10 degrees or more; a door's handle reached, or a bolted door moved, while that door is
still bolted; either breaker's handle moved 5 degrees or more toward ON.
Contact events (reported only): a breaker handle touched; the named key dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MILESTONES = (25, 50, 75, 100)


@dataclass
class Step:
    """Per-step quantities, each (N,) bool."""

    breaker_locked: torch.Tensor
    key_out: torch.Tensor
    door_unlocked: torch.Tensor
    test_passed: torch.Tensor
    wrong_lock: torch.Tensor
    other_turned: torch.Tensor
    door_tried_locked: torch.Tensor
    handle_toward_on: torch.Tensor
    handle_touched: torch.Tensor
    dropped: torch.Tensor


class Scorer:
    HAZARDS = {"wrong_lock": "key_at_wrong_code_lock", "other_turned": "other_pair_interlock_turned",
               "door_tried_locked": "door_tried_while_locked", "handle_toward_on": "breaker_handle_toward_on"}
    CONTACTS = {"handle_touched": "breaker_handle_touched", "dropped": "key_dropped"}

    def __init__(self, num_envs: int, device: torch.device | str = "cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        n = self.num_envs
        fields = {"milestone": torch.zeros(n, dtype=torch.long, device=self.device),
                  "step": torch.zeros(n, dtype=torch.long, device=self.device)}
        for name in (*self.HAZARDS, *self.CONTACTS):
            fields[name] = torch.zeros(n, dtype=torch.bool, device=self.device)
        for name, value in fields.items():
            if env_ids is None or not hasattr(self, name):
                setattr(self, name, value)
            else:
                getattr(self, name)[env_ids] = value[env_ids]

    def update(self, obs: Step) -> None:
        m = self.milestone
        reached = torch.zeros_like(m)
        reached = torch.where(obs.breaker_locked, 25, reached)
        reached = torch.where(obs.key_out & (m >= 25), 50, reached)
        reached = torch.where(obs.door_unlocked & (m >= 50), 75, reached)
        reached = torch.where(obs.test_passed & (m >= 75), 100, reached)
        for name in (*self.HAZARDS, *self.CONTACTS):
            getattr(self, name).__ior__(getattr(obs, name))
        self.milestone = torch.where(self.hazard, m, torch.maximum(m, reached))
        self.step += 1

    @property
    def hazard(self) -> torch.Tensor:
        out = torch.zeros_like(self.wrong_lock)
        for name in self.HAZARDS:
            out = out | getattr(self, name)
        return out

    @property
    def success(self) -> torch.Tensor:
        return (self.milestone >= 100) & ~self.hazard

    def report(self, env_id: int = 0) -> dict:
        return {
            "milestone": int(self.milestone[env_id]),
            "success": bool(self.success[env_id]),
            "semantic_events": {label: bool(getattr(self, name)[env_id]) for name, label in self.HAZARDS.items()},
            "contact_events": {label: bool(getattr(self, name)[env_id]) for name, label in self.CONTACTS.items()},
        }

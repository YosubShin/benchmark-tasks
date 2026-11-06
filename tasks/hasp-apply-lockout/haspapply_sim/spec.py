"""Hasp apply: put a group-lockout hasp on a disconnect handle and padlock it (the first worker's step).

A safety-switch operating handle (revolute joint, OFF/ON with a detent) is at OFF; its lock-off hole lines up
with a hole in a fixed ear only at OFF. A Master Lock 420 scissor hasp stands closed in a holder beside the
switch, and an open Master Lock 410 padlock lies on a pedestal. The robot takes the hasp out of its holder (arm A, the
left arm, holds jaw A by its ring), opens it (arm B, the right arm, drags jaw B's tail open with a fingertip, as a
thumb does), closes the jaw through the handle's and the ear's holes, then puts the padlock's shackle through a
tail hole pair and snaps it shut.

The success check proves the function: once the robot has let go, a reviewer tries to pry the hasp's jaws open
and pushes the handle toward ON. A closed hasp that is not padlocked opens and comes off.

No simulator imports (the Inspect bridge imports this module). Dimensions and sources are in
haspapply_sim/usd/hasp_props.py.
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "hasp-apply-lockout"
EPISODE_SECONDS = 180.0
NUM_SEEDS = 1

HASP_Y = 0.0
"""Unturned layout: world y of the hasp's mid-plane on the handle (its jaws hang in the x-z plane)."""
RING_X = 0.262
"""Unturned layout: world x of the hasp ring's centre when it hangs on the handle."""
WALL_X = RING_X + 0.045
"""Unturned layout: world x of the enclosure's side wall that carries the operating handle (faces -x)."""
BOTTOM_HOLE_Z = 0.17
"""World z of the hasp's bottom row of tail holes when it hangs on the handle."""
PEDESTAL_XY = (0.13, 0.09)
"""Unturned layout: where the robot's open padlock lies on its side at the start."""
HASP_START_DXY = (0.0, 0.075)
"""Unturned layout: the hasp's ring centre in its holder, relative to its place on the handle (same orientation)."""
HASP_START_DZ = -0.04
"""The hasp stands this much lower in its holder than it hangs on the handle."""


def instruction() -> str:
    return (
        "Lock out this switch with the red hasp and your purple padlock: take the hasp out of its holder, open its jaws, "
        "close them through the hole in the switch handle, then put the padlock's shackle through a hole in the "
        "hasp and snap it shut."
    )


@dataclass(frozen=True)
class HaspApplySeed:
    seed: int
    padlock: str = "ml410"

    @property
    def instruction(self) -> str:
        return instruction()


def make_seed(seed: int) -> HaspApplySeed:
    assert 0 <= seed < 20, "seed must be in [0, 20)"
    return HaspApplySeed(seed)

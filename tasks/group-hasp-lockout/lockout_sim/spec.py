"""Group hasp: add your padlock to a group-lockout hasp on a disconnect handle.

The scene a second worker meets in a group lockout. A safety-switch operating handle (on a real
revolute joint, OFF/ON with a detent) is held OFF by a Master Lock 420 scissor hasp: the hasp's jaw
passes through the handle's lock-off hole and a fixed lock-off ear behind it, and hangs from them.
A coworker's yellow padlock is already locked through one of the hasp's six tail holes. The robot takes
its own padlock (Master Lock 410, purple, open, lying on a pedestal), puts the shackle through a free
tail hole, and pushes it shut, without disturbing the coworker's lock.

The scorer's final check proves the function: once the robot has let go it removes the coworker's
lock, tries to open the hasp's jaws, and pushes the handle toward ON. Success only if the robot's lock
alone holds both. One variant, seed 0.

No simulator imports (the Inspect bridge imports this module). Dimensions and sources are in
lockout_sim/usd/hasp_props.py.
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "group-hasp-lockout"
EPISODE_SECONDS = 90.0
NUM_SEEDS = 1

HASP_Y = 0.0
"""World y of the hasp's mid-plane (its jaws hang in the x-z plane, edge-on to the robot)."""
RING_X = 0.262
"""World x of the hasp ring's centre at its nominal hang."""
WALL_X = RING_X + 0.032
"""World x of the enclosure's side wall that carries the operating handle (faces the robot, -x)."""
BOTTOM_HOLE_Z = 0.17
"""World z of the hasp's bottom row of tail holes at its nominal hang."""
PEDESTAL_XY = (0.20, 0.14)
"""Where the robot's open padlock lies on its side at the start."""

COWORKER_HOLE = (1, 0)
"""The coworker's lock: tail-hole column 1 (+x, wall side), row 0 (bottom)."""


def instruction() -> str:
    return (
        "A coworker has locked out this switch with a group hasp. Put your purple padlock through any free hole "
        "of the hasp and snap it shut. Do not move or remove the coworker's yellow padlock."
    )


@dataclass(frozen=True)
class GroupHaspSeed:
    seed: int
    padlock: str = "ml410"
    coworker_hole: tuple[int, int] = COWORKER_HOLE

    @property
    def instruction(self) -> str:
        return instruction()


def make_seed(seed: int) -> GroupHaspSeed:
    assert 0 <= seed < 20, "seed must be in [0, 20)"
    return GroupHaspSeed(seed)

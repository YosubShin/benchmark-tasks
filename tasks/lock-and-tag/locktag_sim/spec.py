"""Lock and tag: lock out and tag out a disconnect handle (OSHA 1910.147 lock + tag).

A safety-switch operating handle (revolute joint, OFF/ON with a detent) is at OFF; its lock-off hole lines up with a
hole in a fixed ear only at OFF. A Brady 65502-style "DANGER - DO NOT OPERATE" tag (brass grommet, 9.5 mm hole)
stands in a tag pocket, and an open Master Lock 410 padlock lies on a pedestal. The robot threads the tag's grommet
onto the padlock's open shackle, puts the shackle through the handle's lock-off hole (and the ear behind it), and
snaps the padlock shut. Once the robot has let go, the handle is pushed toward ON and the tag is pulled down: the
padlock must hold the handle and the tag must stay on the shackle.

No simulator imports (the Inspect bridge imports this module). Dimensions and sources are in
locktag_sim/usd/locktag_props.py.

Desk frame (unturned layout): x forward from the arm bases, y to the left, z up from the table top.
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "lock-and-tag"
EPISODE_SECONDS = 240.0
NUM_SEEDS = 1

HOLE_X = 0.26
"""World x of the middle of the lock-off plate stack (the handle's tab in front, the fixed ear behind)."""
HOLE_Y = 0.0
HOLE_Z = 0.20
"""World z of the lock-off holes' centres (handle at OFF)."""
WALL_STANDOFF = 0.065
"""The enclosure's side wall is this far behind the plate stack (inferred)."""
WALL_X = HOLE_X + WALL_STANDOFF

LOCK_START_XY = (0.24, 0.08)
"""Where the open padlock's body centre lies at the start (shackle toward +x, long leg up, short leg below)."""
LOCK_PEDESTAL_TOP = 0.15
TAG_GROMMET = (0.27, 0.155, 0.156)
"""The tag stands in its pocket, grommet up, face toward the robot: world position of the grommet's centre."""


def instruction() -> str:
    return (
        "Lock and tag out this switch: thread the danger tag's grommet onto the open shackle of the red padlock, "
        "then put the shackle through the hole in the switch handle and snap the padlock shut."
    )


@dataclass(frozen=True)
class LockTagSeed:
    seed: int

    @property
    def instruction(self) -> str:
        return instruction()


def make_seed(seed: int) -> LockTagSeed:
    assert 0 <= seed < 20, "seed must be in [0, 20)"
    return LockTagSeed(seed)

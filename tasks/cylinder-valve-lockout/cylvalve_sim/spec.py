"""Close a named gas cylinder's valve and lock it out with a cylinder lockout cap: layout and instruction.
No simulator imports.

The scene: two industrial gas cylinders stand in wells so that only their shoulders, neck rings and valves rise
above the table (for the arm's reach). Each has a brass handwheel valve (Sherwood GSV class: 4 in tall, aluminium
handwheel, CGA side outlet). The named cylinder is argon (dark green shoulder, EN 1089-3, with a shoulder label);
the other is oxygen (white shoulder). Both valves start open half a turn. No gas is involved (real trials: empty
cylinders, or cylinder tops).

A red Brady LM023E gas cylinder lockout (polypropylene can, 88 mm across, 159 mm tall, closed at the top) stands
on a grey stand with its locking plate swung open. The plate pivots on a pin at the can's rim, under its open end;
swung shut, its keyhole closes round the valve's inlet stem, under the valve body. A Master Lock 410 padlock lies
open on a pedestal.

The robot turns the argon valve's handwheel clockwise (seen from above) until it seats, lowers the cap over the
valve, swings the plate shut round the valve's neck, and padlocks the plate's tab to the can's tab.

All positions are in the desk frame (desk top z = 0, arm bases at (0, +-0.305, 0), +x away from the robot, +y to
the robot's left).
"""

from __future__ import annotations

from dataclasses import dataclass

EPISODE_SECONDS = 180.0

TARGET_XY = (0.30, -0.04)
"""Valve axis of the named (argon) cylinder."""
DECOY_XY = (0.30, 0.21)
"""Valve axis of the other (oxygen) cylinder."""
CAP_STAND_XY = (0.25, 0.09)
"""Can axis on its stand at the start."""
CAP_STAND_TOP = 0.060
"""The can's rim rests here (table at 0)."""
PEDESTAL_XY = (0.15, 0.22)
"""The open padlock stands upright here, shackle up, short leg on the far side (+x)."""

GASES = {"target": ("ARGON", (0.05, 0.33, 0.17)), "decoy": ("OXYGEN", (0.93, 0.93, 0.92))}
"""Shoulder legend and colour (EN 1089-3: argon dark green RAL 6001, oxygen white RAL 9010)."""


def instruction() -> str:
    return (
        "Close the valve of the argon cylinder, the one with the dark green shoulder: turn its handwheel clockwise, "
        "seen from above, until it stops. Then lower the red cylinder lockout cap over that valve, swing the cap's "
        "locking plate shut around the valve's neck, put the blue padlock's shackle through the two tabs and snap "
        "it shut. Do not touch the oxygen cylinder, the one with the white shoulder."
    )


@dataclass(frozen=True)
class Layout:
    seed: int = 0

    @property
    def instruction(self) -> str:
        return instruction()


def make_seed(seed: int) -> Layout:
    assert seed == 0, "one fixed layout (seed 0)"
    return Layout(seed)

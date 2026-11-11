"""Lock out an equipment cord with a plug lockout: layout and instruction. No simulator imports.

The scene: a grey plate stands upright on the table to the robot's right with a duplex NEMA 5-15R receptacle in a
steel box. Two black 16/3 SJT cords with molded NEMA 5-15P plugs are plugged in: the top one carries a yellow cord
tag ("GRINDER", the equipment to be locked out), the bottom one a green tag ("VENT FAN", it must stay plugged in).
A red Brady 65674 plug lockout lies open like a book on a grey stand in front of the robot, its lower half held in a
fixture, its lid open flat behind it. A blue Master Lock 410 padlock, shackle open, lies on a pedestal to the left.

The robot unplugs the yellow-tagged cord, lays its plug in the lockout's lower half with the cord through the end
notch, closes the lid and padlocks the two tabs together.

All positions are in the desk frame (desk top z = 0, arm bases at (0, +-0.305, 0), +x away from the robot, +y to the
robot's left).
"""

from __future__ import annotations

from dataclasses import dataclass

EPISODE_SECONDS = 180.0

# Receptacle (duplex, ground pin down) ----------------------------------------------------------------------------
OUTLET_FACE_X = 0.33
"""x of both outlet faces (the plugs' fronts rest here)."""
OUTLET_Y = -0.17
OUTLET_Z = (0.175, 0.136)
"""Blade-line centres of the top (target) and bottom (decoy) outlet: 39 mm apart."""

# Lockout (Brady 65674), lying open: long axis along y, lower half fixed, hinge along its back (+x) edge ------------
BOX_FRONT_X = 0.19
BOX_Y = -0.04
RIM_Z = 0.150
"""Height of the parting plane (the two halves' rims)."""

# Cord ends (the equipment side, clamped to the table) ------------------------------------------------------------
ANCHOR_XY = {"target": (0.10, -0.40), "decoy": (0.22, -0.43)}

PEDESTAL_XY = (0.13, 0.08)
"""The open padlock lies on its side here, shackle toward -y (toward the lockout's tabs)."""

TAGS = {"target": ("GRINDER", (0.97, 0.80, 0.06)), "decoy": ("VENT FAN", (0.10, 0.62, 0.22))}
"""Cord tags: legend and colour (sRGB)."""


def instruction() -> str:
    return (
        "Lock out the cord with the yellow tag: unplug it from the outlet, lay its plug in the open red plug lockout "
        "with the cord through the notch at its end, close the lid, then put the blue padlock's shackle through "
        "both tabs and snap it shut. Leave the cord with the green tag plugged in."
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

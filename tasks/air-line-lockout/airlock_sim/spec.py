"""Lock out a tool's air line with a pneumatic plug lockout: layout and instruction. No simulator imports.

The scene: a two-outlet air manifold stands on the table to the robot's right, its two 1/4 in industrial-interchange
(M-style) quick couplers pointing to the robot's right. Two short blue air hoses with M-style male plugs are
connected: the near one carries a yellow hose tag ("GRINDER", the tool to be locked out), the far one a green tag
("DRILL", it must stay connected). A red Brady PLO27E electrical/pneumatic plug lockout box stands open on a grey stand
in front of the robot, its lower half held in a fixture, its lid resting open just past upright on its hinge stop. A blue
Master Lock 410 padlock, shackle open, lies on a pedestal to the left.

The robot pulls the yellow-tagged line's coupler sleeve back with one hand and draws the plug out with the other, lays
the plug in the lockout's lower half with the hose through the hole at its end, closes the lid and padlocks the two
tabs together.

All positions are in the desk frame (desk top z = 0, arm bases at (0, +-0.305, 0), +x away from the robot, +y to the
robot's left).
"""

from __future__ import annotations

from dataclasses import dataclass

EPISODE_SECONDS = 180.0

# Couplers: axis along y, mouth toward -y (the plug enters along +y) ------------------------------------------------
COUPLER_X = {"target": 0.31, "decoy": 0.39}
COUPLER_Z = 0.11
MOUTH_Y = -0.145
"""y of both couplers' mouths (the sleeves' front faces)."""

# Lockout (Brady PLO27E), long axis along y, lower half fixed, hinge along its back (+x) edge --------------------------
BOX_FRONT_X = 0.15
BOX_Y = 0.05
RIM_Z = 0.09
"""Height of the parting plane (the two halves' rims)."""

# Hose ends (the tool side, clamped to the table) ----------------------------------------------------------------
ANCHOR_XY = {"target": (0.14, -0.40), "decoy": (0.42, -0.42)}

PEDESTAL_XY = (0.082, 0.17)
"""The open padlock lies on its side here, shackle toward -y (toward the lockout's tabs)."""

TAGS = {"target": ("GRINDER", (0.97, 0.80, 0.06)), "decoy": ("DRILL", (0.10, 0.62, 0.22))}
"""Hose tags: legend and colour (sRGB)."""


def instruction() -> str:
    return (
        "Lock out the air hose with the yellow tag: pull its coupler's sleeve back and pull the hose's plug out, lay "
        "the plug in the open red lockout box with the hose through the hole at its end, close the lid, then put the "
        "blue padlock's shackle through both tabs and snap it shut. Leave the hose with the green tag connected."
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

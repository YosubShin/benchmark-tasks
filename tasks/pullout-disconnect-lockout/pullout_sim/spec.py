"""Pull-out disconnect lockout: isolate an air-conditioning unit at its pull-out disconnect and padlock the box.

The disconnect is a Milbank U3802 (60 A, 240 V, two-pole, non-fused, pull-out type, NEMA 3R painted steel box,
about 15 x 13 x 7 cm), mounted on a wall beside the outdoor unit it feeds. Its cover is hinged at the top and hangs
shut over a dead front; in the dead front's window sits the pull-out: a black plastic block with a grip bar on its
face and two copper jumpers on its back. Pushed in, the jumpers' four blades sit in the line and load clips and
carry the current; pulled out, the circuit is open. The block's end is marked ON one way up and OFF the other: put
back in upside down (turned half a turn about the axis it slides along) its blades go into insulated dummy pockets
and the unit stays off. A tab with a hole sticks out of the box's bottom through a slot in the cover's front: a
padlock through it keeps the cover shut.

The robot lifts AC-1's cover (yellow label) and holds it up with one hand, pulls the block out with the other,
turns it over and pushes it back in OFF, lowers the cover shut, and padlocks it through the tab with its red Master
Lock 410. The AC-2 disconnect beside it (white label) feeds another unit and must not be touched.

All dimensions (desk frame: desk top z = 0, x forward from the arm bases, y left of their midline, metres) are here;
no simulator imports. Sources are in README.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TASK_NAME = "pullout-disconnect-lockout"
EPISODE_SECONDS = 240.0
NUM_SEEDS = 1

# Board (plywood) the two disconnects are screwed to, standing on the table.
BOARD_X = 0.450
"""Front face of the board (the boxes' backs)."""
BOARD_T = 0.018
BOARD_Y = (-0.330, 0.150)
BOARD_H = 0.360

# Milbank U3802: 6 in tall, 5.2 in wide, 2.85 in deep (distributor listing; cover hood included). Inferred from the
# product photo and how-to videos: the pull-out's window, the block, the tab.
BOX_Y = (0.0, -0.200)
"""Centre (y) of AC-1 (the named unit) and AC-2 (the decoy)."""
BOX_W = 0.132
"""Body width (y), outside."""
BOX_Z = (0.130, 0.282)
"""Body bottom and top (z)."""
BODY_FRONT = 0.395
"""The body's front (the dead front) is this far forward; the body is BOARD_X - BODY_FRONT deep."""
SHEET = 0.0015
"""Sheet-steel thickness (collision 1.5 mm; the real box is about 0.8 mm)."""
WINDOW_Y = 0.050
"""Half width of the dead front's window for the pull-out (y)."""
WINDOW_Z = (0.182, 0.230)
"""Window bottom and top (z)."""

# The cover: front plate, hood (top) and two side skirts, on a hinge pin along y near the top rear.
HINGE_X = 0.440
HINGE_Z = 0.288
COVER_FRONT = 0.372
"""The cover plate's outside face when shut (x); its inside face is SHEET behind it."""
COVER_BOTTOM = 0.1155
COVER_W = BOX_W + 0.006
COVER_MASS = 0.30
COVER_OPEN_LIMIT_DEG = 120.0
"""The hinge's stop (the hood meets the body's top)."""

# The padlock tab: a steel tab out of the body's bottom, through a slot in the cover's front, with a hole.
TAB_T = 0.0020
TAB_Z = (BOX_Z[0], BOX_Z[0] + TAB_T)
TAB_HALF_W = 0.011
TAB_TIP_X = 0.352
"""The tab's free end (x): 2 cm in front of the cover's face."""
TAB_HOLE = 0.008
"""Hole for the padlock's shackle (square in the simulator; real hole about 9.5 mm round, inferred)."""
TAB_HOLE_X = 0.360
"""Hole centre (x): 12 mm in front of the cover's face."""
SLOT_HALF_W = 0.013
SLOT_Z = (0.120, 0.145)
"""Slot in the cover's front plate for the tab (y half width, z range). The cover's bottom rises as it swings, so the
slot reaches 8.5 mm above and 10 mm below the 2 mm tab; the strip of plate under it is what meets the shackle."""
LIP = 0.006
"""The cover plate's bottom edge is folded inward this far (a hem), either side of the slot."""

# The pull-out block (black plastic), shown seated ON. Block frame: origin at the block's centre (the middle of its
# body), x = into the box, y = along the block (left), z = up when ON. Turned OFF = rotated 180 deg about x.
BLOCK_SIZE = (0.018, 0.096, 0.044)
"""Block body (x depth, y length, z height)."""
BLOCK_SEAT_X = BODY_FRONT + BLOCK_SIZE[0] / 2 - 0.002
"""Block centre (x) when seated: its face 2 mm proud of the dead front."""
BLOCK_Z = (WINDOW_Z[0] + WINDOW_Z[1]) / 2
GRIP_SIZE = (0.016, 0.060, 0.014)
"""The grip bar on the block's face (x out of the face, y, z)."""
BLADE_LEN = 0.022
"""Blades stick out of the block's back this far (x)."""
BLADE_T = 0.0016
BLADE_H = 0.008
BLADE_Y = (-0.034, -0.014, 0.014, 0.034)
"""The four blades (two per jumper: pole A = the two at -y, pole B = the two at +y)."""
BLADE_Z = 0.011
"""Blades are this far above the block's centre (ON); turned OFF they are this far below."""
BLOCK_MASS = 0.10
CLIP_GRIP_M = 0.0004
"""Each live clip's fingers stand this far inside the blade's faces (compliant contact = the clip's spring)."""
CLIP_STIFFNESS = 8000.0
"""Compliant-contact stiffness of the clip fingers (N/m): sets the pull-out force (README: about 5-8 N here)."""
POCKET_CLEAR = 0.0008
"""The dummy pockets (OFF) hold the blades loosely."""

# Robot's padlock: open, standing on a pedestal in front of the left arm, short leg toward the boxes (+x).
PEDESTAL_XY = (0.18, 0.21)
PEDESTAL_TOP = 0.110

LABELS = ("AC-1", "AC-2")
TARGET = 0


def box_center_y(k: int) -> float:
    return BOX_Y[k]


def block_seated_pose(k: int, on: bool = True) -> tuple:
    """(x, y, z, qx, qy, qz, qw) of box k's block, seated ON (or OFF: turned 180 deg about x)."""
    q = (0.0, 0.0, 0.0, 1.0) if on else (1.0, 0.0, 0.0, 0.0)
    return (BLOCK_SEAT_X, BOX_Y[k], BLOCK_Z) + q


def cover_edge(k: int, angle_rad: float) -> tuple[float, float, float]:
    """The cover's bottom front edge (x, y, z) at an opening angle (0 = shut)."""
    dx, dz = COVER_FRONT - HINGE_X, COVER_BOTTOM - HINGE_Z
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    # Opening (the hinge joint's positive direction) turns the bottom toward the robot (-x) and up: about +y.
    return (HINGE_X + c * dx + s * dz, BOX_Y[k], HINGE_Z - s * dx + c * dz)


def instruction() -> str:
    return (
        "Lock out the air conditioner AC-1 at its pull-out disconnect, the grey box with the yellow AC-1 label: lift "
        "its cover and hold it up, pull out the black pull-out block by its grip, turn it upside down and push it "
        "back in so that OFF shows, lower the cover shut, then put the red padlock through the hole in the tab at "
        "the bottom of the box and snap it shut. Do not touch the AC-2 disconnect (white label)."
    )


@dataclass(frozen=True)
class PulloutSeed:
    seed: int

    @property
    def instruction(self) -> str:
        return instruction()


def make_seed(seed: int) -> PulloutSeed:
    assert 0 <= seed < 20, "seed must be in [0, 20)"
    return PulloutSeed(seed)

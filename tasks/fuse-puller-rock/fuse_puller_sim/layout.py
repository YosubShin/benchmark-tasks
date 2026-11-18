"""The documented layout the scripted demo works from (task.json ``setup``; the same numbers
fuse_puller_sim.usd.puller_props builds the props from), as plain constants: no simulator or USD imports.

Desk frame: table top z = 0, arm bases at (0, +-0.305, 0), x forward from the line joining them, y left. Puller frame:
origin at the hinge (the loop end, where the arms join), +x along the tong toward the jaw tips, the arms part along
y. It stands in its holder, and is used, pointing straight down (PULLER_QUAT), perpendicular to the fuses: handles
at GRASP_X from the hinge, the jaw grooves (across the tong) at JAW_X.
"""

from __future__ import annotations

import math

from fuse_puller_sim.spec import (BLOCK_X, CIRCUITS, DUCT_FAR_X, DUCT_TOP, LABEL_RGB, NAMED, PAN_TOP, PULLER_HOLDER_XY,
                                  SLOT_Y, TRAY_XY, WALL_TOP)

AXIS_Z = PAN_TOP + 0.0247
"""The fuses' axes, 5.3 cm above the table."""
BLOCK_L = 0.0762
"""Bussmann BM6031B block length (along x)."""
FUSE_L, CAP_L = 0.0381, 0.0100
"""10 x 38 midget fuse: 38.1 mm long, 10 mm ferrules."""
CLIP_CENTRE_X = 0.5 * (0.0091 + 0.0173)
"""Each clip's centre from the block's centre (the clips are 26.4 mm apart)."""
TOP_X = 0.121
"""The puller's jaw tips from its hinge (Ideal 34-015: 127 mm overall)."""
JAW_X = 0.114
"""The jaw grooves' centre from the hinge."""
GRASP_X = 0.040
"""Where the demo pinches the notched handles, from the hinge."""
SMOOTH_GRASP_X = 0.090
"""Control run: where it pinches the smooth arms just above the jaws."""
PULLER_QUAT = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
"""The puller pointing straight down (its +x along -z), jaw grooves along the fuses."""
HOLDER_FLOOR, HOLDER_TOP = 0.040, 0.060
"""The holder's pocket floor (the puller stands on its jaw tips) and rim, above the table."""
LABEL = (0.034, 0.020)
LABEL_X = 0.5 * (DUCT_FAR_X[0] + DUCT_FAR_X[1])
"""The circuit labels (34 x 20 mm) on the far wire duct's cover, one above each block."""


def fuse_seat(slot: int) -> tuple[float, float, float]:
    return (BLOCK_X, SLOT_Y[slot], AXIS_Z)


def clip_centre(slot: int, end: int) -> tuple[float, float, float]:
    """Centre of clip ``end`` (0 = near the robot, 1 = far) of block ``slot``."""
    return (BLOCK_X + (1 if end else -1) * CLIP_CENTRE_X, SLOT_Y[slot], AXIS_Z)


def puller_rest_pose() -> tuple:
    """The puller standing in its holder: (hinge x, y, z) + quaternion."""
    return (PULLER_HOLDER_XY[0], PULLER_HOLDER_XY[1], HOLDER_FLOOR + TOP_X) + PULLER_QUAT


def puller_on_fuse_pose(slot: int) -> tuple:
    """The puller pointing down at fuse ``slot``, its jaws round the middle of the fuse body."""
    return (BLOCK_X, SLOT_Y[slot], AXIS_Z + JAW_X) + PULLER_QUAT


def label_world(slot: int) -> tuple[float, float, float]:
    return (LABEL_X, SLOT_Y[slot], DUCT_TOP + 0.0005)


__all__ = ["CIRCUITS", "LABEL_RGB", "NAMED", "TRAY_XY", "WALL_TOP", "math"]

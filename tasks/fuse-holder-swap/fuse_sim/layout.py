"""The documented layout the scripted demo works from (task.json ``setup``; the same numbers
fuse_sim.usd.fuse_props builds the props from), as plain constants: no simulator or USD imports.

Desk frame: table top z = 0, arm bases at (0, +-0.305, 0), x forward from the line joining them, y left. Holder frame
(per holder): d forward from the carrier's front face (negative toward the robot), y left, z up from the holder's
bottom. The carrier frame is the holder's hinge, turned by -theta about +y as the carrier opens by theta.
"""

from __future__ import annotations

import math

from fuse_sim.spec import (BASE_Z, CIRCUITS, FRONT_X, NAMED, RATING_COLOURS, SLOT_Y, SPARE_AXIS_Z, SPARE_RATINGS,
                           SPARE_X, SPARE_Y, TRAY_XY)

MM = 0.001
H = 73.9 * MM
"""Holder height (Bussmann CHM1DU)."""
HINGE_D, HINGE_Z = 4.0 * MM, 29.5 * MM
"""The carrier's hinge: 4 mm behind the face, 29.5 mm above the holder's bottom."""
OPEN_DEG = 72.0
"""The open carrier rests on its stop at 72 degrees."""
TAB_D = (-9.0 * MM, 0.0)
TAB_Z = (52.0 * MM, 61.0 * MM)
"""The red grip tab: 9 mm proud of the carrier face, 52-61 mm above the bottom."""
FUSE_R, FUSE_L, CAP_L = 5.15 * MM, 38.1 * MM, 10.0 * MM
"""10 x 38 midget fuse: 10.3 mm across, 38.1 mm long, 10 mm end caps."""
AXIS_D, FUSE_ZC = 18.0 * MM, 40.0 * MM
"""A seated fuse's axis: 18 mm behind the carrier's face, centred 40 mm above the holder's bottom."""
LEGEND_W, LEGEND_H, LEGEND_Z0, LEGEND_TILT_DEG = 0.030, 0.025, H + 0.008, 45.0
"""Legend plate above each holder (30 x 25 mm, rating colour on its lower half), leaning back 45 degrees, its lower
edge 8 mm above the holder's top and 1.8 cm in front of the pan."""
PAN_FRONT_X = 0.330
CARTON_T, CARTON_H = 0.6 * MM, 23.0 * MM
"""The spare cartons (opened 10-packs): board thickness and outside height."""
LID_OPEN_DEG = 100.0
"""Each carton's lid, hinged at its far top edge, stands open 100 degrees; its inside faces the robot."""


def holder_to_world(slot: int, d: float, y: float, z: float) -> tuple[float, float, float]:
    return (FRONT_X + d, SLOT_Y[slot] + y, BASE_Z + z)


def hinge_world(slot: int) -> tuple[float, float, float]:
    return holder_to_world(slot, HINGE_D, 0.0, HINGE_Z)


def seat_in_carrier() -> tuple[float, float, float]:
    """The seated fuse's centre in the carrier (hinge) frame; its axis is the carrier's +z."""
    return (AXIS_D - HINGE_D, 0.0, FUSE_ZC - HINGE_Z)


def spare_pose(k: int) -> tuple:
    """A spare lying in its carton, axis along +x, centred between its ends: (x, y, z, qx, qy, qz, qw)."""
    s = math.sqrt(0.5)
    return (0.5 * (SPARE_X[0] + SPARE_X[1]), SPARE_Y[k], SPARE_AXIS_Z, 0.0, s, 0.0, s)


def tab_world(slot: int, theta: float) -> tuple[float, float, float]:
    """Centre of a carrier's grip tab with the carrier opened by theta (rad)."""
    hx, hy, hz = hinge_world(slot)
    x, z = 0.5 * (TAB_D[0] + TAB_D[1]) - HINGE_D, 0.5 * (TAB_Z[0] + TAB_Z[1]) - HINGE_Z
    c, s = math.cos(theta), math.sin(theta)
    return (hx + x * c - z * s, hy, hz + x * s + z * c)


def legend_field_world(slot: int) -> tuple[float, float, float]:
    """Middle of a legend's coloured rating field (its lower half)."""
    t = math.radians(LEGEND_TILT_DEG)
    x_back = PAN_FRONT_X  # the plate leans back from its lower edge (1.8 cm in front of the pan) to the pan
    zb = BASE_Z + LEGEND_Z0
    f = 0.25  # a quarter of the way up the plate
    return (x_back - LEGEND_H * math.sin(t) * (1 - f), SLOT_Y[slot], zb + LEGEND_H * math.cos(t) * f)


def carton_label_world(k: int) -> tuple[float, float, float]:
    """Middle of the rating's colour field on carton k's open lid (40% of the way up from the hinge)."""
    x0 = SPARE_X[0] - 0.5 * MM - CARTON_T
    x1 = SPARE_X[1] + 0.5 * MM + CARTON_T
    z1 = SPARE_AXIS_Z - FUSE_R - CARTON_T + CARTON_H
    a = math.radians(LID_OPEN_DEG)
    d = 0.40 * (x1 - x0)
    return (x1 - d * math.cos(a), SPARE_Y[k], z1 + d * math.sin(a))


__all__ = ["CIRCUITS", "NAMED", "RATING_COLOURS", "SPARE_RATINGS", "TRAY_XY"]

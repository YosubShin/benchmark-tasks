"""The documented setup (task.json ``setup``) as the numbers the scripted demo may use. No simulator access.

These are the design positions and part sizes the props are built from (``usd/hasp_props.py``,
``usd/padlock_props.py``), in the desk frame (table top z = 0, x forward from the line joining the arm bases, y to the
robot's left). The whole prop assembly is documented as turned 30 degrees counterclockwise (seen from above) about
the vertical through the hasp's ring. The demo uses these numbers as nominal waypoints and corrects them with what its
cameras see and what its joints report. Nothing here is read from the running simulation.
"""

from __future__ import annotations

import math

import numpy as np

from lockout_sim.kit import q_axis, qmul, rotate
from lockout_sim.spec import COWORKER_HOLE, HASP_Y, PEDESTAL_XY, RING_X
from lockout_sim.usd import hasp_props as H
from lockout_sim.usd import padlock_props as PP

LAYOUT_YAW_DEG = 30.0
YAW_Q = q_axis((0, 0, 1), math.radians(LAYOUT_YAW_DEG))
PIVOT_XY = (RING_X, HASP_Y)


def turn(v) -> np.ndarray:
    """A direction of the unturned layout, turned with the assembly."""
    return np.asarray(rotate(YAW_Q, v), dtype=float)


def turn_point(p) -> np.ndarray:
    x, y, z = p
    dx, dy, _ = rotate(YAW_Q, (x - PIVOT_XY[0], y - PIVOT_XY[1], 0.0))
    return np.array([PIVOT_XY[0] + dx, PIVOT_XY[1] + dy, z])


EX, EY, EZ = turn((1.0, 0.0, 0.0)), turn((0.0, 1.0, 0.0)), np.array([0.0, 0.0, 1.0])
"""The hasp's axes: EX across its tail (toward the wall side), EY through its plane (the robot's lock comes from -EY),
EZ up."""

PADLOCK = PP.PADLOCKS["ml410"]
OPEN_TRAVEL = PP.OPEN_TRAVEL
LOCK_LYING_QUAT = qmul(YAW_Q, qmul(q_axis((1, 0, 0), math.pi / 2), q_axis((0, 0, 1), math.pi / 2)))
"""The robot's padlock on its side on the pedestal: width (local x) up, shackle (local z) toward the hasp."""
LOCK_START = turn_point((PEDESTAL_XY[0], PEDESTAL_XY[1], H.PEDESTAL_TOP + PADLOCK.body[0] / 2))
"""Its body centre lying on the pedestal (pedestal top 13.8 cm)."""

HASP_RING = turn_point(H.nominal_hasp_pos())
"""The hasp's ring centre at its documented hang (the jaws' frame origin)."""
HOLE_COLS, HOLE_ROWS, HOLE = H.HOLE_COLS, H.HOLE_ROWS, H.HOLE
TAIL_W, TAIL_TOP, TAIL_BOTTOM, JAW_T = H.TAIL_W, H.TAIL_TOP, H.TAIL_BOTTOM, H.JAW_T
COWORKER = COWORKER_HOLE


def hasp_point(local, ring=None) -> np.ndarray:
    """A point given in the hasp's frame (x across the tail, y through its plane, z up; origin at the ring centre)."""
    ring = HASP_RING if ring is None else ring
    return np.asarray(ring) + turn(local)


def hole(col: int, row: int, ring=None) -> np.ndarray:
    return hasp_point((HOLE_COLS[col], 0.0, HOLE_ROWS[row]), ring)


def tail_points(step: float = 0.0012, inset: float = 0.0006, y: float = 0.0):
    """(red points on the tail's face, points inside its six holes) in the hasp frame."""
    red, holes = [], []
    hs = HOLE / 2
    for x in np.arange(-TAIL_W / 2 + inset, TAIL_W / 2 - inset + 1e-9, step):
        for z in np.arange(TAIL_BOTTOM + inset, TAIL_TOP - inset + 1e-9, step):
            inside = any(abs(x - cx) < hs + inset and abs(z - cz) < hs + inset for cx in HOLE_COLS for cz in HOLE_ROWS)
            if not inside:
                red.append((x, y, z))
    for cx in HOLE_COLS:
        for cz in HOLE_ROWS:
            for x in np.linspace(cx - hs + 0.0012, cx + hs - 0.0012, 5):
                for z in np.linspace(cz - hs + 0.0012, cz + hs - 0.0012, 5):
                    holes.append((x, y, z, cx, cz))
    return np.array(red), np.array(holes)


def tail_outline(step: float = 0.0012, inset: float = 0.0006, y: float = JAW_T):
    """The tail as a 4 mm slab (both jaws' plates) seen from its +y side, in the hasp frame: (points on its red faces:
    the front face, the top edge face and the two side faces; points where it is not red: just beyond its long sides
    and above its top edge beside the steel neck, at the front and back faces' depths, and the middle of each square
    hole, where the background shows through both plates)."""
    hs = HOLE / 2
    in_hole = lambda x, z: any(abs(x - cx) < hs + inset and abs(z - cz) < hs + inset  # noqa: E731
                               for cx in HOLE_COLS for cz in HOLE_ROWS)
    face = [(x, y, z) for x in np.arange(-TAIL_W / 2 + inset, TAIL_W / 2 - inset + 1e-9, step)
            for z in np.arange(TAIL_BOTTOM + inset, TAIL_TOP - inset + 1e-9, step) if not in_hole(x, z)]
    for yy in np.linspace(-y + inset, y - inset, 3):
        face += [(x, yy, TAIL_TOP) for x in np.arange(-TAIL_W / 2 + inset, TAIL_W / 2 - inset + 1e-9, step)]
        face += [(sx * TAIL_W / 2, yy, z) for sx in (-1, 1) for z in np.arange(TAIL_BOTTOM + inset, TAIL_TOP, step)]
    out = []
    for yy in (-y, y):
        for d in (0.0008, 0.0018):
            out += [(sx * (TAIL_W / 2 + d), yy, z) for sx in (-1, 1) for z in np.arange(TAIL_BOTTOM + 0.002, TAIL_TOP, step)]
            out += [(x, yy, TAIL_TOP + d) for x in np.arange(-TAIL_W / 2, TAIL_W / 2 + 1e-9, step) if abs(x) > 0.0105]
    for cx in HOLE_COLS:
        for cz in HOLE_ROWS:
            out += [(x, 0.0, z) for x in np.linspace(cx - 0.0015, cx + 0.0015, 3) for z in np.linspace(cz - 0.0012, cz + 0.0012, 3)]
    return np.array(face), np.array(out)

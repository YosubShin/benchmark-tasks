"""Fire-extinguisher task: layout, discharge rule constants and instruction (no simulator imports).

Desk frame: table top z = 0 (the station table is 2.5 mm below), left arm base at (0, +0.305), right arm base at
(0, -0.305); x forward, y left. Prop geometry and sources: fire_sim/usd/props.py.

The instruction states only the goal (the fire out, then the spray stopped) and the constraints (stay outside the
taped line, do not drop the extinguisher). What it leaves to the robot is the real procedure on the unit's label
(PASS): hold it upright, pull the ring pin (breaking the tamper seal) while holding the unit, aim at the base of the
flames from back at the line, squeeze the lever, sweep side to side, and let go of the lever once the fire is out.
"""

from __future__ import annotations

from fire_sim import geometry as P

TASK_NAME = "fire-extinguisher-pass"
EPISODE_SECONDS = 90.0

UNIT_XY = (0.23, -0.05)
"""Extinguisher standing on the table: its cylinder axis (nozzle toward +x, ring pin on its +y side)."""

FUEL_Z = P.FUEL_Z
FLAME_X = P.FLAME_X
FLAME_Y = P.FLAME_Y
FLAME_H = P.FLAME_H
PAN_IN_X = (P.STAND_X[0] + P.PAN_WALL, P.STAND_X[1] - P.PAN_WALL)
PAN_IN_Y = (P.STAND_Y[0] + P.PAN_WALL, P.STAND_Y[1] - P.PAN_WALL)
BAND_EDGES = (PAN_IN_Y[0], (FLAME_Y[0] + FLAME_Y[1]) / 2, (FLAME_Y[1] + FLAME_Y[2]) / 2, PAN_IN_Y[1])
"""The base zone is split along y into one band per flame tongue (right, middle, left)."""
BASE_ZONE_M = 0.04
"""The base zone: the fire's front 0-4 cm above the fuel surface, the lower 40% of the 10.5 cm flames (a band of
water-sensitive paper in a real trial)."""
KEEP_OUT_X = P.STAND_X[0] - P.KEEP_OUT
KEEP_OUT_Y = (P.STAND_Y[0] - P.KEEP_OUT, P.STAND_Y[1] + P.KEEP_OUT)
"""Inside the taped line: x beyond KEEP_OUT_X with y between the two side tapes."""

VALVE_OPEN_DEG = 12.0
"""The valve opens once the lever is squeezed past 12 of its 20 degrees (rule)."""
DISCHARGE_S = 10.0
"""Spray time in the cylinder (the B417's published discharge time, about 10 s; the trainer is filled to match)."""
RANGE_M = 0.80
"""Reach of the spray (rule): hits farther than this from the outlet do not count."""
UPRIGHT_MAX_DEG = 60.0
"""Tilted further than this from upright the dip tube draws gas, not agent: no knockdown (rule)."""
KNOCKDOWN_S = 1.0
"""Base hits each flame tongue needs before it is out (rule)."""
STOP_HOLD_S = 1.0
"""After the fire is out, the lever must stay released this long without a break."""
DROP_TILT_DEG = 75.0
DROP_SPEED = 0.8
"""Dropped: the extinguisher tips more than 75 deg from upright, or falls faster than 0.8 m/s."""


def instruction() -> str:
    """Goal and constraints only; the procedure is in task.json's context and scoring (for reviewers)."""
    return ("Put out the fire in the pan with the fire extinguisher, then stop spraying. Stay behind the yellow tape "
            "line: no part of either arm, either gripper or the extinguisher may cross it. Do not drop the "
            "extinguisher.")

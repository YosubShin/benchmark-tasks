"""Pipe insulation shield task: layout, part dimensions and instruction (no simulator imports).

Desk frame: table top z = 0, x forward (away from the robot), y to the robot's left, left arm base at (0, +0.305, 0),
right arm base at (0, -0.305, 0). pipeshield_sim.station maps it into the station (z + 0.75).

Real parts (README.md "References"; files and pages in out/compare/pipe-insulation-shield/ref/SOURCES.txt):
- Pipe: 1/2 in Type L copper tube (0.625 in OD, 0.545 in ID; 0.285 lb/ft, 0.386 lb/ft full of water: CDA Copper
  Tube Handbook Table 14.2b) with Johns Manville Micro-Lok HP fiberglass pipe insulation, nominal 1 in wall (actual
  1.09 in, OD 2.83 in: JM CI-214 p3), about 0.22 lb/ft (from JM's 5.2 pcf). Full of water: 0.606 lb/ft (8.84 N/m).
- Shield: Eaton B-Line B3151-3X insulation protection shield (for 1/2 in tube with 1 in insulation): 18 ga
  galvanized, 12 in long, 2.88 in inside diameter, half-round (MSS Type 40), 75 lb per 100 (B-Line PH20 shields
  section, printed p164-166). Anvil Fig. 167 size 2A is the same: 18 ga, 12 in, 2.88 in.
- Hanger: B-Line B3100-2-1/2 clevis hanger (the size B-Line pairs with the 3X shield), 1/2-13 rod, centre of pipe
  to top of hanger 3-15/16 in (B-Line hanger catalog printed p50); lower strap 1 in wide (Anvil Fig. 260 size 2-1/2,
  dimension H). Strap thickness and the strap's inside radius are not published (inferred).
- Hanger spacing for 1/2 in copper: 5 ft maximum (MSS SP-69, as B-Line's catalog quotes it), 6 ft (IPC 308.5).
"""

from __future__ import annotations

import math

TASK_NAME = "pipe-insulation-shield"
EPISODE_SECONDS = 150.0
IN = 0.0254
LBFT = 0.45359237 / 0.3048
"""1 lb/ft in kg/m."""

# -- pipe -------------------------------------------------------------------------------------------------------------
PIPE_X = 0.420
"""The pipe runs left-right (along y), its axis this far in front of the arm bases."""
INS_R = 2.83 / 2 * IN
"""Outside radius of the insulation (35.9 mm)."""
CU_R = 0.625 / 2 * IN
CU_RI = 0.545 / 2 * IN
PIPE_Y = (-0.66, 0.58)
"""The section's ends (1.24 m). Right end 8 cm past hanger B; the left end overhangs hanger A by 58 cm."""
PIPE_W = (0.285 + 0.101 + 0.22) * LBFT
"""kg/m: Type L copper full of water, plus the insulation."""
PIPE_MASS = PIPE_W * (PIPE_Y[1] - PIPE_Y[0])
PIPE_CY = (PIPE_Y[0] + PIPE_Y[1]) / 2

# -- hangers ---------------------------------------------------------------------------------------------------------
HANGER_A_Y = 0.0
"""The hanger the shield goes into (A)."""
HANGER_B_Y = -0.58
"""The second hanger (B), 0.58 m to the right; its shield is already in."""
STRAP_W = 1.0 * IN
STRAP_T = 3 / 16 * IN
"""Lower strap thickness (inferred)."""
CLEVIS_R = 0.0405
"""Inside radius of the lower strap's bottom (inferred: clears the shield's 37.8 mm outside radius by 2.7 mm)."""
Z_CB = 0.405
"""Top of hanger A's lower strap at its bottom (the strap's inside, where the shield's middle sits)."""
TOP_ABOVE_PIPE = 3.9375 * IN
"""Centre of pipe to top of hanger (B3100-2-1/2 'C')."""
ROD_D = 0.5 * IN
BOLT_D = 0.5 * IN
BOLT_ABOVE_PIPE = 0.070
"""Cross bolt's axis above the pipe's axis (inferred from 'C' less the yoke: about 1 in clear above the insulation, so
the pipe can be raised some 28 mm at the hanger before it meets the bolt)."""
BEAM_Z = 0.85
"""Underside of the overhead strut the hanger rods hang from."""

# -- shield (B3151-3X) -------------------------------------------------------------------------------------------------
SH_RI = 2.88 / 2 * IN
SH_T = 0.0478 * IN
"""18 ga (1.21 mm)."""
SH_RO = SH_RI + SH_T
SH_L = 12.0 * IN
SH_MASS = 0.75 * 0.45359237
SH_SEGS = 18
MARK_W = 0.012
"""Red tape round the shield's middle (its centre mark)."""
"""The half-round modelled as 18 flat strips of 10 degrees."""

# -- resting heights ---------------------------------------------------------------------------------------------------
PIPE_Z_BARE = Z_CB + INS_R
"""Pipe axis height at hanger A with the insulation directly on the strap (the start)."""
PIPE_Z_ON_SHIELD = Z_CB + SH_T + INS_R
"""Pipe axis height at hanger A resting on a seated shield (the goal)."""
Z_CB_B = Z_CB - SH_T
"""Hanger B's strap is 1.2 mm lower: its shield is in, so the pipe starts level."""

# -- shield's start: balanced on a wooden block in front of the pipe ----------------------------------------------------------
SH0_X = 0.330
SH0_Y = -0.060
SPINE0 = 0.303
"""Height of the shield's underside (its spine) resting on its post."""
BLOCK_SIZE = (0.038, 0.038, SPINE0)
"""One 2x2 offcut (1-1/2 in square) standing on end under the shield's middle, as tall as its rest height."""

# -- scoring -----------------------------------------------------------------------------------------------------------
UNDER_X_TOL = 0.006
"""Shield's axis within this of the pipe's axis, sideways."""
ARC_DOWN_DEG = 15.0
CENTRE_TOL = 0.010
"""Shield's middle within 10 mm of the hanger's centre line, along the pipe."""
SEAT_TOL = 0.002
"""Shield's spine within this of the strap; pipe's axis within this of its height resting on the shield."""
RELEASE_CLEAR = 0.030
HOLD_S = 1.0
LIFT_MAX = 0.020
"""Hazard: the pipe raised more than 20 mm at hanger A."""
SHIFT_MAX = 0.015
"""Hazard: the pipe pushed or pulled along its run (or sideways) more than 15 mm."""
DROP_Z = 0.230
"""Hazard: the shield's spine below this height (fallen off its post or out of the hand)."""


def shield_start() -> tuple:
    """(x, y, z) of the shield's origin (its axis, the centre of its curvature) on its post."""
    return (SH0_X, SH0_Y, SPINE0 + SH_RO)


def pipe_start() -> tuple:
    return (PIPE_X, PIPE_CY, PIPE_Z_BARE)


def pipe_z_at(y_pipe_c: float, z_pipe_c: float, pitch_rad: float, y: float) -> float:
    """Height of the pipe's axis at position y along the run (pitch about x, + raises the left end)."""
    return z_pipe_c + math.tan(pitch_rad) * (y - y_pipe_c)


def instruction() -> str:
    return ("Fit the insulation shield into the pipe hanger under the pipe, centred on the hanger, so the pipe rests "
            "on it. Lift the pipe no more than 2 cm and do not shift it along its run, do not drop the shield, "
            "and let go when you are done.")

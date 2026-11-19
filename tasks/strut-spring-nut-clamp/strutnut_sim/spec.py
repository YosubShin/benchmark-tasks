"""Strut spring-nut task: layout, part dimensions and instruction (no simulator imports).

Desk frame: table top z = 0, x forward (away from the robot), y to the robot's left, left arm base at (0, +0.305, 0),
right arm base at (0, -0.305, 0). strutnut_sim.station maps it into the station (z + 0.75).

Real parts (README.md "References"; Unistrut General Engineering Catalog No. 17 = "GEC17"):
- Channel: Unistrut P1000, 1-5/8 x 1-5/8 in (41.3 mm), 12 ga (0.105 in, 2.7 mm), slot 7/8 in (22.2 mm) between the
  inturned lips; each lip 3/8 in (9.5 mm) wide at the top and turned in and down 9/32 in (7.1 mm) from the top face
  (GEC17 p24). Perma-Green (GR) painted finish.
- Spring nut: Unistrut P1010 (1/2-13, B-Line N225 equivalent), 1/2 in (12.7 mm) thick, about 3/4 in wide and
  1.36 in long (distributor listing "3/4 x 1-3/4 x 1-23/64 in, std spring": width, height with spring, length),
  12 lb per 100 (54 g). "Nuts are rectangular with ends shaped to permit a quarter turn clockwise ... Two toothed
  grooves in the top of the nut engage the inturned edges of the channel" (GEC17 p10). Groove width and depth and the
  spring's rate are not published (inferred).
- Fitting: Unistrut P1064 flat plate fitting (square washer), 1-5/8 in square, 1/4 in thick, 9/16 in hole for a
  1/2 in bolt, 17 lb per 100 (77 g) (GEC17 p75).
- Bolt: 1/2-13 x 15/16 in hex head cap screw ("All 9/16 in diameter holes use 1/2 x 15/16 in hex head cap screws and
  1/2 in nuts - P1010", GEC17 p71); ASME B18.2.1: 3/4 in across flats, 5/16 in head; 13 threads per inch.
- Recommended final torque 50 ft-lb (68 N m) for 1/2 in (GEC17 p71, B-Line p78): an installer's wrench job after
  the fittings are positioned; this task ends at hand-snug.
"""

from __future__ import annotations

import math

TASK_NAME = "strut-spring-nut-clamp"
EPISODE_SECONDS = 150.0
IN = 0.0254

# -- channel and stand --------------------------------------------------------------------------------------------
CH_W = 1.625 * IN
"""Channel width and height (41.3 mm)."""
CH_T = 0.105 * IN
"""12 ga wall (2.67 mm)."""
SLOT = 0.875 * IN
"""Opening between the lips (22.2 mm)."""
LIP_RETURN = 9 / 32 * IN
"""The lips turn in and down to 7.1 mm below the channel's top face."""
LIP_ROUND = 0.0018
"""Chamfer standing in for the lip's rounded bend at the slot edge (inferred from the section drawing)."""
CH_X = 0.300
"""Channel axis, forward of the arm bases; the channel runs left-right (along y)."""
CH_Y = (-0.305, 0.305)
"""A 2 ft (610 mm) length, centred on the midline."""
CH_Z0 = 0.0
"""Underside of the channel (its back): it is bolted to the bench top, open side up (surface mounted)."""
CH_TOP = CH_Z0 + CH_W
INNER_HALF = CH_W / 2 - CH_T
"""Half the clear width inside the channel, between the walls (17.98 mm)."""
FLOOR_Z = CH_Z0 + CH_T
"""Top of the channel's back (inside floor)."""
ANCHOR_Y = (-0.25, -0.13, 0.13, 0.25)
"""Where the channel is bolted down through its back (visible bolt heads and washers in the slot)."""
MARK_Y = -0.040
"""Centre line of the red mark (12 mm red tape across the near wall and lip) where the bolt has to go."""
MARK_W = 0.012
INSERT_Y = MARK_Y - 0.050
"""Where the demo puts the nut into the channel (any point of the slot works)."""

# -- spring nut (P1010) ---------------------------------------------------------------------------------------------
NUT_W = 0.75 * IN
"""19.05 mm across (fits the 22.2 mm slot edge-on)."""
NUT_L = 1.36 * IN
"""34.5 mm long (less than the 35.96 mm between the channel's walls)."""
NUT_T = 0.5 * IN
NUT_MASS = 0.054
NUT_END_R = 0.0176
"""The two leading corners (for a clockwise turn) are rounded to this radius about the nut's centre, so the nut can
turn a quarter turn clockwise inside the channel; the other two corners are square and stop it just past 90 deg."""
GROOVE_C = SLOT / 2 + CH_T / 2
"""Groove centre lines, from the nut's centre along its length: under the lip returns (12.45 mm)."""
GROOVE_W = 0.0042
GROOVE_D = 0.0016
"""Groove width and depth (inferred; the catalogs show two toothed grooves)."""
NUT_CHAMFER = 0.0015
"""Bevel on the nut's lower long edges (inferred) that guides it into the slot."""
SEATED_TOP = CH_TOP - LIP_RETURN + GROOVE_D
"""Height of the nut's top face when its grooves sit on the lip returns."""

SPRING_FREE = 1.75 * IN - NUT_T
"""Spring free length below the nut (31.8 mm: the P1010's listed 1-3/4 in height with spring, less the nut)."""
SPRING_K = 440.0
"""N/m (inferred): about 5 N holding the seated nut up against the lips, about 6 N to push it below them."""
SPRING_D = 2.0
FOOT_R = 0.008
FOOT_T = 0.002
FOOT_MASS = 0.004
SPRING_MAX = 0.024
"""Furthest the spring can be compressed (coils solid; inferred)."""

# -- bolt (1/2-13 x 15/16 HHCS) ---------------------------------------------------------------------------------------
PITCH = IN / 13
BOLT_D = 0.5 * IN
BOLT_L = 15 / 16 * IN
"""Length under the head (23.8 mm)."""
HEAD_AF = 0.75 * IN
HEAD_H = 5 / 16 * IN
BOLT_MASS = 0.040
START_TURNS = 2.0
"""The bolt is started this many turns into the nut when the kit is handed over (installer's pre-assembly)."""
SLIDE_MASS = 0.004

# -- fitting (P1064 square washer) ---------------------------------------------------------------------------------
FIT_W = 1.625 * IN
FIT_T = 0.25 * IN
FIT_HOLE = 9 / 16 * IN
FIT_MASS = 0.077

# -- thread model ---------------------------------------------------------------------------------------------------
CLAMP_K = 2.0e5
"""N/m: axial stiffness of the started thread and the clamped stack (rule-driven screw coupling, as a drive)."""
CLAMP_D = 300.0
CLAMP_MAX_N = 1500.0
TURN_K = 20.0
TURN_D = 0.01
TURN_ARMATURE = 2e-5
RUN_NM = 0.03
"""Running torque of the started 1/2-13 thread with no load (inferred): enough to carry the free nut round with the
bolt while it is below the lips, far less than the nut's square corners resist once they reach the walls."""
NUT_K = 0.2
"""Nut factor in T = K F d (the bolt formula)."""

# -- holder (where the kit stands at the start) -----------------------------------------------------------------------
HOLDER_XY = (0.215, -0.170)
HOLDER_SIZE = (0.060, 0.060, 0.066)
POCKET = (0.024, 0.038)
"""Pocket the nut and its spring hang in (x across, y along the nut)."""
POCKET_DEPTH = 0.061

# -- scoring --------------------------------------------------------------------------------------------------------
NUT_ACROSS_DEG = 10.0
SEAT_TOL = 0.0012
CENTRE_TOL = 0.003
MARK_TOL = 0.003
ENGAGED_TURNS = 5.0
FIT_FLAT_TOL = 0.0006
FIT_TILT_DEG = 2.0
FIT_SQUARE_DEG = 3.0
HEAD_GAP_TOL = 0.0005
RELEASE_CLEAR = 0.030
TEST_WAIT_S = 1.0
TEST_S = 2.0
TEST_RAMP_S = 0.5
TEST_PULL_N = 20.0
TEST_MAX_MOVE = 0.001
DROP_Z = 0.030
"""A kit part below this height outside the holder's pocket and outside the channel has fallen to the table."""
NUT_OFF_TURNS = 0.5


def head_rest_above_nut(engaged_m: float) -> float:
    """Head underside above the nut's top face for a given engaged thread length."""
    return BOLT_L - engaged_m


def kit_start() -> dict:
    """Start poses: the kit stands in the holder, the fitting flat on the holder's top, the bolt's head on it, the nut
    hanging on the bolt in the pocket with its spring free, its long axis along y (as the slot will need)."""
    hx, hy = HOLDER_XY
    top = HOLDER_SIZE[2]
    e0 = START_TURNS * PITCH
    nut_top = top + FIT_T - head_rest_above_nut(e0)
    return {"fit": (hx, hy, top), "nut": (hx, hy, nut_top), "e0": e0}


def nut_outline(n_arc: int = 6):
    """Counter-clockwise xy outline of the nut (local frame: x across its width, y along its length), with the two
    leading corners for a clockwise turn, (+x, +y) and (-x, -y), cut by an arc of radius NUT_END_R."""
    hw, hl, r = NUT_W / 2, NUT_L / 2, NUT_END_R
    # Arc on the (+x, +y) corner, from the long side (x = hw) to the end (y = hl).
    a0 = math.asin(min(1.0, hl / r)) if hl < r else math.pi / 2
    y_at_hw = math.sqrt(max(r * r - hw * hw, 0.0))
    x_at_hl = math.sqrt(max(r * r - hl * hl, 0.0))
    t0, t1 = math.atan2(y_at_hw, hw), math.atan2(hl, x_at_hl)
    arc = [(r * math.cos(t0 + (t1 - t0) * k / n_arc), r * math.sin(t0 + (t1 - t0) * k / n_arc))
           for k in range(n_arc + 1)]
    del a0
    pts = [(hw, -hl)] + arc + [(-hw, hl)] + [(-x, -y) for x, y in arc]
    # Order: (hw,-hl) -> up the +x side -> arc to the +y end -> (-hw, hl) -> down the -x side -> arc to -y end.
    return pts


def instruction() -> str:
    return ("Fasten the square plate fitting to the strut at the red mark with its spring nut and bolt. Leave the bolt "
            "hand-snug, not torqued. Do not drop any of the parts, and let go of them when you are done.")

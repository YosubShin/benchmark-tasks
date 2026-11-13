"""Cylinder chain task: layout, part dimensions and instruction (no simulator imports).

Desk frame: table top z = 0, x forward (away from the robot), y to the robot's left, left arm base at (0, +0.305, 0),
right arm base at (0, -0.305, 0). cylchain_sim.station maps it into the station (z + 0.75).

Real parts (README.md "References"):
- Wall bracket: Bristol Maid CYB/DE2 cylinder wall bracket for two D/E medical cylinders (one-piece steel, grey,
  359 x 107 x 88 mm W x D x H, "drop over chains to secure the cylinder", "positioned on the wall at approx half
  cylinder height"). A horizontal plate with two U-shaped cradles, folded down at the back into a 88 mm wall flange.
  Each bay's chain is fixed at the tip of the tongue on the bay's right (seen from the front); on the tongue on the
  bay's left a round-bar J-hook hangs under the plate, its tip turned up: a link of the chain is dropped over it.
  Hook and chain sizes are read from the data-sheet photos (inferred, see README).
- Chain: zinc-plated welded long-link chain, 4 mm wire, links 28 x 11 mm inside (inferred from the photos).
- Cylinder: Catalina/Luxfer-type aluminium medical oxygen cylinder, size MD (111 mm diameter, 424 mm to the top
  of the neck, about 2.5 kg empty), CGA 870 post valve with toggle; green shoulder (US medical oxygen colour).
"""

from __future__ import annotations

import math

TASK_NAME = "cylinder-chain-secure"
EPISODE_SECONDS = 120.0

# -- wall and bracket ------------------------------------------------------------------------------------------
WALL_X = 0.43
"""Front face of the plywood wall panel the bracket is screwed to."""
WALL_T = 0.018
WALL_Y = (-0.30, 0.20)
WALL_H = 0.50
PLATE_TOP = 0.330
"""Top of the bracket's plate above the table: half the height of an E cylinder (0.66 m), as the data sheet says
for the D/E range; an MD/D cylinder in it gets its chain in its upper third."""
PLATE_T = 0.003
PLATE_D = 0.107
FLANGE_H = 0.088
PLATE_FRONT_X = WALL_X - PLATE_D
BAY_W = 0.117
"""Width of each U-shaped cradle (D/E cylinders up to about 111 mm)."""
BAY_BACK = 0.012
"""Steel left behind the cradle's back."""
END_TONGUE = 0.030
CENTRE_TONGUE = 0.359 - 2 * BAY_W - 2 * END_TONGUE
BAY_Y = (0.0, -(BAY_W + CENTRE_TONGUE))
"""Bay centres: the left bay (the robot's cylinder) on the midline, the right bay empty."""
PLATE_Y = (BAY_Y[1] - BAY_W / 2 - END_TONGUE, BAY_W / 2 + END_TONGUE)
BUMPER_T = 0.002
TONGUE_TIP_X = PLATE_FRONT_X + 0.010
"""Where the chain hole sits on its tongue (10 mm behind the plate's front edge)."""
ANCHOR = (TONGUE_TIP_X, -(BAY_W / 2 + 0.022), PLATE_TOP - PLATE_T - 0.004)
"""The chain's fixed end: through a hole in the centre tongue, on the bay's right."""
HOOK_Y = BAY_W / 2 + 0.017
"""The J-hook hangs under the left end tongue's tip."""
HOOK_X = PLATE_FRONT_X + 0.006
"""Its stem, 6 mm behind the plate's front edge."""
HOOK_R = 0.003
"""Round bar, 6 mm."""
HOOK_STEM = 0.050
"""Stem length below the plate's underside (to the bottom of the bend)."""
HOOK_W = 0.022
"""Stem to tip, centre to centre (16 mm clear, less than a flat link's 19 mm width): the bend opens forward, so the tip stands 10 mm in front of the plate's edge with
nothing above it, and a link is dropped over it from above."""
HOOK_TIP_H = 0.022
"""The tip rises this much above the bottom of the bend."""


def hook_points():
    """Polyline of the J-hook's centre line (desk frame): stem top, stem bottom, bend, tip."""
    top = PLATE_TOP - PLATE_T
    bot = top - HOOK_STEM
    x = HOOK_X
    return [(x, HOOK_Y, top), (x, HOOK_Y, bot + HOOK_W / 2), (x - HOOK_W / 2, HOOK_Y, bot),
            (x - HOOK_W, HOOK_Y, bot + HOOK_W / 2), (x - HOOK_W, HOOK_Y, bot + HOOK_TIP_H)]


# -- cylinder ------------------------------------------------------------------------------------------------
CYL_R = 0.0556
CYL_BODY_H = 0.370
CYL_NECK_R = 0.016
CYL_NECK_Z = 0.418
CYL_TOP = 0.424
VALVE = ((-0.016, 0.016), (-0.015, 0.015), (CYL_TOP, CYL_TOP + 0.045))
TOGGLE = ((-0.004, 0.004), (-0.004, 0.004), (CYL_TOP + 0.045, CYL_TOP + 0.075))
CYL_MASS = 3.3
"""2.5 kg empty, 0.25 kg valve, 0.55 kg oxygen (15 cu ft)."""
CYL_COM_Z = 0.20
CYL_FRICTION = 0.35
CYL_XY = (WALL_X - BAY_BACK - BUMPER_T - CYL_R - 0.001, 0.0)
"""Standing upright in the left bay, 1 mm in front of the cradle's back."""
SHOULDER_PUSH_Z = 0.40

# -- chain -----------------------------------------------------------------------------------------------------
LINK_PITCH = 0.028
LINK_WIRE = 0.004
LINK_IN_W = 0.011
LINK_MASS = 0.007
N_LINKS = 14
"""Free links below the fixed end (the chain is fitted for the bracket's larger E cylinders: its last links lie on
the table)."""
CHAIN_LEN = N_LINKS * LINK_PITCH
TAUT_LEN = 0.290
"""Chain length from the fixed end to the hook's bend when pulled tight round the cylinder's front under the plate
(measured in the sim by pulling the end; the analytic wrap below, taut_path_length, ignores the plate and the
links' width and gives 0.17 m; the eleventh link, index 10, is the last one short of the hook)."""

# -- task thresholds -----------------------------------------------------------------------------------------
BAND = (0.25, 0.36)
"""Upper third of the cylinder's straight wall (two thirds of 0.37 m = 0.247 m) up to below the shoulder (0.37 m)."""
SLACK_MAX = 0.035
"""Snug: the hooked link leaves at most 35 mm (one and a quarter links) of chain beyond the length the chain takes
when pulled tight round the cylinder to the hook (TAUT_LEN)."""
SEATED_GAP = 0.006
UPRIGHT_DEG = 3.0
TIP_HAZARD_DEG = 10.0
VALVE_MARGIN = 0.015
TEST_WAIT_S = 1.0
RELEASE_CLEAR = 0.05
TEST_PULL_N = 10.0
TEST_S = 2.0
TEST_MAX_TILT_DEG = 6.0
"""Measured: with the snug link in the hook the 10 N pull still leans the cylinder about 4.2 degrees (the long links
straighten and the link has play in the hook's bend); a loose cylinder falls over (it tips at about 4 N). 6 degrees
separates the two with margin for a phone inclinometer (about 0.5 degree); snugness itself is checked by the link
count (the 0.75 conditions must hold when the check starts)."""


def instruction() -> str:
    """Goal and constraints only (no procedure)."""
    return ("Secure the gas cylinder to the wall bracket with the bracket's chain so that it cannot tip over. "
            "Keep the cylinder upright the whole time, do not touch its valve, and finish with both grippers away "
            "from the cylinder and the chain.")


def taut_path_length(anchor, end, centre_xy, r_chain=CYL_R + 0.004) -> float:
    """Shortest chain path from the fixed end (cylinder's right, y < 0) round the cylinder's front to a point on
    its left (y > 0): tangent, arc through x < centre, tangent; the height difference spread along it."""
    cx, cy = centre_xy

    def tangent(p, sign):
        dx, dy = p[0] - cx, p[1] - cy
        d = max(math.hypot(dx, dy), r_chain + 1e-6)
        t = math.sqrt(d * d - r_chain * r_chain)
        ang = math.atan2(dy, dx) + sign * math.acos(r_chain / d)
        return t, ang

    ta, aa = tangent(anchor, -1)  # right side: its tangent point is further round toward the front
    tb, ab = tangent(end, +1)
    arc = (aa % (2 * math.pi)) - (ab % (2 * math.pi))
    if arc < 0:
        arc += 2 * math.pi
    horiz = ta + tb + r_chain * arc
    return math.hypot(horiz, anchor[2] - end[2])

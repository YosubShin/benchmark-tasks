"""Generate the props: a strip of WAGO TOPJOB S push-in terminal blocks on a DIN rail above a wiring duct, a 2.5 mm2
fine-stranded wire stripped 10 mm (no ferrule) whose flexible conductor comes out of the duct, the stand its end lies
on, and a WAGO 210-720 operating tool on its own stand.

Real parts: WAGO 2002-1201 (grey), 2002-1204 (blue, N) and 2002-1207 (green-yellow, PE) TOPJOB S 2-conductor
terminal blocks, 5.2 mm wide, Push-in CAGE CLAMP. Published (wago.com/us/p/2002-1201 and its datasheet): 5.2 x 48.5 x
39.5 mm (w x h x d), 32.9 mm above the rail's top, strip length 10-12 mm; solid conductors (1-4 mm2) and fine-stranded
conductors with ferrules (1-2.5 mm2) push in directly; fine-stranded conductors without ferrules (0.25-4 mm2) are
connected "via operating tool" (datasheet p12: the tool goes into the operating slot "vertically", square to the face,
the entry is under 15 deg to it). On TOPJOB S the tool stays in the slot by itself while it holds the clamp open
(WAGO: "hands-free wiring").

Modelled (geometry):
- ``strip.usda``: one fixed-base articulation. ``base`` holds the mounting plate, the TS 35 rail, the wiring duct, the
  six housings and, per terminal, the entry block in the wire frame W (side walls, busbar wall, a thin floor, end
  stop) and the operating slot (``_make_slot``: a passage straight into the face, a tapered mouth, a ceiling that
  guides the blade). ``leaf_k``: the clamping leg on a revolute joint, pressed closed by the task's drive (the spring)
  against a hard stop, its clamping edge LEAF_INTERFERENCE into the strand bundle's path; plus a tab under the
  channel that the tool's blade rides on: up a ramp (turning the leg open) onto a flat (where the spring presses the
  blade square to its travel, so it stays put).
- ``wire.usda``: one fixed-base articulation: ``anchor`` (fixed in the duct), CABLE_N capsule links (the flexible
  conductor, springs toward the routed shape), ``wire_end`` (the last 80 mm of insulation, rigid; origin at its front
  face) and the stripped end: STRAND_N bare-strand links on bending joints, the first on a sprung slide along the wire
  (``strand_slide``): the bundle's axial give, so pushed against the closed leg it bunches up instead of opening it.
- ``holder.usda``: the wire's stand and the tool's stand.
- ``tool.usda``: the operating tool, one rigid body (origin at the blade's tip).

Frames. Desk frame: desk top z = 0, arm bases at (0, +-0.305, 0) (the station harness raises it by 0.75 m). The
station table is 2.5 mm below desk z = 0 (TABLE_Z). Wire frame W of terminal k: origin at the centre of its upper
entry's mouth (M0_k), +x (u) along the entry channel into the terminal (toward the plate and down, ENTRY_DEG below
horizontal), +y along the rail, +z (v) perpendicular (toward the plate and up). The busbar is on the +v side, the
clamping leg on the -v side.

Inferred (to measure on a bought part): the entry angle and channel size, the leg's length, angle, spring torque,
the ferrule's friction and the cable's bending stiffness.

Usage: python -m terminal_tool_sim.usd.terminal_props [<out dir>]   (default: terminal_tool_sim/assets)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from terminal_tool_sim.spec import FRONT_X, HOLDER_TOP, HOLDER_XY, N_TERMINALS, PITCH, TERMINAL_KINDS, TOP_Z, terminal_y
from terminal_tool_sim.usd import usd_kit as kit

MM = 0.001
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TABLE_Z = -0.0025
"""The station table's top in the desk frame."""

# WAGO 2002-1201 (published) ---------------------------------------------------------------------------------
LENGTH = 48.5 * MM
HEIGHT = 32.9 * MM
"""Above the rail's top face."""
RAIL_FROM_END = 19.9 * MM
RAIL_DEPTH = 7.5 * MM
RAIL_W = 35.0 * MM
"""TS 35 x 7.5 (EN 60715)."""

RAIL_X = FRONT_X + HEIGHT
PANEL_X = RAIL_X + RAIL_DEPTH
BOT_Z = TOP_Z - LENGTH
RAIL_Z = BOT_Z + RAIL_FROM_END

# Entry (datasheet p12: "the conductor entry is less than 15 degrees"; the rest inferred from the profile photos) -
ENTRY_DEG = 15.0
_C, _S = math.cos(math.radians(ENTRY_DEG)), math.sin(math.radians(ENTRY_DEG))
MOUTH_FROM_END = 4.0 * MM
CH_W = 3.1 * MM
CH_V = (-1.65 * MM, 1.25 * MM)
CH_LEN = 12.0 * MM
FUNNEL_U = 2.0 * MM
FUNNEL_V = 0.6 * MM
"""Tapered mouth (real entries are moulded funnels): over the first FUNNEL_U the channel narrows from FUNNEL_V wider
on both sides (across, v) to its size."""
FUNNEL_Y = 0.4 * MM
"""...and from FUNNEL_Y wider on both sides along the rail (y)."""
BLOCK_V = (-6.0 * MM, 3.5 * MM)
BLOCK_U = 13.0 * MM
CORNER_Z = 13.5 * MM
CORNER_X = 13.8 * MM

# Clamping leg (inferred) ------------------------------------------------------------------------------------
LEAF_U = 7.0 * MM
LEAF_L = 5.0 * MM
LEAF_REACH = 5.0 * MM
"""Hinge to clamping edge (as in the ferrule task)."""
LEAF_PLATE_FROM = 3.5 * MM
"""The leg's contact plate (collision) covers only its outer part, from here to the edge; the operating tool's blade
passes just above the hinge, under it. (The real leg is the free end of a bent spring.)"""
LEAF_REST_DEG = 64.0
LEAF_T = 0.8 * MM
LEAF_WIDTH = 2.4 * MM
LEAF_INTERFERENCE = 0.5 * MM
LEAF_MAX_DEG = 40.0
"""The leg's travel stop (the spring's room in its pocket; inferred). At 25 deg a ferrule pushed in tilted jammed against
the leg on its stop, 1 mm short of home."""
LEAF_SIGN = 1
LEAF_MASS = 0.2
"""Far heavier than a real leg (~1 g): light links let the leg's edge sink into the sleeve under its own spring."""
SPRING_K = 0.10
"""N*m/rad (inferred, as in the ferrule task). Needs 480 Hz physics."""
SPRING_PRELOAD_RAD = 0.25
"""Drive target this far beyond the closed stop: 0.025 N*m on the stop. The pull-out probe holds 50 N with 0.3 mm of
creep at this setting; 0.15 rad let the wire out at about 50 N, and 0.05 N*m/rad with 0.15 rad at about 23 N."""
SPRING_DAMPING = 0.002

# Wire: 2.5 mm2 fine-stranded (H07V-K class 5), stripped 10 mm, NO ferrule ------------------------------------
STRAND_LEN = 10.0 * MM
"""Strip length (WAGO 2002-1201: 10-12 mm)."""
STRAND_W = 2.0 * MM
"""The bare strand bundle (2.5 mm2 of copper, about 2 mm across; square in collision)."""
STRAND_N = 4
STRAND_LINK = STRAND_LEN / STRAND_N
STRAND_LINK_MASS = 0.001
STRAND_LINK_INERTIA = (1e-7, 1e-7, 1e-7)
"""Raised far above a 2.5 mm copper link's (~1e-9) so the soft joints stay stable at 480 Hz."""
STRAND_BEND_K = 0.01
"""N*m/rad per bending joint (inferred): the stripped end holds its shape under its own weight (2 deg sag)."""
STRAND_BEND_DAMPING = 0.0002
STRAND_BEND_LIMIT_DEG = 45.0
STRAND_COLLAPSE_K = 500.0
"""N/m: the bundle's axial give (inferred). Loose strands carry little compression: pushed against the closed
clamping leg they bow out and bunch (birdcage) instead of opening it. Modelled as an axial spring: about 2.5 N at
4.5 mm of collapse, against the ~5 N the leg needs."""
STRAND_COLLAPSE_PRELOAD = 0.0005
"""(The no-robot probe used 200 N/m and strand friction 0.6; 500 N/m lets a slightly misaligned bundle slip past the
mouth's lip under the robot's push, still well short of what opens the leaf.)"""
STRAND_COLLAPSE_MAX = 0.006
STRAND_COLLAPSE_DAMPING = 0.5
INSUL_LEN = 80.0 * MM
INSUL_W = 3.6 * MM
"""H07V-K 2.5 mm2: 3.4-4.1 mm over the insulation (Lapp). Wider than the 2.9 mm channel: it stops at the mouth (the
insulation stop)."""
COLLAR_W = 4.6 * MM
"""Collar of the ferrules on the pre-landed lower wires (visual)."""
HOME_U = 0.0
"""Where the insulation's front face stops (on the mouth: its square section meets the taper's walls at the face)."""
WIRE_MASS = 0.03
"""The rigid end (80 mm of insulation): real ~3 g; heavier for a stable contact against the 0.2 kg leg."""
TAIL_X = -INSUL_LEN
"""The rigid end's tail (where the flexible cable joins it), wire-end frame (origin: the insulation's front face)."""
GRIP_X = -25.0 * MM
FRICTION_SLEEVE = 0.8
"""Bare copper strands on the stainless leg and the copper busbar (inferred; higher than the ferrule's 0.6, as loose
strands bed around the clamping edge). The wedge that holds the wire needs it."""
FRICTION_GRIP = 2.5
"""Insulation against the finger pads (combine max). As in the ferrule task: the station's gripper model squeezes with
about 9 N per finger, so 1.5 let the wire slide back in the pads. Not measured on the real gripper."""

# Operating tool: WAGO 210-720 (blade 3.5 x 0.5 mm, partially insulated shaft; listed for the 2002-1201) --------
BLADE_W = 3.5 * MM
BLADE_T = 0.5 * MM
BLADE_LEN = 10.0 * MM
"""Blade length past the shaft: the shaft (wider than the slot) stops on the face with the blade 10 mm in."""
SHAFT_W = 3.0 * MM
SHAFT_LEN = 80.0 * MM
SHAFT_BARE = 8.0 * MM
HANDLE_W = 13.0 * MM
HANDLE_LEN = 100.0 * MM
TOOL_MASS = 0.05
"""Real ~25 g (inferred); heavier for stable contacts."""
TOOL_GRIP_X = -170.0 * MM
"""Where the hand holds the tool: the handle's rear end, tool frame (origin at the blade's tip, +x toward the tip)."""
FRICTION_BLADE = 0.2
"""Steel blade on the steel spring and the housing (inferred)."""
FRICTION_HANDLE = 2.0
TOOL_XY = (0.27, 0.02)
"""The blade's tip at the start, lying on its stand pointing at the plate (+x)."""
TOOL_Z = 0.09
"""The tool's axis on its stand."""

# Operating slot (inferred from WAGO's photos: a rectangular opening inboard of the entry; the tool goes in square
# to the front face, "vertically", the entry at under 15 deg to it) ---------------------------------------------
SLOT_W = 4.0 * MM
"""Along the rail."""
SLOT_H = 1.2 * MM
SLOT_FUNNEL = 2.0 * MM
SLOT_FUNNEL_Z = 0.8 * MM
SLOT_FUNNEL_Y = 0.5 * MM
SLOT_BELOW_PIVOT = -1.0 * MM
"""The slot's ceiling this far below the leg's hinge (negative: above it; the blade's underside 0.5 mm above it)."""
TAB_R = 5.2 * MM
"""The leg's tab, which the blade rides on, this far in from the hinge (its lever)."""
TAB_FLAT = 0.6 * MM
TAB_T = 0.8 * MM
TAB_RAMP_DEG = 15.0
"""The ramp's slope with the clamp held open (36 deg at rest)."""
TOOL_OPEN_RAD = 0.44
"""How far the fully inserted blade holds the leg open (25 deg): its edge drops 1.2 mm, below the middle of the
channel."""

# Flexible cable (inferred) ----------------------------------------------------------------------------------
CABLE_LEN = 0.40
CABLE_N = 16
CABLE_LINK = CABLE_LEN / CABLE_N
CABLE_R = INSUL_W / 2
CABLE_LINK_MASS = 0.003
"""Real H07V-K 2.5 mm2 is about 35 g/m (0.9 g per 25 mm); heavier links keep the chain stiff enough in PhysX."""
CABLE_LINK_INERTIA = (5e-8, 2.5e-7, 2.5e-7)
"""About the link's centre (x along it). The capsule's own is 5e-9 about its axis and 1.6e-7 across; raised so the
twist joints do not ring at 480 Hz."""
CABLE_EI = 1.5e-4
"""Bending stiffness, N*m^2 (inferred: a 2.5 mm2 fine-stranded PVC conductor)."""
CABLE_K = CABLE_EI / CABLE_LINK
CABLE_DAMPING = 0.002
CABLE_LIMIT_DEG = 100.0
CABLE_FRICTION = (0.5, 0.4)
"""PVC on the desk and the duct."""
AXES = ("Y", "Z", "X")
"""Joint axes cycle: the two bending axes, then twist."""

# Duct (wiring duct below the strip) ---------------------------------------------------------------------------
DUCT_X = (PANEL_X - 0.045, PANEL_X)
DUCT_Z = (0.03, BOT_Z - 0.030)
SLOT_Y = -0.14
"""Where the wire leaves the duct, through a slot in its bottom wall."""
ANCHOR = (DUCT_X[0] + 0.012, SLOT_Y, DUCT_Z[0] - 0.0025)
"""The cable's fixed end: just under the duct's bottom wall, the cable leaving straight down."""

SOLVER_ITERATIONS = 32
CONTACT_OFFSET = 0.0004

COLORS = {
    "grey": (0.62, 0.63, 0.64),
    "blue": (0.10, 0.32, 0.72),
    "green_yellow": (0.20, 0.55, 0.15),
    "yellow_stripe": (0.92, 0.80, 0.10),
    "dark": (0.06, 0.06, 0.07),
    "plate": (0.72, 0.74, 0.70),
    "rail": (0.80, 0.81, 0.83),
    "tin": (0.80, 0.81, 0.83),
    "copper": (0.80, 0.45, 0.20),
    "collar_blue": (0.05, 0.22, 0.80),
    "wire_blue": (0.30, 0.55, 0.85),
    "duct": (0.55, 0.57, 0.58),
    "holder": (0.25, 0.25, 0.27),
    "white": (0.95, 0.95, 0.93),
    "steel": (0.65, 0.67, 0.70),
    "brown": (0.36, 0.20, 0.10),
    "black": (0.04, 0.04, 0.04),
    "wire_grey": (0.45, 0.46, 0.47),
    "tool_green": (0.42, 0.74, 0.22),
    "tool_grey": (0.30, 0.31, 0.33),
}
LOWER_WIRES = {1: "brown", 2: "black", 3: "wire_grey", 4: "wire_blue", 5: "green_yellow", 6: "brown"}
"""The wires already landed in the lower entries (IEC 60445 colours: L1 brown, L2 black, L3 grey, N blue, PE green-
yellow), 1.5 mm2 with black ferrules (WAGO 216-244)."""

W_QUAT = (0.0, math.sin(math.radians(ENTRY_DEG) / 2), 0.0, math.cos(math.radians(ENTRY_DEG) / 2))
"""xyzw: world -> wire frame W (a turn of ENTRY_DEG about +y: +x goes to (cos, 0, -sin))."""


def u_hat() -> tuple:
    return (_C, 0.0, -_S)


def v_hat() -> tuple:
    return (_S, 0.0, _C)


def mouth(n: int) -> tuple[float, float, float]:
    """Centre of terminal n's upper entry mouth (M0), on the front face."""
    return (FRONT_X, terminal_y(n), TOP_Z - MOUTH_FROM_END)


def w_to_world(n: int, u: float, y: float, v: float) -> tuple[float, float, float]:
    m = mouth(n)
    return (m[0] + u * _C + v * _S, m[1] + y, m[2] - u * _S + v * _C)


def leaf_pivot_w() -> tuple[float, float]:
    tip_v = CH_V[1] - STRAND_W + LEAF_INTERFERENCE
    a = math.radians(LEAF_REST_DEG)
    h = LEAF_T / 2
    return (LEAF_U - LEAF_REACH * math.cos(a) + h * math.sin(a), tip_v - LEAF_REACH * math.sin(a) - h * math.cos(a))


def _sx(u: float, v: float) -> tuple[float, float]:
    """Wire-frame (u, v) -> the slot frame's (x, z): from the mouth, x straight into the face, z up."""
    return (u * _C + v * _S, -u * _S + v * _C)


def _uv(x: float, z: float) -> tuple[float, float]:
    return (x * _C - z * _S, x * _S + z * _C)


def pivot_s() -> tuple[float, float]:
    """The leg's hinge in the slot frame (x in, z up; origin the entry's mouth)."""
    return _sx(*leaf_pivot_w())


def slot_ceiling_z() -> float:
    return pivot_s()[1] - SLOT_BELOW_PIVOT


def slot_centre(n: int) -> tuple[float, float, float]:
    """World centre of terminal n's operating-slot opening on the front face."""
    m = mouth(n)
    return (m[0], m[1], m[2] + slot_ceiling_z() - SLOT_H / 2)


def tool_home_tip(n: int) -> tuple:
    """Tool pose (blade tip, xyzw) fully inserted in terminal n's slot: blade riding the ceiling, shaft on the face."""
    m = mouth(n)
    return (m[0] + BLADE_LEN, m[1], m[2] + slot_ceiling_z() - BLADE_T / 2, 0.0, 0.0, 0.0, 1.0)


def tool_start() -> tuple:
    return (TOOL_XY[0], TOOL_XY[1], TOOL_Z, 0.0, 0.0, 0.0, 1.0)


def _rot2(p, c, a):
    """Rotate point p about c by a (rad, counter-clockwise in the (x, z) plane, x right, z up)."""
    dx, dz = p[0] - c[0], p[1] - c[1]
    return (c[0] + dx * math.cos(a) - dz * math.sin(a), c[1] + dx * math.sin(a) + dz * math.cos(a))


def tab_outline_s(opening: float = 0.0) -> list[tuple[float, float]]:
    """The leg's tab (slot frame x, z) with the leg opened by ``opening`` rad. Designed open: its corner at the blade's
    underside TAB_R in from the hinge, a flat beyond, a ramp (TAB_RAMP_DEG) before it; at rest everything is turned
    back (counter-clockwise) by TOOL_OPEN_RAD, so the ramp stands in the blade's path."""
    px, pz = pivot_s()
    zb = slot_ceiling_z() - BLADE_T
    k = (px + TAB_R, zb)
    e = (k[0] + TAB_FLAT, zb)
    a = math.radians(TAB_RAMP_DEG)
    ramp = 4.2 * MM
    f = (k[0] - ramp * math.cos(a), zb - ramp * math.sin(a))
    pts = [f, k, e, (e[0], zb - TAB_T), (f[0] + TAB_T / math.tan(a) * 0 + 0.0, f[1] - TAB_T)]
    return [_rot2(q, (px, pz), TOOL_OPEN_RAD - opening) for q in pts]


def wire_home_root(n: int) -> tuple:
    """Wire end pose fully inserted into terminal n: insulation on the mouth's taper, strands on the busbar."""
    return w_to_world(n, HOME_U, 0.0, CH_V[1] - STRAND_W / 2) + W_QUAT


def wire_aim_root(n: int) -> tuple:
    """Where to aim the wire end on terminal n's entry axis: the channel's centre line."""
    return w_to_world(n, 0.0, 0.0, (CH_V[0] + CH_V[1]) / 2) + W_QUAT


def wire_start_root() -> tuple:
    """Wire end pose lying on its stand: ferrule toward the panel (+x), sloping down ENTRY_DEG like the entry."""
    return (HOLDER_XY[0], HOLDER_XY[1], HOLDER_TOP + 0.0002) + W_QUAT


def holder_posts() -> list[tuple[str, float, float]]:
    """(name, x along the wire end from its root, half-width of what it carries)."""
    return [("front", -6.0 * MM, INSUL_W / 2), ("rear", TAIL_X + 6 * MM, INSUL_W / 2)]


# Rotations ---------------------------------------------------------------------------------------------------
def _rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == "Z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _qmat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _mat_q(m):
    from scipy.spatial.transform import Rotation

    return tuple(float(v) for v in Rotation.from_matrix(m).as_quat())


ANCHOR_ROT = _rot("Y", math.pi / 2)
"""The anchor's +x points straight down (out of the duct's bottom slot)."""


def joint_axis(k: int) -> str:
    return AXES[k % len(AXES)]


def cable_fk(q):
    """Frames (pos, rot) of the CABLE_N links and of the wire end for joint angles q (CABLE_N + 1 of them)."""
    p, r = np.asarray(ANCHOR, float), ANCHOR_ROT.copy()
    links = []
    for k in range(CABLE_N):
        r = r @ _rot(joint_axis(k), q[k])
        links.append((p.copy(), r.copy()))
        p = p + r @ np.array([CABLE_LINK, 0, 0])
    r = r @ _rot(joint_axis(CABLE_N), q[CABLE_N])
    end = (p - r @ np.array([TAIL_X, 0, 0]), r)
    return links, end


def solve_route(verbose=True):
    """Joint angles that route the cable from the duct slot down to the desk, along it and up to the wire end lying on
    its stand; the cable lies low and keeps clear of the stand's posts and of the hand's path to the wire."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    start = wire_start_root()
    sp, sr = np.asarray(start[:3]), _qmat(start[3:])
    grasp = sp + sr @ np.array([GRIP_X, 0, 0])
    posts = [sp + sr @ np.array([x, 0, 0]) for _, x, _ in holder_posts()]
    floor = TABLE_Z + CABLE_R + 0.0003

    def points(links):
        pts = []
        for p, r in links:
            pts += [p + r @ np.array([f * CABLE_LINK, 0, 0]) for f in (0.5, 1.0)]
        return pts

    def residual(q):
        links, (pos, rot) = cable_fk(q)
        err_p = (pos - sp) * 300.0
        err_r = Rotation.from_matrix(sr.T @ rot).as_rotvec() * 30.0
        pts = points(links)
        pen = [max(0.0, floor - e[2]) * 300.0 for e in pts]
        pen += [1.5 * (e[2] - floor) for e in pts[:-2]]  # lie low
        for post in posts:  # off the stand's posts (2 x 16 mm), below their tops
            pen += [max(0.0, 0.016 - math.hypot(e[0] - post[0], e[1] - post[1])) * 200.0 * (e[2] < post[2] + 0.004)
                    for e in pts[:-3]]
        # Out of the hand's path: the fingers come down onto the grasp point from behind at 45 deg; keep the cable
        # 3 cm under that line within 4 cm of the wire's vertical plane, and out of a 4.5 cm column above the grasp.
        pen += [max(0.0, e[2] - (grasp[2] + (grasp[0] - e[0]) - 0.03)) * 150.0 * (abs(e[1] - grasp[1]) < 0.04)
                * (e[0] < grasp[0] + 0.02) for e in pts[:-2]]
        pen += [max(0.0, 0.045 - math.hypot(e[0] - grasp[0], e[1] - grasp[1])) * 150.0 * (e[2] > 0.06) for e in pts[:-2]]
        # In front of the plate and out of the duct (below it, except where it leaves the slot), off the arm bases.
        pen += [max(0.0, e[0] - (DUCT_X[0] - CABLE_R - 0.001)) * 200.0 * (e[2] > DUCT_Z[0] - 0.004) for e in pts[2:]]
        pen += [max(0.0, e[0] - (PANEL_X - CABLE_R - 0.002)) * 200.0 for e in pts]
        pen += [max(0.0, 0.03 - e[0]) * 200.0 for e in pts]
        pen += [max(0.0, 0.18 - math.hypot(e[0], e[1] + 0.305)) * 200.0 for e in pts]
        # Clear of the tool's stand.
        pen += [max(0.0, 0.035 - abs(e[1] - TOOL_XY[1])) * 200.0 * (TOOL_XY[0] - 0.17 < e[0] < TOOL_XY[0])
                for e in pts]
        return np.concatenate([err_p, err_r, 0.25 * np.asarray(q), pen])

    lim = math.radians(CABLE_LIMIT_DEG) * 0.8
    best = None
    for trial in range(12):
        q0 = np.random.default_rng(trial).uniform(-0.5, 0.5, CABLE_N + 1)
        sol = least_squares(residual, q0, bounds=(-lim, lim))
        if best is None or sol.cost < best.cost:
            best = sol
    links, (pos, rot) = cable_fk(best.x)
    if verbose:
        pts = points(links)
        print(f"route: end pos error {np.linalg.norm(pos - sp) * 1000:.2f} mm, rot error "
              f"{np.degrees(np.linalg.norm(Rotation.from_matrix(sr.T @ rot).as_rotvec())):.2f} deg, lowest "
              f"{min(e[2] for e in pts) * 1000:.1f} mm, x {min(e[0] for e in pts):.3f}..{max(e[0] for e in pts):.3f}, "
              f"y {min(e[1] for e in pts):.3f}..{max(e[1] for e in pts):.3f}, max |q| "
              f"{np.degrees(np.abs(best.x).max()):.0f} deg, cost {best.cost:.4f}")
    return best.x


# USD helpers -----------------------------------------------------------------------------------------------
def _material(stage, root, name, color, roughness=0.5):
    path = f"{root}/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _tex_material(stage, root, name, png_name):
    path = f"{root}/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    reader = UsdShade.Shader.Define(stage, f"{path}/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"{path}/tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{png_name}")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _mesh(stage, path, points, faces, material, st=None, collide=False):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([len(f) for f in faces]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in faces for i in f]))
    mesh.CreateSubdivisionSchemeAttr("none")
    if st is not None:
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray([Gf.Vec2f(*t) for t in st]))
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    return mesh


def _prism(stage, path, outline_xz, y0, width, material, collide=True):
    """A prism along y from an (x, z) convex outline."""
    n = len(outline_xz)
    pts = [(x, y0 - width / 2, z) for x, z in outline_xz] + [(x, y0 + width / 2, z) for x, z in outline_xz]
    faces = [list(range(n))[::-1], list(range(n, 2 * n))]
    faces += [[k, (k + 1) % n, n + (k + 1) % n, n + k] for k in range(n)]
    return _mesh(stage, path, pts, faces, material, collide=collide)


def _box(stage, path, x, y, z, mat, collide=True):
    c = ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2)
    s = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    return kit.box(stage, path, c, s, mat, collide=collide)


def _slab(stage, path, origin, axis, deg, x, y, z, mat):
    """A collision box with local ranges x, y, z in a frame at ``origin`` turned ``deg`` about local ``axis``."""
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*origin))
    (xf.AddRotateYOp() if axis == "Y" else xf.AddRotateZOp()).Set(deg)
    return _box(stage, f"{path}/box", x, y, z, mat)


def _xform(stage, path, translate, rot_y_deg=0.0):
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if rot_y_deg:
        xf.AddRotateYOp().Set(rot_y_deg)
    return xf


def _body(stage, path, pos, rot, mass, inertia=None):
    """A rigid body at a world pose; ``inertia`` (diagonal, kg m^2, about its origin) for links whose colliders are
    too thin (or absent) for a usable computed inertia."""
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    x, y, z, w = _mat_q(rot)
    xform.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
    mass_api.CreateMassAttr(mass)
    if inertia is not None:
        com, diag = inertia
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*diag))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
    return xform


def _capsule_x(stage, path, length, radius, material):
    """A collision capsule along +x from 0 to ``length`` (body frame)."""
    cap = UsdGeom.Capsule.Define(stage, path)
    cap.CreateAxisAttr("X")
    cap.CreateRadiusAttr(radius)
    cap.CreateHeightAttr(float(length))
    UsdGeom.Xformable(cap).AddTranslateOp().Set(Gf.Vec3d(length / 2, 0, 0))
    UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    return cap


def _quad_x(stage, path, x, y, z, material):
    """A visual quad facing -x (toward the robot), y and z ranges; the image's left edge at +y."""
    pts = [(x, y[1], z[0]), (x, y[0], z[0]), (x, y[0], z[1]), (x, y[1], z[1])]
    st = [(0, 0), (1, 0), (1, 1), (0, 1)]
    mesh = _mesh(stage, path, pts, [[0, 1, 2, 3]], material, st=st)
    mesh.CreateDoubleSidedAttr(True)
    return mesh


def _offsets(stage) -> None:
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def write_textures(out_dir: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    for n in range(1, N_TERMINALS + 1):
        w, h = 200, 250
        img = Image.new("RGB", (w, h), (243, 243, 238))
        d = ImageDraw.Draw(img)
        d.text((w / 2, h / 2), str(n), font=ImageFont.truetype(FONT, 150), fill=(15, 15, 15), anchor="mm")
        img.save(out_dir / f"marker_{n}.png")
    w, h = 300, 250
    img = Image.new("RGB", (w, h), (243, 243, 238))
    ImageDraw.Draw(img).text((w / 2, h / 2), "X1", font=ImageFont.truetype(FONT, 140), fill=(15, 15, 15), anchor="mm")
    img.save(out_dir / "marker_X1.png")


def _make_slot(stage, tp: str, n: int, mat) -> None:
    """Terminal n's operating slot: a passage straight into the face (world-aligned), its ceiling SLOT_BELOW_PIVOT
    under the leg's hinge, a tapered mouth, and an end wall. Collision only (hidden)."""
    sp = f"{tp}_opslot"
    _xform(stage, sp, mouth(n))
    zc = slot_ceiling_z()
    zf = zc - SLOT_H
    hw, outer = SLOT_W / 2, PITCH / 2 - 0.05 * MM
    fu, fz, fy = SLOT_FUNNEL, SLOT_FUNNEL_Z, SLOT_FUNNEL_Y
    depth = BLADE_LEN + 2.0 * MM
    lo, hi = zf - 2.0 * MM, zc + 1.8 * MM
    parts = {
        # The ceiling guides the blade all the way in, a thin plate under the spring's pocket (the leg moves through
        # it: the base and the legs do not collide).
        "ceiling": ((fu, 8.0 * MM), (-outer, outer), (zc, zc + 0.6 * MM)),
        "floor": ((fu, depth), (-outer, outer), (lo, zf)),
        "side_a": ((fu, depth), (hw, outer), (lo, hi)),
        "side_b": ((fu, depth), (-outer, -hw), (lo, hi)),
        "end": ((depth, depth + 1.0 * MM), (-outer, outer), (lo, hi)),
        "ceiling_mouth": ((0.0, fu), (-outer, outer), (zc + fz, hi)),
        "floor_mouth": ((0.0, fu), (-outer, outer), (lo, zf - fz)),
        "side_a_mouth": ((0.0, fu), (hw + fy, outer), (lo, hi)),
        "side_b_mouth": ((0.0, fu), (-outer, -hw - fy), (lo, hi)),
    }
    for name, (x, y, z) in parts.items():
        _box(stage, f"{sp}/{name}", x, y, z, mat)
    tv, ty = math.degrees(math.atan2(fz, fu)), math.degrees(math.atan2(fy, fu))
    lv, ly = math.hypot(fu, fz), math.hypot(fu, fy)
    _slab(stage, f"{sp}/taper_ceiling", (0.0, -hw - fy, zc + fz), "Y", tv, (0.0, lv), (0, 2 * hw + 2 * fy), (0.0, 0.8 * MM), mat)
    _slab(stage, f"{sp}/taper_floor", (0.0, -hw - fy, zf - fz), "Y", -tv, (0.0, lv), (0, 2 * hw + 2 * fy), (-0.8 * MM, 0.0), mat)
    _slab(stage, f"{sp}/taper_side_a", (0.0, hw + fy, zf - fz), "Z", -ty, (0.0, ly), (0.0, 0.6 * MM), (0, SLOT_H + 2 * fz), mat)
    _slab(stage, f"{sp}/taper_side_b", (0.0, -hw - fy, zf - fz), "Z", ty, (0.0, ly), (-0.6 * MM, 0.0), (0, SLOT_H + 2 * fz), mat)
    for prim in stage.GetPrimAtPath(sp).GetChildren():
        kit.hide(stage, str(prim.GetPath()))


# Strip -----------------------------------------------------------------------------------------------------
def make_strip(out_path: Path) -> None:
    root = "/terminal_strip"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
        SOLVER_ITERATIONS)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    mats = {name: _material(stage, root, name, c) for name, c in COLORS.items()}
    markers = {n: _tex_material(stage, root, f"marker_{n}", f"marker_{n}.png") for n in range(1, N_TERMINALS + 1)}
    group = _tex_material(stage, root, "marker_X1", "marker_X1.png")

    base = kit.rigid_body(stage, f"{root}/base", (0, 0, 0), 5.0)
    y_hi = terminal_y(1) + PITCH / 2
    y_lo = terminal_y(N_TERMINALS) - PITCH / 2
    _box(stage, f"{root}/base/plate", (PANEL_X, PANEL_X + 0.003), (-0.20, 0.03), (0.0, 0.26), mats["plate"])
    _box(stage, f"{root}/base/rail", (RAIL_X, PANEL_X), (y_lo - 0.030, y_hi + 0.030),
         (RAIL_Z - RAIL_W / 2, RAIL_Z + RAIL_W / 2), mats["rail"])
    # Wiring duct (collision), its lid's edge and the slot the wire leaves by (visual).
    duct_y = (y_lo - 0.05, y_hi + 0.05)
    _box(stage, f"{root}/base/duct", DUCT_X, duct_y, DUCT_Z, mats["duct"])
    for k, z in enumerate(np.linspace(DUCT_Z[0] + 0.004, DUCT_Z[1] - 0.004, 2)):
        _box(stage, f"{root}/base/duct_lid_{k}", (DUCT_X[0] - 0.0004, DUCT_X[0]), duct_y, (z - 0.0006, z + 0.0006),
             mats["dark"], collide=False)
    _box(stage, f"{root}/base/duct_slot", (DUCT_X[0] + 0.004, DUCT_X[1] - 0.004), (SLOT_Y - 0.003, SLOT_Y + 0.003),
         (DUCT_Z[0] - 0.0004, DUCT_Z[0]), mats["dark"], collide=False)
    _box(stage, f"{root}/base/housing_lower", (FRONT_X, RAIL_X), (y_lo, y_hi), (BOT_Z, TOP_Z - CORNER_Z),
         mats["grey"], collide=True)
    _box(stage, f"{root}/base/housing_back", (FRONT_X + CORNER_X, RAIL_X), (y_lo, y_hi), (TOP_Z - CORNER_Z, TOP_Z),
         mats["grey"], collide=True)
    for prim_name in ("housing_lower", "housing_back"):
        kit.hide(stage, f"{root}/base/{prim_name}")
    ch = 1.0 * MM
    outline = [(FRONT_X, BOT_Z + ch), (FRONT_X + ch, BOT_Z), (RAIL_X, BOT_Z), (RAIL_X, TOP_Z),
               (FRONT_X + ch, TOP_Z), (FRONT_X, TOP_Z - ch)]
    pivot_u, pivot_v = leaf_pivot_w()
    hw = CH_W / 2
    for n in range(1, N_TERMINALS + 1):
        y = terminal_y(n)
        tp = f"{root}/base/t{n}"
        kind = TERMINAL_KINDS[n]
        _prism(stage, f"{tp}_body", outline, y, PITCH - 0.15 * MM, mats[kind], collide=False)
        if kind == "green_yellow":  # yellow stripes on the green housing (front and upper end)
            for k, (z0, z1) in enumerate(((BOT_Z + 12 * MM, BOT_Z + 17 * MM), (TOP_Z - 20 * MM, TOP_Z - 15 * MM))):
                _box(stage, f"{tp}_stripe_{k}", (FRONT_X - 0.1 * MM, RAIL_X), (y - PITCH / 2 + 0.1 * MM,
                                                                             y + PITCH / 2 - 0.1 * MM), (z0, z1),
                     mats["yellow_stripe"], collide=False)
        zc = (TOP_Z + BOT_Z) / 2
        _quad_x(stage, f"{tp}_marker", FRONT_X - 0.25 * MM, (y - 2.4 * MM, y + 2.4 * MM), (zc - 3.0 * MM, zc + 3.0 * MM),
                markers[n])
        sc = slot_centre(n)
        _box(stage, f"{tp}_slot_0", (FRONT_X - 0.2 * MM, FRONT_X), (y - SLOT_W / 2 - SLOT_FUNNEL_Y, y + SLOT_W / 2 + SLOT_FUNNEL_Y),
             (sc[2] - SLOT_H / 2 - SLOT_FUNNEL_Z, sc[2] + SLOT_H / 2 + SLOT_FUNNEL_Z), mats["dark"], collide=False)
        zs = BOT_Z + (TOP_Z - sc[2])
        _box(stage, f"{tp}_slot_1", (FRONT_X - 0.2 * MM, FRONT_X), (y - SLOT_W / 2 - SLOT_FUNNEL_Y, y + SLOT_W / 2 + SLOT_FUNNEL_Y),
             (zs - SLOT_H / 2 - SLOT_FUNNEL_Z, zs + SLOT_H / 2 + SLOT_FUNNEL_Z), mats["dark"], collide=False)
        zm = TOP_Z - MOUTH_FROM_END
        _box(stage, f"{tp}_mouth_top", (FRONT_X - 0.15 * MM, FRONT_X), (y - hw, y + hw), (zm - 1.5 * MM, zm + 1.5 * MM),
             mats["dark"], collide=False)
        zb = BOT_Z + MOUTH_FROM_END
        _box(stage, f"{tp}_mouth_bot", (FRONT_X - 0.15 * MM, FRONT_X), (y - hw, y + hw), (zb - 1.5 * MM, zb + 1.5 * MM),
             mats["dark"], collide=False)
        # The wire already landed in the lower entry (visual): collar, a run out of the mouth, then down into the duct.
        bot = f"{tp}_bottom_wire"
        _xform(stage, bot, (FRONT_X, y, zb), -ENTRY_DEG)
        lw = mats[LOWER_WIRES[n]]
        _box(stage, f"{bot}/collar", (-7.0 * MM, 0.0), (-COLLAR_W / 2, COLLAR_W / 2), (-COLLAR_W / 2, COLLAR_W / 2),
             mats["dark"], collide=False)
        _box(stage, f"{bot}/run", (-16 * MM, -7.0 * MM), (-INSUL_W / 2, INSUL_W / 2), (-INSUL_W / 2, INSUL_W / 2),
             lw, collide=False)
        x_end, z_end = FRONT_X - 16 * MM * _C, zb - 16 * MM * _S
        _box(stage, f"{tp}_wire_down", (x_end - INSUL_W / 2, x_end + INSUL_W / 2), (y - INSUL_W / 2, y + INSUL_W / 2),
             (DUCT_Z[1], z_end + INSUL_W / 2), lw, collide=False)
        if LOWER_WIRES[n] == "green_yellow":
            for j in range(3):
                z0 = DUCT_Z[1] + 0.003 + j * 0.006
                _box(stage, f"{tp}_wire_down_stripe_{j}", (x_end - INSUL_W / 2 - 0.05 * MM, x_end + INSUL_W / 2),
                     (y - INSUL_W / 2 - 0.05 * MM, y + INSUL_W / 2 + 0.05 * MM), (z0, z0 + 0.003),
                     mats["yellow_stripe"], collide=False)
        e = f"{tp}_entry"
        _xform(stage, e, mouth(n), ENTRY_DEG)
        # The side walls only bound the channel: below it the spring's pocket is open to the operating slot.
        side_v = (CH_V[0] - 0.3 * MM, BLOCK_V[1])
        _box(stage, f"{e}/side_a", (FUNNEL_U, BLOCK_U), (hw, PITCH / 2 - 0.05 * MM), side_v, mats[kind])
        _box(stage, f"{e}/side_b", (FUNNEL_U, BLOCK_U), (-PITCH / 2 + 0.05 * MM, -hw), side_v, mats[kind])
        _box(stage, f"{e}/side_a_mouth", (0.0, FUNNEL_U), (hw + FUNNEL_Y, PITCH / 2 - 0.05 * MM), BLOCK_V, mats[kind])
        _box(stage, f"{e}/side_b_mouth", (0.0, FUNNEL_U), (-PITCH / 2 + 0.05 * MM, -hw - FUNNEL_Y), BLOCK_V, mats[kind])
        _box(stage, f"{e}/busbar", (FUNNEL_U, BLOCK_U), (-hw, hw), (CH_V[1], BLOCK_V[1]), mats["copper"])
        _box(stage, f"{e}/busbar_mouth", (0.0, FUNNEL_U), (-hw, hw), (CH_V[1] + FUNNEL_V, BLOCK_V[1]), mats[kind])
        _box(stage, f"{e}/lower", (FUNNEL_U, 4.6 * MM), (-hw, hw), (CH_V[0] - 0.8 * MM, CH_V[0]), mats[kind])
        _box(stage, f"{e}/lower_mouth", (0.0, FUNNEL_U), (-hw, hw), (CH_V[0] - FUNNEL_V - 0.8 * MM, CH_V[0] - FUNNEL_V),
             mats[kind])
        _box(stage, f"{e}/stop", (CH_LEN, BLOCK_U), (-hw, hw), CH_V, mats[kind])
        # The funnel's tapered faces: slabs from the wide mouth (u = 0) to the channel (u = FUNNEL_U).
        fy = FUNNEL_Y
        tv = math.degrees(math.atan2(FUNNEL_V, FUNNEL_U))
        ty_ = math.degrees(math.atan2(FUNNEL_Y, FUNNEL_U))
        lv, ly = math.hypot(FUNNEL_U, FUNNEL_V), math.hypot(FUNNEL_U, FUNNEL_Y)
        _slab(stage, f"{e}/taper_busbar", (0.0, -hw - fy, CH_V[1] + FUNNEL_V), "Y", tv, (0.0, lv), (0, 2 * hw + 2 * fy),
              (0.0, 0.8 * MM), mats[kind])
        _slab(stage, f"{e}/taper_lower", (0.0, -hw - fy, CH_V[0] - FUNNEL_V), "Y", -tv, (0.0, lv), (0, 2 * hw + 2 * fy),
              (-0.8 * MM, 0.0), mats[kind])
        _slab(stage, f"{e}/taper_side_a", (0.0, hw + fy, CH_V[0] - FUNNEL_V), "Z", -ty_, (0.0, ly), (0.0, 0.6 * MM),
              (0, CH_V[1] - CH_V[0] + 2 * FUNNEL_V), mats[kind])
        _slab(stage, f"{e}/taper_side_b", (0.0, -hw - fy, CH_V[0] - FUNNEL_V), "Z", ty_, (0.0, ly), (-0.6 * MM, 0.0),
              (0, CH_V[1] - CH_V[0] + 2 * FUNNEL_V), mats[kind])
        for part in ("side_a", "side_b", "side_a_mouth", "side_b_mouth", "busbar", "busbar_mouth", "lower", "lower_mouth", "stop", "taper_busbar",
                     "taper_lower", "taper_side_a", "taper_side_b"):
            kit.hide(stage, f"{e}/{part}")
        _make_slot(stage, tp, n, mats[kind])
    _box(stage, f"{root}/base/end_plate", (FRONT_X + 1 * MM, RAIL_X), (y_lo - 0.8 * MM, y_lo), (BOT_Z, TOP_Z),
         mats["grey"], collide=False)
    for k, (ya, yb) in enumerate(((y_hi, y_hi + 9 * MM), (y_lo - 0.8 * MM - 9 * MM, y_lo - 0.8 * MM))):
        _box(stage, f"{root}/base/end_stop_{k}", (FRONT_X + 8 * MM, RAIL_X), (ya, yb), (RAIL_Z - 13 * MM, RAIL_Z + 13 * MM),
             mats["grey"], collide=False)
    _quad_x(stage, f"{root}/base/group_marker", FRONT_X + 8 * MM - 0.25 * MM, (y_hi + 1 * MM, y_hi + 8 * MM),
            (RAIL_Z - 3 * MM, RAIL_Z + 3 * MM), group)
    kit.fix_to_world(stage, root, base.GetPath())

    for n in range(1, N_TERMINALS + 1):
        pivot = w_to_world(n, pivot_u, 0.0, pivot_v)
        lp = f"{root}/leaf_{n}"
        link = _xform(stage, lp, pivot, ENTRY_DEG)
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateMassAttr(LEAF_MASS)
        _xform(stage, f"{lp}/arm", (0, 0, 0), -LEAF_REST_DEG)
        _box(stage, f"{lp}/arm/plate", (LEAF_PLATE_FROM, LEAF_REACH), (-LEAF_WIDTH / 2, LEAF_WIDTH / 2), (-LEAF_T / 2, LEAF_T / 2),
             mats["steel"])
        # The tab the operating tool's blade rides on (the spring's back, reached through the operating slot).
        pu, pv = leaf_pivot_w()
        tab = [tuple(a - b for a, b in zip(_uv(x, z), (pu, pv))) for x, z in tab_outline_s(0.0)]
        _prism(stage, f"{lp}/tab", tab, 0.0, LEAF_WIDTH, mats["steel"], collide=True)
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/joints/leaf_{n}_joint")
        joint.CreateBody0Rel().SetTargets([base.GetPath()])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(lp)])
        joint.CreateAxisAttr("Y")
        joint.CreateLocalPos0Attr(Gf.Vec3f(*pivot))
        joint.CreateLocalRot0Attr(Gf.Quatf(W_QUAT[3], *W_QUAT[:3]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        lo, hi = sorted((0.0, LEAF_SIGN * LEAF_MAX_DEG))
        joint.CreateLowerLimitAttr(lo)
        joint.CreateUpperLimitAttr(hi)
        UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    kit.friction(stage, root, [f"{root}/base/duct"], static=CABLE_FRICTION[0], dynamic=CABLE_FRICTION[1],
                 name="duct_pvc", combine="min")
    _offsets(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


# Wire --------------------------------------------------------------------------------------------------------
def _joint_between(stage, path, body0, frame0, body1, frame1, world_point, world_rot, axis):
    """A revolute joint at ``world_point`` whose frame is ``world_rot`` in both bodies' authored poses (angle 0)."""
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(str(body0))])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(str(body1))])
    joint.CreateAxisAttr(axis)
    for (p, r), pos_attr, rot_attr in ((frame0, joint.CreateLocalPos0Attr, joint.CreateLocalRot0Attr),
                                       (frame1, joint.CreateLocalPos1Attr, joint.CreateLocalRot1Attr)):
        pos_attr(Gf.Vec3f(*[float(v) for v in r.T @ (np.asarray(world_point) - p)]))
        x, y, z, w = _mat_q(r.T @ world_rot)
        rot_attr(Gf.Quatf(float(w), float(x), float(y), float(z)))
    joint.CreateLowerLimitAttr(-CABLE_LIMIT_DEG)
    joint.CreateUpperLimitAttr(CABLE_LIMIT_DEG)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    return joint


def make_wire(out_path: Path, q=None) -> None:
    """The anchor in the duct, the cable's links and the rigid wire end, one fixed-base articulation whose authored
    pose (all joints at 0) is the solved route."""
    root = "/wire"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
        SOLVER_ITERATIONS)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    mats = {name: _material(stage, root, name, c) for name, c in COLORS.items()}
    q = solve_route() if q is None else q
    links, (end_p, end_r) = cable_fk(q)

    # The root link sits at the asset's origin with no rotation (Arena/Isaac Lab write the articulation's default root
    # pose, the asset's placement, to it on reset); the cable's first joint is at ANCHOR in its frame.
    anchor = _body(stage, f"{root}/anchor", (0.0, 0.0, 0.0), np.eye(3), 0.01, (ANCHOR, (1e-6, 1e-6, 1e-6)))
    # The cable's run inside the duct (visual).
    kit.cylinder(stage, f"{root}/anchor/inside", (ANCHOR[0], ANCHOR[1], ANCHOR[2] + 0.015), CABLE_R, 0.03,
                 mats["wire_blue"], axis="Z")
    kit.fix_to_world(stage, root, anchor.GetPath())
    prev, prev_frame = anchor.GetPath(), (np.zeros(3), np.eye(3))
    for k, (p, r) in enumerate(links):
        body = _body(stage, f"{root}/cable_{k:02d}", p, r, CABLE_LINK_MASS, ((CABLE_LINK / 2, 0, 0), CABLE_LINK_INERTIA))
        _capsule_x(stage, f"{root}/cable_{k:02d}/insulation", CABLE_LINK, CABLE_R, mats["wire_blue"])
        _joint_between(stage, f"{root}/joints/cable_j{k:02d}", prev, prev_frame, body.GetPath(), (p, r), p, r,
                       joint_axis(k))
        prev, prev_frame = body.GetPath(), (p, r)
    end = _body(stage, f"{root}/wire_end", end_p, end_r, WIRE_MASS)
    tail = end_p + end_r @ np.array([TAIL_X, 0, 0])
    _joint_between(stage, f"{root}/joints/cable_j{CABLE_N:02d}", prev, prev_frame, end.GetPath(), (end_p, end_r),
                   tail, end_r, joint_axis(CABLE_N))
    e = f"{root}/wire_end"
    end.GetPrim().AddAppliedSchema("PhysxRigidBodyAPI")
    end.GetPrim().CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(0.5)
    hi, hs = INSUL_W / 2, STRAND_W / 2
    _box(stage, f"{e}/insulation", (TAIL_X, 0.0), (-hi, hi), (-hi, hi), mats["wire_blue"])
    # The stripped end: STRAND_N links of bare strands. strand_00 slides on wire_end (the bundle's axial give, a
    # sprung prismatic joint, collapse only); the links bend on revolute joints (Y, Z, Y).
    strand_paths = []
    prev = end.GetPath()
    for k in range(STRAND_N):
        sp = end_p + end_r @ np.array([k * STRAND_LINK, 0, 0])
        body = _body(stage, f"{root}/strand_{k:02d}", sp, end_r, STRAND_LINK_MASS,
                     ((STRAND_LINK / 2, 0, 0), STRAND_LINK_INERTIA))
        _box(stage, f"{root}/strand_{k:02d}/copper", (0.0, STRAND_LINK), (-hs, hs), (-hs, hs), mats["copper"])
        kit.hide(stage, f"{root}/strand_{k:02d}/copper")
        kit.cylinder(stage, f"{root}/strand_{k:02d}/look", (STRAND_LINK / 2, 0, 0), hs * 0.98, STRAND_LINK + 0.2 * MM,
                     mats["copper"], axis="X")
        for j in range(6):  # outer strands, slightly proud, so the bundle reads as stranded copper up close
            a = j * math.pi / 3 + k * 0.5
            kit.cylinder(stage, f"{root}/strand_{k:02d}/s{j}", (STRAND_LINK / 2, 0.8 * hs * math.cos(a),
                                                                0.8 * hs * math.sin(a)), 0.3 * MM,
                         STRAND_LINK + 0.2 * MM, mats["tin"], axis="X")
        strand_paths.append(f"{root}/strand_{k:02d}/copper")
        if k == 0:
            joint = UsdPhysics.PrismaticJoint.Define(stage, f"{root}/joints/strand_slide")
            joint.CreateBody0Rel().SetTargets([end.GetPath()])
            joint.CreateBody1Rel().SetTargets([body.GetPath()])
            joint.CreateAxisAttr("X")
            joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            joint.CreateLowerLimitAttr(-STRAND_COLLAPSE_MAX)
            joint.CreateUpperLimitAttr(0.0)
            UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        else:
            j = _joint_between(stage, f"{root}/joints/strand_j{k:02d}", prev, (prev_sp, end_r), body.GetPath(),
                               (sp, end_r), sp, end_r, ("Y", "Z")[(k - 1) % 2])
            j.GetLowerLimitAttr().Set(-STRAND_BEND_LIMIT_DEG)
            j.GetUpperLimitAttr().Set(STRAND_BEND_LIMIT_DEG)
        prev, prev_sp = body.GetPath(), sp
    kit.friction(stage, root, strand_paths, static=FRICTION_SLEEVE, dynamic=FRICTION_SLEEVE, name="copper",
                 combine="max")
    kit.friction(stage, root, [f"{e}/insulation"], static=FRICTION_GRIP, dynamic=FRICTION_GRIP,
                 name="pvc", combine="max")
    kit.friction(stage, root, [f"{root}/cable_{k:02d}/insulation" for k in range(CABLE_N)], static=CABLE_FRICTION[0],
                 dynamic=CABLE_FRICTION[1], name="cable_pvc", combine="min")
    _offsets(stage)
    stage.GetRootLayer().Save()
    np.save(out_path.with_name("cable_route_q.npy"), np.asarray(q))
    print(f"wrote {out_path}")


def tool_posts() -> list[tuple[str, float, float]]:
    """(name, x along the tool from its tip, half-width of what it carries)."""
    return [("tool_front", -(BLADE_LEN + 20 * MM), SHAFT_W / 2),
            ("tool_rear", -140 * MM, HANDLE_W / 2)]


def make_tool(out_path: Path) -> None:
    """WAGO 210-720 operating tool (blade 3.5 x 0.5 mm, partially insulated shaft): one rigid body, origin at the
    blade's tip, +x toward the tip, the blade's 3.5 mm across along y. Box colliders; round looks."""
    root = "/op_tool"
    stage, root_prim = kit.new_stage(out_path, root)
    mats = {name: _material(stage, root, name, c) for name, c in COLORS.items()}
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    mass = UsdPhysics.MassAPI.Apply(root_prim)
    mass.CreateMassAttr(TOOL_MASS)
    x_shaft = -BLADE_LEN
    x_handle = x_shaft - SHAFT_LEN
    x_end = x_handle - HANDLE_LEN
    mass.CreateCenterOfMassAttr(Gf.Vec3f((x_handle + x_end) / 2 + 0.01, 0, 0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(2e-6, 6e-5, 6e-5))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
    root_prim.AddAppliedSchema("PhysxRigidBodyAPI")
    root_prim.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(0.5)
    root_prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(SOLVER_ITERATIONS)
    hb, hs, hh = BLADE_W / 2, SHAFT_W / 2, HANDLE_W / 2
    _box(stage, f"{root}/blade", (-BLADE_LEN, 0.0), (-hb, hb), (-BLADE_T / 2, BLADE_T / 2), mats["steel"])
    _box(stage, f"{root}/shaft", (x_handle, x_shaft), (-hs, hs), (-hs, hs), mats["steel"])
    _box(stage, f"{root}/handle", (x_end, x_handle), (-hh, hh), (-hh, hh), mats["tool_green"])
    for name in ("shaft", "handle"):
        kit.hide(stage, f"{root}/{name}")
    kit.cylinder(stage, f"{root}/look_bare", (x_shaft - SHAFT_BARE / 2, 0, 0), hs, SHAFT_BARE, mats["steel"], axis="X")
    ins = SHAFT_LEN - SHAFT_BARE
    kit.cylinder(stage, f"{root}/look_sleeve", (x_handle + ins / 2, 0, 0), hs + 0.3 * MM, ins, mats["tool_green"],
                 axis="X")
    kit.cylinder(stage, f"{root}/look_handle", ((x_handle + x_end) / 2, 0, 0), hh, HANDLE_LEN - 10 * MM,
                 mats["tool_green"], axis="X")
    kit.cylinder(stage, f"{root}/look_nose", (x_handle - 5 * MM, 0, 0), hh * 0.8, 10 * MM, mats["tool_grey"], axis="X")
    kit.cylinder(stage, f"{root}/look_cap", (x_end + 6 * MM, 0, 0), hh + 0.2 * MM, 12 * MM, mats["tool_grey"], axis="X")
    kit.friction(stage, root, [f"{root}/blade", f"{root}/shaft"], static=FRICTION_BLADE, dynamic=FRICTION_BLADE,
                 name="steel", combine="min")
    kit.friction(stage, root, [f"{root}/handle"], static=FRICTION_HANDLE, dynamic=FRICTION_HANDLE, name="grip",
                 combine="max")
    _offsets(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_holder(out_path: Path) -> None:
    """Two narrow posts the wire end lies on, sloping 15 deg down toward the ferrule: one under the collar, one under
    the insulation's tail. The finger pads close on the insulation between them."""
    root = "/wire_stand"
    stage, _ = kit.new_stage(out_path, root)
    mats = {name: _material(stage, root, name, c) for name, c in COLORS.items()}
    cx, cy = HOLDER_XY
    for name, xw, half in holder_posts():
        x = cx + xw * _C
        top = HOLDER_TOP - xw * _S - half / _C
        _box(stage, f"{root}/{name}", (x - 1.0 * MM, x + 1.0 * MM), (cy - 0.008, cy + 0.008), (TABLE_Z, top),
             mats["holder"])
        for k, sy in enumerate((-1, 1)):
            y0 = cy + sy * (half + 0.3 * MM)
            _box(stage, f"{root}/{name}_lip_{k}", (x - 1.0 * MM, x + 1.0 * MM), tuple(sorted((y0, y0 + sy * 2 * MM))),
                 (top, top + 1.2 * MM), mats["holder"])
    tx, ty = TOOL_XY
    for name, xw, half in tool_posts():
        x = tx + xw
        top = TOOL_Z - half
        _box(stage, f"{root}/{name}", (x - 1.0 * MM, x + 1.0 * MM), (ty - 0.010, ty + 0.010), (TABLE_Z, top),
             mats["holder"])
        for k, sy in enumerate((-1, 1)):
            y0 = ty + sy * (half + 0.3 * MM)
            _box(stage, f"{root}/{name}_lip_{k}", (x - 1.0 * MM, x + 1.0 * MM), tuple(sorted((y0, y0 + sy * 2 * MM))),
                 (top, top + 1.5 * MM), mats["holder"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_textures(out_dir)
    make_strip(out_dir / "strip.usda")
    make_wire(out_dir / "wire.usda")
    make_holder(out_dir / "holder.usda")
    make_tool(out_dir / "tool.usda")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets")

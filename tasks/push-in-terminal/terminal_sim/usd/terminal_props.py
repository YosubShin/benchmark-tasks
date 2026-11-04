"""Generate the props: a strip of WAGO TOPJOB S push-in terminal blocks on a DIN rail above a wiring duct, a ferruled
2.5 mm2 wire whose flexible conductor comes out of the duct, and the stand its end lies on.

Real parts: WAGO 2002-1201 (grey), 2002-1204 (blue, N) and 2002-1207 (green-yellow, PE) TOPJOB S 2-conductor
terminal blocks, 5.2 mm wide, Push-in CAGE CLAMP. Published (wago.com/us/p/2002-1201 and its datasheet): 5.2 x 48.5 x
39.5 mm (w x h x d), 32.9 mm above the rail's top, the rail's centre 19.9 mm from one end, strip length 10-12 mm,
0.25-4 mm2, "direct insertion of solid and fine-stranded conductors with ferrules without tools". (The 2002-1207
earth block's rail foot differs; here all six share the 2002-1201 profile.)

How the real one clamps (WAGO datasheet p12, WAGO animation "TOPJOB S For all Conductor Types"): each conductor entry
is a hole in the front face near the block's outer end, its axis tilted under 15 deg from straight-in so the conductor
moves toward the block's middle as it goes in, down to the busbar. A stainless clamping spring's leg rests across the
entry channel. A pushed-in conductor deflects the leg; the leg's edge points in the insertion direction, so pulling
the conductor drags the edge into it and wedges it against the busbar (easy in, hard out). To remove it, a
screwdriver in the operating slot next to the entry opens the leg.

Modelled (geometry):
- ``strip.usda``: one fixed-base articulation. ``base`` holds the mounting plate, the TS 35 rail, the wiring duct
  (collision; the wire leaves it through a slot in its bottom wall), the six housings and, per terminal, the entry
  block in the wire frame W: the channel's side walls, the busbar wall it is pressed against, the lower wall up to
  the spring's pocket, and the end stop. The collar of the ferrule is wider than the channel, so it stops at the
  mouth. ``leaf_k`` (k = 1..6): the clamping leg of terminal k on a revolute joint across the channel, pressed closed
  by the task's drive (the spring) against a hard stop. Its clamping edge stands LEAF_INTERFERENCE into the ferrule's
  path. Each lower entry holds a landed 1.5 mm2 wire, fixed to ``base`` (a visual tube, no collision):
  its ferrule collar, then the insulation bent down and back into the duct's top comb (``lower_wire_path``).
- ``wire.usda``: one fixed-base articulation: ``anchor`` (fixed in the duct, at the slot), CABLE_N capsule links of
  CABLE_LINK (the flexible conductor; revolute joints cycling about the two bending axes and the twist axis, each with
  a spring toward the routed shape, set by the task), and ``wire_end``: the last 64 mm of insulation, the ferrule's
  collar and sleeve, one rigid body (root of the old rigid wire: collar's front face, +x toward the ferrule tip).
  The cable's starting route (duct slot -> desk -> up onto the stand) is solved here (``solve_route``) and is the
  joints' zero.
- ``holder.usda``: two posts the wire end lies on.

Frames. Desk frame: desk top z = 0, arm bases at (0, +-0.305, 0) (the station harness raises it by 0.75 m). The
station table is 2.5 mm below desk z = 0 (TABLE_Z). Wire frame W of terminal k: origin at the centre of its upper
entry's mouth (M0_k), +x (u) along the entry channel into the terminal (toward the plate and down, ENTRY_DEG below
horizontal), +y along the rail, +z (v) perpendicular (toward the plate and up). The busbar is on the +v side, the
clamping leg on the -v side.

Inferred (to measure on a bought part): the entry angle and channel size, the leg's length, angle, spring torque,
the ferrule's friction and the cable's bending stiffness.

Usage: python -m terminal_sim.usd.terminal_props [<out dir>]   (default: terminal_sim/assets)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from terminal_sim.spec import FRONT_X, HOLDER_TOP, HOLDER_XY, N_TERMINALS, PITCH, TERMINAL_KINDS, TOP_Z, terminal_y
from terminal_sim.usd import usd_kit as kit

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
"""N*m/rad (inferred). Needs 480 Hz physics."""
SPRING_PRELOAD_RAD = 0.25
"""Drive target this far beyond the closed stop: 0.025 N*m on the stop. The pull-out probe holds 50 N with 0.3 mm of
creep at this setting; 0.15 rad let the wire out at about 50 N, and 0.05 N*m/rad with 0.15 rad at about 23 N."""
SPRING_DAMPING = 0.002

# Wire: 2.5 mm2, insulated ferrule WAGO 216-246 (DIN 46228-4, blue) -----------------------------------------
SLEEVE_LEN = 10.0 * MM
SLEEVE_W = 2.5 * MM
COLLAR_LEN = 7.0 * MM
COLLAR_W = 4.6 * MM
INSUL_LEN = 64.0 * MM
INSUL_W = 3.6 * MM
WIRE_MASS = 0.03
"""The rigid end (80 mm): real ~3 g; heavier for a stable contact against the 0.2 kg leg."""
TAIL_X = -(COLLAR_LEN + INSUL_LEN)
"""The rigid end's tail (where the flexible cable joins it), wire-end frame."""
GRIP_X = -32.0 * MM
FRICTION_SLEEVE = 0.6
"""Tinned copper on the stainless leg and the copper busbar (inferred). The wedge that holds the wire needs it."""
FRICTION_GRIP = 2.5
"""Insulation and collar against the finger pads (combine max). Raised from 1.5: the station's gripper model squeezes
with about 9 N per finger (5000 N/m on a 1.8 mm closing), and at 1.5 the wire slid back in the pads at about 5 N of
push, less than the clamp needs to seat. Not measured on the real gripper."""

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
    "wire_green": (0.03, 0.30, 0.05),
    "dark": (0.06, 0.06, 0.07),
    "plate": (0.72, 0.74, 0.70),
    "rail": (0.80, 0.81, 0.83),
    "tin": (0.80, 0.81, 0.83),
    "copper": (0.80, 0.45, 0.20),
    "collar_blue": (0.05, 0.22, 0.80),
    "wire_blue": (0.01, 0.16, 0.45),
    "duct": (0.55, 0.57, 0.58),
    "holder": (0.25, 0.25, 0.27),
    "white": (0.95, 0.95, 0.93),
    "steel": (0.65, 0.67, 0.70),
    "brown": (0.15, 0.05, 0.015),
    "black": (0.04, 0.04, 0.04),
    "wire_grey": (0.25, 0.27, 0.29),
}
LOWER_WIRES = {1: "brown", 2: "black", 3: "wire_grey", 4: "wire_blue", 5: "green_yellow", 6: "brown"}
"""The wires already landed in the lower entries (IEC 60445 colours: L1 brown, L2 black, L3 grey, N blue, PE green-
yellow), 1.5 mm2 H07V-K (2.9 mm across) with black insulated ferrules (DIN 46228-4 colour code for 1.5 mm2)."""
LOWER_INSUL_W = 2.9 * MM
LOWER_COLLAR_W = 4.0 * MM
LOWER_COLLAR_LEN = 7.0 * MM
LOWER_BEND_R = 12.0 * MM
"""About 4x the insulation's diameter, the usual minimum bend radius for fixed PVC wiring."""
LOWER_STRAIGHT = 9.0 * MM
"""Straight run out of the entry (from the mouth, along its axis) before the first bend."""
LOWER_BACK_DEG = 330.0
"""Heading (in the x-z plane, 180 = toward the robot, 270 = down) at the S-bend's inflection: the wire leaves the
entry sloping 15 deg down toward the robot, turns down and back toward the plate, then straight down into the duct."""
LOWER_DUCT_IN = 3.5 * MM
"""Where the wires drop into the duct's top comb: this far behind the duct's front face."""
LOWER_DUCT_DEPTH = 12.0 * MM
"""How far below the duct's top the visible tube ends (inside the solid duct)."""

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
    tip_v = CH_V[1] - SLEEVE_W + LEAF_INTERFERENCE
    a = math.radians(LEAF_REST_DEG)
    h = LEAF_T / 2
    return (LEAF_U - LEAF_L * math.cos(a) + h * math.sin(a), tip_v - LEAF_L * math.sin(a) - h * math.cos(a))


def wire_home_root(n: int) -> tuple:
    """Wire end pose fully inserted into terminal n: collar on the mouth, pressed on the busbar."""
    return w_to_world(n, 0.0, 0.0, CH_V[1] - SLEEVE_W / 2) + W_QUAT


def wire_aim_root(n: int) -> tuple:
    """Where to aim the wire end on terminal n's entry axis: the channel's centre line."""
    return w_to_world(n, 0.0, 0.0, (CH_V[0] + CH_V[1]) / 2) + W_QUAT


def wire_start_root() -> tuple:
    """Wire end pose lying on its stand: ferrule toward the panel (+x), sloping down ENTRY_DEG like the entry."""
    return (HOLDER_XY[0], HOLDER_XY[1], HOLDER_TOP + 0.0002) + W_QUAT


def holder_posts() -> list[tuple[str, float, float]]:
    """(name, x along the wire end from its root, half-width of what it carries)."""
    return [("front", -4.0 * MM, COLLAR_W / 2), ("rear", TAIL_X + 6 * MM, INSUL_W / 2)]


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


# Pre-landed lower wires ------------------------------------------------------------------------------------------
def lower_wire_path(step=1.0 * MM) -> list[tuple[float, float]]:
    """(x, z) points of a lower-entry wire's centre line, from inside its collar to inside the duct: straight out of
    the entry along its axis, a bend of LOWER_BEND_R turning down and back toward the plate, a short straight at
    LOWER_BACK_DEG, a second bend of LOWER_BEND_R back to vertical, then straight down through the duct's top comb.
    The same for every terminal (in the plane y = the terminal's y)."""
    zb = BOT_Z + MOUTH_FROM_END
    out_dir = np.array([-_C, -_S])
    p0 = np.array([FRONT_X, zb]) + out_dir * (LOWER_COLLAR_LEN - 1.0 * MM)
    p1 = np.array([FRONT_X, zb]) + out_dir * LOWER_STRAIGHT
    r = LOWER_BEND_R
    a0, a1 = math.radians(180.0 + ENTRY_DEG), math.radians(LOWER_BACK_DEG)
    x_target = DUCT_X[0] + LOWER_DUCT_IN
    dx_arcs = r * (math.sin(a1) - math.sin(a0)) + r * (1.0 + math.sin(a1))
    straight = (x_target - p1[0] - dx_arcs) / math.cos(a1)
    assert straight >= 0.0, straight
    pts = [p0, p1]
    # first bend: counter-clockwise (heading a0 -> a1), centre to the heading's left
    c = p1 + r * np.array([-math.sin(a0), math.cos(a0)])
    n = max(2, int(r * (a1 - a0) / step))
    for t in np.linspace(a0, a1, n + 1)[1:]:
        pts.append(c + r * np.array([math.sin(t), -math.cos(t)]))
    d1 = np.array([math.cos(a1), math.sin(a1)])
    pts.append(pts[-1] + d1 * straight)
    # second bend: clockwise (heading a1 -> 270 deg), centre to the heading's right
    c = pts[-1] + r * np.array([math.sin(a1), -math.cos(a1)])
    a2 = math.radians(270.0)
    n = max(2, int(r * (a1 - a2) / step))
    for t in np.linspace(a1, a2, n + 1)[1:]:
        pts.append(c + r * np.array([-math.sin(t), math.cos(t)]))
    end_z = DUCT_Z[1] - LOWER_DUCT_DEPTH
    assert pts[-1][1] > end_z, pts[-1]
    pts.append(np.array([pts[-1][0], end_z]))
    return [(float(x), float(z)) for x, z in pts]


def _tube(stage, path, centre, radius, material, sides=12, sectors=None, collide=False):
    """A tube mesh along the 3-D polyline ``centre`` (it must stay in one plane y = const); ``sectors`` limits it to
    those of the ``sides`` faces around (for a stripe). Smooth normals; open ends."""
    centre = [np.asarray(p, float) for p in centre]
    rings = []
    for k, p in enumerate(centre):
        a = centre[max(k - 1, 0)]
        b = centre[min(k + 1, len(centre) - 1)]
        t = (b - a) / np.linalg.norm(b - a)
        side = np.array([0.0, 1.0, 0.0])
        nrm = np.cross(t, side)
        nrm /= np.linalg.norm(nrm)
        rings.append([(math.cos(2 * math.pi * j / sides) * side + math.sin(2 * math.pi * j / sides) * nrm)
                      for j in range(sides)])
    pts, normals = [], []
    for p, ring in zip(centre, rings):
        for d in ring:
            pts.append(tuple(p + radius * d))
            normals.append(tuple(d))
    faces = []
    for k in range(len(centre) - 1):
        for j in (range(sides) if sectors is None else sectors):
            j1 = (j + 1) % sides
            faces.append([k * sides + j, k * sides + j1, (k + 1) * sides + j1, (k + 1) * sides + j])
    mesh = _mesh(stage, path, pts, faces, material)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in normals]))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if collide:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    return mesh


def lower_wire(stage, path, n, mats) -> None:
    """Terminal n's pre-landed lower wire: a black ferrule collar on the entry's mouth, the insulation (its IEC colour)
    bent down and back into the duct. Fixed with the strip, visual only: with convex colliders on them, the scripted
    demo's line-up was pushed 8 mm sideways (most likely the loose cable, hanging from the raised wire end toward its
    duct slot, resting on the X1:5-X1:6 bends) and the official run scored 0.75 twice."""
    y = terminal_y(n)
    zb = BOT_Z + MOUTH_FROM_END
    out_dir = np.array([-_C, 0.0, -_S])
    mouth_b = np.array([FRONT_X, y, zb])
    UsdGeom.Xform.Define(stage, path)
    collar = [mouth_b + out_dir * 0.2 * MM, mouth_b + out_dir * LOWER_COLLAR_LEN]
    _tube(stage, f"{path}/collar", collar, LOWER_COLLAR_W / 2, mats["dark"])
    centre = [np.array([x, y, z]) for x, z in lower_wire_path()]
    r = LOWER_INSUL_W / 2
    kind = LOWER_WIRES[n]
    if kind == "green_yellow":  # green insulation with a yellow stripe along it
        _tube(stage, f"{path}/insulation", centre, r, mats["wire_green"], sectors=range(0, 7))
        _tube(stage, f"{path}/stripe", centre, r, mats["yellow_stripe"], sectors=range(7, 12))
    else:
        _tube(stage, f"{path}/insulation", centre, r, mats[kind])


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
        for k, (z0, z1) in enumerate(((TOP_Z - 9.5 * MM, TOP_Z - 6.5 * MM), (BOT_Z + 6.5 * MM, BOT_Z + 9.5 * MM))):
            _box(stage, f"{tp}_slot_{k}", (FRONT_X - 0.2 * MM, FRONT_X), (y - 0.9 * MM, y + 0.9 * MM), (z0, z1),
                 mats["dark"], collide=False)
        zm = TOP_Z - MOUTH_FROM_END
        _box(stage, f"{tp}_mouth_top", (FRONT_X - 0.15 * MM, FRONT_X), (y - hw, y + hw), (zm - 1.5 * MM, zm + 1.5 * MM),
             mats["dark"], collide=False)
        zb = BOT_Z + MOUTH_FROM_END
        _box(stage, f"{tp}_mouth_bot", (FRONT_X - 0.15 * MM, FRONT_X), (y - hw, y + hw), (zb - 1.5 * MM, zb + 1.5 * MM),
             mats["dark"], collide=False)
        # The wire already landed in the lower entry: ferrule collar, then the insulation bent down and back into
        # the duct's top comb (fixed, visual only; see lower_wire_path).
        lower_wire(stage, f"{tp}_lower_wire", n, mats)
        e = f"{tp}_entry"
        _xform(stage, e, mouth(n), ENTRY_DEG)
        _box(stage, f"{e}/side_a", (FUNNEL_U, BLOCK_U), (hw, PITCH / 2 - 0.05 * MM), BLOCK_V, mats[kind])
        _box(stage, f"{e}/side_b", (FUNNEL_U, BLOCK_U), (-PITCH / 2 + 0.05 * MM, -hw), BLOCK_V, mats[kind])
        _box(stage, f"{e}/side_a_mouth", (0.0, FUNNEL_U), (hw + FUNNEL_Y, PITCH / 2 - 0.05 * MM), BLOCK_V, mats[kind])
        _box(stage, f"{e}/side_b_mouth", (0.0, FUNNEL_U), (-PITCH / 2 + 0.05 * MM, -hw - FUNNEL_Y), BLOCK_V, mats[kind])
        _box(stage, f"{e}/busbar", (FUNNEL_U, BLOCK_U), (-hw, hw), (CH_V[1], BLOCK_V[1]), mats["copper"])
        _box(stage, f"{e}/busbar_mouth", (0.0, FUNNEL_U), (-hw, hw), (CH_V[1] + FUNNEL_V, BLOCK_V[1]), mats[kind])
        _box(stage, f"{e}/lower", (FUNNEL_U, pivot_u - 0.5 * MM), (-hw, hw), (BLOCK_V[0], CH_V[0]), mats[kind])
        _box(stage, f"{e}/lower_mouth", (0.0, FUNNEL_U), (-hw, hw), (BLOCK_V[0], CH_V[0] - FUNNEL_V), mats[kind])
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
        _box(stage, f"{lp}/arm/plate", (0.0, LEAF_L), (-LEAF_WIDTH / 2, LEAF_WIDTH / 2), (-LEAF_T / 2, LEAF_T / 2),
             mats["steel"])
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
    hs, hc, hi = SLEEVE_W / 2, COLLAR_W / 2, INSUL_W / 2
    _box(stage, f"{e}/sleeve", (0.0, SLEEVE_LEN), (-hs, hs), (-hs, hs), mats["tin"])
    _box(stage, f"{e}/collar", (-COLLAR_LEN, 0.0), (-hc, hc), (-hc, hc), mats["collar_blue"])
    _box(stage, f"{e}/insulation", (TAIL_X, -COLLAR_LEN), (-hi, hi), (-hi, hi), mats["wire_blue"])
    kit.friction(stage, root, [f"{e}/sleeve"], static=FRICTION_SLEEVE, dynamic=FRICTION_SLEEVE, name="tin",
                 combine="max")
    kit.friction(stage, root, [f"{e}/insulation", f"{e}/collar"], static=FRICTION_GRIP, dynamic=FRICTION_GRIP,
                 name="pvc", combine="max")
    kit.friction(stage, root, [f"{root}/cable_{k:02d}/insulation" for k in range(CABLE_N)], static=CABLE_FRICTION[0],
                 dynamic=CABLE_FRICTION[1], name="cable_pvc", combine="min")
    _offsets(stage)
    stage.GetRootLayer().Save()
    np.save(out_path.with_name("cable_route_q.npy"), np.asarray(q))
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
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_textures(out_dir)
    make_strip(out_dir / "strip.usda")
    make_wire(out_dir / "wire.usda")
    make_holder(out_dir / "holder.usda")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets")

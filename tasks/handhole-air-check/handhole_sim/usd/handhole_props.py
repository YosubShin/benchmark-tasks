"""Generate the props (sample a handhole's air through its pull slot, open only on a clean reading) as USD.

Parts (each follows a real product; "inferred" values are not published and are to measure). References are in
the README.

- Handhole (`box.usda`), a fixed-base articulation: a Hubbell Quazite PG1118 box (sales drawing PG1118BA12:
  top 514 x 340 mm outside, cover recess 464 x 289 mm, opening below the ledge 400 x 225 mm, two 3/8-7 EZ-nut
  inserts at diagonal corners) with its two cover bolts (PG1118CA00 parts list: C080173 3/8-7 x 3.00 in hex auger
  bolt, C080210 washer 1.0 in OD x 0.08 in). Each bolt is a screw: a prismatic "slider" on the box and the bolt
  turning on it; the task couples rise to turn (7 threads per inch, 3.63 mm per turn) while the thread is
  engaged, so the washer clamps the cover. Simplifications: the box is cut to 75 mm above the desk (real 12 in
  deep, set in the ground); the thread engages for ENGAGE_TURNS (real: about 6-12 turns, inferred from the 3 in
  bolt through a 1-3/4 in cover); the 1-1/4 in x 1/2 in counterbore is left out so the head stands on the cover
  (the YAM fingers would not fit the 32 mm bore); the hex head (9/16 in, 14.3 mm across flats) is a square of
  14.3 mm and 12 mm tall (real about 6 mm) so a fingertip pinch clears the washer.
- Cover (`cover.usda`), a free rigid body: Quazite PG1118CA00 / HA00, 460 x 286 x 44 mm, two 5/8 in bolt holes at
  diagonal corners (15-3/4 x 8-7/8 in spacing), and the 1/2 x 2 in pull slot with a 3/16 in centre pin, centred on
  the short axis, 6-9/16 in (167 mm) from the centre. Real mass 12 kg (27 lb); here 0.6 kg (a declared light
  stand-in). Tan polymer concrete.
- Monitor (`monitor.usda`), a floating articulation: a Honeywell BW GasAlertMax XT II (integral pump;
  131 x 70 x 52 mm, 328 g, user manual p.87) lying face up on a block, with its sample hose (BW Tygon, 1/4 in OD,
  1/8 in ID; sampling brochure) as a chain of rigid 50 mm links on alternating hinges, an inline filter the robot
  holds (20 x 20 x 40 mm; size and position inferred), and the last 100 mm of hose (after a 30 mm rigid stub) to the open end as one rigid
  link on a two-axis gimbal at the stub (so it hangs straight down whatever the
  filter's pitch in the fingers). A real 1/4 in Tygon end hangs roughly straight over this length; the three-hinge
  version swung and buckled against the slot walls. The case is yellow with a black bezel round the LCD, as on the
  real unit; its two visual alarm bars (red) sit at the end of the face and flash in alarm (task rule).
  Real hoses are 10-75 ft; 0.63 m here.
- Stand (`stand.usda`), static: the meter's low block and the pad the filter lies on.

Frames: everything is written in the env (world) frame; desk top z = 0.

Usage: python -m handhole_sim.usd.handhole_props handhole_sim/assets
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from handhole_sim.usd import usd_kit as kit


def _capsule(stage, path, p0, p1, radius, material):
    """A collision capsule from p0 to p1 (axis-aligned segments only)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    axis = "XYZ"[int(np.argmax(np.abs(d)))]
    cap = UsdGeom.Capsule.Define(stage, path)
    cap.CreateAxisAttr(axis)
    cap.CreateRadiusAttr(radius)
    cap.CreateHeightAttr(float(np.linalg.norm(d)))
    UsdGeom.Xformable(cap).AddTranslateOp().Set(Gf.Vec3d(*((p0 + p1) / 2)))
    UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    return cap


def _body(stage, path, pos, quat, mass):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    x, y, z, w = quat
    xform.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(mass)
    return xform


def _joint(stage, path, body0, body1, axis, pos0, rot0, pos1, lower_deg, upper_deg):
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(str(body0))])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(str(body1))])
    joint.CreateAxisAttr(axis)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    x, y, z, w = rot0
    joint.CreateLocalRot0Attr(Gf.Quatf(float(w), float(x), float(y), float(z)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*pos1))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLowerLimitAttr(lower_deg)
    joint.CreateUpperLimitAttr(upper_deg)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    return joint

IN = 0.0254

# Box and cover (Quazite PG1118) -------------------------------------------------------------------
BX, BY = 0.30, -0.02
"""Box centre in the world. The long axis (18 in) runs along y."""
Z_TOP = 0.075
"""Cover top (flush with the box top). The box is cut to 75 mm so the hose carry stays inside the YAM's
fingers-down reach (above about 0.21 m the station IK cannot hold fingers straight down near the arm base)."""
COVER = (11.25 * IN, 18.125 * IN, 1.75 * IN)
"""x, y, thickness: 286 x 460 x 44 mm."""
COVER_MASS = 0.6
Z_COVER_BOTTOM = Z_TOP - COVER[2]
BOX_OUT = (13.375 * IN, 20.25 * IN)
RECESS = (11.375 * IN, 18.25 * IN)
OPENING = (8.875 * IN, 15.75 * IN)
FLOOR_T = 0.008
BOLT_DX, BOLT_DY = 8.875 * IN / 2, 15.75 * IN / 2
BOLTS = ((BX + BOLT_DX, BY + BOLT_DY), (BX - BOLT_DX, BY - BOLT_DY))
"""Bolt 0 (far, +y: arm A) and bolt 1 (near, -y: arm B), at diagonal corners as on the drawing."""
BOLT_HOLE = 0.016
SLOT_XY = (BX, BY + 6.5625 * IN)
SLOT = (0.5 * IN, 2.0 * IN)
"""Pull slot: 13 mm along x, 51 mm along y."""
PIN = 0.1875 * IN
PIN_Z = Z_TOP - 0.012
SLOT_OPENING_Y = SLOT_XY[1] - (SLOT[1] / 2 + PIN / 2) / 2
"""Centre of the slot's half toward the cover's centre (beside the pin), where the hose goes in."""

PITCH = IN / 7
ENGAGE_TURNS = 1.5
WASHER = (1.0 * IN, 0.002)
HEAD = (0.0143, 0.012)
SHANK = 0.0095
SHANK_LEN = COVER[2] + ENGAGE_TURNS * PITCH + 0.002
"""Shank below the washer: through the cover and the engaged thread."""
SLIDER_MAX = 0.25
BOLT_MASS = 0.03

INTERIOR = (OPENING[0] / 2, OPENING[1] / 2, FLOOR_T, Z_COVER_BOTTOM)
"""Half x, half y, floor z, ceiling z of the box's air space (below the ledge)."""

# Monitor and hose -----------------------------------------------------------------------------------
METER = (0.131, 0.070, 0.052)
METER_MASS = 0.328
HOSE_R = 0.25 * IN / 2
LINK_L = 0.05
LINK_MASS = 0.004
FILTER = (0.040, 0.020, 0.020)
FILTER_MASS = 0.015
STUB_L = 0.030
"""The hose's first 30 mm past the filter's outlet, rigid with the filter (it keeps the free end's pivot clear of the
fingers that hold the filter)."""
TAIL_L = 0.10
TAIL_MASS = 0.003
HOSE_HEADINGS = (0, 0, 90, 90, 180, 180, 180, 180, 180, 180)
"""Heading (deg about z) of each hose link from the meter to the filter: a U lying at the hose height (the meter
behind the box's far-left corner, in the top camera's view; the filter beside the box, pointing toward the robot)."""
HOSE_AXES = ("Z", "Y", "Z", "Y", "Z", "Y", "Z", "Y", "Z", "Y")
FILTER_XY = (0.37, 0.38)
FILTER_DIR = math.cos(math.radians(HOSE_HEADINGS[-1]))
"""+1 or -1: the filter (and the tail past it) lies along this x direction from FILTER_XY."""
"""Where the filter starts (its hose end); it lies along the last link's heading, the tail sticking out past the pad."""
TAIL_N = 1
Z_HOSE = 0.040
"""Height of the hose and the filter's axis at rest: the filter lies on a pad (top 30 mm), the meter on a low block."""
PAD_TOP = Z_HOSE - FILTER[2] / 2
LINK_LIMIT_DEG = 110.0
TAIL_LIMIT_DEG = 175.0
"""The tail's hinge at the filter: free enough to hang straight down whatever the filter's pitch in the fingers."""


COLORS = {
    "concrete": (0.60, 0.52, 0.39), "concrete_dark": (0.45, 0.39, 0.29), "steel": (0.74, 0.75, 0.78),
    "meter_black": (0.08, 0.08, 0.09), "meter_yellow": (0.95, 0.75, 0.08), "hose": (0.80, 0.86, 0.88),
    "lcd_bg": (0.66, 0.72, 0.58), "lcd_alarm": (0.85, 0.30, 0.25), "lcd_seg": (0.03, 0.03, 0.03),
    "alarm_off": (0.22, 0.22, 0.24),
    "filter": (0.92, 0.92, 0.90), "stand": (0.40, 0.40, 0.42), "black": (0.05, 0.05, 0.05),
}

DISPLAY_STATES = ("room", "go", "nogo")
READINGS = {
    # BW order on the LCD: H2S, CO, O2, LEL. Room air, and the two seeded box atmospheres.
    "room": ("0", "0", "20.9", "0", False),
    "go": ("0", "0", "20.9", "0", False),  # clean box air reads like room air, as on a real monitor
    "nogo": ("18", "5", "20.4", "0", True),
}
"""Alarm low setpoints (BW Max XT II manual p.55): H2S 10 ppm, CO 35 ppm, O2 19.5 %, LEL 10 %."""


ALARM_BAR = (0.028, 0.026)
"""Each of the two visual alarm lenses on the face (meter frame x, y size), side by side at the end away from the
inlet (the real unit's wide-angle alarm bars; enlarged so they read at the policy camera's 320 x 176)."""


def _emissive(stage, root, name, color):
    path = f"{root}/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _mats(stage, root, names):
    out = {}
    for name in names:
        path = f"{root}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[name]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        out[name] = material
    return out


def _plate_with_holes(stage, path, x_range, y_range, z_range, holes, material):
    """A plate (horizontal) with rectangular holes, as a grid of boxes. holes: (cx, cy, sx, sy)."""
    xs = sorted({x_range[0], x_range[1], *[h[0] - h[2] / 2 for h in holes], *[h[0] + h[2] / 2 for h in holes]})
    ys = sorted({y_range[0], y_range[1], *[h[1] - h[3] / 2 for h in holes], *[h[1] + h[3] / 2 for h in holes]})
    zc, zt = (z_range[0] + z_range[1]) / 2, z_range[1] - z_range[0]
    k = 0
    for i in range(len(xs) - 1):  # merge cells along y within each x column
        x0, x1 = xs[i], xs[i + 1]
        run = None
        for j in range(len(ys) - 1):
            y0, y1 = ys[j], ys[j + 1]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            in_hole = any(abs(cx - h[0]) < h[2] / 2 and abs(cy - h[1]) < h[3] / 2 for h in holes)
            if not in_hole:
                run = (run[0], y1) if run else (y0, y1)
            if (in_hole or j == len(ys) - 2) and run:
                kit.box(stage, f"{path}_{k}", ((x0 + x1) / 2, (run[0] + run[1]) / 2, zc),
                        (x1 - x0, run[1] - run[0], zt), material)
                k += 1
                run = None


def _ring(stage, path, outer, inner, z_range, center, material):
    """Four walls between an outer and an inner rectangle (x, y sizes) about center."""
    cx, cy = center
    zc, zt = (z_range[0] + z_range[1]) / 2, z_range[1] - z_range[0]
    wx, wy = (outer[0] - inner[0]) / 2, (outer[1] - inner[1]) / 2
    for k, s in enumerate((-1, 1)):
        kit.box(stage, f"{path}_x{k}", (cx + s * (inner[0] / 2 + wx / 2), cy, zc), (wx, outer[1], zt), material)
        kit.box(stage, f"{path}_y{k}", (cx, cy + s * (inner[1] / 2 + wy / 2), zc), (inner[0], wy, zt), material)


def _thin(stage, prefix, offset=0.002):
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(offset)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def make_box(out_path: Path) -> None:
    root = "/box"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(16)
    m = _mats(stage, root, ("concrete", "concrete_dark", "steel"))
    base = kit.rigid_body(stage, f"{root}/body", (0, 0, 0), 20.0)
    kit.fix_to_world(stage, root, base.GetPath())
    p = f"{root}/body"
    kit.box(stage, f"{p}/floor", (BX, BY, FLOOR_T / 2), (BOX_OUT[0], BOX_OUT[1], FLOOR_T), m["concrete_dark"])
    _ring(stage, f"{p}/lower", BOX_OUT, OPENING, (FLOOR_T, Z_COVER_BOTTOM), (BX, BY), m["concrete"])
    _ring(stage, f"{p}/upper", BOX_OUT, (RECESS[0], RECESS[1]), (Z_COVER_BOTTOM, Z_TOP), (BX, BY), m["concrete"])
    # Bolt corners: the ledge is solid where the EZ-nut inserts sit (the opening is chamfered there on the drawing).
    for k, (x, y) in enumerate(BOLTS):
        sx = 1 if x > BX else -1
        sy = 1 if y > BY else -1
        cx = x - sx * 0.01
        cy = y - sy * 0.01
        kit.box(stage, f"{p}/boss{k}", (cx, cy, (FLOOR_T + Z_COVER_BOTTOM) / 2), (0.06, 0.06, Z_COVER_BOTTOM - FLOOR_T),
                m["concrete"])
    for k, (x, y) in enumerate(BOLTS):
        slider = kit.rigid_body(stage, f"{root}/slider{k}", (x, y, Z_TOP), 0.01)
        bolt = kit.rigid_body(stage, f"{root}/bolt{k}", (x, y, Z_TOP), BOLT_MASS)
        b = f"{root}/bolt{k}"
        kit.box(stage, f"{b}/washer", (0, 0, WASHER[1] / 2), (WASHER[0], WASHER[0], WASHER[1]), m["steel"])
        kit.box(stage, f"{b}/head", (0, 0, WASHER[1] + HEAD[1] / 2), (HEAD[0], HEAD[0], HEAD[1]), m["steel"])
        kit.box(stage, f"{b}/shank", (0, 0, -SHANK_LEN / 2), (SHANK, SHANK, SHANK_LEN), m["steel"])
        kit.box(stage, f"{b}/mark", (HEAD[0] / 2 - 0.002, 0, WASHER[1] + HEAD[1] + 0.0003), (0.004, 0.003, 0.0006),
                m["concrete_dark"], collide=False)
        kit.prismatic(stage, f"{root}/joints/slider{k}", base.GetPath(), slider.GetPath(), "Z", (x, y, Z_TOP), 0.0,
                      SLIDER_MAX)
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/joints/bolt{k}")
        joint.CreateBody0Rel().SetTargets([slider.GetPath()])
        joint.CreateBody1Rel().SetTargets([bolt.GetPath()])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    kit.friction(stage, root, [f"{root}/bolt{k}" for k in range(2)], static=1.0, dynamic=0.9, name="steel_grip")
    _thin(stage, f"{root}/bolt")
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_cover(out_path: Path) -> None:
    root = "/cover"
    stage, root_prim = kit.new_stage(out_path, root)
    m = _mats(stage, root, ("concrete", "concrete_dark", "steel"))
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(COVER_MASS)
    holes = [(x - BX, y - BY, BOLT_HOLE, BOLT_HOLE) for x, y in BOLTS]
    holes.append((SLOT_XY[0] - BX, SLOT_XY[1] - BY, SLOT[0], SLOT[1]))
    hx, hy, t = COVER[0] / 2, COVER[1] / 2, COVER[2]
    _plate_with_holes(stage, f"{root}/plate", (-hx, hx), (-hy, hy), (-t / 2, t / 2), holes, m["concrete"])
    sx, sy = SLOT_XY[0] - BX, SLOT_XY[1] - BY
    kit.box(stage, f"{root}/pin", (sx, sy, PIN_Z - (Z_TOP - t / 2)), (SLOT[0] + 0.002, PIN, PIN), m["steel"])
    # Logo panel and tier stamp (visual).
    kit.box(stage, f"{root}/logo", (0.0, -0.01, t / 2 + 0.0004), (0.07, 0.17, 0.0008), m["concrete_dark"],
            collide=False)
    _thin(stage, root)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def cover_pose() -> tuple:
    return (BX, BY, Z_TOP - COVER[2] / 2, 0.0, 0.0, 0.0, 1.0)


# Monitor -----------------------------------------------------------------------------------------------
def _qz(deg):
    a = math.radians(deg) / 2
    return np.array([0.0, 0.0, math.sin(a), math.cos(a)])


def _qy(deg):
    a = math.radians(deg) / 2
    return np.array([0.0, math.sin(a), 0.0, math.cos(a)])


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def _rot(q, v):
    qv = np.array([*v, 0.0])
    c = np.array([-q[0], -q[1], -q[2], q[3]])
    return _qmul(_qmul(q, qv), c)[:3]


def hose_layout():
    """Rest poses (world pos, quat) of meter, hose links, filter and tail links; plus each joint's rest rotation
    (in its parent body's frame) and axis. Built backwards from the filter in its fork."""
    n = len(HOSE_HEADINGS)
    top = np.array([FILTER_XY[0], FILTER_XY[1], Z_HOSE])
    starts = [None] * n
    end = top.copy()
    for k in range(n - 1, -1, -1):
        h = math.radians(HOSE_HEADINGS[k])
        starts[k] = end - LINK_L * np.array([math.cos(h), math.sin(h), 0.0])
        end = starts[k]
    inlet = starts[0]
    h0 = HOSE_HEADINGS[0]
    meter_q = _qz(h0)
    meter_pos = inlet - _rot(meter_q, (METER[0] / 2, 0, 0))
    bodies = [("meter", meter_pos, meter_q)]
    joints = []
    prev_q, prev_h = meter_q, h0
    for k in range(n):
        q = _qz(HOSE_HEADINGS[k])
        bodies.append((f"hose_{k:02d}", starts[k], q))
        joints.append((HOSE_AXES[k], _qz(HOSE_HEADINGS[k] - prev_h)))
        prev_q, prev_h = q, HOSE_HEADINGS[k]
    # Filter: lies flat, continuing the last link's heading; the tail continues straight past the pad's edge. It hangs
    # from the filter on a gimbal (a free hinge about the filter's local y, then one about the gimbal's local z), so it
    # hangs straight down whatever way the filter turns in the fingers.
    ident = np.array([0.0, 0.0, 0.0, 1.0])
    bodies.append(("filter", top, prev_q))
    joints.append(("Z", ident))
    p = top + _rot(prev_q, (FILTER[0] + STUB_L, 0, 0))
    bodies.append(("tail_gimbal", p.copy(), prev_q))
    joints.append(("Y", ident))
    for k in range(TAIL_N):
        bodies.append((f"tail_{k}", p.copy(), prev_q))
        joints.append(("Z", ident))
        p = p + _rot(prev_q, (TAIL_L, 0, 0))
    return bodies, joints


JOINT_NAMES = tuple([f"hose_j{k:02d}" for k in range(len(HOSE_HEADINGS))] + ["filter_j"]
                    + ["tail_j0"] + [f"tail_j{k + 1}" for k in range(TAIL_N)])
TIP_BODY = f"tail_{TAIL_N - 1}"


def _quad_up(stage, path, center, size, material, u_range):
    """A horizontal textured quad facing +z: text right along local +x, text up along local -y."""
    cx, cy, cz = center
    sx, sy = size
    x0, x1, y0, y1 = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2
    pts = [(x1, y0, cz), (x0, y0, cz), (x0, y1, cz), (x1, y1, cz)]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    u0, u1 = u_range
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(u0, 0), Gf.Vec2f(u1, 0), Gf.Vec2f(u1, 1), Gf.Vec2f(u0, 1)]))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


# The LCD as geometry: seven-segment digits made of thin visual boxes, one group per display state (the task shows
# one group). A textured version was replaced by another task's texture at render time (renderer texture cache).
SEGMENTS = {
    "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg", "5": "acdfg", "6": "acdefg", "7": "abc",
    "8": "abcdefg", "9": "abcdfg",
}
LCD_CENTER = (0.012, 0.0)
LCD_SIZE = (0.064, 0.054)
"""Meter frame: text reads along local +x, its up is local +y (the reviewer camera's view of the meter)."""


def _seven_seg(stage, path, text, x0, y0, h, material):
    """Draw text (digits and '.') with its lower-left corner at (x0, y0) in the meter's top plane."""
    w, t = 0.55 * h, 0.13 * h
    x = x0
    k = 0
    for ch in text:
        if ch == ".":
            kit.box(stage, f"{path}_{k}", (x + t / 2, y0 + t / 2, 0), (t, t, 0.0006), material, collide=False)
            x += 2 * t
            k += 1
            continue
        segs = {"a": (x + w / 2, y0 + h - t / 2, w, t), "d": (x + w / 2, y0 + t / 2, w, t),
                "g": (x + w / 2, y0 + h / 2, w, t), "b": (x + w - t / 2, y0 + 3 * h / 4, t, h / 2),
                "c": (x + w - t / 2, y0 + h / 4, t, h / 2), "e": (x + t / 2, y0 + h / 4, t, h / 2),
                "f": (x + t / 2, y0 + 3 * h / 4, t, h / 2)}
        for sg in SEGMENTS[ch]:
            cx, cy, sx, sy = segs[sg]
            kit.box(stage, f"{path}_{k}", (cx, cy, 0), (sx, sy, 0.0006), material, collide=False)
            k += 1
        x += w + 0.35 * h


def _lcd_geometry(stage, body, top_z, m):
    cx, cy = LCD_CENTER
    sx, sy = LCD_SIZE
    z = top_z + 0.0004
    kit.box(stage, f"{body}/lcd_panel", (cx, cy, z), (sx, sy, 0.0008), m["lcd_bg"], collide=False)
    # Quadrant dividers.
    kit.box(stage, f"{body}/lcd_div_x", (cx, cy, z + 0.0005), (0.0008, sy - 0.004, 0.0004), m["lcd_seg"],
            collide=False)
    kit.box(stage, f"{body}/lcd_div_y", (cx, cy - 0.004, z + 0.0005), (sx - 0.004, 0.0008, 0.0004), m["lcd_seg"],
            collide=False)
    h = 0.0075
    # Quadrants (BW order): H2S top-left, CO top-right, O2 bottom-left, LEL bottom-right; a band at the top for ALARM.
    origins = [(cx - sx / 2 + 0.003, cy + 0.007), (cx + 0.003, cy + 0.007), (cx - sx / 2 + 0.003, cy - 0.017),
               (cx + 0.003, cy - 0.017)]
    for state in DISPLAY_STATES:
        h2s, co, o2, lel, alarm = READINGS[state]
        g = f"{body}/lcd_{state}"
        xf = UsdGeom.Xform.Define(stage, g)
        xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, z + 0.0007))
        if alarm:
            kit.box(stage, f"{g}/alarm_band", (cx, cy + sy / 2 - 0.004, 0), (sx - 0.002, 0.006, 0.0005),
                    m["lcd_alarm"], collide=False)
            kit.box(stage, f"{g}/alarm_h2s", (cx - sx / 4, cy + 0.011, -0.0002), (sx / 2 - 0.002, h + 0.006, 0.0004),
                    m["lcd_alarm"], collide=False)
        for q, (val, (ox, oy)) in enumerate(zip((h2s, co, o2, lel), origins)):
            _seven_seg(stage, f"{g}/v{q}", val, ox, oy, h, m["lcd_seg"])


def make_monitor(out_dir: Path) -> None:
    root = "/monitor"
    out_path = out_dir / "monitor.usda"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    m = _mats(stage, root, ("meter_black", "meter_yellow", "hose", "filter", "black", "lcd_bg", "lcd_alarm",
                            "lcd_seg", "alarm_off"))
    m["alarm_on"] = _emissive(stage, root, "alarm_on", (1.0, 0.08, 0.05))
    bodies, joints = hose_layout()
    paths = {}
    for name, pos, q in bodies:
        mass = {"meter": METER_MASS, "filter": FILTER_MASS, "tail_gimbal": 0.001}.get(
            name, TAIL_MASS if name.startswith("tail") else LINK_MASS)
        body = _body(stage, f"{root}/{name}", pos, q, mass)
        if name == "tail_gimbal":  # no collider: give it an inertia
            UsdPhysics.MassAPI(body.GetPrim()).CreateDiagonalInertiaAttr(Gf.Vec3f(1e-7, 1e-7, 1e-7))
            paths[name] = body.GetPath()
            continue
        paths[name] = body.GetPath()
        b = f"{root}/{name}"
        if name == "meter":
            lx, ly, lz = METER
            kit.box(stage, f"{b}/case", (0, 0, 0), METER, m["meter_yellow"])
            cx, cy = LCD_CENTER
            kit.box(stage, f"{b}/bezel", (cx, cy, lz / 2 + 0.0002), (LCD_SIZE[0] + 0.012, LCD_SIZE[1] + 0.010, 0.0004),
                    m["meter_black"], collide=False)
            for k, sy in enumerate((-1, 1)):
                c = (-lx / 2 + 0.004 + ALARM_BAR[0] / 2, sy * (ALARM_BAR[1] / 2 + 0.002))
                for state in ("off", "on"):
                    kit.box(stage, f"{b}/alarm_{state}_{k}", (c[0], c[1], lz / 2 + (0.0012 if state == "on" else 0.0008)),
                            (ALARM_BAR[0], ALARM_BAR[1], 0.0008), m[f"alarm_{state}"], collide=False)
            kit.box(stage, f"{b}/inlet", (lx / 2 + 0.004, 0, 0), (0.008, 0.012, 0.012), m["black"], collide=False)
            _lcd_geometry(stage, b, lz / 2, m)
        elif name.startswith("hose"):
            _capsule(stage, f"{b}/tube", (0, 0, 0), (LINK_L, 0, 0), HOSE_R, m["hose"])
        elif name == "filter":
            kit.box(stage, f"{b}/can", (FILTER[0] / 2, 0, 0), FILTER, m["filter"])
            _capsule(stage, f"{b}/stub", (FILTER[0], 0, 0), (FILTER[0] + STUB_L, 0, 0), HOSE_R, m["hose"])
        else:
            _capsule(stage, f"{b}/tube", (0, 0, 0), (TAIL_L, 0, 0), HOSE_R, m["hose"])
    names = [b[0] for b in bodies]
    for k, (axis, rest) in enumerate(joints):
        parent, child = names[k], names[k + 1]
        if parent == "meter":
            pos0 = (METER[0] / 2, 0, 0)
        elif parent == "filter":
            pos0 = (FILTER[0] + STUB_L, 0, 0)
        elif parent == "tail_gimbal":
            pos0 = (0, 0, 0)
        elif parent.startswith("tail"):
            pos0 = (TAIL_L, 0, 0)
        else:
            pos0 = (LINK_L, 0, 0)
        lim = TAIL_LIMIT_DEG if JOINT_NAMES[k].startswith("tail") else LINK_LIMIT_DEG
        _joint(stage, f"{root}/joints/{JOINT_NAMES[k]}", paths[parent], paths[child], axis, pos0, tuple(rest),
               (0, 0, 0), -lim, lim)
    kit.friction(stage, root, [f"{root}/{n}" for n in names if n.startswith(("hose", "tail"))], static=0.15,
                 dynamic=0.12, name="tygon", combine="min")
    kit.friction(stage, root, [f"{root}/filter"], static=1.2, dynamic=1.0, name="grip")
    kit.friction(stage, root, [f"{root}/meter"], static=0.9, dynamic=0.8, name="rubber")
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_stand(out_path: Path) -> None:
    root = "/stand"
    stage, _ = kit.new_stage(out_path, root)
    m = _mats(stage, root, ("stand",))
    bodies, _ = hose_layout()
    mpos = bodies[0][1]
    block_top = mpos[2] - METER[2] / 2
    kit.box(stage, f"{root}/meter_block", (mpos[0], mpos[1], block_top / 2), (0.12, 0.09, block_top), m["stand"])
    # Pad under the filter (the tail sticks out past its edge).
    fx, fy = FILTER_XY
    kit.box(stage, f"{root}/pad", (fx + FILTER_DIR * FILTER[0] / 2, fy, PAD_TOP / 2), (0.05, 0.05, PAD_TOP), m["stand"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    make_box(out_dir / "box.usda")
    make_cover(out_dir / "cover.usda")
    make_monitor(out_dir)
    make_stand(out_dir / "stand.usda")

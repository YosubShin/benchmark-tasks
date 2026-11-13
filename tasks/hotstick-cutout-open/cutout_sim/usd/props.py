"""Generate the props of the hook-stick cutout task as USD: the cutout on its crossarm (one articulation, also carrying
the stick rack) and the hook stick (a free rigid body).

Real parts and the 1:2 scale (values marked "inferred" are not published and are to measure on the build):

- S&C Type XS fuse cutout, extra-heavy-duty pole-top style, 15 kV class (catalog 89021R10), S&C Specification
  Bulletin 351-31 p. 7: overall height N = 514 mm, mounting-to-hinge T = 248 mm, insulator and fuse tube inclined
  20 deg from vertical, pull ring on the upper ferrule facing away from the pole, fuse tube hinged at its lower end
  (trunnion in the hinge) and drawn hanging open below the hinge. The 15 kV tube is about 305 mm from the hinge axis
  to the cap (measured on the drawing). Upper contact: spring pressure 25-50 lb on the cap and a detent latch; pull-
  out force 30-40 lb (133-178 N) perpendicular to the tube (Information Sheet 351-450, 2.1(d)). The tube "rises
  slightly in the hinge during operation" (2.1(e)) and can be lifted out of the hinge with a pole (Instruction Sheet
  351-500 p. 12). Pull ring: "about 1 1/4 to 1 1/2 inch diameter" (US 5,532,668).
  Mock at 1:2 (lengths halved): hinge axis to cap 150 mm, 16 mm square tube (a round 32 mm tube halved), pull ring
  18 mm inside / 26 mm outside, 4 mm bar, its plane the tube's swing plane, centre 124 mm up the tube and 26 mm out
  in front of its axis; insulator 28 mm square behind the tube; top contact (hood and contact fingers) over the cap;
  hinge cheeks either side of the trunnion. Forces are not scaled by geometry: the mock's detent releases at 6 N at
  the ring (a light spring-plunger detent, inferred buildable), within the station's 15 N.
  Built as a fixed-base articulation, all in the frame of the hinge axis (x forward, away from the robot; z up):
  * base (fixed): post, crossarm, bracket, insulator, top contact, hinge cheeks; and the stick rack (two U-cradles and
    an end stop) on the table, carried by the same body so the scene has one fixed articulation.
  * carrier (prismatic along z, 0..25 mm): the trunnion seat; 0 = seated in the hinge. Lifted past the hinge hooks'
    8 mm depth, the tube counts as out of its hinge.
  * tube (revolute about y at the hinge axis): fuse tube, ferrules, cap, pull ring. Joint angle 0 = closed, negative
    = opening (the top swings toward the robot); limit -178 deg.
  Latch (rule, cutout_sim.task): a detent drive holds the tube closed up to its release torque; past 3 deg it is
  switched off and the tube falls open by gravity. While latched the carrier is held down (the upper contact's
  spring pressure on the cap); unlatched, only the tube's weight keeps the trunnion seated.
- Hook stick: a short hot stick (fibreglass, black handgrip, red hand-guard band, aluminium head) with a hook prong:
  380 mm overall, 16 mm square shaft (square so it cannot roll in the cradle or the fingers), 0.12 kg (inferred); a
  5 mm steel prong 40 mm long, sideways from the head and swept back 20 deg toward the handle (a J-hook), so a pulled
  ring slides toward its root.

Usage: python -m cutout_sim.usd.props <out dir>
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from cutout_sim.usd import usd_kit as kit

MM = 0.001
TILT_DEG = 20.0
"""Insulator and fuse tube lean from vertical; the tube's top leans toward the robot (-x)."""

# Desk-frame placement (m): desk top z = 0, arm bases at (0, +-0.305, 0), x forward.
HINGE = (0.50, -0.06, 0.20)
"""The hinge axis centre (trunnion axis) on the cutout's mid-plane."""

# Tube, in the tilted frame T (mm): origin on the hinge axis, z' along the tube (closed), x' toward the insulator.
TUBE_HALF = 8.0
TUBE_Z = (12.0, 128.0)
FERRULE_LO = (11.0, 12.0, (-6.0, 14.0))
FERRULE_HI = (10.0, 10.0, (118.0, 140.0))
CAP = (11.0, 11.0, (140.0, 150.0))
RING_C = (-26.0, 124.0)
"""Pull-ring centre (x', z') in the swing plane."""
RING_R_IN, RING_R_OUT, RING_T = 9.0, 13.0, 4.0
"""Octagonal ring: inside apothem 9 mm (18 mm across), outside 13 mm, 4 mm thick (y)."""
RING_ARM = ((-14.0, -9.0), (2.0,), (120.0, 128.0))
TUBE_MASS = 0.15
CARRIER_MASS = 0.02
LIFT_MAX = 0.025
OPEN_LIMIT_DEG = 178.0

# Base, frame T (mm).
INSULATOR = ((36.0, 64.0), 14.0, (10.0, 150.0))
SHEDS = [40.0 + 18.0 * k for k in range(6)]
HOOD = ((-16.0, 50.0), 16.0, (152.0, 160.0))
CONTACT = ((-8.0, 36.0), 12.0, (150.5, 152.0))
ATTACH_HOOK = ((44.0, 50.0), (16.0, 22.0), (156.0, 172.0))
TOP_CONTACT = (0.0, 156.0)
"""The keep-out centre: the middle of the top contact, just above the cap (x', z')."""
HINGE_CHEEK = ((-6.0, 30.0), (12.0, 16.0), (-14.0, 8.0))
HINGE_BACK = ((26.0, 38.0), 16.0, (-14.0, 12.0))

# Base, desk-aligned parts relative to the hinge (m).
CROSSARM = ((0.060, 0.105), (-0.20, 0.20), (0.070, 0.127))
POST = ((0.105, 0.165), (-0.03, 0.03), (-0.20, 0.20))
FOOT = ((0.040, 0.230), (-0.10, 0.10), (-0.20, -0.19))
BRACKET = ((0.030, 0.062), (-0.010, 0.010), (0.085, 0.105))

SIGN = ((0.054, 0.058), (0.10, 0.17), (0.072, 0.125))
"""Warning sign on the crossarm's front face, left of the cutout (hinge frame, m): a yellow plate with a black
lightning bolt (boxes), as on distribution poles; the colours are what a camera can pick out."""

# Stick rack (desk frame, m): the stick lies along +x, its butt against the end stop.
STICK_BUTT = (0.14, -0.25)
STICK_AXIS_Z = 0.034
CRADLE_X = (0.27, 0.42)
CRADLE_FLOOR = STICK_AXIS_Z - 0.008
CRADLE_INNER = 0.018
CRADLE_TOP = 0.052
END_STOP_X = (0.125, 0.137)

# Stick, frame S (mm): origin at the butt's centre, +x toward the head; the prong points +y (swept back).
STICK_HALF = 8.0
BUTT = (0.0, 12.0, 9.5)
HANDLE = (12.0, 100.0, 9.0)
GUARD = (100.0, 108.0, 9.5)
SHAFT = (108.0, 345.0, STICK_HALF)
HEAD = (345.0, 375.0, 9.0)
PRONG_ROOT = (366.0, 0.0)
PRONG_LEN = 40.0
PRONG_W = 5.0
PRONG_SWEEP_DEG = 20.0
GRIP_X = 56.0
"""Where the robot is meant to hold it: the middle of the handgrip."""
STICK_MASS = 0.12

CONTACT_OFFSET = 0.001

COLORS = dict(kit.COLORS)
COLORS.update({
    "tube": (0.56, 0.58, 0.60),
    "bronze": (0.62, 0.53, 0.40),
    "porcelain": (0.74, 0.74, 0.71),
    "wood": (0.56, 0.40, 0.22),
    "post": (0.36, 0.25, 0.15),
    "ring": (0.95, 0.28, 0.02),
    "alu": (0.78, 0.79, 0.80),
    "stick": (0.96, 0.80, 0.08),
    "rack": (0.30, 0.32, 0.36),
    "sign": (0.98, 0.80, 0.02),
})


def tilt(xp, zp):
    """Frame T (x', z') mm -> hinge frame (x, z) m."""
    c, s = math.cos(math.radians(TILT_DEG)), math.sin(math.radians(TILT_DEG))
    return ((xp * c - zp * s) * MM, (xp * s + zp * c) * MM)


def ring_center_hinge():
    x, z = tilt(*RING_C)
    return (x, 0.0, z)


def top_contact_hinge():
    x, z = tilt(*TOP_CONTACT)
    return (x, 0.0, z)


def _mats(stage, root, names):
    out = {}
    for name in names:
        path = f"{root}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[name]))
        metal = name in ("steel", "alu", "bronze")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4 if metal else 0.6)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.5 if metal else 0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        out[name] = material
    return out


def obox(stage, path, center, size, material, ry_deg=0.0, rz_deg=0.0, collide=True, rx_deg=0.0):
    """A box (m) rotated about x, then y, then z (degrees, right-handed), centred at ``center`` in its body's frame."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    if rz_deg:
        xf.AddRotateZOp().Set(rz_deg)
    if ry_deg:
        xf.AddRotateYOp().Set(ry_deg)
    if rx_deg:
        xf.AddRotateXOp().Set(rx_deg)
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def tbox(stage, path, xr, yr, zr, material, collide=True):
    """A box given in frame T (mm ranges; yr a (lo, hi) pair or a half-width)."""
    if not isinstance(yr, tuple):
        yr = (-yr, yr)
    elif len(yr) == 1:
        yr = (-yr[0], yr[0])
    cx, cz = tilt((xr[0] + xr[1]) / 2, (zr[0] + zr[1]) / 2)
    size = (abs(xr[1] - xr[0]) * MM, abs(yr[1] - yr[0]) * MM, abs(zr[1] - zr[0]) * MM)
    return obox(stage, path, (cx, (yr[0] + yr[1]) / 2 * MM, cz), size, material, ry_deg=-TILT_DEG, collide=collide)


def abox(stage, path, xr, yr, zr, material, collide=True):
    """An axis-aligned box from ranges in metres."""
    c = ((xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2, (zr[0] + zr[1]) / 2)
    return obox(stage, path, c, (xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]), material, collide=collide)


def _thin(stage, prefix):
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def make_cutout(out_path: Path) -> None:
    root = "/cutout"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    m = _mats(stage, root, ("tube", "bronze", "porcelain", "wood", "post", "ring", "steel", "rack", "alu", "grey",
                            "sign", "black"))

    base = kit.rigid_body(stage, f"{root}/base", (0, 0, 0), 20.0)
    kit.fix_to_world(stage, root, base.GetPath())
    b = f"{root}/base"
    abox(stage, f"{b}/crossarm", *CROSSARM, m["wood"])
    abox(stage, f"{b}/post", *POST, m["post"])
    abox(stage, f"{b}/foot", *FOOT, m["grey"])
    abox(stage, f"{b}/bracket", *BRACKET, m["steel"])
    (ix0, ix1), ih, (iz0, iz1) = INSULATOR
    tbox(stage, f"{b}/insulator", (ix0, ix1), ih, (iz0, iz1), m["porcelain"])
    for k, z in enumerate(SHEDS):
        tbox(stage, f"{b}/shed{k}", (ix0 - 6, ix1 + 6), ih + 6, (z, z + 5), m["porcelain"], collide=False)
    tbox(stage, f"{b}/hood", HOOD[0], HOOD[1], HOOD[2], m["steel"])
    tbox(stage, f"{b}/contact", CONTACT[0], CONTACT[1], CONTACT[2], m["bronze"])
    tbox(stage, f"{b}/upper_mount", (14.0, 40.0), 10.0, (146.0, 152.0), m["steel"])
    for s, name in ((1, "l"), (-1, "r")):
        yr = tuple(sorted((s * ATTACH_HOOK[1][0], s * ATTACH_HOOK[1][1])))
        tbox(stage, f"{b}/attach_hook_{name}", ATTACH_HOOK[0], yr, ATTACH_HOOK[2], m["steel"], collide=False)
        yc = tuple(sorted((s * HINGE_CHEEK[1][0], s * HINGE_CHEEK[1][1])))
        tbox(stage, f"{b}/hinge_cheek_{name}", HINGE_CHEEK[0], yc, HINGE_CHEEK[2], m["steel"])
    tbox(stage, f"{b}/hinge_back", HINGE_BACK[0], HINGE_BACK[1], HINGE_BACK[2], m["steel"])
    abox(stage, f"{b}/sign", *SIGN, m["sign"], collide=False)
    yc, zc = (SIGN[1][0] + SIGN[1][1]) / 2, (SIGN[2][0] + SIGN[2][1]) / 2
    xs = SIGN[0][0] - 0.0015
    for k, (dy, dz, ang) in enumerate(((0.008, 0.013, 25.0), (0.0, 0.0, -50.0), (-0.008, -0.013, 25.0))):
        obox(stage, f"{b}/bolt{k}", (xs, yc + dy, zc + dz), (0.002, 0.006, 0.024), m["black"], rx_deg=ang,
             collide=False)

    # Stick rack on the table (desk positions relative to the hinge).
    hx, hy, hz = HINGE
    sy = STICK_BUTT[1] - hy
    for k, cx in enumerate(CRADLE_X):
        x = (cx - 0.012 - hx, cx + 0.012 - hx)
        abox(stage, f"{b}/cradle{k}_floor", x, (sy - 0.025, sy + 0.025), (-hz, CRADLE_FLOOR - hz), m["rack"])
        for s, name in ((1, "l"), (-1, "r")):
            yy = tuple(sorted((sy + s * CRADLE_INNER, sy + s * (CRADLE_INNER + 0.006))))
            abox(stage, f"{b}/cradle{k}_{name}", x, yy, (CRADLE_FLOOR - hz, CRADLE_TOP - hz), m["rack"])
    abox(stage, f"{b}/end_stop", (END_STOP_X[0] - hx, END_STOP_X[1] - hx), (sy - 0.02, sy + 0.02),
         (-hz, CRADLE_TOP - hz), m["rack"])

    # Carrier (trunnion seat) and tube.
    carrier = kit.rigid_body(stage, f"{root}/carrier", (0, 0, 0), CARRIER_MASS)
    obox(stage, f"{root}/carrier/trunnion", (0, 0, 0), (0.008, 0.030, 0.008), m["bronze"], collide=False)
    lift = UsdPhysics.PrismaticJoint.Define(stage, f"{root}/joints/lift_joint")
    lift.CreateBody0Rel().SetTargets([base.GetPath()])
    lift.CreateBody1Rel().SetTargets([carrier.GetPath()])
    lift.CreateAxisAttr("Z")
    lift.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    lift.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    lift.CreateLowerLimitAttr(0.0)
    lift.CreateUpperLimitAttr(LIFT_MAX)
    UsdPhysics.DriveAPI.Apply(lift.GetPrim(), "linear")

    tube = kit.rigid_body(stage, f"{root}/tube", (0, 0, 0), TUBE_MASS)
    t = f"{root}/tube"
    tbox(stage, f"{t}/tube", (-TUBE_HALF, TUBE_HALF), TUBE_HALF, TUBE_Z, m["tube"])
    for name, (hx_, hy_, zr) in (("ferrule_lo", FERRULE_LO), ("ferrule_hi", FERRULE_HI), ("cap", CAP)):
        tbox(stage, f"{t}/{name}", (-hx_, hx_), hy_, zr, m["bronze" if name != "cap" else "alu"])
    tbox(stage, f"{t}/ring_arm", RING_ARM[0], RING_ARM[1][0], RING_ARM[2], m["bronze"])
    r_mid = (RING_R_IN + RING_R_OUT) / 2
    seg = 2 * RING_R_OUT * math.tan(math.radians(22.5)) + 0.4
    for k in range(8):
        a = math.radians(45.0 * k)
        xp, zp = RING_C[0] + r_mid * math.cos(a), RING_C[1] + r_mid * math.sin(a)
        cx, cz = tilt(xp, zp)
        # Segment k: radial along angle a in the (x', z') plane; its length runs tangentially.
        obox(stage, f"{t}/ring{k}", (cx, 0.0, cz), ((RING_R_OUT - RING_R_IN) * MM, RING_T * MM, seg * MM), m["ring"], ry_deg=-TILT_DEG - 45.0 * k)
    swing = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/joints/swing_joint")
    swing.CreateBody0Rel().SetTargets([carrier.GetPath()])
    swing.CreateBody1Rel().SetTargets([tube.GetPath()])
    swing.CreateAxisAttr("Y")
    swing.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    swing.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    swing.CreateLowerLimitAttr(-OPEN_LIMIT_DEG)
    swing.CreateUpperLimitAttr(1.0)
    UsdPhysics.DriveAPI.Apply(swing.GetPrim(), "angular")
    kit.friction(stage, root, [f"{t}/ring{k}" for k in range(8)], static=0.4, dynamic=0.35, name="steel")
    _thin(stage, root)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_stick(out_path: Path) -> None:
    root = "/stick"
    stage, root_prim = kit.new_stage(out_path, root)
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(STICK_MASS)
    m = _mats(stage, root, ("black", "red", "stick", "alu", "steel"))
    for name, (x0, x1, h), mat in (("butt", BUTT, "black"), ("handle", HANDLE, "black"), ("guard", GUARD, "red"),
                                   ("shaft", SHAFT, "stick"), ("head", HEAD, "alu")):
        abox(stage, f"{root}/{name}", (x0 * MM, x1 * MM), (-h * MM, h * MM), (-h * MM, h * MM), m[mat])
    a = math.radians(PRONG_SWEEP_DEG)
    d = (-math.sin(a), math.cos(a))
    c = (PRONG_ROOT[0] + d[0] * PRONG_LEN / 2, d[1] * PRONG_LEN / 2)
    obox(stage, f"{root}/prong", (c[0] * MM, c[1] * MM, 0.0), (PRONG_W * MM, PRONG_LEN * MM, PRONG_W * MM), m["steel"],
         rz_deg=PRONG_SWEEP_DEG)
    kit.friction(stage, root, [f"{root}/handle", f"{root}/butt", f"{root}/shaft"], static=1.0, dynamic=0.9)
    kit.friction(stage, root, [f"{root}/prong", f"{root}/head"], static=0.4, dynamic=0.35, name="steel")
    _thin(stage, root)
    root_prim.AddAppliedSchema("PhysxRigidBodyAPI")
    root_prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def prong_points_stick(n=5):
    """Points along the prong's centre line (stick frame, m), root to tip."""
    a = math.radians(PRONG_SWEEP_DEG)
    return [((PRONG_ROOT[0] - math.sin(a) * PRONG_LEN * k / (n - 1)) * MM, math.cos(a) * PRONG_LEN * k / (n - 1) * MM,
             0.0) for k in range(n)]


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    make_cutout(out_dir / "cutout.usda")
    make_stick(out_dir / "stick.usda")

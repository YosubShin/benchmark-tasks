"""Generate the props: the tops of two gas cylinders with their handwheel valves, the Brady LM023E gas cylinder
lockout (can and swinging locking plate) on its stand, the padlock's pedestal, and the Master Lock 410 padlock.

Real parts (README References):
- Valve: Sherwood GSV-class brass cylinder valve (datasheet: 4.00 in / 101.6 mm tall, of which the 3/4 in NGT inlet
  is 1.89 in / 48 mm; CGA outlet face 1.42 in / 36.1 mm from the axis; aluminium handwheel; operating torque
  1.1-2.2 N m, closing (seating) torque 2.2-3.3 N m). Ours: the inlet shows 14 mm above the neck ring (27 mm across,
  an octagon in collision), a 42 mm square wrench block on top of it (the valve's collar), the body, the side outlet
  (36 mm from the axis), the bonnet and a 57 mm handwheel (12-sided in collision, six lobes in the visual).
- Cylinder: a 184 mm (7 1/4 in) industrial cylinder; only its shoulder, an 80 mm neck ring (the cap-thread collar)
  and the valve show above the table: each cylinder stands in a well so the arm can reach the valve.
- Lockout: Brady LM023E (Prinzing) gas cylinder lockout, red polypropylene, 6 1/4 in (159 mm) tall, 3.47 in
  (88 mm) across, one padlock with a shackle up to 0.34 in (8.6 mm); stems up to 1.41 in (36 mm). A can closed at
  the top; on its open end a flat locking plate on a rivet at the rim swings across the opening. Its keyhole slot
  lets the valve's stem in as it swings; shut, its round end closes round the stem under the valve body and a flat
  tab on the plate lies under a flat tab on the can, both with a vertical hole for the padlock's shackle (the padlock
  hangs below them, as in Brady's installed photo) (Brady product photos; ABUS's
  animation of the same design).

How it holds (geometry): shut, the plate's hole (33 mm) is smaller than the wrench block above it (42 mm), so the
can cannot be lifted off over the valve; the can encloses the handwheel, so the wheel cannot be reached or turned;
a shackle down through both tabs stops the plate swinging open (the plate's tab would have to move sideways across
the shackle).
Without the padlock the plate swings open freely (a light pivot friction only) and the can lifts off.

Files (desk frame):
``cylinders.usda``: fixed-base articulation: ``base`` (both cylinder tops and valve bodies, at the asset origin)
and ``wheel_target`` / ``wheel_decoy`` on revolute joints about the valve axis (+z; 0 = seated shut, positive =
anticlockwise from above = opening).
``cap.usda``: floating articulation: ``can`` (frame: origin at the centre of the open end's rim, +z along the axis
toward the closed top) and ``plate`` on the ``plate_joint`` revolute about a vertical pin at the rim (0 = shut,
positive = swung open, anticlockwise from above).
``stand.usda``: the cap's stand and the padlock's pedestal.
``padlock_ml410.usda``: the blue Master Lock 410 (cylvalve_sim/usd/padlock_410.py).

Usage: python -m cylvalve_sim.usd.cylvalve_props [<out dir>]   (default: cylvalve_sim/assets)
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from cylvalve_sim.spec import CAP_STAND_TOP, CAP_STAND_XY, DECOY_XY, GASES, PEDESTAL_XY, TARGET_XY
from cylvalve_sim.usd import usd_kit as kit

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TABLE_Z = -0.0025
"""The station table's top in the desk frame."""

# Cylinder top and valve (heights above the table) -----------------------------------------------------------------
CYL_R = 0.092
DOME_TOP = 0.045
DOME_H = 0.060
"""Visual shoulder: an ellipsoid cap (radius CYL_R, height DOME_H) whose top is at DOME_TOP; the neck ring (with
the neck below it) stands 6 cm tall on it, so a padlock can hang from the cap's tabs beside it (the rest of the
cylinder is below, in the well)."""
RING_R = 0.040
RING_Z = (0.040, 0.100)
STEM_R = 0.0135
STEM_Z = (RING_Z[1], 0.114)
COLLAR_HW = 0.021
COLLAR_Z = (STEM_Z[1], 0.128)
BODY_HW = 0.015
BODY_Z = (COLLAR_Z[1], 0.151)
OUTLET_R = 0.0125
OUTLET_Z = 0.138
OUTLET_REACH = 0.036
OUTLET_DIR = (0.0, 1.0)
"""The outlet points to the robot's left (+y)."""
BONNET_R = 0.012
BONNET_Z = (BODY_Z[1], 0.159)
WHEEL_R = 0.0285
WHEEL_Z = (0.159, 0.173)
WHEEL_MASS = 0.06
WHEEL_OPEN_DEG = 180.0
"""Both valves start open half a turn."""

# Lockout (Brady LM023E), can frame: origin at the rim's centre, +z up the axis ---------------------------------------
CAN_R = 0.044
CAN_WALL = 0.003
CAN_H = 0.159
CAN_TOP_T = 0.003
CAN_MASS = 0.17
PLATE_T = 0.004
PLATE_GAP = 0.0005
PLATE_Z = (-PLATE_GAP - PLATE_T, -PLATE_GAP)
PLATE_MASS = 0.03
PIVOT = (CAN_R - 0.004, 0.0)
"""The plate's rivet: at the rim, on the side away from the tabs."""
HOLE_R = 0.0165
"""The keyhole's round end (the 27 mm stem passes it with 3 mm to spare; Brady: stems up to 36 mm)."""
PLATE_OPEN_DEG = 150.0
"""Swung open this far the plate clears the valve's wrench block as the can goes down over it."""
PLATE_MAX_DEG = 155.0
TAB_R = (CAN_R - 0.003, 0.066)
"""Radial extent of both tabs (toward the robot, -x): flat, horizontal, one above the other when the plate is shut."""
TAB_HW = 0.010
TAB_T = 0.0035
TAB_HOLE = 0.0095
TAB_HOLE_X = -0.056
TAB_Z = {"can": (0.0, TAB_T), "plate": PLATE_Z}
"""The can's tab is a flange at the rim; the plate's tab is the plate's own extension, just under it."""
CAP_FRICTION = 0.5

STAND_ARC_DEG = (30.0, 290.0)
"""The stand holds the can's rim along this arc (can frame, anticlockwise from +x); the open plate is elsewhere."""
STAND_PAD_T = 0.006
PEDESTAL_TOP = 0.080

CONTACT_OFFSET = 0.0003
SOLVER_ITERATIONS = 32

COLORS = {
    "brass": (0.80, 0.66, 0.36),
    "alu": (0.75, 0.76, 0.78),
    "steel": (0.62, 0.64, 0.67),
    "dark": (0.08, 0.08, 0.08),
    "lockout_red": (0.80, 0.07, 0.05),
    "label_yellow": (0.98, 0.80, 0.10),
    "stand": (0.40, 0.40, 0.42),
    "well": (0.20, 0.20, 0.22),
    "target_shoulder": GASES["target"][1],
    "decoy_shoulder": GASES["decoy"][1],
    "body_grey": (0.55, 0.56, 0.58),
}


def _rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == "Z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _mat_q(m):
    from scipy.spatial.transform import Rotation

    return tuple(float(v) for v in Rotation.from_matrix(np.asarray(m)).as_quat())


def valve_xy(which: str) -> tuple[float, float]:
    return TARGET_XY if which == "target" else DECOY_XY


# Plate keyhole -----------------------------------------------------------------------------------------------------
def stem_path(n: int = 60) -> np.ndarray:
    """The stem's centre in the plate's frame as the plate swings from shut (0) to fully open."""
    th = np.radians(np.linspace(0.0, PLATE_MAX_DEG, n))
    px, py = PIVOT
    return np.stack([px - px * np.cos(th), px * np.sin(th)], axis=1)


CELL = 0.002


def plate_cells():
    """The plate as row strips of CELL x CELL squares: inside the disc, clear of the keyhole by HOLE_R.
    Returns [(x0, x1, y0, y1)]."""
    path = stem_path(200)
    half_diag = CELL / math.sqrt(2)
    strips = []
    n = int(math.ceil(CAN_R / CELL))
    for j in range(-n, n):
        yc = (j + 0.5) * CELL
        run = None
        for i in range(-n, n + 1):
            xc = (i + 0.5) * CELL
            keep = (math.hypot(xc, yc) <= CAN_R - CELL / 2
                    and np.min(np.hypot(path[:, 0] - xc, path[:, 1] - yc)) - half_diag >= HOLE_R)
            if keep and run is None:
                run = xc - CELL / 2
            if not keep and run is not None:
                strips.append((run, xc - CELL / 2, yc - CELL / 2, yc + CELL / 2))
                run = None
    return strips


# USD helpers -----------------------------------------------------------------------------------------------------
def _material(stage, root, name, color, roughness=0.5):
    path = f"{root}/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _mats(stage, root):
    return {name: _material(stage, root, name, c) for name, c in COLORS.items()}


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


def _quad(stage, path, corners, material):
    """A textured quad from four corners (counterclockwise seen from the side it faces), uv (0,0) at the first."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in p]) for p in corners]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateSubdivisionSchemeAttr("none")
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _box(stage, path, x, y, z, mat, collide=True):
    c = ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2)
    s = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    return kit.box(stage, path, c, s, mat, collide=collide)


def _rbox(stage, path, center, size, yaw_deg, mat, collide=True):
    """A box turned about z by yaw_deg (an Xform holding a unit cube)."""
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddRotateZOp().Set(float(yaw_deg))
    return kit.box(stage, f"{path}/box", (0.0, 0.0, 0.0), size, mat, collide=collide)


def _poly(stage, path, cx, cy, z, r_flat, n, mat, yaw0=0.0, collide=True):
    """A regular n-gon prism (flats at r_flat) as n/2 crossed boxes. Returns the collider paths."""
    out = []
    w = 2 * r_flat * math.tan(math.pi / n) + 1e-4
    for k in range(n // 2):
        yaw = yaw0 + k * 360.0 / n
        out.append(_rbox(stage, f"{path}_{k}", (cx, cy, (z[0] + z[1]) / 2), (2 * r_flat, w, z[1] - z[0]), yaw, mat,
                         collide))
    return [str(b.GetPath()) for b in out]


def _cyl(stage, path, cx, cy, z, r, mat, axis="Z"):
    return kit.cylinder(stage, path, (cx, cy, (z[0] + z[1]) / 2), r, z[1] - z[0], mat, axis=axis)


def _body(stage, path, pos, rot, mass, inertia=None):
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


def _offsets(stage) -> None:
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def _solver(root_prim) -> None:
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
        SOLVER_ITERATIONS)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)


def _joint(stage, path, body0, frame0, body1, frame1, world_point, axis, lower, upper):
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(str(body0))])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(str(body1))])
    joint.CreateAxisAttr(axis)
    for (p, r), pos_attr, rot_attr in ((frame0, joint.CreateLocalPos0Attr, joint.CreateLocalRot0Attr),
                                       (frame1, joint.CreateLocalPos1Attr, joint.CreateLocalRot1Attr)):
        pos_attr(Gf.Vec3f(*[float(v) for v in r.T @ (np.asarray(world_point) - p)]))
        x, y, z, w = _mat_q(r.T)
        rot_attr(Gf.Quatf(float(w), float(x), float(y), float(z)))
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    return joint


# Textures ---------------------------------------------------------------------------------------------------
def write_textures(out_dir: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    to8 = lambda c: tuple(int(round(v * 255)) for v in c)  # noqa: E731
    for which, (legend, color) in GASES.items():
        w, h = 900, 400
        bg = to8(color)
        ink = (250, 250, 250) if sum(color) < 1.5 else (15, 15, 15)
        img = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(img)
        draw.rectangle([12, 12, w - 12, h - 12], outline=ink, width=10)
        size = 220
        while ImageFont.truetype(FONT, size).getlength(legend) > w * 0.84:
            size -= 4
        draw.text((w / 2, h / 2), legend, font=ImageFont.truetype(FONT, size), fill=ink, anchor="mm")
        img.save(out_dir / f"label_{which}.png")
    # The lockout's label: yellow panel, red "CYLINDER LOCKOUT" (as on Brady's can).
    w, h = 300, 800
    img = Image.new("RGB", (w, h), to8(COLORS["label_yellow"]))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, w - 10, h - 10], outline=(200, 20, 15), width=12)
    txt = Image.new("RGB", (h - 60, w - 60), to8(COLORS["label_yellow"]))
    d2 = ImageDraw.Draw(txt)
    d2.text(((h - 60) / 2, (w - 60) / 2), "CYLINDER LOCKOUT", font=ImageFont.truetype(FONT, 62), fill=(200, 20, 15),
            anchor="mm")
    img.paste(txt.rotate(90, expand=True), (30, 30))
    img.save(out_dir / "cap_label.png")


# Cylinders and valves ----------------------------------------------------------------------------------------------
def _cylinder_top(stage, base, which, mats, label):
    cx, cy = valve_xy(which)
    root = f"{base}/{which}"
    UsdGeom.Xform.Define(stage, root)
    cols = []
    # Well edge (visual), shoulder (visual ellipsoid), shoulder colliders.
    kit.cylinder(stage, f"{root}/well", (cx, cy, 0.0008), CYL_R + 0.012, 0.0016, mats["well"])
    sph = UsdGeom.Sphere.Define(stage, f"{root}/shoulder")
    sph.CreateRadiusAttr(1.0)
    xf = UsdGeom.Xformable(sph)
    xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, DOME_TOP - DOME_H))
    xf.AddScaleOp().Set(Gf.Vec3f(CYL_R, CYL_R, DOME_H))
    UsdShade.MaterialBindingAPI.Apply(sph.GetPrim()).Bind(mats[f"{which}_shoulder"])
    cols += _poly(stage, f"{root}/dome_lo", cx, cy, (TABLE_Z, 0.025), 0.050, 8, mats["dark"], collide=True)
    cols += _poly(stage, f"{root}/dome_hi", cx, cy, (0.025, RING_Z[0] + 0.001), 0.032, 8, mats["dark"], collide=True)
    for p in cols:
        kit.hide(stage, p)
    # Neck ring (steel, with thread grooves), inlet stem, collar, body, outlet, bonnet.
    _cyl(stage, f"{root}/ring_v", cx, cy, RING_Z, RING_R, mats["steel"])
    for k, z in enumerate(np.linspace(RING_Z[0] + 0.006, RING_Z[1] - 0.005, 6)):
        _cyl(stage, f"{root}/groove_{k}", cx, cy, (z - 0.0006, z + 0.0006), RING_R + 0.0003, mats["body_grey"])
    cols += _poly(stage, f"{root}/ring", cx, cy, RING_Z, RING_R, 12, mats["steel"], collide=True)
    _cyl(stage, f"{root}/stem_v", cx, cy, STEM_Z, STEM_R, mats["brass"])
    cols += _poly(stage, f"{root}/stem", cx, cy, STEM_Z, STEM_R, 8, mats["brass"], yaw0=22.5)
    cols.append(str(_box(stage, f"{root}/collar", (cx - COLLAR_HW, cx + COLLAR_HW), (cy - COLLAR_HW, cy + COLLAR_HW),
                         COLLAR_Z, mats["brass"]).GetPath()))
    cols.append(str(_box(stage, f"{root}/body", (cx - BODY_HW, cx + BODY_HW), (cy - BODY_HW, cy + BODY_HW), BODY_Z,
                         mats["brass"]).GetPath()))
    ox, oy = OUTLET_DIR
    mid = (BODY_HW + OUTLET_REACH) / 2
    kit.cylinder(stage, f"{root}/outlet_v", (cx + ox * mid, cy + oy * mid, OUTLET_Z), OUTLET_R,
                 OUTLET_REACH - BODY_HW, mats["brass"], axis="Y" if oy else "X")
    kit.cylinder(stage, f"{root}/outlet_bore", (cx + ox * (OUTLET_REACH + 0.0002), cy + oy * (OUTLET_REACH + 0.0002),
                                                 OUTLET_Z), 0.0075, 0.0006, mats["dark"], axis="Y" if oy else "X")
    span = (BODY_HW - 0.001, OUTLET_REACH)
    oxr = (cx - OUTLET_R, cx + OUTLET_R) if oy else (cx + span[0], cx + span[1])
    oyr = (cy + span[0], cy + span[1]) if oy else (cy - OUTLET_R, cy + OUTLET_R)
    out = _box(stage, f"{root}/outlet", oxr, oyr, (OUTLET_Z - OUTLET_R, OUTLET_Z + OUTLET_R), mats["brass"])
    kit.hide(stage, str(out.GetPath()))
    cols.append(str(out.GetPath()))
    _cyl(stage, f"{root}/bonnet_v", cx, cy, BONNET_Z, BONNET_R, mats["brass"])
    cols += _poly(stage, f"{root}/bonnet", cx, cy, BONNET_Z, BONNET_R, 6, mats["brass"], collide=True)
    # Shoulder label facing the robot and the top camera (tilted 35 degrees back).
    t = math.radians(35)
    r0, zc, hw, hh = 0.062, 0.031, 0.030, 0.013
    p = np.array([cx - r0, cy, zc])
    u = np.array([0.0, -1.0, 0.0])
    v = np.array([math.sin(t), 0.0, math.cos(t)])
    corners = [p - hw * u - hh * v, p + hw * u - hh * v, p + hw * u + hh * v, p - hw * u + hh * v]
    _quad(stage, f"{root}/label", corners, label)
    return cols


def _wheel(stage, path, which, mats):
    cx, cy = valve_xy(which)
    wheel = _body(stage, path, (cx, cy, WHEEL_Z[0]), np.eye(3), WHEEL_MASS,
                  inertia=((0, 0, 0.007), (1.4e-5, 1.4e-5, 2.4e-5)))
    zl = (0.0, WHEEL_Z[1] - WHEEL_Z[0])
    cols = _poly(stage, f"{path}/rim", 0.0, 0.0, zl, WHEEL_R - 0.002, 12, mats["alu"])
    for p in cols:
        kit.hide(stage, p)
    _cyl(stage, f"{path}/disc_v", 0.0, 0.0, (zl[0], zl[1] - 0.002), WHEEL_R - 0.004, mats["alu"])
    for k in range(6):
        a = math.radians(30 + 60 * k)
        _cyl(stage, f"{path}/lobe_{k}", (WHEEL_R - 0.008) * math.cos(a), (WHEEL_R - 0.008) * math.sin(a),
             (zl[0] + 0.001, zl[1] - 0.001), 0.008, mats["alu"])
    _cyl(stage, f"{path}/hub_v", 0.0, 0.0, (zl[1] - 0.003, zl[1] + 0.001), 0.009, mats["body_grey"])
    _cyl(stage, f"{path}/nut_v", 0.0, 0.0, (zl[1], zl[1] + 0.003), 0.005, mats["brass"])
    # A dark mark on one lobe so the turn shows on camera.
    _box(stage, f"{path}/mark", (WHEEL_R - 0.012, WHEEL_R - 0.004), (-0.0025, 0.0025), (zl[1] - 0.0005, zl[1] + 0.0004),
         mats["dark"], collide=False)
    return wheel, cols


def make_cylinders(out_path: Path) -> None:
    root = "/cylinders"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    _solver(root_prim)
    mats = _mats(stage, root)
    base = _body(stage, f"{root}/base", np.zeros(3), np.eye(3), 5.0)
    cols = []
    for which in ("target", "decoy"):
        label = _tex_material(stage, root, f"label_{which}", f"label_{which}.png")
        cols += _cylinder_top(stage, f"{root}/base", which, mats, label)
    wheel_cols = []
    for which in ("target", "decoy"):
        wheel, wc = _wheel(stage, f"{root}/wheel_{which}", which, mats)
        wheel_cols += wc
        cx, cy = valve_xy(which)
        _joint(stage, f"{root}/joints/wheel_{which}_joint", base.GetPath(), (np.zeros(3), np.eye(3)), wheel.GetPath(),
               (np.array([cx, cy, WHEEL_Z[0]]), np.eye(3)), (cx, cy, WHEEL_Z[0]), "Z", 0.0, 720.0)
    kit.fix_to_world(stage, root, base.GetPath())
    kit.friction(stage, root, cols, static=0.5, dynamic=0.4, name="metal", combine="min")
    # The handwheel's rim: the finger pads' rubber decides the grip.
    kit.friction(stage, root, wheel_cols, static=1.2, dynamic=1.0, name="wheel_grip", combine="max")
    _offsets(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


# Lockout cap --------------------------------------------------------------------------------------------------------
def _tab(stage, path, which, mats):
    """A flat tab with its vertical padlock hole, in the can frame (shut). Returns the collider paths."""
    z = TAB_Z[which]
    h = TAB_HOLE / 2
    x0, x1 = -TAB_R[1], -TAB_R[0]
    out = []
    for k, (xs, ys) in enumerate([((TAB_HOLE_X + h, x1), (-TAB_HW, TAB_HW)), ((x0, TAB_HOLE_X - h), (-TAB_HW, TAB_HW)),
                                  ((TAB_HOLE_X - h, TAB_HOLE_X + h), (h, TAB_HW)),
                                  ((TAB_HOLE_X - h, TAB_HOLE_X + h), (-TAB_HW, -h))]):
        out.append(str(_box(stage, f"{path}_{k}", xs, ys, z, mats["lockout_red"]).GetPath()))
    return out


def make_cap(out_path: Path) -> None:
    root = "/cap"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    _solver(root_prim)
    mats = _mats(stage, root)
    red = mats["lockout_red"]
    can = _body(stage, f"{root}/can", np.zeros(3), np.eye(3), CAN_MASS,
                inertia=((0, 0, 0.085), (8.0e-4, 8.0e-4, 3.0e-4)))
    cols = []
    n = 24
    r_mid = CAN_R - CAN_WALL / 2
    w = 2 * CAN_R * math.tan(math.pi / n) + 0.001
    for k in range(n):
        a = 2 * math.pi * k / n
        cols.append(str(_rbox(stage, f"{root}/can/wall_{k}", (r_mid * math.cos(a), r_mid * math.sin(a), CAN_H / 2),
                              (CAN_WALL, w, CAN_H), math.degrees(a), red).GetPath()))
    cols += _poly(stage, f"{root}/can/top", 0.0, 0.0, (CAN_H - CAN_TOP_T, CAN_H), CAN_R - 0.001, 12, red)
    # The dark inside of the open end (visual): the bore seen from below.
    _cyl(stage, f"{root}/can/bore_v", 0.0, 0.0, (0.0003, 0.0009), CAN_R - CAN_WALL, mats["dark"])
    cols += _tab(stage, f"{root}/can/tab", "can", mats)
    # Pivot rivet (visual).
    _cyl(stage, f"{root}/can/rivet", PIVOT[0], PIVOT[1], (-0.0002, 0.0015), 0.0035, mats["steel"])
    # Label (visual), facing the robot.
    label = _tex_material(stage, root, "cap_label", "cap_label.png")
    x = -CAN_R - 0.0006
    hw, z0, z1 = 0.016, 0.040, 0.140
    _quad(stage, f"{root}/can/label", [(x, hw, z0), (x, -hw, z0), (x, -hw, z1), (x, hw, z1)], label)

    plate = _body(stage, f"{root}/plate", np.zeros(3), np.eye(3), PLATE_MASS)
    pcols = []
    for k, (x0, x1, y0, y1) in enumerate(plate_cells()):
        pcols.append(str(_box(stage, f"{root}/plate/cell_{k}", (x0, x1), (y0, y1), PLATE_Z, red).GetPath()))
    pcols += _tab(stage, f"{root}/plate/tab", "plate", mats)
    _joint(stage, f"{root}/joints/plate_joint", can.GetPath(), (np.zeros(3), np.eye(3)), plate.GetPath(),
           (np.zeros(3), np.eye(3)), (PIVOT[0], PIVOT[1], 0.0), "Z", 0.0, PLATE_MAX_DEG)
    kit.friction(stage, root, cols + pcols, static=CAP_FRICTION, dynamic=CAP_FRICTION, name="pp", combine="min")
    # Where the fingers take hold (the can's sides, the tabs): the finger pads' rubber decides the grip.
    grip = [c for c in cols + pcols if "/wall_" in c or "/tab_" in c]
    kit.friction(stage, root, grip, static=1.2, dynamic=1.0, name="grip", combine="max")
    _offsets(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}: plate {len(pcols)} colliders")


def cap_start_pose() -> tuple:
    x, y = CAP_STAND_XY
    return (x, y, CAP_STAND_TOP + 0.0005, 0.0, 0.0, 0.0, 1.0)


# Stand and pedestal --------------------------------------------------------------------------------------------------
def make_stand(out_path: Path) -> None:
    root = "/stand"
    stage, _ = kit.new_stage(out_path, root)
    mats = _mats(stage, root)
    cx, cy = CAP_STAND_XY
    a0, a1 = (math.radians(v) for v in STAND_ARC_DEG)
    n = 9
    r_in, r_out = CAN_R - 0.007, CAN_R + 0.006
    r_mid = (r_in + r_out) / 2
    seg = (a1 - a0) / n
    w = 2 * r_out * math.tan(seg / 2) + 0.001
    z = (CAP_STAND_TOP - STAND_PAD_T, CAP_STAND_TOP)
    for k in range(n):
        a = a0 + (k + 0.5) * seg
        _rbox(stage, f"{root}/pad_{k}", (cx + r_mid * math.cos(a), cy + r_mid * math.sin(a), sum(z) / 2),
              (r_out - r_in, w, z[1] - z[0]), math.degrees(a), mats["stand"])
    # Column under the pads (below the hanging plate's level).
    _box(stage, f"{root}/column", (cx - 0.050, cx + 0.010), (cy - 0.020, cy + 0.050), (TABLE_Z, z[0]), mats["stand"])
    # The padlock's pedestal: a column under the standing padlock's body, narrower than the body so the fingers
    # close on the body's broad faces past it.
    x, y = PEDESTAL_XY
    _box(stage, f"{root}/ped_foot", (x - 0.025, x + 0.025), (y - 0.025, y + 0.025), (TABLE_Z, TABLE_Z + 0.01),
         mats["stand"])
    _box(stage, f"{root}/ped_column", (x - 0.008, x + 0.008), (y - 0.007, y + 0.007), (TABLE_Z, PEDESTAL_TOP),
         mats["stand"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def main(out_dir: Path) -> None:
    from cylvalve_sim.usd import padlock_410, padlock_props

    out_dir.mkdir(parents=True, exist_ok=True)
    write_textures(out_dir)
    make_cylinders(out_dir / "cylinders.usda")
    make_cap(out_dir / "cap.usda")
    make_stand(out_dir / "stand.usda")
    blue = replace(padlock_props.PADLOCKS["ml410"], color="lock_blue")
    padlock_410.make_padlock_410(out_dir / "padlock_ml410.usda", blue)
    _stiffen_padlock(out_dir / "padlock_ml410.usda")


def _stiffen_padlock(path: Path) -> None:
    """For this task the short leg is a solid square bar (it bears the plate's tab, which pushes it sideways in the
    stacked holes), and the lock gets the same solver iterations as the cap, with CCD on the shackle."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    root = stage.GetPrimAtPath("/padlock")
    _solver(root)
    for prim in stage.Traverse():
        name = prim.GetName()
        if prim.GetParent().GetName() == "short_leg" and prim.HasAPI(UsdPhysics.CollisionAPI) or (
                prim.GetName() == "box" and prim.GetParent().GetParent().GetName() == "short_leg"):
            xf = UsdGeom.Xformable(prim)
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    sx, sy, sz = op.Get()
                    r = max(sx, sy)
                    op.Set(Gf.Vec3f(r, r, sz))
    sh = stage.GetPrimAtPath("/padlock/shackle")
    sh.AddAppliedSchema("PhysxRigidBodyAPI")
    sh.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets")

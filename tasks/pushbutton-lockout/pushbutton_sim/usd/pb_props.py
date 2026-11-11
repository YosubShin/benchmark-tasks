"""Generate the props: the push-button station (two 22 mm flush push buttons on spring-return plungers, each with a
Master Lock S2152AST install base under its bezel), the Master Lock S2151 removable push-button cover, the block it
stands on, the padlock's pedestal and the Master Lock 410 padlock.

Real parts (README References):
- Push buttons: Schneider Harmony XB4BA31 (green flush, 1 NO) and XB4BA42 (red flush, 1 NC) style: 22.5 mm panel hole,
  round chrome bezel 30 mm across standing 9 mm proud of the panel, flush cap; operating travel 2.6 mm (contact
  changes state) at 3.8 N, total travel 4.3 mm (Schneider data sheet).
- Master Lock S2151 removable push-button cover (polycarbonate, clear; fits 22.5 and 30.5 mm operators and buttons up
  to 50 mm across and 45 mm tall): a cylindrical clamshell of two half-shells joined by a hinge along one vertical
  seam; at the other seam each half carries a padlock ear (the two ears lie face to face when the cover is shut, the
  shackle passes through both). Each half's bottom edge has an inward lip that goes under the rim of the install
  base when the cover is closed around it (product photos, sell sheet). Overall size inferred from the button size it
  takes: 64.4 mm across, 50 mm tall, 1.8 mm walls.
- Master Lock S2152AST install base: a clear flat ring captured under the operator's bezel, with a raised outer rim
  (inferred: rim 2 mm wide, 5 mm tall, with a 1.8 mm outward flange at its top).

How the cover holds (geometry): closed, both halves' bottom lips sit under the base's flange, so the cover cannot be
lifted off; it encloses the button with 38 mm of clearance above the cap, so the button cannot be pressed; to take the
cover off it has to be opened, and opening it slides the two ears past each other across the shackle that goes
through both, which the shackle stops. Without the padlock nothing holds the halves shut (hinge damping only).

Files (desk frame):
``station.usda``: fixed-base articulation: ``enclosure`` (the steel box, both install bases and bezels) and
``plunger_start`` / ``plunger_stop`` on prismatic joints ``press_start`` / ``press_stop`` (0 up, -4.3 mm fully in;
the task's preloaded joint drive is the return spring).
``cover.usda``: free articulation: ``half_b`` (root; the back half, +x side when the cover is set as in the demo,
with the hinge barrel and the inner ear) and ``half_f`` (front half, outer ear) on the ``hinge`` revolute joint
(0 shut, positive open). Cover frame: origin on the cover's axis at its bottom edge, the hinge along -y, the ears
at +y; the halves meet in the y-z plane.
``stand.usda``: the block the cover starts on and the padlock's pedestal.
``padlock_ml410.usda``: the blue Master Lock 410 (pushbutton_sim/usd/padlock_410.py).

Usage: python -m pushbutton_sim.usd.pb_props [<out dir>]   (default: pushbutton_sim/assets)
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from pushbutton_sim.spec import (
    BLOCK_TOP, BUTTON_XY, COVER_START_XY, LEGENDS, PANEL_Z, PEDESTAL_XY, STATION_X, STATION_Y,
)
from pushbutton_sim.usd import usd_kit as kit

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TABLE_Z = -0.0025
"""The station table's top in the desk frame."""

# Push button (22 mm, flush), button frame: origin on the panel at the button's axis -------------------------------
BEZEL_R = (0.012, 0.015)
BEZEL_Z = (0.001, 0.010)
CAP_HALF = 0.008
"""Cap collider: a 16 mm square (inside the 24 mm bezel opening)."""
CAP_R = 0.0117
CAP_TOP = 0.0095
CAP_Z = (0.0035, CAP_TOP)
TRAVEL = 0.0043
"""Total travel (Schneider XB4 data sheet)."""
CONTACT_TRAVEL = 0.0026
"""Travel at which the contact changes state (the button counts as pressed)."""
SPRING_K = 692.0
"""Return spring, N/m: with SPRING_PRELOAD_M it pushes 2.0 N at rest and 3.8 N at the 2.6 mm contact point (the data
sheet's operating force; the preload is inferred)."""
SPRING_PRELOAD_M = 0.0029
PLUNGER_MASS = 0.005
PLUNGER_DAMPING = 1.0

# Install base (S2152AST), button frame -----------------------------------------------------------------------------
BASE_DISC_R = (0.01125, 0.0265)
BASE_DISC_T = 0.001
RIM_R = (0.0265, 0.0284)
RIM_TOP = 0.005
FLANGE_R = (0.0284, 0.0302)
FLANGE_Z = (0.0036, RIM_TOP)

# Cover (S2151), cover frame ----------------------------------------------------------------------------------------
R_IN = 0.0305
R_OUT = 0.0322
HEIGHT = 0.050
TOP_T = 0.0018
LIP_R = (0.0292, R_IN)
LIP_Z = (0.0003, 0.003)
LIP_SPAN_DEG = 50.0
"""Each half's lip spans +-50 degrees about the half's middle (none near the hinge, where it could not swing in)."""
HINGE_R = R_OUT + 0.005
"""The hinge axis: at (0, -HINGE_R) in the cover frame (the knuckles stand 5 mm off the wall)."""
BARREL = 0.007
EAR_HW = 0.007
EAR_T = 0.0025
EAR_GAP = 0.0002
EAR_Y_IN = (R_OUT + 0.0004, R_OUT + 0.0004 + EAR_T)
EAR_Y_OUT = (EAR_Y_IN[1] + EAR_GAP, EAR_Y_IN[1] + EAR_GAP + EAR_T)
EAR_Z = (HEIGHT - 0.006, HEIGHT + 0.012)
EAR_HOLE = 0.009
EAR_HOLE_Z = HEIGHT + 0.006
EAR_HOLE_Y = (EAR_Y_IN[0] + EAR_Y_OUT[1]) / 2
"""The ear holes' centre (cover frame); the holes' axis is along y (radial at the seam)."""
HALF_MASS = 0.018
HINGE_DAMPING = 0.05
"""N*m*s/rad (inferred: the moulded hinge pin turns freely; a hanging half does not swing on its own)."""
PC_FRICTION = 0.4
N_SEG = 18
"""Wall and lip segments per half (10 degrees each)."""

STATION_COLOR = (0.78, 0.79, 0.76)
"""RAL 7035 light grey."""

CONTACT_OFFSET = 0.0003
SOLVER_ITERATIONS = 32

COLORS = {
    "station": STATION_COLOR,
    "chrome": (0.80, 0.81, 0.83),
    "cap_green": (0.06, 0.55, 0.16),
    "cap_red": (0.80, 0.06, 0.05),
    "stand": (0.40, 0.40, 0.42),
    "legend": (0.04, 0.04, 0.04),
    "dark": (0.15, 0.15, 0.16),
}
CLEAR = (0.78, 0.86, 0.94)
CLEAR_OPACITY = 0.55


def button_root(which: str) -> tuple[float, float, float]:
    x, y = BUTTON_XY[which]
    return (x, y, PANEL_Z)


def cover_start_pose(open_deg: float | None = None) -> tuple:
    """Root (half_b) pose at the start: standing on the block, world-aligned."""
    x, y = COVER_START_XY
    return (x, y, BLOCK_TOP + 0.0003, 0.0, 0.0, 0.0, 1.0)


def hinge_point() -> np.ndarray:
    return np.array([0.0, -HINGE_R, 0.0])


def half_f_point(p_closed, theta_rad: float) -> np.ndarray:
    """A point of the front half (cover frame, given with the cover shut) with the hinge open by theta."""
    h = hinge_point()
    d = np.asarray(p_closed, dtype=float) - h
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return h + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])


# USD helpers -------------------------------------------------------------------------------------------------------
def _material(stage, root, name, color, roughness=0.5, opacity=None):
    path = f"{root}/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    if opacity is not None:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)
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


def _rbox(stage, path, center, size, yaw_rad, mat, collide=True):
    """A box turned about z by yaw (collider or visual)."""
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddRotateZOp().Set(math.degrees(yaw_rad))
    return kit.box(stage, f"{path}/box", (0.0, 0.0, 0.0), size, mat, collide=collide)


def _ring(stage, path, center, r, z, mat, a0_deg=0.0, a1_deg=360.0, n=36, collide=True):
    """Box segments approximating the ring r[0]..r[1] between angles a0..a1 (degrees from +x), height z[0]..z[1]."""
    out = []
    span = math.radians(a1_deg - a0_deg)
    step = span / n
    for k in range(n):
        a = math.radians(a0_deg) + (k + 0.5) * step
        rm = (r[0] + r[1]) / 2
        c = (center[0] + rm * math.cos(a), center[1] + rm * math.sin(a), center[2] + (z[0] + z[1]) / 2)
        length = 2 * r[1] * math.tan(step / 2) * 1.02
        out.append(_rbox(stage, f"{path}_{k:02d}", c, (r[1] - r[0], length, z[1] - z[0]), a, mat, collide))
    return out


def _tube(stage, path, center, r, z, material, a0_deg=0.0, a1_deg=360.0, n=48):
    """A visual tube (or part of one) as a mesh: inner and outer surfaces, end rings and side caps."""
    a0, a1 = math.radians(a0_deg), math.radians(a1_deg)
    full = abs(a1_deg - a0_deg) >= 359.9
    m = n if full else n + 1
    pts = []
    for k in range(m):
        a = a0 + (a1 - a0) * k / n
        for rr in r:
            for zz in z:
                pts.append((center[0] + rr * math.cos(a), center[1] + rr * math.sin(a), center[2] + zz))
    faces = []
    idx = lambda k, i, j: (k % m) * 4 + i * 2 + j  # noqa: E731
    for k in range(n):
        k1 = k + 1
        faces.append([idx(k, 1, 0), idx(k1, 1, 0), idx(k1, 1, 1), idx(k, 1, 1)])  # outer
        faces.append([idx(k, 0, 0), idx(k, 0, 1), idx(k1, 0, 1), idx(k1, 0, 0)])  # inner
        faces.append([idx(k, 0, 1), idx(k, 1, 1), idx(k1, 1, 1), idx(k1, 0, 1)])  # top
        faces.append([idx(k, 0, 0), idx(k1, 0, 0), idx(k1, 1, 0), idx(k, 1, 0)])  # bottom
    if not full:
        faces.append([idx(0, 0, 0), idx(0, 1, 0), idx(0, 1, 1), idx(0, 0, 1)])
        faces.append([idx(n, 0, 0), idx(n, 0, 1), idx(n, 1, 1), idx(n, 1, 0)])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in p]) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in faces for i in f]))
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _disc(stage, path, center, r, z, material, a0_deg=0.0, a1_deg=360.0, n=48):
    """A visual (partial) disc slab: a tube from radius 0."""
    return _tube(stage, path, center, (0.0, r), z, material, a0_deg, a1_deg, n)


def _body(stage, path, pos, mass):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(mass)
    return xform


def _joint(stage, path, kind, body0, pos0, body1, pos1, axis, lower, upper):
    joint = (UsdPhysics.RevoluteJoint if kind == "revolute" else UsdPhysics.PrismaticJoint).Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(str(body0))])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(str(body1))])
    joint.CreateAxisAttr(axis)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in pos0]))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*[float(v) for v in pos1]))
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular" if kind == "revolute" else "linear")
    return joint


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


def _paths(prims):
    return [str(p.GetPath()) for p in prims]


# Textures ----------------------------------------------------------------------------------------------------------
def write_textures(out_dir: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    for which, legend in LEGENDS.items():
        w, h = 800, 280
        img = Image.new("RGB", (w, h), (12, 12, 12))
        draw = ImageDraw.Draw(img)
        draw.rectangle([8, 8, w - 9, h - 9], outline=(150, 150, 150), width=6)
        size = 190
        while ImageFont.truetype(FONT, size).getlength(legend) > w * 0.8:
            size -= 4
        draw.text((w / 2, h / 2), legend, font=ImageFont.truetype(FONT, size), fill=(240, 240, 240), anchor="mm")
        img.save(out_dir / f"legend_{which}.png")


# Station -----------------------------------------------------------------------------------------------------------
def make_station(out_path: Path) -> None:
    root = "/station"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    _solver(root_prim)
    mats = _mats(stage, root)
    clear = _material(stage, root, "clear", CLEAR, roughness=0.1, opacity=CLEAR_OPACITY)
    enc = _body(stage, f"{root}/enclosure", (0.0, 0.0, 0.0), 2.0)
    cols = [_box(stage, f"{root}/enclosure/box", STATION_X, STATION_Y, (TABLE_Z, PANEL_Z), mats["station"])]
    for which in ("start", "stop"):
        c = button_root(which)
        cap = mats["cap_green" if which == "start" else "cap_red"]
        b = f"{root}/enclosure/{which}"
        UsdGeom.Xform.Define(stage, b)
        # Install base: the flat ring under the bezel (visual), its rim and flange (colliders), clear.
        _tube(stage, f"{b}/base_disc", c, BASE_DISC_R, (0.0, BASE_DISC_T), clear)
        _tube(stage, f"{b}/base_rim_v", c, (RIM_R[0], FLANGE_R[1]), (FLANGE_Z[0], RIM_TOP), clear)
        _tube(stage, f"{b}/base_wall_v", c, RIM_R, (0.0, FLANGE_Z[0]), clear)
        cols += _ring(stage, f"{b}/rim", c, RIM_R, (0.0, RIM_TOP), clear, collide=True)
        cols += _ring(stage, f"{b}/flange", c, FLANGE_R, FLANGE_Z, clear, collide=True)
        for p in _paths(cols[-72:]):
            kit.hide(stage, p)
        # Bezel (chrome ring) and the collar under it.
        _tube(stage, f"{b}/bezel_v", c, BEZEL_R, BEZEL_Z, mats["chrome"])
        cols += _ring(stage, f"{b}/bezel", c, BEZEL_R, BEZEL_Z, mats["chrome"], n=16, collide=True)
        for p in _paths(cols[-16:]):
            kit.hide(stage, p)
        _tube(stage, f"{b}/well_v", c, (0.0, BEZEL_R[0]), (0.0005, 0.001), mats["dark"])
        # Legend plate to the button's right (as seen by the robot), 40 x 12 mm.
        tex = _tex_material(stage, root, f"legend_{which}", f"legend_{which}.png")
        x0, x1 = c[0] - 0.007, c[0] + 0.007
        y0, y1 = c[1] - 0.063 - 0.020, c[1] - 0.063 + 0.020
        z = PANEL_Z + 0.0004
        _quad(stage, f"{b}/legend", [(x0, y1, z), (x0, y0, z), (x1, y0, z), (x1, y1, z)], tex)
        # Plunger.
        pl = _body(stage, f"{root}/plunger_{which}", c, PLUNGER_MASS)
        cols.append(_box(stage, f"{pl.GetPath()}/cap", (-CAP_HALF, CAP_HALF), (-CAP_HALF, CAP_HALF), CAP_Z, cap))
        kit.hide(stage, cols[-1].GetPath())
        _disc(stage, f"{pl.GetPath()}/cap_v", (0.0, 0.0, 0.0), CAP_R, (CAP_Z[0] + 0.0005, CAP_TOP), cap)
        _joint(stage, f"{root}/joints/press_{which}", "prismatic", enc.GetPath(), c, pl.GetPath(), (0.0, 0.0, 0.0),
               "Z", -TRAVEL, 0.0)
    kit.fix_to_world(stage, root, enc.GetPath())
    _offsets(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


# Cover -------------------------------------------------------------------------------------------------------------
def _half(stage, path, front: bool, mats, clear):
    """One half-shell in the cover frame (cover shut). The back half spans angles -90..90 deg from +x, the front half
    90..270. Returns (collider paths, grip collider paths)."""
    a0 = 90.0 if front else -90.0
    mid = a0 + 90.0
    cols = []
    cols += _ring(stage, f"{path}/wall", (0, 0, 0), (R_IN, R_OUT), (0.0, HEIGHT - TOP_T), clear, a0, a0 + 180.0, N_SEG)
    cols += _ring(stage, f"{path}/lip", (0, 0, 0), LIP_R, LIP_Z, clear, mid - LIP_SPAN_DEG, mid + LIP_SPAN_DEG, 10)
    # Top plate: strips across the half (parallel to the seam line, along y).
    sign = -1.0 if front else 1.0
    n_strip = 6
    for k in range(n_strip):
        u0, u1 = R_OUT * k / n_strip, R_OUT * (k + 1) / n_strip
        hy = math.sqrt(max(R_OUT ** 2 - u0 ** 2, 0.0))
        xs = tuple(sorted((sign * u0, sign * u1)))
        cols.append(_box(stage, f"{path}/top_{k}", xs, (-hy, hy), (HEIGHT - TOP_T, HEIGHT), clear))
    for p in _paths(cols):
        kit.hide(stage, p)
    # Visual shell.
    _tube(stage, f"{path}/wall_v", (0, 0, 0), (R_IN, R_OUT), (0.0, HEIGHT - TOP_T), clear, a0, a0 + 180.0)
    _disc(stage, f"{path}/top_v", (0, 0, 0), R_OUT, (HEIGHT - TOP_T, HEIGHT), clear, a0, a0 + 180.0)
    _tube(stage, f"{path}/lip_v", (0, 0, 0), LIP_R, LIP_Z, clear, mid - LIP_SPAN_DEG, mid + LIP_SPAN_DEG, 16)
    # Ear at the free seam (+y): a plate in the x-z plane with a square hole; the front half's ear is the outer one.
    ey = EAR_Y_OUT if front else EAR_Y_IN
    h = EAR_HOLE / 2
    grip = []
    for k, (x, z) in enumerate([((-EAR_HW, -h), EAR_Z), ((h, EAR_HW), EAR_Z),
                                ((-h, h), (EAR_HOLE_Z + h, EAR_Z[1])), ((-h, h), (EAR_Z[0], EAR_HOLE_Z - h))]):
        grip.append(_box(stage, f"{path}/ear_{k}", x, ey, z, clear))
    # Web joining the ear to its own half's wall (visual).
    wx = (-EAR_HW, 0.0) if front else (0.0, EAR_HW)
    _box(stage, f"{path}/ear_web", wx, (R_OUT - 0.0015, ey[0]), (EAR_Z[0], HEIGHT), clear, collide=False)
    # Hinge: knuckles along -y; the back half carries the barrel's collider (where a hand can hold it).
    for k, (z0, z1) in enumerate(((0.002, 0.016), (0.018, 0.032), (0.034, 0.048))):
        mine = (k % 2 == 1) == front
        if mine:
            kit.cylinder(stage, f"{path}/knuckle_{k}", (0.0, -HINGE_R, (z0 + z1) / 2), BARREL / 2, z1 - z0, clear)
    if not front:
        grip.append(_box(stage, f"{path}/barrel", (-BARREL / 2, BARREL / 2), (-HINGE_R - BARREL / 2, -HINGE_R + BARREL / 2),
                         (0.001, HEIGHT), clear))
        kit.hide(stage, grip[-1].GetPath())
    wx = (-0.0015, 0.0) if front else (0.0, 0.0015)
    _box(stage, f"{path}/hinge_web", wx, (-HINGE_R, -R_OUT + 0.0015), (0.002, 0.048), clear, collide=False)
    for p in _paths(grip[:4]):
        kit.hide(stage, p)
    _ear_visual(stage, f"{path}/ear_v", ey, clear)
    return _paths(cols), _paths(grip)


def _ear_visual(stage, path, ey, material):
    """The ear's visible plate: a frame around the hole (four boxes, so the hole shows)."""
    h = EAR_HOLE / 2
    for k, (x, z) in enumerate([((-EAR_HW, -h), EAR_Z), ((h, EAR_HW), EAR_Z),
                                ((-h, h), (EAR_HOLE_Z + h, EAR_Z[1])), ((-h, h), (EAR_Z[0], EAR_HOLE_Z - h))]):
        _box(stage, f"{path}_{k}", x, ey, z, material, collide=False)


def make_cover(out_path: Path) -> None:
    root = "/cover"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    _solver(root_prim)
    mats = _mats(stage, root)
    clear = _material(stage, root, "clear", CLEAR, roughness=0.1, opacity=CLEAR_OPACITY)
    back = _body(stage, f"{root}/half_b", (0.0, 0.0, 0.0), HALF_MASS)
    front = _body(stage, f"{root}/half_f", (0.0, 0.0, 0.0), HALF_MASS)
    cb, gb = _half(stage, f"{root}/half_b", False, mats, clear)
    cf, gf = _half(stage, f"{root}/half_f", True, mats, clear)
    h = hinge_point()
    _joint(stage, f"{root}/joints/hinge", "revolute", back.GetPath(), h, front.GetPath(), h, "Z", 0.0, 100.0)
    kit.friction(stage, root, cb + cf, static=PC_FRICTION, dynamic=PC_FRICTION, name="pc", combine="min")
    # The ears and barrel are where fingers take hold: the finger pads' rubber decides the grip.
    kit.friction(stage, root, gb + gf, static=1.2, dynamic=1.0, name="grip", combine="max")
    _offsets(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


# Stand -------------------------------------------------------------------------------------------------------------
LOCK_Z = PANEL_Z + EAR_HOLE_Z
"""The lying padlock's centre height on its pedestal: level with the ear holes."""
LOCK_HALF_T = 0.0095


def make_stand(out_path: Path) -> None:
    root = "/stand"
    stage, _ = kit.new_stage(out_path, root)
    mats = _mats(stage, root)
    x, y = COVER_START_XY
    _box(stage, f"{root}/block", (x - 0.06, x + 0.06), (y - 0.06, y + 0.06), (TABLE_Z, BLOCK_TOP), mats["stand"])
    # The padlock's pedestal: a level slab under the shackle half of the lying body (the fingers take the other end).
    px, py = PEDESTAL_XY
    top = LOCK_Z - LOCK_HALF_T - 0.0005
    _box(stage, f"{root}/ped_foot", (px - 0.025, px + 0.025), (py - 0.045, py + 0.005), (TABLE_Z, TABLE_Z + 0.01),
         mats["stand"])
    _box(stage, f"{root}/ped_slab", (px - 0.019, px + 0.019), (py - 0.026, py - 0.004), (top - 0.010, top),
         mats["stand"])
    _box(stage, f"{root}/ped_column", (px - 0.006, px + 0.006), (py - 0.023, py - 0.007), (TABLE_Z, top - 0.010),
         mats["stand"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def lip_clearance_report() -> None:
    """Check the geometry: closing the front half about the hinge never takes a lip point inside its shut radius, and
    the lips clear the flange when open by the starting angle; the back half's lips clear the flange 3 mm off."""
    from pushbutton_sim.spec import COVER_OPEN_DEG

    pts = []
    for a in np.linspace(math.radians(180 - LIP_SPAN_DEG), math.radians(180 + LIP_SPAN_DEG), 41):
        for r in LIP_R:
            pts.append((r * math.cos(a), r * math.sin(a), 0.0))
    worst = 1.0
    for th in np.linspace(0.0, math.radians(COVER_OPEN_DEG), 61)[1:]:
        for p in pts:
            q = half_f_point(p, th)
            worst = min(worst, math.hypot(q[0], q[1]) - math.hypot(p[0], p[1]))
    inner_open = min(math.hypot(*half_f_point((LIP_R[0] * math.cos(a), LIP_R[0] * math.sin(a), 0), math.radians(
        COVER_OPEN_DEG))[:2]) for a in np.linspace(math.radians(180 - LIP_SPAN_DEG), math.radians(180 + LIP_SPAN_DEG), 41))
    off = min(math.hypot(LIP_R[0] * math.cos(a) + 0.003, LIP_R[0] * math.sin(a))
              for a in np.linspace(-math.radians(LIP_SPAN_DEG), math.radians(LIP_SPAN_DEG), 41))
    print(f"lip check: min radial change while closing {worst * 1000:.2f} mm (>= 0 ok); front lips' inner radius "
          f"at {COVER_OPEN_DEG:.0f} deg open {inner_open * 1000:.1f} mm, back lips 3 mm off {off * 1000:.1f} mm "
          f"(flange outer {FLANGE_R[1] * 1000:.1f} mm)")


def main(out_dir: Path) -> None:
    from pushbutton_sim.usd import padlock_410, padlock_props

    out_dir.mkdir(parents=True, exist_ok=True)
    write_textures(out_dir)
    make_station(out_dir / "station.usda")
    make_cover(out_dir / "cover.usda")
    make_stand(out_dir / "stand.usda")
    blue = replace(padlock_props.PADLOCKS["ml410"], color="lock_blue")
    padlock_410.make_padlock_410(out_dir / "padlock_ml410.usda", blue)
    lip_clearance_report()
    print(f"ear holes at {np.round(np.array(button_root('start')) + [0, EAR_HOLE_Y, EAR_HOLE_Z], 4)}; "
          f"button force at rest {SPRING_K * SPRING_PRELOAD_M:.1f} N, at contact "
          f"{SPRING_K * (SPRING_PRELOAD_M + CONTACT_TRAVEL):.1f} N")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets")

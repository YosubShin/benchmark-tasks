"""Generate the props: a plate with a duplex NEMA 5-15R receptacle, two cords with molded NEMA 5-15P plugs plugged
into it, the Brady 65674 plug lockout lying open on its stand, the Master Lock 410 padlock and its pedestal.

Real parts (README References):
- NEMA 5-15P / 5-15R (NEMA WD 6 as summarised in public references): two flat blades 6.35 mm wide, 1.5 mm thick,
  15.9 mm long, 12.7 mm apart; a round ground pin 4.8 mm across, 3.2 mm longer, 11.9 mm below the blades' line.
  A duplex receptacle's two outlets are 39 mm apart (ground pin down).
- The plug body (38 x 30 x 30 mm with an 18 mm strain relief) is a typical molded 5-15P on 16/3 SJT cord (8 mm
  across, about 0.10 kg/m): inferred, not one vendor's drawing.
- Brady 65674 small plug lockout (polypropylene, red; distributor data: 3.5 in long, about 2 in across; plugs up to
  1.7 in across and 3.2 in long; cord hole 1/2 in): two half-shells on a hinge along one long edge; a semicircular
  notch at one end of each half makes the 1/2 in cord hole when closed; one flat tab per half, at mid-length on the
  free edge, perpendicular to the long axis; closed, the two tabs lie face to face and their holes line up for a
  padlock shackle (Brady product photos and animation).

How the lockout holds (geometry, nothing else): the plug body (38 x 30 mm) cannot pass the 12.7 mm cord hole; the
blades are inside the closed shell, so the plug cannot be plugged in; a shackle through both tabs' holes stops the
lid turning on its hinge (the lid's tab would have to move across the shackle). Without the padlock the lid swings
open freely (a light hinge damping only).

How the receptacle holds a plug: behind each blade slot a contact pad on a radial slide presses the blade against a
fixed contact wall (the task's preloaded joint drive); the friction on both faces gives about 5 N of withdrawal force.

Files (all in the desk frame):
``outlet.usda``: one fixed-base articulation: ``base`` (plate, cover plate, both outlet faces, contact walls) and
``pad_<outlet>_<blade>`` contact pads on prismatic joints (outlet = top or bottom, blade = 0 (-y) or 1 (+y)).
``cord_target.usda`` / ``cord_decoy.usda``: fixed-base articulations: ``anchor`` (fixed under the cord clamp), CORD_N
capsule links (revolute joints cycling about the two bending axes and the twist axis, springs toward the routed shape),
``plug`` (frame: origin on the plug's front face on the blade line, +x along the blades, +z toward the blades' side
away from the ground pin). The starting route is solved here and is the joints' zero.
``lockout.usda``: fixed-base articulation: ``base`` (the lower half-shell, held by its fixture) and ``lid`` on the
``hinge`` revolute joint (0 closed, 180 deg open flat). Box frame: origin at the closed box's centre on the parting
plane, world-aligned; the long axis is y, the hinge along the back edge (+x), the tabs on the front edge (-x).
``stand.usda``: the lockout's stand and fixture clips, and the padlock's pedestal.
``padlock_ml410.usda``: the blue Master Lock 410 (pluglock_sim/usd/padlock_410.py).

Usage: python -m pluglock_sim.usd.pluglock_props [<out dir>]   (default: pluglock_sim/assets)
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from pluglock_sim.spec import (
    ANCHOR_XY, BOX_FRONT_X, BOX_Y, OUTLET_FACE_X, OUTLET_Y, OUTLET_Z, PEDESTAL_XY, RIM_Z, TAGS,
)
from pluglock_sim.usd import usd_kit as kit

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TABLE_Z = -0.0025
"""The station table's top in the desk frame."""

# Plug (molded NEMA 5-15P), plug frame ----------------------------------------------------------------------------
BODY_X = (-0.030, 0.0)
BODY_HW = 0.019
BODY_Z = (-0.020, 0.010)
RELIEF_X = (-0.048, BODY_X[0])
RELIEF_HW = 0.008
CORD_Z = -0.005
"""The cord leaves the strain relief on the body's centre line."""
BLADE_L = 0.0159
BLADE_W = 0.00635
BLADE_T = 0.0015
BLADE_Y = 0.00635
PIN_L = 0.0191
PIN_D = 0.0048
PIN_Z = -0.0119
PLUG_MASS = 0.05
GRIP_X = -0.022
"""Where the fingers hold the plug (plug frame x): the middle of the body."""
PLUG_POINTS = ([(x, y, z) for x in BODY_X for y in (-BODY_HW, BODY_HW) for z in BODY_Z]
               + [(RELIEF_X[0], y, CORD_Z + dz) for y in (-RELIEF_HW, RELIEF_HW) for dz in (-RELIEF_HW, RELIEF_HW)]
               + [(BLADE_L, s * (BLADE_Y + BLADE_T / 2), dz) for s in (-1, 1) for dz in (-BLADE_W / 2, BLADE_W / 2)]
               + [(PIN_L, dy, PIN_Z + dz) for dy in (-PIN_D / 2, PIN_D / 2) for dz in (-PIN_D / 2, PIN_D / 2)])
"""Corners of the plug's parts (plug frame): body, strain relief end, blade tips, ground-pin tip. The in-lockout test
checks all of them."""

# Receptacle (duplex 5-15R), outlet frame: world-aligned, origin on the outlet face at the blade line ------------
FACE_T = 0.003
FACE_HW = 0.017
FACE_Z = (-0.021, 0.013)
SLOT_HW = 0.0011
SLOT_HH = 0.004
PIN_HOLE_H = 0.00275
WALL_Y = BLADE_Y + BLADE_T / 2
"""Fixed contact wall's inner face (the blade's outer face rests on it)."""
CONTACT_X = (FACE_T + 0.002, FACE_T + 0.012)
PAD_T = 0.0015
PAD_INTERFERENCE = 0.0003
PAD_REST = BLADE_Y - BLADE_T / 2 + PAD_INTERFERENCE
"""|y| of a pad's outer face at its stop: 0.3 mm into the blade's inner face."""
CONTACT_K = 2000.0
CONTACT_PRELOAD_M = 0.0015
"""Drive target beyond the stop: about 3.6 N on a blade; with brass friction 0.35 on both faces of both blades the
plug comes out at about 5 N. A real receptacle holds harder (UL 498 asks a 15 A receptacle to need more than 13 N to
withdraw its test plug): set low because the station arm's soft wrist cannot pull much harder without the plug
pitching in the hand and its blades binding in the slots."""
CONTACT_DAMPING = 5.0
CONTACT_MASS = 0.01
BRASS_FRICTION = 0.35
PLATE_T = 0.003
PLATE_Y = (OUTLET_Y - 0.10, OUTLET_Y + 0.10)
PLATE_TOP = 0.215
COVER = (0.070, 0.115)

# Cord: 16/3 SJT, black ----------------------------------------------------------------------------------------
CORD_LEN = 0.60
CORD_N = 24
CORD_LINK = CORD_LEN / CORD_N
CORD_R = 0.004
CORD_LINK_MASS = 0.10 * CORD_LINK
CORD_LINK_INERTIA = (1e-7, 2e-7, 2e-7)
CORD_EI = 0.0008
"""Bending stiffness, N*m^2 (inferred: scaled from 10/3 SJOOW by diameter^4; 10 cm held out droops about 1 cm)."""
CORD_K = CORD_EI / CORD_LINK
CORD_DAMPING = 0.004
CORD_LIMIT_DEG = 100.0
CORD_FRICTION = (0.6, 0.5)
AXES = ("Y", "Z", "X")
RELIEF_JOINTS = 2
RELIEF_K = 8.0 * CORD_K
"""The last two cord joints (at the plug and one link out) are stiffer and start straight: the molded strain relief
keeps the first 2.5 to 5 cm of cord in line with the plug (inferred)."""
TAG_LINK = CORD_N - 3
"""The cord tag is wrapped round this link (5 to 7.5 cm from the plug)."""
TAG_SIZE = (0.045, 0.022)

# Lockout (Brady 65674), box frame ------------------------------------------------------------------------------
BOX_HL = 0.0445
"""Half length (along y): 89 mm long."""
BOX_HW = 0.0255
"""Half width (along x): 51 mm across."""
HALF_D = 0.0255
"""Depth of each half-shell (along z)."""
SHELL_T = 0.0025
HOLE_R = 0.00635
"""Cord hole 1/2 in: a 12.7 mm square opening (half in each shell's -y end wall)."""
TAB_X = (-(BOX_HW + 0.024), -(BOX_HW + 0.0005))
TAB_HZ = 0.012
TAB_T = 0.003
TAB_GAP = 0.0002
TAB_HOLE = 0.0095
TAB_HOLE_X = -(BOX_HW + 0.0135)
BOX_CENTER = (BOX_FRONT_X + BOX_HW, BOX_Y, RIM_Z)
HINGE = (BOX_HW, 0.0, 0.0)
"""Hinge axis (along y) in the box frame: the back edge of the parting plane."""
LID_MASS = 0.035
BASE_MASS = 0.035
SHELL_FRICTION = 0.5
INTERIOR = ((-BOX_HW + SHELL_T, BOX_HW - SHELL_T), (-BOX_HL + SHELL_T, BOX_HL - SHELL_T),
            (-HALF_D + SHELL_T, HALF_D - SHELL_T))
STAND_X = (-BOX_HW + 0.0055, HINGE[0] + 2 * BOX_HW + 0.004)
"""Stand top in the box frame: set back 5.5 mm from the front so a padlock hanging from the tabs clears it; deep
enough to carry the lid lying open flat."""
STAND_HY = 0.060
PEDESTAL_TOP = 0.120
PED_TILT_DEG = 15.0
"""The pedestal's top slopes down toward the lockout by this much (the padlock lies on it shackle end down)."""

CONTACT_OFFSET = 0.0003
SOLVER_ITERATIONS = 32

COLORS = {
    "plug_black": (0.05, 0.05, 0.05),
    "brass": (0.80, 0.66, 0.36),
    "steel": (0.72, 0.74, 0.77),
    "ivory": (0.90, 0.88, 0.80),
    "cover": (0.80, 0.81, 0.82),
    "plate": (0.55, 0.57, 0.60),
    "slot": (0.08, 0.08, 0.08),
    "cord": (0.03, 0.03, 0.03),
    "clamp": (0.70, 0.71, 0.72),
    "lockout_red": (0.78, 0.05, 0.04),
    "stand": (0.40, 0.40, 0.42),
    "fixture": (0.25, 0.25, 0.27),
    "white": (0.95, 0.95, 0.95),
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


def outlet_root(which: str) -> tuple[float, float, float]:
    return (OUTLET_FACE_X, OUTLET_Y, OUTLET_Z[0] if which == "target" else OUTLET_Z[1])


def box_center() -> tuple[float, float, float]:
    return BOX_CENTER


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


def _joint(stage, path, kind, body0, frame0, body1, frame1, world_point, world_rot, axis, lower, upper):
    joint = (UsdPhysics.RevoluteJoint if kind == "revolute" else UsdPhysics.PrismaticJoint).Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(str(body0))])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(str(body1))])
    joint.CreateAxisAttr(axis)
    for (p, r), pos_attr, rot_attr in ((frame0, joint.CreateLocalPos0Attr, joint.CreateLocalRot0Attr),
                                       (frame1, joint.CreateLocalPos1Attr, joint.CreateLocalRot1Attr)):
        pos_attr(Gf.Vec3f(*[float(v) for v in r.T @ (np.asarray(world_point) - p)]))
        x, y, z, w = _mat_q(r.T @ world_rot)
        rot_attr(Gf.Quatf(float(w), float(x), float(y), float(z)))
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular" if kind == "revolute" else "linear")
    return joint


# Textures ---------------------------------------------------------------------------------------------------
def write_textures(out_dir: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    to8 = lambda c: tuple(int(round(v * 255)) for v in c)  # noqa: E731
    for which, (legend, color) in TAGS.items():
        w, h = 900, 440
        img = Image.new("RGB", (w, h), to8(color))
        draw = ImageDraw.Draw(img)
        size = 200
        while ImageFont.truetype(FONT, size).getlength(legend) > w * 0.88:
            size -= 4
        draw.text((w / 2, h / 2), legend, font=ImageFont.truetype(FONT, size), fill=(10, 10, 10), anchor="mm")
        img.save(out_dir / f"tag_{which}.png")
    # Lockout label (DANGER / LOCKED OUT, as on Brady's English label).
    w, h = 1000, 460
    img = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 40, 530, 200], fill=(10, 10, 10))
    draw.ellipse([50, 58, 510, 182], fill=(205, 20, 20))
    draw.text((280, 120), "DANGER", font=ImageFont.truetype(FONT, 96), fill=(255, 255, 255), anchor="mm")
    draw.text((280, 330), "LOCKED OUT", font=ImageFont.truetype(FONT, 66), fill=(10, 10, 10), anchor="mm")
    draw.line([570, 40, 570, 420], fill=(120, 120, 120), width=4)
    draw.text((780, 190), "DO NOT", font=ImageFont.truetype(FONT, 76), fill=(10, 10, 10), anchor="mm")
    draw.text((780, 290), "REMOVE", font=ImageFont.truetype(FONT, 76), fill=(10, 10, 10), anchor="mm")
    img.save(out_dir / "lockout_label.png")


# Outlet ------------------------------------------------------------------------------------------------------
def _outlet_face(stage, base, name, c, mats):
    """One outlet's face (box pieces around two blade slots and the ground-pin hole), its contact walls and stop."""
    x0, yc, zc = c
    fx = (x0, x0 + FACE_T)
    ys = lambda a, b: (yc + a, yc + b)  # noqa: E731
    zs = lambda a, b: (zc + a, zc + b)  # noqa: E731
    pieces = [
        (ys(-FACE_HW, FACE_HW), zs(SLOT_HH, FACE_Z[1])),
        (ys(-FACE_HW, -BLADE_Y - SLOT_HW), zs(-SLOT_HH, SLOT_HH)),
        (ys(-BLADE_Y + SLOT_HW, BLADE_Y - SLOT_HW), zs(-SLOT_HH, SLOT_HH)),
        (ys(BLADE_Y + SLOT_HW, FACE_HW), zs(-SLOT_HH, SLOT_HH)),
        (ys(-FACE_HW, FACE_HW), zs(PIN_Z + PIN_HOLE_H, -SLOT_HH)),
        (ys(-FACE_HW, -PIN_HOLE_H), zs(PIN_Z - PIN_HOLE_H, PIN_Z + PIN_HOLE_H)),
        (ys(PIN_HOLE_H, FACE_HW), zs(PIN_Z - PIN_HOLE_H, PIN_Z + PIN_HOLE_H)),
        (ys(-FACE_HW, FACE_HW), zs(FACE_Z[0], PIN_Z - PIN_HOLE_H)),
    ]
    for k, (y, z) in enumerate(pieces):
        _box(stage, f"{base}/{name}_face_{k}", fx, y, z, mats["ivory"])
    # Dark slot and pin-hole insides (visual).
    for k, s in enumerate((-1, 1)):
        _box(stage, f"{base}/{name}_slot_{k}", (x0 + 0.0005, x0 + FACE_T), (yc + s * BLADE_Y - SLOT_HW,
             yc + s * BLADE_Y + SLOT_HW), zs(-SLOT_HH, SLOT_HH), mats["slot"], collide=False)
    _box(stage, f"{base}/{name}_pin_hole", (x0 + 0.0005, x0 + FACE_T), ys(-PIN_HOLE_H, PIN_HOLE_H),
         zs(PIN_Z - PIN_HOLE_H, PIN_Z + PIN_HOLE_H), mats["slot"], collide=False)
    # Contact walls (fixed, outside each blade) and the stop behind the blade tips.
    walls = []
    for k, s in enumerate((-1, 1)):
        y = (yc + s * WALL_Y, yc + s * (WALL_Y + 0.002))
        walls.append(_box(stage, f"{base}/{name}_wall_{k}", (x0 + FACE_T, x0 + 0.020), tuple(sorted(y)),
                          zs(-0.004, 0.004), mats["brass"], collide=True).GetPath())
    _box(stage, f"{base}/{name}_stop", (x0 + 0.022, x0 + 0.026), ys(-FACE_HW, FACE_HW), zs(FACE_Z[0], FACE_Z[1]),
         mats["slot"], collide=True)
    return walls


def pad_frame(which: str, k: int):
    """(world position of the pad slide's origin, world rotation): local +x = outward from the outlet centre."""
    x0, yc, zc = outlet_root(which)
    s = -1 if k == 0 else 1
    rot = _rot("Z", s * math.pi / 2)
    return np.array([x0, yc + s * PAD_REST, zc]), rot


def make_outlet(out_path: Path) -> None:
    root = "/outlet"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    _solver(root_prim)
    mats = _mats(stage, root)
    base = _body(stage, f"{root}/base", (0.0, 0.0, 0.0), np.eye(3), 2.0)
    b = f"{root}/base"
    x0 = OUTLET_FACE_X
    # Plate (wall), with an opening behind the outlets for the contact chambers.
    px = (x0 + FACE_T, x0 + FACE_T + PLATE_T)
    zc_mid = (OUTLET_Z[0] + OUTLET_Z[1]) / 2
    oy, oz = (OUTLET_Y - 0.03, OUTLET_Y + 0.03), (zc_mid - 0.05, zc_mid + 0.05)
    for k, (y, z) in enumerate([(PLATE_Y, (TABLE_Z, oz[0])), (PLATE_Y, (oz[1], PLATE_TOP)),
                                ((PLATE_Y[0], oy[0]), oz), ((oy[1], PLATE_Y[1]), oz)]):
        _box(stage, f"{b}/plate_{k}", px, y, z, mats["plate"], collide=os.environ.get("PL_NOPLATE") != "1")
    _box(stage, f"{b}/plate_back", (x0 + 0.027, x0 + 0.05), (oy[0] - 0.01, oy[1] + 0.01), (oz[0] - 0.01, oz[1] + 0.01),
         mats["plate"])
    _box(stage, f"{b}/plate_foot", (px[1], px[1] + 0.08), PLATE_Y, (TABLE_Z, TABLE_Z + 0.006), mats["plate"])
    # Stainless cover plate (visual, behind the outlet faces) and the receptacle body around the faces.
    _box(stage, f"{b}/cover", (x0 + 0.0012, x0 + FACE_T), (OUTLET_Y - COVER[0] / 2, OUTLET_Y + COVER[0] / 2),
         (zc_mid - COVER[1] / 2, zc_mid + COVER[1] / 2), mats["cover"], collide=False)
    kit.cylinder(stage, f"{b}/cover_screw", (x0 + 0.0009, OUTLET_Y, zc_mid - 0.004), 0.0035, 0.0012,
                 mats["steel"], axis="X")
    walls = []
    for which in ("target", "decoy"):
        walls += _outlet_face(stage, b, which, outlet_root(which), mats)
    # Contact pads on radial slides.
    pads = []
    for which in ("target", "decoy"):
        for k in (0, 1):
            p, r = pad_frame(which, k)
            name = f"pad_{which}_{k}"
            body = _body(stage, f"{root}/{name}", p, r, CONTACT_MASS)
            # Pad frame: x outward (its outer face at x = 0), y along the outlet's x-axis direction rotated.
            lx = (-PAD_T, 0.0)
            ax = r.T @ np.array([1.0, 0, 0])  # world x in the pad frame
            # The pad spans CONTACT_X along world x; in the pad frame world x maps to +-local y.
            ly_world = np.array(CONTACT_X)
            ly = tuple(sorted(float(v) for v in (ax[1] * ly_world[0], ax[1] * ly_world[1])))
            _box(stage, f"{root}/{name}/pad", lx, ly, (-0.003, 0.003), mats["brass"])
            pads.append(f"{root}/{name}/pad")
            _joint(stage, f"{root}/joints/{name}_joint", "prismatic", base.GetPath(), (np.zeros(3), np.eye(3)),
                   body.GetPath(), (p, r), p, r, "X", -0.002, 0.0)
    kit.fix_to_world(stage, root, base.GetPath())
    kit.friction(stage, root, [str(w) for w in walls] + pads, static=BRASS_FRICTION, dynamic=BRASS_FRICTION,
                 name="brass", combine="min")
    _offsets(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


# Cords and plugs -----------------------------------------------------------------------------------------------
ANCHOR_HEADING_DEG = {"target": 60.0, "decoy": 85.0}
TAIL = np.array([RELIEF_X[0], 0.0, CORD_Z])


def anchor(which: str):
    x, y = ANCHOR_XY[which]
    return np.array([x, y, TABLE_Z + CORD_R]), _rot("Z", math.radians(ANCHOR_HEADING_DEG[which]))


def joint_axis(k: int) -> str:
    return AXES[k % len(AXES)]


def cord_fk(which: str, q):
    """Frames (pos, rot) of the CORD_N links and of the plug for joint angles q (CORD_N + 1 of them)."""
    p, r = anchor(which)
    p, r = p.copy(), r.copy()
    links = []
    for k in range(CORD_N):
        r = r @ _rot(joint_axis(k), q[k])
        links.append((p.copy(), r.copy()))
        p = p + r @ np.array([CORD_LINK, 0, 0])
    r = r @ _rot(joint_axis(CORD_N), q[CORD_N])
    return links, (p - r @ TAIL, r)


def plug_start(which: str):
    return np.array(outlet_root(which)), np.eye(3)


def _points(links):
    pts = []
    for p, r in links:
        pts += [p + r @ np.array([f * CORD_LINK, 0, 0]) for f in (0.25, 0.5, 0.75, 1.0)]
    return pts


def solve_route(which: str, avoid=(), verbose=True):
    """Joint angles that route the cord from its clamp along the table, up to its plug in the outlet. The cord leaves
    the plug heading toward the robot, stays off the other plug, the other cord, the plate and the arm base, and keeps
    out of the space above the plug where the hand comes down."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    sp, sr = plug_start(which)
    floor = TABLE_Z + CORD_R + 0.0003
    other = "decoy" if which == "target" else "target"
    ox, oy, oz = outlet_root(other)
    avoid = np.asarray(avoid).reshape(-1, 3) if len(avoid) else np.zeros((0, 3))

    tx, ty, tz = outlet_root("target")
    lo = np.array([ox + RELIEF_X[0], oy - BODY_HW, oz + BODY_Z[0]])
    hi = np.array([ox + 0.003, oy + BODY_HW, oz + BODY_Z[1]])

    def residual(q):
        links, (pos, rot) = cord_fk(which, q)
        err_p = (pos - sp) * 3000.0
        err_r = Rotation.from_matrix(sr.T @ rot).as_rotvec() * 300.0
        P = np.asarray(_points(links))
        pen = [np.maximum(0.0, floor - P[:, 2]) * 300.0, 0.6 * (P[:-12, 2] - floor)]
        E = P[:-2]
        # Off the other plug (its body and strain relief, with a margin).
        d = np.minimum(E - lo, hi - E).min(axis=1) + 0.006 + CORD_R
        pen.append(np.maximum(0.0, d) * 400.0)
        # Off the plate, inside the table, off the right arm's base.
        pen.append(np.maximum(0.0, E[:, 0] - (OUTLET_FACE_X - 0.004)) * 300.0)
        pen.append(np.maximum(0.0, 0.03 - E[:, 0]) * 300.0)
        pen.append(np.maximum(0.0, 0.12 - np.hypot(E[:, 0], E[:, 1] + 0.305)) * 300.0)
        # The hand's way down to the target plug.
        box = (E[:, 0] > tx - 0.07) & (E[:, 0] < tx) & (np.abs(E[:, 1] - ty) < 0.04)
        pen.append(np.maximum(0.0, E[:, 2] - (tz + 0.012)) * 300.0 * box)
        if len(avoid):
            dd = np.linalg.norm(P[:, None] - avoid[None], axis=2).min(axis=1)
            pen.append(np.maximum(0.0, 2 * CORD_R + 0.004 - dd) * 400.0)
        # The plug's strain relief keeps the cord straight for its first few centimetres.
        relief = 3.0 * np.asarray(q[-RELIEF_JOINTS:])
        return np.concatenate([err_p, err_r, 0.1 * np.asarray(q), relief, *pen])

    lim = math.radians(CORD_LIMIT_DEG) * 0.8
    best = None
    for trial in range(6):
        q0 = np.random.default_rng(trial).uniform(-0.4, 0.4, CORD_N + 1)
        sol = least_squares(residual, q0, bounds=(-lim, lim))
        if best is None or sol.cost < best.cost:
            best = sol
    links, (pos, rot) = cord_fk(which, best.x)
    if verbose:
        pts = _points(links)
        print(f"route {which}: plug pos error {np.linalg.norm(pos - sp) * 1000:.2f} mm, rot error "
              f"{np.degrees(np.linalg.norm(Rotation.from_matrix(sr.T @ rot).as_rotvec())):.2f} deg, lowest "
              f"{min(e[2] for e in pts) * 1000:.1f} mm, x {min(e[0] for e in pts):.3f}..{max(e[0] for e in pts):.3f}, "
              f"y {min(e[1] for e in pts):.3f}..{max(e[1] for e in pts):.3f}, max |q| "
              f"{np.degrees(np.abs(best.x).max()):.0f} deg, cost {best.cost:.4f}")
    return best.x


def _capsule_x(stage, path, length, radius, material):
    cap = UsdGeom.Capsule.Define(stage, path)
    cap.CreateAxisAttr("X")
    cap.CreateRadiusAttr(radius)
    cap.CreateHeightAttr(float(length))
    UsdGeom.Xformable(cap).AddTranslateOp().Set(Gf.Vec3d(length / 2, 0, 0))
    UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    return cap


def _plug_geometry(stage, c, mats):
    """The molded 5-15P in the plug frame; returns (body colliders, blade colliders)."""
    body = [str(_box(stage, f"{c}/body", BODY_X, (-BODY_HW, BODY_HW), BODY_Z, mats["plug_black"]).GetPath())]
    body.append(str(_box(stage, f"{c}/relief", RELIEF_X, (-RELIEF_HW, RELIEF_HW), (CORD_Z - RELIEF_HW,
                                                                                    CORD_Z + RELIEF_HW),
                         mats["plug_black"]).GetPath()))
    # Grip ribs (visual) on both sides.
    for s in (-1, 1):
        for j in range(5):
            x = BODY_X[0] + 0.006 + 0.004 * j
            _box(stage, f"{c}/rib_{'ab'[s > 0]}{j}", (x - 0.001, x + 0.001), tuple(sorted((s * BODY_HW,
                 s * (BODY_HW + 0.0006)))), (BODY_Z[0] + 0.006, BODY_Z[1] - 0.006), mats["plug_black"], collide=False)
    blades = []
    for k, s in enumerate((-1, 1)):
        blades.append(str(_box(stage, f"{c}/blade_{k}", (-0.001, BLADE_L), (s * BLADE_Y - BLADE_T / 2,
                               s * BLADE_Y + BLADE_T / 2), (-BLADE_W / 2, BLADE_W / 2), mats["brass"]).GetPath()))
    blades.append(str(_box(stage, f"{c}/pin", (-0.001, PIN_L), (-PIN_D / 2, PIN_D / 2),
                           (PIN_Z - PIN_D / 2, PIN_Z + PIN_D / 2), mats["brass"], collide=True).GetPath()))
    kit.hide(stage, f"{c}/pin")
    kit.cylinder(stage, f"{c}/pin_vis", (PIN_L / 2 - 0.0005, 0.0, PIN_Z), PIN_D / 2, PIN_L + 0.001, mats["brass"],
                 axis="X")
    return body, blades


def make_cord(out_path: Path, which: str, q=None) -> None:
    root = f"/cord_{which}"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    _solver(root_prim)
    mats = _mats(stage, root)
    q = solve_route(which) if q is None else q
    links, (plug_p, plug_r) = cord_fk(which, q)
    a_p, a_r = anchor(which)
    anc = _body(stage, f"{root}/anchor", (0.0, 0.0, 0.0), np.eye(3), 0.05, (tuple(a_p), (1e-5, 1e-5, 1e-5)))
    a = f"{root}/anchor"
    # Cord clamp (visual): a strap screwed to the table over the cord, and the cord's tail under it.
    hx = a_r @ np.array([1.0, 0, 0])
    yaw = math.degrees(math.atan2(hx[1], hx[0]))
    clamp = UsdGeom.Xform.Define(stage, f"{a}/clamp")
    clamp.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in a_p]))
    clamp.AddRotateZOp().Set(yaw)
    _box(stage, f"{a}/clamp/strap", (-0.006, 0.006), (-0.014, 0.014), (CORD_R, CORD_R + 0.002), mats["clamp"],
         collide=False)
    _box(stage, f"{a}/clamp/tail", (-0.04, 0.0), (-CORD_R, CORD_R), (-CORD_R, CORD_R), mats["cord"], collide=False)
    kit.fix_to_world(stage, root, anc.GetPath())

    prev, prev_frame = anc.GetPath(), (np.zeros(3), np.eye(3))
    jackets = []
    for k, (p, r) in enumerate(links):
        body = _body(stage, f"{root}/cord_{k:02d}", p, r, CORD_LINK_MASS, ((CORD_LINK / 2, 0, 0), CORD_LINK_INERTIA))
        jackets.append(str(_capsule_x(stage, f"{root}/cord_{k:02d}/jacket", CORD_LINK, CORD_R, mats["cord"]).GetPath()))
        _joint(stage, f"{root}/joints/cord_j{k:02d}", "revolute", prev, prev_frame, body.GetPath(), (p, r), p, r,
               joint_axis(k), -CORD_LIMIT_DEG, CORD_LIMIT_DEG)
        prev, prev_frame = body.GetPath(), (p, r)
        if k == TAG_LINK:
            _cord_tag(stage, f"{root}/cord_{k:02d}", which, r, mats)
    plug = _body(stage, f"{root}/plug", plug_p, plug_r, PLUG_MASS)
    tail = plug_p + plug_r @ TAIL
    _joint(stage, f"{root}/joints/cord_j{CORD_N:02d}", "revolute", prev, prev_frame, plug.GetPath(),
           (plug_p, plug_r), tail, plug_r, joint_axis(CORD_N), -CORD_LIMIT_DEG, CORD_LIMIT_DEG)
    c = f"{root}/plug"
    plug.GetPrim().AddAppliedSchema("PhysxRigidBodyAPI")
    plug.GetPrim().CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(0.5)
    body, blades = _plug_geometry(stage, c, mats)
    kit.friction(stage, root, blades, static=BRASS_FRICTION, dynamic=BRASS_FRICTION, name="brass", combine="min")
    kit.friction(stage, root, body, static=1.5, dynamic=1.2, name="grip", combine="max")
    kit.friction(stage, root, jackets, static=CORD_FRICTION[0], dynamic=CORD_FRICTION[1], name="jacket",
                 combine="min")
    _offsets(stage)
    stage.GetRootLayer().Save()
    np.save(out_path.with_name(f"cord_{which}_route_q.npy"), np.asarray(q))
    print(f"wrote {out_path}")


def _cord_tag(stage, link_path, which, link_rot, mats):
    """A wrap-around cord tag (visual): a sleeve round the link and a flag standing up from it (world up at the start
    pose), the legend printed on both faces of the flag."""
    up = link_rot.T @ np.array([0.0, 0.0, 1.0])
    up = up - up[0] * np.array([1.0, 0, 0])
    up /= np.linalg.norm(up)
    side = np.cross(up, np.array([1.0, 0, 0]))
    color = _material(stage, link_path, f"tag_{which}_color", TAGS[which][1])
    tex = _tex_material(stage, link_path, f"tag_{which}_legend", f"tag_{which}.png")
    xf = UsdGeom.Xform.Define(stage, f"{link_path}/tag")
    m = np.stack([np.array([1.0, 0, 0]), side, up], axis=1)
    x, y, z, w = _mat_q(m)
    xf.AddTranslateOp().Set(Gf.Vec3d(CORD_LINK / 2, 0, 0))
    xf.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
    t = f"{link_path}/tag"
    L, H = TAG_SIZE
    _box(stage, f"{t}/sleeve", (-L / 2, L / 2), (-CORD_R - 0.0008, CORD_R + 0.0008), (-CORD_R - 0.0008, CORD_R + 0.0008),
         color, collide=False)
    _box(stage, f"{t}/flag", (-L / 2, L / 2), (-0.0006, 0.0006), (CORD_R, CORD_R + H), color, collide=False)
    z0, z1 = CORD_R + 0.002, CORD_R + H - 0.002
    for s, name in ((1, "a"), (-1, "b")):
        yv = s * 0.0008
        xs = (-L / 2 + 0.002, L / 2 - 0.002) if s < 0 else (L / 2 - 0.002, -L / 2 + 0.002)
        _quad(stage, f"{t}/legend_{name}", [(xs[0], yv, z0), (xs[1], yv, z0), (xs[1], yv, z1), (xs[0], yv, z1)], tex)
    # Top edge, visible from above.
    _box(stage, f"{t}/flag_top", (-L / 2, L / 2), (-0.0012, 0.0012), (CORD_R + H - 0.001, CORD_R + H), color,
         collide=False)


# Lockout ------------------------------------------------------------------------------------------------------
def _shell(stage, path, sign, mats, label_tex=None):
    """One half-shell in the box frame with the lid closed: sign -1 the lower half (z <= 0), +1 the lid (z >= 0).
    Returns the collider paths."""
    red = mats["lockout_red"]
    gap = 0.0003 if sign > 0 else 0.0
    z_out, z_in = sign * HALF_D, sign * (HALF_D - SHELL_T)
    zr = tuple(sorted((sign * gap, z_out)))
    out = []
    out.append(_box(stage, f"{path}/floor", (-BOX_HW, BOX_HW), (-BOX_HL, BOX_HL), tuple(sorted((z_in, z_out))), red))
    out.append(_box(stage, f"{path}/front", (-BOX_HW, -BOX_HW + SHELL_T), (-BOX_HL, BOX_HL), zr, red))
    out.append(_box(stage, f"{path}/back", (BOX_HW - SHELL_T, BOX_HW), (-BOX_HL, BOX_HL), zr, red))
    out.append(_box(stage, f"{path}/end_pos", (-BOX_HW, BOX_HW), (BOX_HL - SHELL_T, BOX_HL), zr, red))
    # -y end wall with the half cord hole.
    ye = (-BOX_HL, -BOX_HL + SHELL_T)
    out.append(_box(stage, f"{path}/end_neg_0", (-BOX_HW, -HOLE_R), ye, zr, red))
    out.append(_box(stage, f"{path}/end_neg_1", (HOLE_R, BOX_HW), ye, zr, red))
    out.append(_box(stage, f"{path}/end_neg_2", (-HOLE_R, HOLE_R), ye, tuple(sorted((sign * HOLE_R, z_out))), red))
    # The padlock tab: a plate across the long axis at mid-length on the front edge, with its hole.
    ty = (TAB_GAP, TAB_GAP + TAB_T) if sign > 0 else (-TAB_GAP - TAB_T, -TAB_GAP)
    h = TAB_HOLE / 2
    for k, (x, z) in enumerate([((TAB_HOLE_X + h, TAB_X[1]), (-TAB_HZ, TAB_HZ)),
                                ((TAB_X[0], TAB_HOLE_X - h), (-TAB_HZ, TAB_HZ)),
                                ((TAB_HOLE_X - h, TAB_HOLE_X + h), (h, TAB_HZ)),
                                ((TAB_HOLE_X - h, TAB_HOLE_X + h), (-TAB_HZ, -h))]):
        out.append(_box(stage, f"{path}/tab_{k}", x, ty, z, red))
    _box(stage, f"{path}/tab_root", (TAB_X[1], -BOX_HW + 0.0002), ty, (-TAB_HZ, TAB_HZ), red, collide=False)
    # Ribs on the outer face (visual), as on the real shell.
    for j, y in enumerate((-0.022, -0.017, 0.017, 0.022)):
        _box(stage, f"{path}/rib_{j}", (-BOX_HW - 0.0006, -BOX_HW), (y - 0.0012, y + 0.0012),
             tuple(sorted((sign * 0.004, sign * (HALF_D - 0.003)))), red, collide=False)
    if label_tex is not None:
        zl = sign * (HALF_D + 0.0004)
        lx, ly = 0.017, 0.035
        _quad(stage, f"{path}/label", [(-lx, ly, zl), (-lx, -ly, zl), (lx, -ly, zl), (lx, ly, zl)], label_tex)
    return [str(p.GetPath()) for p in out]


def make_lockout(out_path: Path) -> None:
    root = "/lockout"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    _solver(root_prim)
    mats = _mats(stage, root)
    c = np.array(BOX_CENTER)
    # The fixed root link sits at the asset's origin (Isaac Lab writes the root pose there at reset); its shell is
    # authored under an Xform at the box centre. The lid's frame is the box frame (lid closed).
    base = _body(stage, f"{root}/base", np.zeros(3), np.eye(3), BASE_MASS)
    UsdGeom.Xform.Define(stage, f"{root}/base/shell").AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in c]))
    lid = _body(stage, f"{root}/lid", c, np.eye(3), LID_MASS)
    label = _tex_material(stage, root, "label", "lockout_label.png")
    cols = _shell(stage, f"{root}/base/shell", -1, mats)
    cols += _shell(stage, f"{root}/lid", 1, mats, label_tex=label)
    # Hinge knuckles (visual) along the back edge.
    for k, (y0, y1, part) in enumerate(((-0.040, -0.026, "base"), (-0.024, -0.008, "lid"), (-0.006, 0.006, "base"),
                                        (0.008, 0.024, "lid"), (0.026, 0.040, "base"))):
        kit.cylinder(stage, f"{root}/{'base/shell' if part == 'base' else part}/knuckle_{k}",
                     (HINGE[0] + 0.001, (y0 + y1) / 2, 0.0), 0.0035, y1 - y0,
                     mats["lockout_red"], axis="Y")
    hinge_w = c + np.array(HINGE)
    joint = _joint(stage, f"{root}/joints/hinge", "revolute", base.GetPath(), (np.zeros(3), np.eye(3)), lid.GetPath(),
                   (c, np.eye(3)), hinge_w, np.eye(3), "Y", 0.0, 182.0)
    kit.fix_to_world(stage, root, base.GetPath())
    tabs = [c for c in cols if "/tab_" in c]
    kit.friction(stage, root, [c for c in cols if c not in tabs], static=SHELL_FRICTION, dynamic=SHELL_FRICTION,
                 name="pp", combine="min")
    # The tabs are where fingers take hold: the finger pads' rubber decides the grip (as on the plug body).
    kit.friction(stage, root, tabs, static=1.2, dynamic=1.0, name="tab_grip", combine="max")
    _offsets(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path} (hinge at {np.round(hinge_w, 4)})")
    return joint


def make_stand(out_path: Path) -> None:
    root = "/stand"
    stage, _ = kit.new_stage(out_path, root)
    mats = _mats(stage, root)
    cx, cy, cz = BOX_CENTER
    top = cz - HALF_D - 0.0003
    _box(stage, f"{root}/block", (cx + STAND_X[0], cx + STAND_X[1]), (cy - STAND_HY, cy + STAND_HY), (TABLE_Z, top),
         mats["stand"])
    # The fixture: two clips that hold the lower half-shell's ends at its back corners (the front corners stay clear
    # for the hand that brings the padlock).
    for kx, xs in enumerate(((BOX_HW - 0.016, BOX_HW - 0.004),)):
        for ky, ys in enumerate(((-BOX_HL - 0.006, -BOX_HL), (BOX_HL, BOX_HL + 0.006))):
            _box(stage, f"{root}/clip_{kx}{ky}", (cx + xs[0], cx + xs[1]), (cy + ys[0], cy + ys[1]),
                 (top, cz - 0.012), mats["fixture"])
    # The padlock's pedestal: a column under the shackle end of the lying padlock's body (its centre of mass, with the
    # shackle, is over the column); the body's other end overhangs it, so the lower finger passes under it.
    x, y = PEDESTAL_XY
    _box(stage, f"{root}/ped_foot", (x - 0.025, x + 0.025), (y - 0.045, y + 0.005), (TABLE_Z, TABLE_Z + 0.01),
         mats["stand"])
    # The slab's top is the padlock's support plane: through (x, y, PEDESTAL_TOP), sloping down toward -y by
    # PED_TILT_DEG; it spans 4 to 26 mm from the body's centre toward its shackle end (the fingers take the other end).
    t = math.radians(PED_TILT_DEG)
    axis = np.array([0.0, -math.cos(t), -math.sin(t)])
    normal = np.array([0.0, -math.sin(t), math.cos(t)])
    centre = np.array([x, y, PEDESTAL_TOP]) + 0.015 * axis - 0.005 * normal
    slab = UsdGeom.Xform.Define(stage, f"{root}/ped_slab")
    slab.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
    slab.AddRotateXOp().Set(PED_TILT_DEG)
    kit.box(stage, f"{root}/ped_slab/top", (0.0, 0.0, 0.0), (0.038, 0.022, 0.010), mats["stand"])
    low = centre - 0.011 * abs(axis[1]) * np.array([0, 1, 0])
    _box(stage, f"{root}/ped_column", (x - 0.006, x + 0.006), (centre[1] - 0.008, centre[1] + 0.008),
         (TABLE_Z, centre[2] - 0.004), mats["stand"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def main(out_dir: Path) -> None:
    from pluglock_sim.usd import padlock_410, padlock_props

    out_dir.mkdir(parents=True, exist_ok=True)
    write_textures(out_dir)
    make_outlet(out_dir / "outlet.usda")
    # The solved routes are kept beside the assets (solving takes minutes); delete them to solve again.
    cached = {w: out_dir / f"cord_{w}_route_q.npy" for w in ("decoy", "target")}
    q_decoy = np.load(cached["decoy"]) if cached["decoy"].exists() else solve_route("decoy")
    links, _ = cord_fk("decoy", q_decoy)
    make_cord(out_dir / "cord_decoy.usda", "decoy", q_decoy)
    q_target = (np.load(cached["target"]) if cached["target"].exists()
                else solve_route("target", avoid=_points(links)))
    make_cord(out_dir / "cord_target.usda", "target", q_target)
    make_lockout(out_dir / "lockout.usda")
    make_stand(out_dir / "stand.usda")
    blue = replace(padlock_props.PADLOCKS["ml410"], color="lock_blue")
    padlock_410.make_padlock_410(out_dir / "padlock_ml410.usda", blue)
    print(f"plug withdrawal (both blades, both faces): {4 * BRASS_FRICTION * CONTACT_K * (CONTACT_PRELOAD_M + PAD_INTERFERENCE):.1f} N")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets")

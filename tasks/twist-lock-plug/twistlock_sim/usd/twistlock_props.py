"""Generate the props: a vertical plate with two NEMA L5-30R twist-lock flanged receptacles (Hubbell HBL2616), the
HBL2611 L5-30P cord cap on its 10/3 SJOOW cord, and the cradle the cap starts on.

Real parts (README References): Hubbell HBL2611 (L5-30P cord cap, drawing M-7899: 57.5 mm body, 50.0 mm face, 69.8 mm
long without blades, 0.070 in blades), Hubbell HBL2616 (L5-30R flanged receptacle, drawing M-7911: 73.2 mm flange,
3.3 mm thick, screws on a 63.5 mm circle), 10/3 SJOOW portable cord (about 14.7 mm across, about 0.34 kg/m).

How the real lock works and how it is modelled (geometry):
- Each of the cap's three curved blades carries a foot near its tip that sticks out radially. The receptacle face
  has three curved slots, each with a wider entry notch at one end that lets a foot through only in the entry
  orientation. Pushed in, the feet pass the face into the contact chamber. Turned clockwise (seen from the front),
  the blades run along their arc slots and the feet go under solid face: a pull presses the feet against the back of
  the face. The slots' far ends stop the turn.
- Here: the face is a stack of convex annular sectors with real gaps for the slots and notches; the blades are convex
  arc sectors, each with its foot. Nothing else holds the cap (no joint, latch or attach).
- Contact springs: behind the face, one brass contact per blade is a small link on a radial slide, pressed outward
  onto the blade's inner face by a preloaded spring (the task's joint drive). It grips the blade from the moment its
  tip passes the face's back until the end of the turn: a few newtons of insertion drag and a turning resistance;
  the turned cap does not turn back by itself.

``panel.usda``: one fixed-base articulation. ``base`` holds the plate, both receptacles and the nameplates;
``contact_<rec>_<j>`` (rec = target or decoy, j = 0..2) are the contact springs on prismatic joints.
``cord_cap.usda``: one fixed-base articulation. ``anchor`` (fixed to the desk under the cord clamp), CORD_N capsule
links (revolute joints cycling about the two bending axes and the twist axis, each with a spring toward the routed
shape, set by the task), and ``cap``: the cord cap, one rigid body (frame: origin on the cap's face, on its axis,
+x out through the blades). The cord's starting route (clamp -> along the desk -> up to the cap's back on its cradle)
is solved here (``solve_route``) and is the joints' zero.
``cradle.usda``: two rails on a post.

Frames. Receptacle frame M (world-aligned): origin on the receptacle axis in its face's front plane, +x into the
receptacle (toward the plate), +z up. Angles theta about +x are measured from +y toward +z, so a positive turn about
+x is clockwise as seen by someone standing in front of the plate. The cap is home in M before the turn; turned to
lock, it is M rotated by about +LOCK_DEG about x.

Usage: python -m twistlock_sim.usd.twistlock_props [<out dir>]   (default: twistlock_sim/assets)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from twistlock_sim.spec import (
    CLAMP_XY, DECOY_Y, FACE_X, LABEL_COLORS, LEGENDS, PLATE_Y, PLUG_START_ROLL_DEG, PLUG_START_XY, REC_Z, TARGET_Y,
)
from twistlock_sim.usd import usd_kit as kit

MM = 0.001
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TABLE_Z = -0.0025
"""The station table's top in the desk frame."""

# Cap (HBL2611; Hubbell drawing M-7899) ----------------------------------------------------------------------------
R_BODY = 0.02875
"""Ribbed black body: 57.5 mm max diameter (published)."""
R_FACE_PLUG = 0.025
"""White front: 50.0 mm face diameter (published)."""
X_FRONT, X_BODY, X_TAIL = -0.015, -0.055, -0.0698
"""White front to -15 mm, black ribbed body to -55 mm, cord-grip end to -69.8 mm (69.8 mm overall without blades,
published; the split is scaled from the product photo)."""
R_TAIL = 0.022
PLUG_MASS = 0.25
"""HBL2611 about 0.25 kg (inferred from shipping weights)."""
GRIP_X = -0.035
"""Where the fingers hold the cap (its frame): the black ribbed body, 35 mm behind the face."""

# L5-30 blade pattern (measured off Hubbell drawings M-7899 / M-7911; NEMA WD 6 is not public) -----------------
R_B = 0.011
BLADE_T = 0.00178
BLADE_L = 0.021
BLADE_ARC_DEG = 45.0
BLADE_THETA = (138.0, 248.0, 0.0)
"""Blade centre angles in the cap frame (not 120 deg apart: the cap fits one way only)."""
NECK_L = 0.0065
TAB_H = 0.0020
TAB_L = BLADE_L - NECK_L
TAB_ARC_DEG = 15.0

# Receptacle (HBL2616; Hubbell drawing M-7911, install sheet PD1257) --------------------------------------------
LOCK_PLAY = 0.0005
FACE_T = NECK_L - LOCK_PLAY
LOCK_X = FACE_T
LOCK_DEG = 35.0
CLR = 0.0004
CLR_DEG = math.degrees(CLR / R_B)
R_FACE = 0.022
R_FLANGE = 0.0366
FLANGE_X = (0.003, 0.0063)
CHAMBER_DEPTH = BLADE_L + 0.004
PLATE_T = 0.003
PLATE_Z = (0.0, 0.235)
"""Plate height: its top stays under the right hand at the arms' ready pose (fingers at about 0.25 m)."""
PLATE_HOLE_HALF = 0.024
LABEL_SIZE = (0.100, 0.036)
"""Nameplate (100 x 36 mm, engraved phenolic): the circuit number in 17 mm letters, the load below it."""
LABEL_Z = REC_Z + R_FLANGE + 0.004 + LABEL_SIZE[1] / 2

R_IN = R_B - BLADE_T / 2 - CLR
R_MID = R_B + BLADE_T / 2 + CLR
R_OUT = R_B + BLADE_T / 2 + TAB_H + CLR

# Contact springs (inferred: the real contacts' force and shape are not published) -------------------------------
CONTACT_X = (FACE_T + 0.0015, FACE_T + 0.0115)
"""Along the receptacle axis: from 1.5 mm behind the face's back, 10 mm long (a blade tip reaches it 13.5 mm short of
home)."""
CONTACT_THETA_OFF = 17.5
"""Angle of each contact from its blade's entry centre: inside the 10 deg window the blade's inner face covers from
the entry orientation through the full 35 deg turn."""
CONTACT_HALF_W = 0.0008
CONTACT_INTERFERENCE = 0.0003
CONTACT_R = (R_B - BLADE_T / 2 - 0.0020, R_B - BLADE_T / 2 + CONTACT_INTERFERENCE)
"""Radial extent at rest: its top stands 0.3 mm into the blade's inner face (the blade pushes it in)."""
CONTACT_LEAD = 0.004
"""The contact's front end ramps 0.8 mm down over 4 mm (the blade tip's lead-in)."""
CONTACT_TRAVEL = 0.002
CONTACT_K = 2000.0
"""Spring rate, N/m (inferred)."""
CONTACT_PRELOAD_M = 0.0020
"""Drive target beyond the rest stop: 4 N at rest, about 4.6 N on a blade (inferred). With brass-on-brass friction
0.35 the three contacts resist the turn with up to about 0.05 N*m and the push with up to about 5 N."""
CONTACT_DAMPING = 5.0
CONTACT_MASS = 0.01
"""A real contact is ~2 g; heavier for a stable contact at 240 Hz."""
CONTACT_FRICTION = 0.35

# Cord: 10/3 SJOOW, black (distributor spec sheets: 0.58 in OD, 0.23 lb/ft) ---------------------------------------
CORD_LEN = 0.60
CORD_N = 24
CORD_LINK = CORD_LEN / CORD_N
CORD_R = 0.0147 / 2
CORD_LINK_MASS = 0.34 * CORD_LINK
CORD_LINK_INERTIA = (5e-7, 6e-7, 6e-7)
"""About the link's centre (x along it); the capsule's own is about 2.3e-7 axial and 5.6e-7 across."""
CORD_EI = 0.0075
"""Bending stiffness, N*m^2 (inferred: 10 cm of the cord held out level droops about 5 mm)."""
CORD_K = CORD_EI / CORD_LINK
CORD_DAMPING = 0.01
CORD_LIMIT_DEG = 100.0
CORD_FRICTION = (0.6, 0.5)
"""Rubber jacket on the desk."""
AXES = ("Y", "Z", "X")
ANCHOR = (CLAMP_XY[0], CLAMP_XY[1], TABLE_Z + CORD_R)
CLAMP_LEN = 0.03

PLUG_FRICTION = 0.35
GRIP_FRICTION = (1.5, 1.2)
CONTACT_OFFSET = 0.0003
SOLVER_ITERATIONS = 32
SEG_DEG = 8.0
DISK_SIDES = 32
"""A full disk (the cap's front, body and tail; the receptacle face's centre; the chamber's back) collides as ONE
convex 32-gon prism (64 vertices, PhysX's hull limit), not as SEG_DEG pie slices. Two flat faces pressed together that
are both cut into pie slices blew the simulation up: wherever a slice of the cap's front overlapped a slice of the
receptacle face by a thin sliver of angle (any turn that is not a multiple of SEG_DEG), the pair's least
penetration was sideways across the sliver, not along the axis, so as soon as the push pressed the faces a fraction
of a millimetre together the contacts pushed sideways and twisted instead of holding the cap out (cap through the
face, or NaN). The slices stay as the visual mesh."""

COLORS = {
    "cap_black": (0.06, 0.06, 0.06),
    "cap_white": (0.88, 0.87, 0.82),
    "brass": (0.78, 0.66, 0.36),
    "face_black": (0.07, 0.07, 0.07),
    "plate": (0.36, 0.39, 0.41),
    "enclosure": (0.42, 0.45, 0.47),
    "screw": (0.75, 0.75, 0.78),
    "cradle": (0.40, 0.40, 0.42),
    "cord": (0.025, 0.025, 0.025),
    "clamp": (0.70, 0.71, 0.72),
    "mark": (0.95, 0.95, 0.95),
}
"""The HBL2611 and HBL2616 are black nylon with a white front ring on the cap (Hubbell product photos)."""


def blade_ranges(theta_c: float):
    lo, hi = theta_c - BLADE_ARC_DEG / 2, theta_c + BLADE_ARC_DEG / 2
    return (lo, hi), (hi - TAB_ARC_DEG, hi)


def slot_ranges(theta_c: float):
    (lo, hi), (tlo, thi) = blade_ranges(theta_c)
    return (lo - CLR_DEG, hi + LOCK_DEG + CLR_DEG), (tlo - CLR_DEG, thi + CLR_DEG)


LOCKED_MIN_DEG = TAB_ARC_DEG + 2 * CLR_DEG + 2.0
"""Turned at least this far, every foot is wholly under solid face."""


def rec_root(which: str = "target") -> tuple[float, float, float]:
    return (FACE_X, TARGET_Y if which == "target" else DECOY_Y, REC_Z)


def plug_start_root() -> tuple[float, float, float]:
    cx, cy = PLUG_START_XY
    return (cx - X_TAIL / 2, cy, REC_Z - 0.010)


def plug_start_quat() -> tuple[float, float, float, float]:
    h = math.radians(PLUG_START_ROLL_DEG) / 2
    return (math.sin(h), 0.0, 0.0, math.cos(h))


def plug_body_center_offset() -> tuple[float, float, float]:
    return (X_TAIL / 2, 0.0, 0.0)


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
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in p]) for p in points]))
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


def _disk_collider(stage, path, r, x0, x1, center):
    """One invisible convex DISK_SIDES-gon prism along x (flats on the circle of radius r): a full disk's collider."""
    cx, cy, cz = center
    ro = r / math.cos(math.pi / DISK_SIDES)
    yz = [(cy + ro * math.cos(2 * math.pi * (k + 0.5) / DISK_SIDES), cz + ro * math.sin(2 * math.pi * (k + 0.5) /
                                                                                         DISK_SIDES))
          for k in range(DISK_SIDES)]
    m = len(yz)
    pts = [(cx + x0, y, z) for y, z in yz] + [(cx + x1, y, z) for y, z in yz]
    faces = [list(range(m))[::-1], list(range(m, 2 * m))]
    faces += [[i, (i + 1) % m, m + (i + 1) % m, m + i] for i in range(m)]
    mesh = _mesh(stage, path, pts, faces, None, collide=True)
    mesh.MakeInvisible()
    return path


def _sector(stage, path, r0, r1, a0_deg, a1_deg, x0, x1, material, center=(0.0, 0.0, 0.0), collide=True):
    """An annular sector prism along x, split into convex pieces of at most SEG_DEG. A full disk (r0 = 0, 360 deg)
    keeps the pie-slice pieces for the look only and collides as one convex prism (see DISK_SIDES)."""
    if collide and r0 <= 1e-6 and a1_deg - a0_deg >= 360.0 - 1e-6:
        _sector(stage, path, r0, r1, a0_deg, a1_deg, x0, x1, material, center, collide=False)
        return [_disk_collider(stage, f"{path}_collider", r1, x0, x1, center)]
    n = max(1, math.ceil((a1_deg - a0_deg) / SEG_DEG - 1e-9))
    cx, cy, cz = center
    out = []
    for k in range(n):
        b0 = math.radians(a0_deg + (a1_deg - a0_deg) * k / n)
        b1 = math.radians(a0_deg + (a1_deg - a0_deg) * (k + 1) / n)
        bm = (b0 + b1) / 2
        ring = [(r0, b0), (r0, b1)] if r0 > 1e-6 else [(0.0, bm)]
        ro = r1 / math.cos((b1 - b0) / 4)
        ring += [(r1, b1), (ro, (bm + b1) / 2), (ro, (b0 + bm) / 2), (r1, b0)]
        yz = [(cy + r * math.cos(a), cz + r * math.sin(a)) for r, a in ring]
        m = len(yz)
        pts = [(cx + x0, y, z) for y, z in yz] + [(cx + x1, y, z) for y, z in yz]
        faces = [list(range(m))[::-1], list(range(m, 2 * m))]
        faces += [[i, (i + 1) % m, m + (i + 1) % m, m + i] for i in range(m)]
        mesh = _mesh(stage, f"{path}_{k}", pts, faces, material, collide=collide)
        mesh.CreateDoubleSidedAttr(True)
        out.append(f"{path}_{k}")
    return out


def _solid_arcs(gaps):
    start = gaps[0][1]
    gaps = sorted(((a - start) % 360.0 + start, (a - start) % 360.0 + start + (b - a)) for a, b in gaps)
    out, cur = [], start
    for a, b in gaps:
        if a > cur + 1e-6:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < start + 360.0 - 1e-6:
        out.append((cur, start + 360.0))
    return out


def _box(stage, path, x, y, z, mat, collide=True):
    c = ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2)
    s = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    return kit.box(stage, path, c, s, mat, collide=collide)


def _quad_x(stage, path, x, yc, zc, size, material):
    """A textured quad in the plane x = const, facing -x (readable from the robot's side)."""
    hy, hz = size[0] / 2, size[1] / 2
    pts = [(x, yc + hy, zc - hz), (x, yc - hy, zc - hz), (x, yc - hy, zc + hz), (x, yc + hy, zc + hz)]
    m = _mesh(stage, path, pts, [[0, 1, 2, 3]], material, st=[(0, 0), (1, 0), (1, 1), (0, 1)])
    m.CreateDoubleSidedAttr(True)
    return m


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


def _capsule_x(stage, path, length, radius, material):
    cap = UsdGeom.Capsule.Define(stage, path)
    cap.CreateAxisAttr("X")
    cap.CreateRadiusAttr(radius)
    cap.CreateHeightAttr(float(length))
    UsdGeom.Xformable(cap).AddTranslateOp().Set(Gf.Vec3d(length / 2, 0, 0))
    UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    return cap


def _offsets(stage) -> None:
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


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

    return tuple(float(v) for v in Rotation.from_matrix(np.asarray(m)).as_quat())


# Nameplates --------------------------------------------------------------------------------------------------
def write_textures(out_dir: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    for k, (big, small) in enumerate(LEGENDS):
        w, h = 1000, 360
        (bg, fg) = LABEL_COLORS[k]
        to8 = lambda c: tuple(int(round(v * 255)) for v in c)  # noqa: E731
        img = Image.new("RGB", (w, h), to8(bg))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, w - 11, h - 11], outline=to8(fg), width=8)
        draw.text((w / 2, h * 0.36), big, font=ImageFont.truetype(FONT, 190), fill=to8(fg), anchor="mm")
        size = 80
        while ImageFont.truetype(FONT, size).getlength(small) > w * 0.86:
            size -= 2
        draw.text((w / 2, h * 0.78), small, font=ImageFont.truetype(FONT, size), fill=to8(fg), anchor="mm")
        img.save(out_dir / f"nameplate_{k}.png")


# Panel ---------------------------------------------------------------------------------------------------------
def contact_frame(which: str, j: int):
    """(world position of the slide's origin, world rotation): +x radial outward, +y tangential, +z along the
    receptacle axis (into it). The contact's collider is authored in this frame."""
    c = rec_root(which)
    th = math.radians(BLADE_THETA[j] + CONTACT_THETA_OFF)
    radial = np.array([0.0, math.cos(th), math.sin(th)])
    axial = np.array([1.0, 0.0, 0.0])
    tangential = np.cross(axial, radial)
    rot = np.stack([radial, tangential, axial], axis=1)
    return np.asarray(c, dtype=float), rot


def _receptacle(stage, base, name, yc, mats):
    c = (FACE_X, yc, REC_Z)
    p = f"{base}/{name}"
    coll = []
    slots = [slot_ranges(t) for t in BLADE_THETA]
    coll += _sector(stage, f"{p}_face_in", 0.0, R_IN, 0.0, 360.0, 0.0, FACE_T, mats["face_black"], c)
    for j, (a, b) in enumerate(_solid_arcs([s[0] for s in slots])):
        coll += _sector(stage, f"{p}_face_blade_{j}", R_IN, R_MID, a, b, 0.0, FACE_T, mats["face_black"], c)
    for j, (a, b) in enumerate(_solid_arcs([s[1] for s in slots])):
        coll += _sector(stage, f"{p}_face_foot_{j}", R_MID, R_OUT, a, b, 0.0, LOCK_X, mats["face_black"], c)
    coll += _sector(stage, f"{p}_face_out", R_OUT, R_FACE, 0.0, 360.0, 0.0, FACE_T, mats["face_black"], c)
    coll += _sector(stage, f"{p}_chamber_back", 0.0, R_FACE, 0.0, 360.0, CHAMBER_DEPTH, CHAMBER_DEPTH + 0.003,
                    mats["face_black"], c)
    for j, t in enumerate(BLADE_THETA):  # brass seen through the slots (visual)
        (lo, hi), _ = blade_ranges(t)
        _sector(stage, f"{p}_brass_{j}", R_MID + 0.0003, R_MID + 0.0011, lo + 6, hi + LOCK_DEG - 4, FACE_T + 0.002,
                CHAMBER_DEPTH - 0.002, mats["brass"], c, collide=False)
    coll += _sector(stage, f"{p}_flange", R_FACE - 0.0005, R_FLANGE, 0.0, 360.0, *FLANGE_X, mats["face_black"], c)
    for j in range(3):
        a = math.radians(90.0 + 120.0 * j)
        kit.cylinder(stage, f"{p}_screw_{j}", (FACE_X + FLANGE_X[0] - 0.0006, yc + 0.03175 * math.cos(a),
                                               REC_Z + 0.03175 * math.sin(a)), 0.0035, 0.0012, mats["screw"], axis="X")
    # Turn mark (a pen line on the flange at 12 o'clock, for reading the cap's turn).
    _box(stage, f"{p}_mark", (FACE_X + FLANGE_X[0] - 0.0004, FACE_X + FLANGE_X[0]), (yc - 0.0008, yc + 0.0008),
         (REC_Z + R_FACE + 0.002, REC_Z + R_FLANGE - 0.002), mats["mark"], collide=False)
    kit.cylinder(stage, f"{p}_body", (FACE_X + CHAMBER_DEPTH / 2 + 0.02, yc, REC_Z), R_FACE + 0.002,
                 CHAMBER_DEPTH + 0.03, mats["face_black"], axis="X")
    return coll


def _contact_points():
    """Convex outline of a contact in its frame (x radial, y tangential, z axial); origin = receptacle centre."""
    r0, r1 = CONTACT_R
    z0, z1 = CONTACT_X
    pts = []
    for y in (-CONTACT_HALF_W, CONTACT_HALF_W):
        pts += [(r0, y, z0), (r1 - 0.0008, y, z0), (r1, y, z0 + CONTACT_LEAD), (r1, y, z1), (r0, y, z1)]
    m = 5
    faces = [list(range(m))[::-1], list(range(m, 2 * m))]
    faces += [[i, (i + 1) % m, m + (i + 1) % m, m + i] for i in range(m)]
    return pts, faces


def make_panel(out_path: Path) -> None:
    root = "/panel"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
        SOLVER_ITERATIONS)
    mats = {name: _material(stage, root, name, c) for name, c in COLORS.items()}
    labels = [_tex_material(stage, root, f"nameplate_{k}", f"nameplate_{k}.png") for k in range(len(LEGENDS))]
    base = kit.rigid_body(stage, f"{root}/base", (0, 0, 0), 5.0)
    b = f"{root}/base"
    coll = []
    px = (FACE_X + FLANGE_X[1], FACE_X + FLANGE_X[1] + PLATE_T)
    ys = sorted((TARGET_Y, DECOY_Y))
    hh = PLATE_HOLE_HALF
    edges = [PLATE_Y[0], ys[0] - hh, ys[0] + hh, ys[1] - hh, ys[1] + hh, PLATE_Y[1]]
    _box(stage, f"{b}/plate_lo", px, PLATE_Y, (PLATE_Z[0], REC_Z - hh), mats["plate"])
    _box(stage, f"{b}/plate_hi", px, PLATE_Y, (REC_Z + hh, PLATE_Z[1]), mats["plate"])
    for j in range(3):
        _box(stage, f"{b}/plate_mid_{j}", px, (edges[2 * j], edges[2 * j + 1]), (REC_Z - hh, REC_Z + hh),
             mats["plate"])
    _box(stage, f"{b}/enclosure", (px[1], px[1] + 0.10), PLATE_Y, PLATE_Z, mats["enclosure"], collide=False)
    for k, (name, yc) in enumerate((("target", TARGET_Y), ("decoy", DECOY_Y))):
        coll += _receptacle(stage, b, name, yc, mats)
        _quad_x(stage, f"{b}/{name}_nameplate", px[0] - 0.0016, yc, LABEL_Z, LABEL_SIZE, labels[k])
        _box(stage, f"{b}/{name}_nameplate_edge", (px[0] - 0.0015, px[0]),
             (yc - LABEL_SIZE[0] / 2, yc + LABEL_SIZE[0] / 2),
             (LABEL_Z - LABEL_SIZE[1] / 2, LABEL_Z + LABEL_SIZE[1] / 2), mats["cap_black"], collide=False)
    kit.friction(stage, root, coll, static=PLUG_FRICTION, dynamic=PLUG_FRICTION, name="nylon", combine="min")
    kit.fix_to_world(stage, root, base.GetPath())
    pts, faces = _contact_points()
    contacts = []
    for name in ("target", "decoy"):
        for j in range(3):
            pos, rot = contact_frame(name, j)
            lp = f"{root}/contact_{name}_{j}"
            _body(stage, lp, pos, rot, CONTACT_MASS)
            _mesh(stage, f"{lp}/spring", pts, faces, mats["brass"], collide=True)
            contacts.append(f"{lp}/spring")
            joint = UsdPhysics.PrismaticJoint.Define(stage, f"{root}/joints/contact_{name}_{j}_joint")
            joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{b}")])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(lp)])
            joint.CreateAxisAttr("X")
            joint.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in pos]))
            x, y, z, w = _mat_q(rot)
            joint.CreateLocalRot0Attr(Gf.Quatf(float(w), float(x), float(y), float(z)))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
            joint.CreateLowerLimitAttr(-CONTACT_TRAVEL)
            joint.CreateUpperLimitAttr(0.0)
            UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    kit.friction(stage, root, contacts, static=CONTACT_FRICTION, dynamic=CONTACT_FRICTION, name="contact_brass",
                 combine="min")
    _offsets(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


# Cord and cap --------------------------------------------------------------------------------------------------
ANCHOR_ROT = _rot("Z", math.pi / 2)
"""The cord leaves its clamp heading +y (toward the robot's midline)."""


def joint_axis(k: int) -> str:
    return AXES[k % len(AXES)]


def cord_fk(q):
    """Frames (pos, rot) of the CORD_N links and of the cap for joint angles q (CORD_N + 1 of them)."""
    p, r = np.asarray(ANCHOR, float), ANCHOR_ROT.copy()
    links = []
    for k in range(CORD_N):
        r = r @ _rot(joint_axis(k), q[k])
        links.append((p.copy(), r.copy()))
        p = p + r @ np.array([CORD_LINK, 0, 0])
    r = r @ _rot(joint_axis(CORD_N), q[CORD_N])
    return links, (p - r @ np.array([X_TAIL, 0, 0]), r)


HAND_BOX = ((0.0, 0.26), (PLUG_START_XY[1] - 0.075, PLUG_START_XY[1] + 0.075), 0.095)
"""Keep the cord out of the hand's path to the cap (x range, y range, above this z), except next to the cap."""


def solve_route(verbose=True):
    """Joint angles that route the cord from its clamp along the desk and up to the cap's back on its cradle; the
    cord drops off the cap's back at once (out of the hand's way) and lies low, off the cradle and the arm base."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    start = plug_start_root() + plug_start_quat()
    sp, sr = np.asarray(start[:3]), _qmat(start[3:])
    floor = TABLE_Z + CORD_R + 0.0003
    cx, cy = PLUG_START_XY

    def points(links):
        pts = []
        for p, r in links:
            pts += [p + r @ np.array([f * CORD_LINK, 0, 0]) for f in (0.5, 1.0)]
        return pts

    def residual(q):
        links, (pos, rot) = cord_fk(q)
        err_p = (pos - sp) * 3000.0
        err_r = Rotation.from_matrix(sr.T @ rot).as_rotvec() * 300.0
        pts = points(links)
        pen = [max(0.0, floor - e[2]) * 300.0 for e in pts]
        pen += [1.0 * (e[2] - floor) for e in pts[:-4]]
        (x0, x1), (y0, y1), z0 = HAND_BOX
        pen += [max(0.0, e[2] - z0) * 200.0 * (x0 < e[0] < x1) * (y0 < e[1] < y1) for e in pts[:-3]]
        # Off the cradle's post (a 40 x 30 mm column under the cap).
        pen += [max(0.0, 0.035 - math.hypot(e[0] - cx, e[1] - cy)) * 200.0 * (e[2] < 0.12) for e in pts[:-3]]
        # In front of the plate, off the right arm's base, inside the table.
        pen += [max(0.0, e[0] - (FACE_X - 0.03)) * 200.0 for e in pts]
        pen += [max(0.0, 0.14 - math.hypot(e[0], e[1] + 0.305)) * 200.0 for e in pts]
        pen += [max(0.0, 0.03 - e[0]) * 200.0 for e in pts]
        return np.concatenate([err_p, err_r, 0.1 * np.asarray(q), pen])

    lim = math.radians(CORD_LIMIT_DEG) * 0.8
    best = None
    for trial in range(10):
        q0 = np.random.default_rng(trial).uniform(-0.4, 0.4, CORD_N + 1)
        sol = least_squares(residual, q0, bounds=(-lim, lim))
        if best is None or sol.cost < best.cost:
            best = sol
    links, (pos, rot) = cord_fk(best.x)
    if verbose:
        pts = points(links)
        print(f"route: cap pos error {np.linalg.norm(pos - sp) * 1000:.2f} mm, rot error "
              f"{np.degrees(np.linalg.norm(Rotation.from_matrix(sr.T @ rot).as_rotvec())):.2f} deg, lowest "
              f"{min(e[2] for e in pts) * 1000:.1f} mm, x {min(e[0] for e in pts):.3f}..{max(e[0] for e in pts):.3f}, "
              f"y {min(e[1] for e in pts):.3f}..{max(e[1] for e in pts):.3f}, max |q| "
              f"{np.degrees(np.abs(best.x).max()):.0f} deg, cost {best.cost:.4f}")
    return best.x


def _joint_between(stage, path, body0, frame0, body1, frame1, world_point, world_rot, axis):
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(str(body0))])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(str(body1))])
    joint.CreateAxisAttr(axis)
    for (p, r), pos_attr, rot_attr in ((frame0, joint.CreateLocalPos0Attr, joint.CreateLocalRot0Attr),
                                       (frame1, joint.CreateLocalPos1Attr, joint.CreateLocalRot1Attr)):
        pos_attr(Gf.Vec3f(*[float(v) for v in r.T @ (np.asarray(world_point) - p)]))
        x, y, z, w = _mat_q(r.T @ world_rot)
        rot_attr(Gf.Quatf(float(w), float(x), float(y), float(z)))
    joint.CreateLowerLimitAttr(-CORD_LIMIT_DEG)
    joint.CreateUpperLimitAttr(CORD_LIMIT_DEG)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    return joint


def _cap_geometry(stage, c, mats):
    """The HBL2611 in the cap body's frame; returns (body colliders, blade colliders)."""
    body = _sector(stage, f"{c}/front", 0.0, R_FACE_PLUG, 0.0, 360.0, X_FRONT, 0.0, mats["cap_white"])
    body += _sector(stage, f"{c}/body", 0.0, R_BODY, 0.0, 360.0, X_BODY, X_FRONT, mats["cap_black"])
    body += _sector(stage, f"{c}/tail", 0.0, R_TAIL, 0.0, 360.0, X_TAIL, X_BODY, mats["cap_black"])
    # Ribs on the body (visual): 18 shallow ribs, as on the HBL2611's grip.
    for j in range(18):
        a = 360.0 * j / 18
        _sector(stage, f"{c}/rib_{j}", R_BODY - 0.0002, R_BODY + 0.0008, a - 3.0, a + 3.0, X_BODY + 0.004,
                X_FRONT - 0.003, mats["cap_black"], collide=False)
    # Turn mark: a pen line on the rim at 12 o'clock in the entry orientation (matches the flange's).
    _sector(stage, f"{c}/mark", R_FACE_PLUG - 0.0004, R_FACE_PLUG + 0.0003, 88.5, 91.5, X_FRONT + 0.001, -0.001,
            mats["mark"], collide=False)
    blades = []
    for j, t in enumerate(BLADE_THETA):
        (lo, hi), (tlo, thi) = blade_ranges(t)
        blades += _sector(stage, f"{c}/blade_{j}", R_B - BLADE_T / 2, R_B + BLADE_T / 2, lo, hi, -0.001, BLADE_L,
                          mats["brass"])
        blades += _sector(stage, f"{c}/foot_{j}", R_B + BLADE_T / 2 - 0.0003, R_B + BLADE_T / 2 + TAB_H, tlo, thi,
                          BLADE_L - TAB_L, BLADE_L, mats["brass"])
    # Strain relief where the cord enters (visual).
    kit.cylinder(stage, f"{c}/strain_relief", (X_TAIL - 0.006, 0.0, 0.0), CORD_R + 0.0025, 0.012, mats["cap_black"],
                 axis="X")
    return body, blades


def make_cord_cap(out_path: Path, q=None) -> None:
    root = "/cord_cap"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
        SOLVER_ITERATIONS)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    mats = {name: _material(stage, root, name, c) for name, c in COLORS.items()}
    q = solve_route() if q is None else q
    links, (cap_p, cap_r) = cord_fk(q)

    anchor = _body(stage, f"{root}/anchor", (0.0, 0.0, 0.0), np.eye(3), 0.05, (ANCHOR, (1e-5, 1e-5, 1e-5)))
    # The cord clamp (visual): a strap over the cord, screwed to the desk, and the cord's tail under it.
    a = f"{root}/anchor"
    ax, ay, az = ANCHOR
    w = 0.016
    _box(stage, f"{a}/strap", (ax - w, ax + w), (ay - 0.006, ay + 0.006), (az + CORD_R, az + CORD_R + 0.002),
         mats["clamp"], collide=False)
    for s, side in enumerate((-1, 1)):
        x_in, x_out = ax + side * (CORD_R + 0.0005), ax + side * w
        _box(stage, f"{a}/leg_{s}", tuple(sorted((x_in, x_out))), (ay - 0.006, ay + 0.006), (TABLE_Z, az + CORD_R),
             mats["clamp"], collide=False)
    kit.cylinder(stage, f"{a}/tail", (ax, ay - 0.02, az), CORD_R, 0.04, mats["cord"], axis="Y")
    kit.fix_to_world(stage, root, anchor.GetPath())

    prev, prev_frame = anchor.GetPath(), (np.zeros(3), np.eye(3))
    for k, (p, r) in enumerate(links):
        body = _body(stage, f"{root}/cord_{k:02d}", p, r, CORD_LINK_MASS, ((CORD_LINK / 2, 0, 0), CORD_LINK_INERTIA))
        _capsule_x(stage, f"{root}/cord_{k:02d}/jacket", CORD_LINK, CORD_R, mats["cord"])
        _joint_between(stage, f"{root}/joints/cord_j{k:02d}", prev, prev_frame, body.GetPath(), (p, r), p, r,
                       joint_axis(k))
        prev, prev_frame = body.GetPath(), (p, r)
    cap = _body(stage, f"{root}/cap", cap_p, cap_r, PLUG_MASS)
    tail = cap_p + cap_r @ np.array([X_TAIL, 0, 0])
    _joint_between(stage, f"{root}/joints/cord_j{CORD_N:02d}", prev, prev_frame, cap.GetPath(), (cap_p, cap_r),
                   tail, cap_r, joint_axis(CORD_N))
    c = f"{root}/cap"
    cap.GetPrim().AddAppliedSchema("PhysxRigidBodyAPI")
    cap.GetPrim().CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(0.5)
    body, blades = _cap_geometry(stage, c, mats)
    kit.friction(stage, root, blades, static=PLUG_FRICTION, dynamic=PLUG_FRICTION, name="brass", combine="min")
    kit.friction(stage, root, body, static=GRIP_FRICTION[0], dynamic=GRIP_FRICTION[1], name="grip", combine="max")
    kit.friction(stage, root, [f"{root}/cord_{k:02d}/jacket" for k in range(CORD_N)], static=CORD_FRICTION[0],
                 dynamic=CORD_FRICTION[1], name="jacket", combine="min")
    _offsets(stage)
    stage.GetRootLayer().Save()
    np.save(out_path.with_name("cord_route_q.npy"), np.asarray(q))
    print(f"wrote {out_path}")


def make_cradle(out_path: Path) -> None:
    root = "/cradle"
    stage, _ = kit.new_stage(out_path, root)
    mats = {name: _material(stage, root, name, c) for name, c in COLORS.items()}
    cx, cy = PLUG_START_XY
    axis_z = plug_start_root()[2]
    dy = 0.013
    top = axis_z - math.sqrt(R_BODY ** 2 - dy ** 2) - 0.0005
    xr = (cx - 0.018, cx + 0.018)
    for k, s in enumerate((-1, 1)):
        _box(stage, f"{root}/rail_{k}", xr, (cy + s * dy - 0.002, cy + s * dy + 0.002), (top - 0.006, top),
             mats["cradle"])
    _box(stage, f"{root}/post", xr, (cy - dy - 0.002, cy + dy + 0.002), (TABLE_Z, top - 0.006), mats["cradle"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_textures(out_dir)
    make_panel(out_dir / "panel.usda")
    make_cord_cap(out_dir / "cord_cap.usda")
    make_cradle(out_dir / "cradle.usda")
    print("slots (arc, notch) deg:", [tuple(tuple(round(v, 1) for v in r) for r in slot_ranges(t)) for t in BLADE_THETA])
    print(f"locked from {LOCKED_MIN_DEG:.1f} deg; contact force at rest {CONTACT_K * CONTACT_PRELOAD_M:.1f} N")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets")

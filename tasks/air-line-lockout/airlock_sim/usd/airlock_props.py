"""Generate the props: a two-outlet air manifold with two 1/4 in industrial-interchange (M-style) quick couplers, two
short air hoses with M-style plugs connected to them, the Brady PLO27E plug lockout standing open on its stand, the
Master Lock 410 padlock and its pedestal.

Real parts (README References):
- Coupler: Milton 715-class 1/4 in M-style (industrial interchange, MIL-C-4109) automatic coupler, brass body and
  sleeve, about 50 mm long and 23.6 mm across the sleeve. Inside, a ring of steel balls sits in holes in the body's
  bore; the spring-loaded sleeve's inner land holds them in the plug's groove, so the plug cannot come out. Pulling
  the sleeve back (toward the air supply) brings a relief over the balls; they can then move outward and the plug's
  lip pushes past them.
- Plug: M-style male plug with a 1/4 in hose barb (about 45 mm long, 14.3 mm hex), on a 1/4 in ID air hose
  (9.5 mm across) with a crimped ferrule. The nipple's lip, groove and shoulder are inferred proportions (the
  interchange drawing is not public); collision is square, not round.
- Brady PLO27E electrical/pneumatic plug lockout (polystyrene, red; 163 x 117 x 96 mm; padlock shackle up to
  7.87 mm, 20 mm clearance; hose/cord hole up to 9.4 mm): two half-shells hinged along one long edge; a hole at
  one end split between the halves takes the hose; one flat tab per half at the middle of the front, side by side,
  their holes lined up when the lid is shut.

How the coupler holds (geometry): four balls (3.2 mm) on radial slides stop at the bore (inner limit, the body's
tapered holes) and at the sleeve's land (outer); with the sleeve forward the balls stand 1.1 mm inside the lip's
radius, so the lip cannot pass; with the sleeve back by 5 mm the land has passed the balls and they rise clear.
How the lockout holds (geometry): the plug's hex and ferrule (14-15 mm) cannot pass the 10 mm hose hole; the
nipple is inside the closed shell, so it cannot be connected; a shackle through both tabs' holes stops the lid
turning on its hinge. Without the padlock the lid swings open freely (a light hinge damping only).

Files (all in the desk frame):
``manifold.usda``: one fixed-base articulation (self-collisions on): ``base`` (stand, manifold block, both coupler
bodies with their bores and plug stops), ``sleeve_<which>`` on prismatic joints (0 forward, + back), ``ball_<which>_<k>``
on radial prismatic joints (0 in the groove, + outward). Self-collisions are on so each sleeve's land meets its balls;
links joined to the base (all of them) never collide with it.
``hose_target.usda`` / ``hose_decoy.usda``: fixed-base articulations: ``anchor`` (fixed under the hose clamp), CORD_N
capsule links (revolute joints cycling about the two bending axes and the twist axis, springs toward the routed
shape), ``plug`` (frame: origin on the nipple's front face, +x along the nipple (into the coupler), +z up at the
start). The starting route is solved here and is the joints' zero.
``lockout.usda``: fixed-base articulation: ``base`` (the lower half-shell, held by its fixture) and ``lid`` on the
``hinge`` revolute joint (0 closed, LID_OPEN_DEG open against its stop). Box frame: origin at the closed box's
centre on the parting plane, world-aligned; the long axis is y, the hinge along the back edge (+x), the tabs on the
front (-x), the hose hole at the -y end.
``stand.usda``: the lockout's stand and fixture clips, and the padlock's pedestal.
``padlock_ml410.usda``: the blue Master Lock 410 (airlock_sim/usd/padlock_410.py).

Usage: python -m airlock_sim.usd.airlock_props [<out dir>]   (default: airlock_sim/assets)
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from airlock_sim.spec import (
    ANCHOR_XY, BOX_FRONT_X, BOX_Y, COUPLER_X, COUPLER_Z, MOUTH_Y, PEDESTAL_XY, RIM_Z, TAGS,
)
from airlock_sim.usd import usd_kit as kit

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TABLE_Z = -0.0025
"""The station table's top in the desk frame."""
MM = 0.001

# Plug (M-style, 1/4 in hose barb), plug frame: x along the nipple, 0 at its front face --------------------------
LIP_X = (-4.0 * MM, 0.0)
GROOVE_X = (-7.5 * MM, -4.0 * MM)
SHANK_X = (-15.5 * MM, -7.5 * MM)
NIP_HW = 4.6 * MM
"""Nipple half width (lip and shoulder): a 9.2 mm square standing in for the round nipple."""
GROOVE_HW = 3.5 * MM
HEX_X = (-27.5 * MM, -15.5 * MM)
HEX_HW = 7.15 * MM
"""14.3 mm across the hex's flats."""
STEM_X = (-32.0 * MM, -27.5 * MM)
STEM_HW = 3.5 * MM
FERRULE_X = (-45.0 * MM, -32.0 * MM)
FERRULE_HW = 7.5 * MM
PLUG_L = -FERRULE_X[0]
PLUG_MASS = 0.035
GRIP_X = -30.0 * MM
"""Where the fingers hold the plug (plug frame x): across the hex's back and the ferrule's front."""
PLUG_POINTS = ([(x, y, z) for x in HEX_X for y in (-HEX_HW, HEX_HW) for z in (-HEX_HW, HEX_HW)]
               + [(x, y, z) for x in FERRULE_X for y in (-FERRULE_HW, FERRULE_HW) for z in (-FERRULE_HW, FERRULE_HW)]
               + [(0.0, y, z) for y in (-NIP_HW, NIP_HW) for z in (-NIP_HW, NIP_HW)])
"""Corners of the plug's parts (plug frame): hex, ferrule, nipple tip. The in-lockout test checks all of them."""

# Coupler (M-style), coupler frame: origin at the mouth's centre, +x into the coupler (world +y) --------------------
R_C = None  # set below (world rotation of a coupler frame)
SEAT_X = 14.5 * MM
"""The plug's nipple front sits this deep in the coupler when connected (its hex 1 mm outside the sleeve)."""
BORE_HW = 4.75 * MM
BORE_T = 2.0 * MM
BODY_L = 50.0 * MM
BALL_R = 1.6 * MM
BALL_X = SEAT_X + (GROOVE_X[0] + GROOVE_X[1]) / 2
"""Balls' axial position: over the groove of a seated plug."""
BALL_R_IN = GROOVE_HW + BALL_R + 0.05 * MM
"""Ball centre's radius at its inner stop (in the groove)."""
BALL_TRAVEL = 1.5 * MM
"""Outward travel the body's holes allow (ball centre up to 6.6 mm from the axis)."""
LAND_R = BALL_R_IN + BALL_R + 0.1 * MM
"""Sleeve land's inner radius: the balls can rise only 0.1 mm with the sleeve forward (their inner edge then stays
1.0 mm inside the lip's 4.6 mm)."""
LAND_X = (BALL_X - 2.5 * MM, BALL_X + 2.5 * MM)
SLEEVE_L = 25.0 * MM
SLEEVE_R = 11.8 * MM
SLEEVE_T = 2.0 * MM
SLEEVE_TRAVEL = 6.0 * MM
SLEEVE_RELEASE = 4.6 * MM
"""Sleeve travel at which the land's front edge passes the balls' tops (the plug can come out beyond about 5 mm)."""
SLEEVE_K = 250.0
SLEEVE_PRELOAD = 8.0 * MM
"""Sleeve spring: 2 N forward at rest, 3.5 N pulled fully back (set low for the station arm; see README)."""
SLEEVE_MASS = 0.03
BALL_MASS = 0.002
"""A 3.2 mm steel ball weighs 0.13 g; raised to 2 g to keep the articulation's mass ratios sane."""
BRASS_FRICTION = 0.3

# Manifold (anodised aluminium block on a steel post) -----------------------------------------------------------
MANIFOLD_X = (COUPLER_X["target"] - 0.035, COUPLER_X["decoy"] + 0.035)
NIPPLE_L = 0.030
"""Each coupler sits on a 1/4 in NPT hex nipple, 30 mm long, screwed into the manifold."""
MANIFOLD_Y = (MOUTH_Y + BODY_L + NIPPLE_L - 0.003, MOUTH_Y + BODY_L + NIPPLE_L + 0.027)
MANIFOLD_Z = (COUPLER_Z - 0.025, COUPLER_Z + 0.025)

# Hose: 1/4 in ID air hose, blue -----------------------------------------------------------------------------------
CORD_LEN = 0.60
CORD_N = 24
CORD_LINK = CORD_LEN / CORD_N
CORD_R = 0.00475
CORD_LINK_MASS = 0.08 * CORD_LINK
CORD_LINK_INERTIA = (1e-7, 2e-7, 2e-7)
CORD_EI = 0.00003
"""Bending stiffness, N*m^2. The joints' springs pull toward the starting route, a memory a real hose barely has, so
they are kept weak (a 10 cm free end hangs nearly straight down): the hose drapes under its own weight like a rope;
this is softer than a real air hose."""
CORD_K = CORD_EI / CORD_LINK
CORD_DAMPING = 0.003
CORD_LIMIT_DEG = 100.0
CORD_FRICTION = (0.6, 0.5)
AXES = ("Y", "Z", "X")
RELIEF_JOINTS = 2
RELIEF_K = 3.0 * CORD_K
"""The last two hose joints (at the ferrule and one link out) are stiffer and start straight (the crimped ferrule and
the hose just behind it)."""
TAG_LINK = CORD_N - 3
"""The hose tag is wrapped round this link (5 to 7.5 cm from the ferrule's end)."""
TAG_SIZE = (0.060, 0.035)

# Lockout (Brady PLO27E), box frame ------------------------------------------------------------------------------
BOX_HL = 0.0815
"""Half length (along y): 163 mm long."""
BOX_HW = 0.0585
"""Half width (along x): 117 mm front to back."""
HALF_D = 0.048
"""Depth of each half-shell (along z): 96 mm tall closed."""
SHELL_T = 0.003
HOLE_R = 0.005
"""Hose hole: a 10 mm square opening (half in each shell's -y end wall), at mid-depth."""
TAB_X = (-(BOX_HW + 0.032), -(BOX_HW + 0.0005))
TAB_HZ = 0.013
TAB_T = 0.003
TAB_GAP = 0.0002
TAB_HOLE = 0.0085
TAB_HOLE_X = -(BOX_HW + 0.0215)
BOX_CENTER = (BOX_FRONT_X + BOX_HW, BOX_Y, RIM_Z)
HINGE = (BOX_HW, 0.0, 0.0)
"""Hinge axis (along y) in the box frame: the back edge of the parting plane."""
LID_OPEN_DEG = 95.0
"""The lid rests open against its hinge stop, 5 degrees past upright (its centre of mass, in the deep lid shell, is
then behind the hinge; it falls shut once turned below about 68 degrees)."""
LID_MASS = 0.12
BASE_MASS = 0.12
SHELL_FRICTION = 0.5
INTERIOR = ((-BOX_HW + SHELL_T, BOX_HW - SHELL_T), (-BOX_HL + SHELL_T, BOX_HL - SHELL_T),
            (-HALF_D + SHELL_T, HALF_D - SHELL_T))
STAND_X = (-BOX_HW + 0.0055, BOX_HW + 0.004)
"""Stand top in the box frame: set back 5.5 mm from the front so a padlock hanging from the tabs clears it."""
STAND_HY = 0.090
PEDESTAL_TOP = 0.060
PED_TILT_DEG = 15.0
"""The pedestal's top slopes down toward the lockout by this much (the padlock lies on it shackle end down)."""

CONTACT_OFFSET = 0.0003
SOLVER_ITERATIONS = 32

COLORS = {
    "brass": (0.80, 0.66, 0.36),
    "steel": (0.72, 0.74, 0.77),
    "zinc": (0.62, 0.64, 0.66),
    "alu": (0.70, 0.72, 0.76),
    "dark": (0.08, 0.08, 0.08),
    "hose": (0.08, 0.30, 0.78),
    "clamp": (0.70, 0.71, 0.72),
    "lockout_red": (0.85, 0.12, 0.05),
    "stand": (0.40, 0.40, 0.42),
    "fixture": (0.25, 0.25, 0.27),
    "white": (0.95, 0.95, 0.95),
    "valve_red": (0.80, 0.08, 0.06),
}


def _rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == "Z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


R_C = _rot("Z", math.pi / 2)
"""Coupler/plug frame in the world at the start: local x -> world +y, local y -> world -x, local z -> up."""


def _mat_q(m):
    from scipy.spatial.transform import Rotation

    return tuple(float(v) for v in Rotation.from_matrix(np.asarray(m)).as_quat())


def coupler_root(which: str) -> np.ndarray:
    """World position of a coupler's mouth centre."""
    return np.array([COUPLER_X[which], MOUTH_Y, COUPLER_Z])


def box_center() -> tuple[float, float, float]:
    return BOX_CENTER


# USD helpers -----------------------------------------------------------------------------------------------------
def _material(stage, root, name, color, roughness=0.5, metallic=0.0):
    path = f"{root}/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    if metallic:
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _mats(stage, root):
    metal = {"brass": 0.6, "steel": 0.6, "zinc": 0.5, "alu": 0.3}
    return {name: _material(stage, root, name, c, roughness=0.35 if name in metal else 0.5,
                            metallic=metal.get(name, 0.0)) for name, c in COLORS.items()}


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


def _hex_prism(stage, path, x0, x1, across_flats, material):
    """A visual hexagonal prism along x (body frame), flats facing +-y."""
    r = across_flats / 2 / math.cos(math.pi / 6)
    ring = [(r * math.cos(math.pi / 6 + k * math.pi / 3), r * math.sin(math.pi / 6 + k * math.pi / 3))
            for k in range(6)]
    pts = [(x0, -a, b) for a, b in ring] + [(x1, -a, b) for a, b in ring]
    faces, counts = [], []
    for k in range(6):
        j = (k + 1) % 6
        faces += [k, j, 6 + j, 6 + k]
        counts.append(4)
    faces += list(range(5, -1, -1)) + list(range(6, 12))
    counts += [6, 6]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in p]) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
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


def _sphere(stage, path, radius, material):
    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(radius)
    UsdShade.MaterialBindingAPI.Apply(sph.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
    return sph


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


class _Frame:
    """Authoring in a local frame (position p, rotation r) under a body whose own frame is (bp, br)."""

    def __init__(self, stage, body_path, body_frame, p, r):
        self.stage, self.body_path = stage, body_path
        bp, br = body_frame
        self.p = br.T @ (np.asarray(p) - bp)
        self.r = br.T @ r

    def xform(self, name):
        xf = UsdGeom.Xform.Define(self.stage, f"{self.body_path}/{name}")
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in self.p]))
        x, y, z, w = _mat_q(self.r)
        xf.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
        return f"{self.body_path}/{name}"


# Textures ---------------------------------------------------------------------------------------------------
def write_textures(out_dir: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    to8 = lambda c: tuple(int(round(v * 255)) for v in c)  # noqa: E731
    for which, (legend, color) in TAGS.items():
        w, h = 900, 480
        img = Image.new("RGB", (w, h), to8(color))
        draw = ImageDraw.Draw(img)
        size = 220
        while ImageFont.truetype(FONT, size).getlength(legend) > w * 0.88:
            size -= 4
        draw.text((w / 2, h / 2), legend, font=ImageFont.truetype(FONT, size), fill=(10, 10, 10), anchor="mm")
        img.save(out_dir / f"airlock_tag_{which}.png")
    # Lockout label (as on the PLO27E's lid: LOCKOUT DEVICE / pneumatic male fitting, electrical plug / DO NOT PLUG IN).
    w, h = 1000, 620
    img = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([18, 18, w - 18, h - 18], outline=(200, 30, 20), width=10)
    draw.text((w / 2, 110), "LOCKOUT DEVICE", font=ImageFont.truetype(FONT, 104), fill=(15, 30, 70), anchor="mm")
    small = ImageFont.truetype(FONT, 44)
    for k, line in enumerate(("Pneumatic", "Male Fitting", "Electrical", "Plug")):
        draw.text((60, 230 + 62 * k), line, font=small, fill=(20, 60, 40), anchor="lm")
    draw.text((720, 330), "DO NOT", font=ImageFont.truetype(FONT, 82), fill=(10, 10, 10), anchor="mm")
    draw.text((720, 440), "PLUG IN", font=ImageFont.truetype(FONT, 82), fill=(200, 20, 20), anchor="mm")
    img.save(out_dir / "airlock_lockout_label.png")


# Manifold and couplers ----------------------------------------------------------------------------------------
def ball_dirs():
    """The four balls' radial directions in the coupler frame (they face the square nipple's four flats)."""
    return [np.array(v, dtype=float) for v in ((0, 0, 1), (0, 1, 0), (0, 0, -1), (0, -1, 0))]


def _coupler_body(stage, base, which, mats):
    """A coupler's fixed parts on the manifold's base link (coupler frame = mouth, R_C): the bore walls, the plug
    stop and the visual brass body. Returns collider paths."""
    f = _Frame(stage, base, (np.zeros(3), np.eye(3)), coupler_root(which), R_C)
    c = f.xform(f"coupler_{which}")
    cols = []
    h, t = BORE_HW, BORE_T
    for k, (y, z) in enumerate([((-h - t, h + t), (h, h + t)), ((-h - t, h + t), (-h - t, -h)),
                                ((h, h + t), (-h, h)), ((-h - t, -h), (-h, h))]):
        cols.append(_box(stage, f"{c}/bore_{k}", (0.0005, BODY_L), y, z, mats["brass"]))
    cols.append(_box(stage, f"{c}/stop", (SEAT_X, SEAT_X + 0.004), (-h, h), (-h, h), mats["dark"]))
    # Visual: the body under the sleeve, its hex at the back, the dark bore.
    kit.cylinder(stage, f"{c}/body_vis", (BODY_L / 2 + 0.006, 0, 0), 9.6 * MM, BODY_L - 0.012, mats["brass"], axis="X")
    _hex_prism(stage, f"{c}/hex", BODY_L - 0.013, BODY_L, 0.022, mats["brass"])
    kit.cylinder(stage, f"{c}/bore_vis", (0.004, 0, 0), BORE_HW + 0.0002, 0.0075, mats["dark"], axis="X")
    return cols


def _sleeve(stage, root, base, which, mats):
    """The sleeve: a body at the mouth (coupler frame) on a prismatic joint along the coupler's axis. Colliders: the
    inner land over the balls and the outer shell (square, 23.6 mm across flats, where fingers hold it)."""
    p = coupler_root(which)
    body = _body(stage, f"{root}/sleeve_{which}", p, R_C, SLEEVE_MASS)
    s = f"{root}/sleeve_{which}"
    cols = []
    w = (-3.2 * MM, 3.2 * MM)
    for k, d in enumerate(ball_dirs()):
        lo, hi = LAND_R, LAND_R + 0.0015
        span = (lo, hi) if d.sum() > 0 else (-hi, -lo)
        y, z = (w, span) if abs(d[2]) > 0.5 else (span, w)
        cols.append(_box(stage, f"{s}/land_{k}", LAND_X, y, z, mats["brass"]))
    R, T = SLEEVE_R, SLEEVE_T
    for k, (y, z) in enumerate([((-R, R), (R - T, R)), ((-R, R), (-R, -R + T)), ((R - T, R), (-R + T, R - T)),
                                ((-R, -R + T), (-R + T, R - T))]):
        cols.append(_box(stage, f"{s}/shell_{k}", (0.0, SLEEVE_L), y, z, mats["brass"]))
        kit.hide(stage, f"{s}/shell_{k}")
    # Visual: knurled brass sleeve (a cylinder with dark rings) and its front face ring.
    kit.cylinder(stage, f"{s}/vis", (SLEEVE_L / 2, 0, 0), R, SLEEVE_L, mats["brass"], axis="X")
    for j in range(5):
        kit.cylinder(stage, f"{s}/knurl_{j}", (0.010 + 0.0028 * j, 0, 0), R + 0.0003, 0.0008, mats["zinc"], axis="X")
    _joint(stage, f"{root}/joints/sleeve_{which}_joint", "prismatic", base, (np.zeros(3), np.eye(3)), body.GetPath(),
           (p, R_C), p, R_C, "X", 0.0, SLEEVE_TRAVEL)
    return cols


def _balls(stage, root, base, which, mats):
    cols = []
    origin = coupler_root(which)
    for k, d in enumerate(ball_dirs()):
        dw = R_C @ d
        p = origin + R_C @ np.array([BALL_X, 0, 0]) + BALL_R_IN * dw
        # Joint frame: x along the ball's outward direction.
        xr = dw
        yr = R_C @ np.array([1.0, 0, 0])
        zr = np.cross(xr, yr)
        r = np.stack([xr, yr, zr], axis=1)
        body = _body(stage, f"{root}/ball_{which}_{k}", p, r, BALL_MASS,
                     ((0, 0, 0), (2e-9, 2e-9, 2e-9)))
        cols.append(str(_sphere(stage, f"{root}/ball_{which}_{k}/ball", BALL_R, mats["steel"]).GetPath()))
        _joint(stage, f"{root}/joints/ball_{which}_{k}_joint", "prismatic", base, (np.zeros(3), np.eye(3)),
               body.GetPath(), (p, r), p, r, "X", 0.0, BALL_TRAVEL)
    return cols


def make_manifold(out_path: Path) -> None:
    root = "/manifold"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=True)
    _solver(root_prim)
    mats = _mats(stage, root)
    base = _body(stage, f"{root}/base", (0.0, 0.0, 0.0), np.eye(3), 3.0)
    b = f"{root}/base"
    # Manifold block on a post with a foot plate, a supply line with a ball valve (handle across = shut).
    _box(stage, f"{b}/block", MANIFOLD_X, MANIFOLD_Y, MANIFOLD_Z, mats["alu"])
    px = (sum(MANIFOLD_X) / 2 - 0.02, sum(MANIFOLD_X) / 2 + 0.02)
    _box(stage, f"{b}/post", px, MANIFOLD_Y, (TABLE_Z, MANIFOLD_Z[0]), mats["stand"])
    _box(stage, f"{b}/foot", (px[0] - 0.04, px[1] + 0.04), (MANIFOLD_Y[0] - 0.02, MANIFOLD_Y[1] + 0.03),
         (TABLE_Z, TABLE_Z + 0.006), mats["stand"])
    yc, zc = sum(MANIFOLD_Y) / 2, COUPLER_Z
    kit.cylinder(stage, f"{b}/supply_nipple", (MANIFOLD_X[1] + 0.012, yc, zc), 0.0065, 0.024, mats["brass"], axis="X")
    _box(stage, f"{b}/valve", (MANIFOLD_X[1] + 0.024, MANIFOLD_X[1] + 0.054), (yc - 0.013, yc + 0.013),
         (zc - 0.013, zc + 0.013), mats["brass"], collide=False)
    _box(stage, f"{b}/valve_stem", (MANIFOLD_X[1] + 0.036, MANIFOLD_X[1] + 0.042), (yc - 0.003, yc + 0.003),
         (zc + 0.013, zc + 0.020), mats["steel"], collide=False)
    _box(stage, f"{b}/valve_handle", (MANIFOLD_X[1] + 0.033, MANIFOLD_X[1] + 0.045), (yc - 0.06, yc + 0.004),
         (zc + 0.020, zc + 0.024), mats["valve_red"], collide=False)
    kit.cylinder(stage, f"{b}/supply_hose", (MANIFOLD_X[1] + 0.104, yc, zc), 0.0075, 0.10, mats["dark"], axis="X")
    cols = []
    for which in ("target", "decoy"):
        cols += _coupler_body(stage, b, which, mats)
        p = coupler_root(which)
        # The coupler's threaded nipple into the manifold (visual).
        kit.cylinder(stage, f"{b}/nipple_{which}", (p[0], MOUTH_Y + BODY_L + NIPPLE_L / 2, p[2]), 0.0065,
                     NIPPLE_L + 0.002, mats["brass"], axis="Y")
        _box(stage, f"{b}/nipple_hex_{which}", (p[0] - 0.0079, p[0] + 0.0079), (MOUTH_Y + BODY_L + 0.010,
             MOUTH_Y + BODY_L + 0.018), (p[2] - 0.0079, p[2] + 0.0079), mats["brass"], collide=False)
    for which in ("target", "decoy"):
        cols += _sleeve(stage, root, base.GetPath(), which, mats)
        cols += _balls(stage, root, base.GetPath(), which, mats)
    kit.fix_to_world(stage, root, base.GetPath())
    kit.friction(stage, root, [str(c.GetPath()) if hasattr(c, "GetPath") else c for c in cols],
                 static=BRASS_FRICTION, dynamic=BRASS_FRICTION, name="brass", combine="min")
    shells = [f"{root}/sleeve_{w}/shell_{k}" for w in ("target", "decoy") for k in range(4)]
    kit.friction(stage, root, shells, static=1.2, dynamic=1.0, name="sleeve_grip", combine="max")
    _offsets(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


# Hoses and plugs -----------------------------------------------------------------------------------------------
ANCHOR_HEADING_DEG = {"target": 70.0, "decoy": 85.0}
TAIL = np.array([-PLUG_L, 0.0, 0.0])


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
    return coupler_root(which) + R_C @ np.array([SEAT_X, 0, 0]), R_C.copy()


def _points(links):
    pts = []
    for p, r in links:
        pts += [p + r @ np.array([f * CORD_LINK, 0, 0]) for f in (0.25, 0.5, 0.75, 1.0)]
    return pts


def solve_route(which: str, avoid=(), verbose=True):
    """Joint angles that route the hose from its clamp along the table, up to its plug in the coupler. The hose leaves
    the plug heading to the robot's right, stays off the other hose, the manifold and the arm base, and keeps out of
    the space above the plug where the hand comes down."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    sp, sr = plug_start(which)
    floor = TABLE_Z + CORD_R + 0.0003
    other = "decoy" if which == "target" else "target"
    op = coupler_root(other)
    lo = np.array([op[0] - 0.012, op[1] - PLUG_L - 0.002, op[2] - 0.012])
    hi = np.array([op[0] + 0.012, op[1] + 0.06, op[2] + 0.012])
    avoid = np.asarray(avoid).reshape(-1, 3) if len(avoid) else np.zeros((0, 3))
    tp = coupler_root(which)

    def residual(q):
        links, (pos, rot) = cord_fk(which, q)
        err_p = (pos - sp) * 3000.0
        err_r = Rotation.from_matrix(sr.T @ rot).as_rotvec() * 300.0
        P = np.asarray(_points(links))
        pen = [np.maximum(0.0, floor - P[:, 2]) * 300.0, 3.0 * (P[:-20, 2] - floor)]
        E = P[:-2]
        d = np.minimum(E - lo, hi - E).min(axis=1) + 0.006 + CORD_R
        pen.append(np.maximum(0.0, d) * 400.0)
        # Off the manifold and its post, off the right arm's base.
        pen.append(np.maximum(0.0, E[:, 1] - (MOUTH_Y - 0.004)) * 300.0)
        pen.append(np.maximum(0.0, 0.12 - np.hypot(E[:, 0], E[:, 1] + 0.305)) * 300.0)
        # The hands' way down to the plug: keep the first 12 cm from the plug low (below the coupler's axis).
        box = (np.abs(E[:, 0] - tp[0]) < 0.05) & (E[:, 1] > tp[1] - 0.12)
        pen.append(np.maximum(0.0, E[:, 2] - (tp[2] + 0.004)) * 300.0 * box)
        if len(avoid):
            dd = np.linalg.norm(P[:, None] - avoid[None], axis=2).min(axis=1)
            pen.append(np.maximum(0.0, 2 * CORD_R + 0.006 - dd) * 400.0)
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
    """The M-style plug in the plug frame; returns (grip colliders, nipple colliders)."""
    sq = lambda hw: (-hw, hw)  # noqa: E731
    nipple = [str(_box(stage, f"{c}/{name}", xs, sq(hw), sq(hw), mats["zinc"]).GetPath())
              for name, xs, hw in (("lip", LIP_X, NIP_HW), ("groove", GROOVE_X, GROOVE_HW),
                                   ("shank", SHANK_X, NIP_HW))]
    for name in ("lip", "groove", "shank"):
        kit.hide(stage, f"{c}/{name}")
    for name, xs, r in (("lip_vis", LIP_X, NIP_HW + 0.0001), ("groove_vis", GROOVE_X, GROOVE_HW + 0.0001),
                        ("shank_vis", SHANK_X, NIP_HW + 0.0001)):
        kit.cylinder(stage, f"{c}/{name}", ((xs[0] + xs[1]) / 2, 0, 0), r, xs[1] - xs[0], mats["zinc"], axis="X")
    grip = [str(_box(stage, f"{c}/hex", HEX_X, sq(HEX_HW), sq(HEX_HW), mats["zinc"]).GetPath()),
            str(_box(stage, f"{c}/stem", STEM_X, sq(STEM_HW), sq(STEM_HW), mats["zinc"]).GetPath()),
            str(_box(stage, f"{c}/ferrule", FERRULE_X, sq(FERRULE_HW), sq(FERRULE_HW), mats["steel"]).GetPath())]
    for name in ("hex", "stem", "ferrule"):
        kit.hide(stage, f"{c}/{name}")
    _hex_prism(stage, f"{c}/hex_vis", HEX_X[0], HEX_X[1], 2 * HEX_HW, mats["zinc"])
    kit.cylinder(stage, f"{c}/stem_vis", ((STEM_X[0] + STEM_X[1]) / 2, 0, 0), STEM_HW, STEM_X[1] - STEM_X[0],
                 mats["zinc"], axis="X")
    kit.cylinder(stage, f"{c}/ferrule_vis", ((FERRULE_X[0] + FERRULE_X[1]) / 2, 0, 0), FERRULE_HW,
                 FERRULE_X[1] - FERRULE_X[0], mats["steel"], axis="X")
    for j in range(3):
        x = FERRULE_X[0] + 0.003 + 0.0035 * j
        kit.cylinder(stage, f"{c}/crimp_{j}", (x, 0, 0), FERRULE_HW + 0.0002, 0.0012, mats["zinc"], axis="X")
    return grip, nipple


def make_hose(out_path: Path, which: str, q=None) -> None:
    root = f"/hose_{which}"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    _solver(root_prim)
    mats = _mats(stage, root)
    q = solve_route(which) if q is None else q
    links, (plug_p, plug_r) = cord_fk(which, q)
    a_p, a_r = anchor(which)
    anc = _body(stage, f"{root}/anchor", (0.0, 0.0, 0.0), np.eye(3), 0.05, (tuple(a_p), (1e-5, 1e-5, 1e-5)))
    a = f"{root}/anchor"
    # Hose clamp (visual): a strap screwed to the table over the hose, and the hose's tail under it.
    hx = a_r @ np.array([1.0, 0, 0])
    yaw = math.degrees(math.atan2(hx[1], hx[0]))
    clamp = UsdGeom.Xform.Define(stage, f"{a}/clamp")
    clamp.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in a_p]))
    clamp.AddRotateZOp().Set(yaw)
    _box(stage, f"{a}/clamp/strap", (-0.007, 0.007), (-0.016, 0.016), (CORD_R, CORD_R + 0.002), mats["clamp"],
         collide=False)
    _box(stage, f"{a}/clamp/tail", (-0.04, 0.0), (-CORD_R, CORD_R), (-CORD_R, CORD_R), mats["hose"], collide=False)
    kit.fix_to_world(stage, root, anc.GetPath())

    prev, prev_frame = anc.GetPath(), (np.zeros(3), np.eye(3))
    jackets = []
    for k, (p, r) in enumerate(links):
        body = _body(stage, f"{root}/cord_{k:02d}", p, r, CORD_LINK_MASS, ((CORD_LINK / 2, 0, 0), CORD_LINK_INERTIA))
        jackets.append(str(_capsule_x(stage, f"{root}/cord_{k:02d}/jacket", CORD_LINK, CORD_R, mats["hose"]).GetPath()))
        _joint(stage, f"{root}/joints/cord_j{k:02d}", "revolute", prev, prev_frame, body.GetPath(), (p, r), p, r,
               joint_axis(k), -CORD_LIMIT_DEG, CORD_LIMIT_DEG)
        prev, prev_frame = body.GetPath(), (p, r)
        if k == TAG_LINK:
            _hose_tag(stage, f"{root}/cord_{k:02d}", which, r, mats)
    plug = _body(stage, f"{root}/plug", plug_p, plug_r, PLUG_MASS)
    tail = plug_p + plug_r @ TAIL
    _joint(stage, f"{root}/joints/cord_j{CORD_N:02d}", "revolute", prev, prev_frame, plug.GetPath(),
           (plug_p, plug_r), tail, plug_r, joint_axis(CORD_N), -CORD_LIMIT_DEG, CORD_LIMIT_DEG)
    c = f"{root}/plug"
    plug.GetPrim().AddAppliedSchema("PhysxRigidBodyAPI")
    plug.GetPrim().CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(0.5)
    grip, nipple = _plug_geometry(stage, c, mats)
    kit.friction(stage, root, nipple, static=BRASS_FRICTION, dynamic=BRASS_FRICTION, name="steel", combine="min")
    kit.friction(stage, root, grip, static=1.2, dynamic=1.0, name="grip", combine="max")
    kit.friction(stage, root, jackets, static=CORD_FRICTION[0], dynamic=CORD_FRICTION[1], name="jacket",
                 combine="min")
    _offsets(stage)
    stage.GetRootLayer().Save()
    np.save(out_path.with_name(f"hose_{which}_route_q.npy"), np.asarray(q))
    print(f"wrote {out_path}")


def _hose_tag(stage, link_path, which, link_rot, mats):
    """A wrap-around hose tag (visual): a sleeve round the link and a flag standing up from it (world up at the start
    pose), the legend printed on both faces of the flag and on its top edge's band."""
    up = link_rot.T @ np.array([0.0, 0.0, 1.0])
    up = up - up[0] * np.array([1.0, 0, 0])
    up /= np.linalg.norm(up)
    side = np.cross(up, np.array([1.0, 0, 0]))
    color = _material(stage, link_path, f"tag_{which}_color", TAGS[which][1])
    tex = _tex_material(stage, link_path, f"tag_{which}_legend", f"airlock_tag_{which}.png")
    xf = UsdGeom.Xform.Define(stage, f"{link_path}/tag")
    m = np.stack([np.array([1.0, 0, 0]), side, up], axis=1)
    x, y, z, w = _mat_q(m)
    xf.AddTranslateOp().Set(Gf.Vec3d(CORD_LINK / 2, 0, 0))
    xf.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
    t = f"{link_path}/tag"
    L, H = TAG_SIZE
    kit.cylinder(stage, f"{t}/sleeve", (0.0, 0.0, 0.0), CORD_R + 0.0025, L, color, axis="X")
    _box(stage, f"{t}/flag", (-L / 2, L / 2), (-0.0006, 0.0006), (CORD_R, CORD_R + H), color, collide=False)
    z0, z1 = CORD_R + 0.002, CORD_R + H - 0.002
    for s, name in ((1, "a"), (-1, "b")):
        yv = s * 0.0008
        xs = (-L / 2 + 0.002, L / 2 - 0.002) if s < 0 else (L / 2 - 0.002, -L / 2 + 0.002)
        _quad(stage, f"{t}/legend_{name}", [(xs[0], yv, z0), (xs[1], yv, z0), (xs[1], yv, z1), (xs[0], yv, z1)], tex)
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
    # -y end wall with the half hose hole.
    ye = (-BOX_HL, -BOX_HL + SHELL_T)
    out.append(_box(stage, f"{path}/end_neg_0", (-BOX_HW, -HOLE_R), ye, zr, red))
    out.append(_box(stage, f"{path}/end_neg_1", (HOLE_R, BOX_HW), ye, zr, red))
    out.append(_box(stage, f"{path}/end_neg_2", (-HOLE_R, HOLE_R), ye, tuple(sorted((sign * HOLE_R, z_out))), red))
    # The padlock tab: a plate across the long axis at mid-length on the front, with its hole.
    ty = (TAB_GAP, TAB_GAP + TAB_T) if sign > 0 else (-TAB_GAP - TAB_T, -TAB_GAP)
    h = TAB_HOLE / 2
    for k, (x, z) in enumerate([((TAB_HOLE_X + h, TAB_X[1]), (-TAB_HZ, TAB_HZ)),
                                ((TAB_X[0], TAB_HOLE_X - h), (-TAB_HZ, TAB_HZ)),
                                ((TAB_HOLE_X - h, TAB_HOLE_X + h), (h, TAB_HZ)),
                                ((TAB_HOLE_X - h, TAB_HOLE_X + h), (-TAB_HZ, -h))]):
        out.append(_box(stage, f"{path}/tab_{k}", x, ty, z, red))
    _box(stage, f"{path}/tab_root", (TAB_X[1], -BOX_HW + 0.0002), ty, (-TAB_HZ, TAB_HZ), red, collide=False)
    # Ribs on the outer faces (visual), as on the real shell.
    for j, y in enumerate((-0.060, -0.030, 0.030, 0.060)):
        _box(stage, f"{path}/rib_{j}", (-BOX_HW - 0.0008, -BOX_HW), (y - 0.0015, y + 0.0015),
             tuple(sorted((sign * 0.005, sign * (HALF_D - 0.004)))), red, collide=False)
    if label_tex is not None:
        zl = sign * (HALF_D + 0.0004)
        lx, ly = 0.040, 0.064
        _quad(stage, f"{path}/label", [(-lx, ly, zl), (-lx, -ly, zl), (lx, -ly, zl), (lx, ly, zl)], label_tex)
    return [str(p.GetPath()) for p in out]


def make_lockout(out_path: Path) -> None:
    root = "/lockout"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    _solver(root_prim)
    mats = _mats(stage, root)
    c = np.array(BOX_CENTER)
    base = _body(stage, f"{root}/base", np.zeros(3), np.eye(3), BASE_MASS)
    UsdGeom.Xform.Define(stage, f"{root}/base/shell").AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in c]))
    lid = _body(stage, f"{root}/lid", c, np.eye(3), LID_MASS)
    label = _tex_material(stage, root, "label", "airlock_lockout_label.png")
    cols = _shell(stage, f"{root}/base/shell", -1, mats)
    cols += _shell(stage, f"{root}/lid", 1, mats, label_tex=label)
    for k, (y0, y1, part) in enumerate(((-0.078, -0.050, "base"), (-0.048, -0.016, "lid"), (-0.014, 0.014, "base"),
                                        (0.016, 0.048, "lid"), (0.050, 0.078, "base"))):
        kit.cylinder(stage, f"{root}/{'base/shell' if part == 'base' else part}/knuckle_{k}",
                     (HINGE[0] + 0.001, (y0 + y1) / 2, 0.0), 0.004, y1 - y0, mats["lockout_red"], axis="Y")
    hinge_w = c + np.array(HINGE)
    joint = _joint(stage, f"{root}/joints/hinge", "revolute", base.GetPath(), (np.zeros(3), np.eye(3)), lid.GetPath(),
                   (c, np.eye(3)), hinge_w, np.eye(3), "Y", 0.0, LID_OPEN_DEG)
    kit.fix_to_world(stage, root, base.GetPath())
    tabs = [c for c in cols if "/tab_" in c]
    kit.friction(stage, root, [c for c in cols if c not in tabs], static=SHELL_FRICTION, dynamic=SHELL_FRICTION,
                 name="ps", combine="min")
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
    for ky, ys in enumerate(((-BOX_HL - 0.006, -BOX_HL), (BOX_HL, BOX_HL + 0.006))):
        _box(stage, f"{root}/clip_{ky}", (cx + BOX_HW - 0.020, cx + BOX_HW - 0.004), (cy + ys[0], cy + ys[1]),
             (top, cz - 0.020), mats["fixture"])
    x, y = PEDESTAL_XY
    _box(stage, f"{root}/ped_foot", (x - 0.025, x + 0.025), (y - 0.045, y + 0.005), (TABLE_Z, TABLE_Z + 0.01),
         mats["stand"])
    t = math.radians(PED_TILT_DEG)
    axis = np.array([0.0, -math.cos(t), -math.sin(t)])
    normal = np.array([0.0, -math.sin(t), math.cos(t)])
    centre = np.array([x, y, PEDESTAL_TOP]) + 0.015 * axis - 0.005 * normal
    slab = UsdGeom.Xform.Define(stage, f"{root}/ped_slab")
    slab.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
    slab.AddRotateXOp().Set(PED_TILT_DEG)
    kit.box(stage, f"{root}/ped_slab/top", (0.0, 0.0, 0.0), (0.038, 0.022, 0.010), mats["stand"])
    _box(stage, f"{root}/ped_column", (x - 0.006, x + 0.006), (centre[1] - 0.008, centre[1] + 0.008),
         (TABLE_Z, centre[2] - 0.004), mats["stand"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def main(out_dir: Path) -> None:
    from airlock_sim.usd import padlock_410, padlock_props

    out_dir.mkdir(parents=True, exist_ok=True)
    write_textures(out_dir)
    make_manifold(out_dir / "manifold.usda")
    cached = {w: out_dir / f"hose_{w}_route_q.npy" for w in ("decoy", "target")}
    q_decoy = np.load(cached["decoy"]) if cached["decoy"].exists() else solve_route("decoy")
    links, _ = cord_fk("decoy", q_decoy)
    make_hose(out_dir / "hose_decoy.usda", "decoy", q_decoy)
    q_target = (np.load(cached["target"]) if cached["target"].exists()
                else solve_route("target", avoid=_points(links)))
    make_hose(out_dir / "hose_target.usda", "target", q_target)
    make_lockout(out_dir / "lockout.usda")
    make_stand(out_dir / "stand.usda")
    blue = replace(padlock_props.PADLOCKS["ml410"], color="lock_blue")
    padlock_410.make_padlock_410(out_dir / "padlock_ml410.usda", blue)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets")

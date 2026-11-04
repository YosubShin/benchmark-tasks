"""Generate the props of the trapped-key interlock task as USD (two coded pairs: breaker, key, door).

Real parts (values marked "inferred" are not published and are to measure):

- Kirk Key Type F isolation interlock, SD series (brass), single cylinder. Published (Sentric/Kirk datasheet
  U-Type F-E-01 (01-26), drawing p. 4): housing 1.750 in (44.45 mm) wide, 2.500 in (63.5 mm) long for one
  cylinder, 1.375 in (34.9 mm) deep; cylinder centre 1.0 in (25.4 mm) from the top end, so 1.5 in (38.1 mm)
  from the bolt end; cylinder about 1.36 in (34.5 mm) across, standing 0.75 in (19 mm) proud of the housing;
  key standing 1.25 in (31.75 mm) above the cylinder; stainless lock bolt 5/8 in (15.9 mm) diameter with a
  3/4 in (19.05 mm) throw; 1.83 lb. Kirk white paper "Key Interlock Operation": "a lock cylinder, a support
  housing, a movable locking bolt, and a cam arranged to move the bolt"; the key is removable only in one bolt
  position (E = bolt extended, key removable).
  Built as a fixed-base articulation per station:
  * housing (part of the station's root body): a box, plus the cylinder's face plate with the keyhole: a 9 mm
    square hole the key's shank turns in, and a notch for the key's bit at one angle only (the coded keyway).
  * plug (revolute about the cylinder axis, 0..90 deg): a 7 mm square socket that the key's 6 mm square tip
    drives, and a crank with a pin at 19.05 mm radius under it (the cam).
  * bolt (prismatic, 0..19.05 mm): the 15.9 mm square rod (a round 5/8 in bolt kept at its width) and a yoke
    whose slot holds the crank pin (a scotch yoke: bolt travel = 19.05 mm x sin(plug angle)). At 90 deg the pin
    is at dead centre, so pushing the bolt back cannot turn the plug.
  * trapping, by geometry: the key's bit sits in a 4 mm gap under the face plate. The notch lines up with the
    bit only at plug = 90 deg (bolt extended), so only there can the key come out. If the bolt cannot extend
    (blocked), the plug cannot reach 90 deg and the key stays trapped.
  Simplifications: a 7-pin tumbler cylinder is replaced by a warded keyway (shank + bit + notch); the key code
  is the shank's cross-section (see CODES): a wrong key's shank does not fit the plug's socket, so it stops on
  the plug's top and cannot turn it (a real wrong-code Kirk key goes in and will not turn). Cam = scotch yoke (Kirk's cam profile is not
  published). Plug turning torque is not published (to measure).
- Key: Kirk SD key, nickel-silver, flat, stamped with its code. Stand-in: a 28 x 24 x 3 mm bow (the gripper
  pinches it), a 6 x 6 mm (A1) or 4 x 7.6 mm (B1) shank and a 5 mm bit; 20 g (inferred).
- Breaker: a molded-case breaker lying face up (Square D PowerPact H class, 3P frame about 105 x 160 x 86 mm,
  inferred from the class), toggle handle on a revolute joint (OFF -20 deg / ON +20 deg, inferred). The Type F
  is bracketed beside the handle (as in the "Kirk Lock Test" video, UPS bypass breakers) so that its extended
  bolt stands in the handle's path to ON: the breaker is locked open (L-O). The handle at ON would be in the
  bolt's way, so the key could not be freed (geometry).
- Door: a fuse-compartment door (Kirk white paper "Examples of Key Interlock Systems", fig. 1) as a 100 x 140 mm
  plate hinged on its far side, with a keeper tab on its lower free corner; a second Type F on the bench in
  front of it pushes its bolt through the keeper (locked closed, L-C), as on the gate and door in Kirk's
  "Trapped Key Interlock Usage in Switchgear" animation (0:29, 0:53). Sizes inferred.

Frames: each station's root sits at its own origin on the desk top (z = 0), axes along the world's.

Usage: python -m interlock_sim.usd.props <out dir>
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from interlock_sim.usd import usd_kit as kit

MM = 0.001

# Kirk Type F, SD series (lock frame L: origin on the cylinder axis at the face's top; z out of the face;
# the bolt leaves along -x) -----------------------------------------------------------------------------------
HOUSING_X = (-38.1, 25.4)
HOUSING_Y = (-22.225, 22.225)
HOUSING_Z = (-53.9, -19.0)
CYL_R = 17.3
FACE_T = 4.0
PLATE = 15.0
HOLE = 4.5
"""(Clearances: shank 6 mm in a 6.4 mm socket, about 4 deg of play; bit 3 mm in a 4 mm notch, about 3.6 deg at
its tip. Together the key can leave only with the plug past about 82 deg, the bolt past 18.8 mm.)
Half-width of the square hole the shank turns in (the shank's half-diagonal is 4.24 mm)."""
NOTCH_W = 2.0
NOTCH_X_TARGET = -9.5
"""The coded notch reaches from the hole to x = -9.5 mm (direction -x: where the bit points at plug 90 deg)."""
CAVITY = 4.0
PLUG_TOP = -(FACE_T + CAVITY)
SOCKET_DEPTH = 8.0
SOCKET_FLOOR = PLUG_TOP - SOCKET_DEPTH
PLUG_HALF = 10.0
CRANK_R = 19.05
"""Pin radius = the bolt's throw (3/4 in)."""
PIN = 2.0
CRANK_Z = (SOCKET_FLOOR - 4.5, SOCKET_FLOOR - 2.0)
PIN_Z = (-27.0, CRANK_Z[0])
BOLT = 7.95
"""Half-width of the bolt rod (5/8 in)."""
BOLT_Z = -38.0
BOLT_LEN = 30.0
THROW = 19.05
YOKE_GAP = 2.2
YOKE_BAR = 5.0
YOKE_Y = (-4.0, 23.0)
YOKE_Z = (PIN_Z[0], CRANK_Z[0] - 1.0)
BRIDGE_Z = (-30.0, PIN_Z[0] - 0.5)
PLUG_MASS, BOLT_MASS = 0.02, 0.04

# Key (key frame K: origin at the tip, z toward the bow; the bit points +y; the bow lies in the x-z plane) -----
SHANK_TOP = 25.0
CODES = {
    "A1": {"shank": (3.0, 3.0), "socket": (3.35, 3.35)},
    "B1": {"shank": (2.0, 3.8), "socket": (2.35, 4.15)},
}
"""The key code (warded stand-in for the pin tumblers): each key's shank cross-section (half-widths along the key's
x and y, mm) and the plug socket it seats in. A1: 6 x 6 mm square shank, 6.7 mm square socket. B1: 4 x 7.6 mm flat
shank, 4.7 x 8.3 mm socket. Key A1 cannot enter a B1 socket (6 mm > 4.7 mm) and key B1 cannot enter an A1 socket
(7.6 mm > 6.7 mm), so a wrong key stops on the plug's top with its bit still in the face plate and cannot turn the
plug. Both shanks turn freely in the face plate's 9 mm square hole (half-diagonals 4.24 and 4.29 mm < 4.5 mm)."""
BIT_RMAX = 8.0
BIT_T = 1.5
BIT_Z = (8.2, 11.2)
BOW = (14.0, 1.5, 24.0)
"""Bow half-width (x), half-thickness (y), height (z)."""
KEY_MASS = 0.02
KEY_TIP_IN_L = SOCKET_FLOOR + 0.3
"""Where the key's tip sits (lock frame z) when fully home."""
GRASP_K = (0.0, 0.0, SHANK_TOP + BOW[2] / 2)
"""The bow's centre, where the reference grasps (key frame, mm)."""

# Breaker station (frame B: centre of the breaker's footprint on the desk) ---------------------------------
BRK = (52.5, 80.0, 86.0)
"""Half-length (x), half-width (y) and height of the molded-case breaker lying face up (inferred class size)."""
HANDLE = (11.0, 6.0, 32.0)
HANDLE_OFF_DEG, HANDLE_ON_DEG = -20.0, 20.0
BRK_LOCK = (52.1, -11.0, 146.0)
"""Breaker lock's L origin in frame B: its bolt (L z -38 -> 108 mm) passes 22 mm above the face, on the ON
side of the handle (bolt y -19..-3 mm; the handle at OFF (-20 deg, leaning +y) reaches y -0.85 mm at that height;
turning toward ON (+) leans it toward -y, into the extended bolt)."""

BRK_LABEL = (-51.0, -17.0, -42.0, 42.0)
"""Breaker face label (x0, x1, y0, y1 in frame B, mm): 34 x 84 mm, on the robot side of the handle."""
BRK_FRONT_LABEL = (-75.0, 75.0, 8.0, 78.0)
"""Label on the case's robot-facing side (y0, y1, z0, z1 in frame B, mm): 150 x 70 mm."""

# Door station (frame D: the door lock's keyway axis on the desk; the lock is turned -90 deg so its bolt goes +y)
DOOR_LOCK_Z = -HOUSING_Z[0]
DOOR_X = (50.0, 53.0)
DOOR_Y = (40.0, 140.0)
DOOR_Z = (3.0, 143.0)
HINGE = (51.5, 140.0)
DOOR_OPEN_DEG = -100.0
KEEPER_Y = (44.0, 47.0)
KEEPER_X = (-13.0, DOOR_X[0])
KEEPER_Z = (3.0, 35.0)
KEEPER_HOLE = 9.0
DOOR_HANDLE = ((37.0, DOOR_X[0]), (50.0, 62.0), (85.0, 115.0))
ENCLOSURE = ((54.0, 145.0), (35.0, 145.0), (0.0, 152.0))
DOOR_MASS = 0.3
DOOR_TOP_LABEL = (62.0, 138.0, 40.0, 140.0)
"""Label on the enclosure's top (x0, x1, y0, y1 in frame D, mm): 76 x 100 mm."""

CONTACT_OFFSET = 0.001
SOLVER_ITERATIONS = 32

COLORS = dict(kit.COLORS)
COLORS.update({
    "brass": (0.78, 0.62, 0.25),
    "nickel": (0.80, 0.80, 0.78),
    "kirk_red": (0.75, 0.08, 0.08),
    "breaker": (0.16, 0.16, 0.17),
    "door": (0.80, 0.80, 0.76),
})

LABEL_COLORS = {"A1": (250, 205, 30), "B1": (90, 170, 235)}
"""Equipment-label background per interlock pair (A1 safety yellow, B1 light blue), so the pairs differ in colour
as well as in text at the policy camera's resolution."""

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size):
    return ImageFont.truetype(FONT_B, size)


def make_textures(tex: Path, codes) -> None:
    """Kirk nameplates (red, white lettering, the key code and the bolt designation) and door labels."""
    tex.mkdir(parents=True, exist_ok=True)
    for code, mode in codes:
        img = Image.new("RGB", (256, 128), (190, 20, 20))
        d = ImageDraw.Draw(img)
        d.rectangle([3, 3, 252, 124], outline=(255, 255, 255), width=3)
        d.text((128, 26), "KIRK KEY INTERLOCK", fill=(255, 255, 255), font=_font(20), anchor="mm")
        d.text((128, 76), code, fill=(255, 255, 255), font=_font(56), anchor="mm")
        d.text((128, 112), mode, fill=(255, 255, 255), font=_font(18), anchor="mm")
        img.save(tex / f"plate_{code}_{mode}.png")
        door = Image.new("RGB", (256, 160), (245, 245, 240))
        d = ImageDraw.Draw(door)
        d.rectangle([3, 3, 252, 156], outline=(20, 20, 20), width=4)
        d.text((128, 38), "FUSE COMPT.", fill=(20, 20, 20), font=_font(30), anchor="mm")
        d.text((128, 104), code, fill=(20, 20, 20), font=_font(72), anchor="mm")
        door.save(tex / f"door_{code}.png")
    for code in sorted({c for c, _ in codes}):
        bg = LABEL_COLORS[code]
        img = Image.new("RGB", (128, 64), bg)
        ImageDraw.Draw(img).text((64, 32), code, fill=(20, 20, 20), font=_font(44), anchor="mm")
        img.save(tex / f"key_{code}.png")
        # Equipment labels (top faces, read by the top camera): the code large, black on the pair's colour.
        for name, (w, h) in (("label", (256, 192)), ("label_strip", (256, 104)), ("label_front", (256, 120))):
            img = Image.new("RGB", (w, h), bg)
            d = ImageDraw.Draw(img)
            d.rectangle([4, 4, w - 5, h - 5], outline=(20, 20, 20), width=5)
            d.text((w // 2, h // 2 + 4), code, fill=(20, 20, 20), font=_font(int(h * 0.78)), anchor="mm")
            img.save(tex / f"{name}_{code}.png")
    for name, text, bg in (("off", "O OFF", (40, 160, 70)), ("on", "I ON", (200, 30, 30))):
        img = Image.new("RGB", (128, 64), bg)
        ImageDraw.Draw(img).text((64, 32), text, fill=(255, 255, 255), font=_font(34), anchor="mm")
        img.save(tex / f"brk_{name}.png")


def _tex_material(stage, root, name, file_rel):
    """Copied from tools/c11_make_props.py (not imported, so its owner can change it)."""
    path = f"{root}/Looks/tex_{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    reader = UsdShade.Shader.Define(stage, f"{path}/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"{path}/tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(file_rel)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _quad(stage, path, corners, material):
    """A textured quad (visual). corners in metres: image bottom-left, bottom-right, top-right, top-left."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


class Frame:
    """Maps lock-frame millimetres to a body's frame (metres): a yaw about z and an offset (mm)."""

    def __init__(self, offset=(0.0, 0.0, 0.0), yaw_deg=0.0):
        self.offset = offset
        self.c, self.s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        self.yaw = yaw_deg

    def p(self, x, y, z):
        return ((self.offset[0] + self.c * x - self.s * y) * MM, (self.offset[1] + self.s * x + self.c * y) * MM,
                (self.offset[2] + z) * MM)

    def box(self, stage, path, xr, yr, zr, material, collide=True):
        """An axis-aligned box in lock-frame mm (the yaw must be a multiple of 90 deg)."""
        c = self.p((xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2, (zr[0] + zr[1]) / 2)
        sx, sy = abs(xr[1] - xr[0]), abs(yr[1] - yr[0])
        if abs(self.s) > 0.5:
            sx, sy = sy, sx
        return kit.box(stage, path, c, (sx * MM, sy * MM, abs(zr[1] - zr[0]) * MM), material, collide)


def _body(stage, path, frame: Frame, mass):
    """A rigid body whose frame is the lock frame (origin and yaw)."""
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*frame.p(0, 0, 0)))
    xform.AddRotateZOp().Set(frame.yaw)
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(mass)
    return xform


def _joint(stage, path, kind, body0, body1, axis, pos0_m, yaw_deg, lower, upper):
    joint = (UsdPhysics.RevoluteJoint if kind == "revolute" else UsdPhysics.PrismaticJoint).Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(str(body0))])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(str(body1))])
    joint.CreateAxisAttr(axis)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pos0_m))
    h = math.radians(yaw_deg) / 2
    joint.CreateLocalRot0Attr(Gf.Quatf(math.cos(h), 0, 0, math.sin(h)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular" if kind == "revolute" else "linear")
    return joint


def _thin(stage, prefix):
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def add_kirk_lock(stage, root, base_path, frame: Frame, code, mats, plate_mat, hand=1, notch_x=None):
    """Housing (into the root body at base_path), plug and bolt bodies and their joints. Returns the joint paths.

    hand=+1: the plug turns 0 -> +90 deg to extend the bolt, the notch points -x (where the key's bit points at
    +90 deg). hand=-1: the mirror-handed lock: 0 -> -90 deg extends the bolt, the notch points +x. Same key.
    """
    f = frame
    b = base_path
    notch_x = NOTCH_X_TARGET if notch_x is None else notch_x
    sx, sy = CODES[code]["socket"]
    f.box(stage, f"{b}/kirk_housing", HOUSING_X, HOUSING_Y, HOUSING_Z, mats["brass"])
    # Face plate with the keyhole (square hole + coded notch toward -x for hand=+1, +x for hand=-1).
    z = (-FACE_T, 0.0)
    m = hand

    def u(a, c):
        return (m * a, m * c)
    f.box(stage, f"{b}/kirk_face0", u(-PLATE, notch_x), (-PLATE, PLATE), z, mats["brass"])
    f.box(stage, f"{b}/kirk_face1", u(HOLE, PLATE), (-PLATE, PLATE), z, mats["brass"])
    f.box(stage, f"{b}/kirk_face2", u(notch_x, HOLE), (HOLE, PLATE), z, mats["brass"])
    f.box(stage, f"{b}/kirk_face3", u(notch_x, HOLE), (-PLATE, -HOLE), z, mats["brass"])
    f.box(stage, f"{b}/kirk_face4", u(notch_x, -HOLE), (NOTCH_W, HOLE), z, mats["brass"])
    f.box(stage, f"{b}/kirk_face5", u(notch_x, -HOLE), (-HOLE, -NOTCH_W), z, mats["brass"])
    cyl = kit.cylinder(stage, f"{b}/kirk_cylinder", f.p(0, 0, (HOUSING_Z[1] - FACE_T - 0.2) / 2), CYL_R * MM,
                       (-HOUSING_Z[1] - FACE_T - 0.2) * MM, mats["brass"])
    del cyl
    # Nameplate on the housing's top face, beside the cylinder (visual).
    zt = HOUSING_Z[1] + 0.3
    x0, x1 = HOUSING_X[0] + 2, -CYL_R - 1
    y0, y1 = (-20.0, 20.0)
    w = [f.p(x0, y0, zt), f.p(x0, y1, zt), f.p(x1, y1, zt), f.p(x1, y0, zt)]
    xs, ys = sorted({round(p[0], 6) for p in w}), sorted({round(p[1], 6) for p in w})
    # Readable from the robot (looking +x): text bottom toward -x, text left toward +y.
    zw = w[0][2]
    pts = [(xs[0], ys[1], zw), (xs[0], ys[0], zw), (xs[1], ys[0], zw), (xs[1], ys[1], zw)]
    _quad(stage, f"{b}/kirk_plate", pts, plate_mat)

    plug = _body(stage, f"{root}/plug", f, PLUG_MASS)
    pb = Frame()
    P = f"{root}/plug"
    pz = (SOCKET_FLOOR, PLUG_TOP)
    pb.box(stage, f"{P}/wall0", (-PLUG_HALF, -sx), (-PLUG_HALF, PLUG_HALF), pz, mats["brass"])
    pb.box(stage, f"{P}/wall1", (sx, PLUG_HALF), (-PLUG_HALF, PLUG_HALF), pz, mats["brass"])
    pb.box(stage, f"{P}/wall2", (-sx, sx), (sy, PLUG_HALF), pz, mats["brass"])
    pb.box(stage, f"{P}/wall3", (-sx, sx), (-PLUG_HALF, -sy), pz, mats["brass"])
    pb.box(stage, f"{P}/floor", (-PLUG_HALF, PLUG_HALF), (-PLUG_HALF, PLUG_HALF), (SOCKET_FLOOR - 2.0, SOCKET_FLOOR),
           mats["brass"])
    pb.box(stage, f"{P}/crank", (-3.0, 3.0), (0.0, m * (CRANK_R + 3.0)), CRANK_Z, mats["steel"])
    # Round pin (a cylinder collider) so the yoke slot can be tight: 4 mm pin in a 4.4 mm slot.
    pin = UsdGeom.Cylinder.Define(stage, f"{P}/pin")
    pin.CreateRadiusAttr(PIN * MM)
    pin.CreateHeightAttr((PIN_Z[1] - PIN_Z[0]) * MM)
    pin.CreateAxisAttr("Z")
    UsdGeom.Xformable(pin).AddTranslateOp().Set(Gf.Vec3d(0.0, m * CRANK_R * MM, (PIN_Z[0] + PIN_Z[1]) / 2 * MM))
    UsdShade.MaterialBindingAPI.Apply(pin.GetPrim()).Bind(mats["steel"])
    UsdPhysics.CollisionAPI.Apply(pin.GetPrim())

    bolt = _body(stage, f"{root}/bolt", f, BOLT_MASS)
    B = f"{root}/bolt"
    x_end = HOUSING_X[0]
    pb.box(stage, f"{B}/rod", (x_end, x_end + BOLT_LEN), (-BOLT, BOLT), (BOLT_Z - BOLT, BOLT_Z + BOLT), mats["steel"])
    yoke_y = (m * YOKE_Y[0], m * YOKE_Y[1])
    pb.box(stage, f"{B}/bridge", (x_end + BOLT_LEN - 2.0, YOKE_GAP + YOKE_BAR), yoke_y, BRIDGE_Z, mats["steel"])
    pb.box(stage, f"{B}/yoke0", (-YOKE_GAP - YOKE_BAR, -YOKE_GAP), yoke_y, YOKE_Z, mats["steel"])
    pb.box(stage, f"{B}/yoke1", (YOKE_GAP, YOKE_GAP + YOKE_BAR), yoke_y, YOKE_Z, mats["steel"])

    pos0 = f.p(0, 0, 0)
    pj = _joint(stage, f"{root}/joints/plug_joint", "revolute", base_path, plug.GetPath(), "Z", pos0, f.yaw,
                *((0.0, 90.0) if hand > 0 else (-90.0, 0.0)))
    bj = _joint(stage, f"{root}/joints/bolt_joint", "prismatic", base_path, bolt.GetPath(), "X", pos0, f.yaw,
                -THROW * MM, 0.0)
    return pj, bj


def _station_root(stage, root):
    root_prim = stage.GetPrimAtPath(root)
    kit.articulation_root(stage, root_prim, self_collisions=True)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
        SOLVER_ITERATIONS)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)


def make_breaker(out_path: Path, code: str) -> None:
    root = "/breaker"
    stage, _ = kit.new_stage(out_path, root)
    _station_root(stage, root)
    mats = kit.materials(stage, root, ("steel", "black", "white", "dark", "stand"))
    mats.update(_extra_mats(stage, root))
    base = kit.rigid_body(stage, f"{root}/base", (0, 0, 0), 5.0)
    kit.fix_to_world(stage, root, base.GetPath())
    o = Frame()
    bx, by, bz = BRK
    o.box(stage, f"{root}/base/case", (-bx, bx), (-by, by), (0.0, bz), mats["breaker"])
    o.box(stage, f"{root}/base/escutcheon", (-16.0, 16.0), (-30.0, 30.0), (bz, bz + 1.0), mats["dark"], collide=False)
    lx, ly, lz = BRK_LOCK
    # Bracket under the lock (over the breaker's face) and a column under its overhang.
    o.box(stage, f"{root}/base/bracket", (lx + HOUSING_X[0], lx + HOUSING_X[1]), (ly + HOUSING_Y[0], ly + HOUSING_Y[1]),
          (bz, lz + HOUSING_Z[0]), mats["steel"])
    o.box(stage, f"{root}/base/column", (bx, lx + HOUSING_X[1]), (ly + HOUSING_Y[0], ly + HOUSING_Y[1]), (0.0, bz),
          mats["stand"])
    tex_plate = _tex_material(stage, root, "plate", f"./tex/plate_{code}_L-O.png")
    add_kirk_lock(stage, root, base.GetPath(), Frame((lx, ly, lz), 0.0), code, mats, tex_plate)
    # Equipment label on the face, between the robot-side edge and the handle (reads from behind the robot).
    lab = _tex_material(stage, root, "label", f"./tex/label_strip_{code}.png")
    zt = (bz + 1.2) * MM
    x0, x1, y0, y1 = BRK_LABEL
    _quad(stage, f"{root}/base/label", [(x0 * MM, y1 * MM, zt), (x0 * MM, y0 * MM, zt), (x1 * MM, y0 * MM, zt),
                                        (x1 * MM, y1 * MM, zt)], lab)
    # And on the case's robot-facing side.
    front = _tex_material(stage, root, "label_front", f"./tex/label_front_{code}.png")
    xf = (-bx - 0.3) * MM
    y0, y1, z0, z1 = BRK_FRONT_LABEL
    _quad(stage, f"{root}/base/label_front", [(xf, y1 * MM, z0 * MM), (xf, y0 * MM, z0 * MM), (xf, y0 * MM, z1 * MM),
                                              (xf, y1 * MM, z1 * MM)], front)
    # OFF / ON marks on the face either side of the handle (OFF toward +y).
    for name, sy in (("off", 1), ("on", -1)):
        m = _tex_material(stage, root, f"brk_{name}", f"./tex/brk_{name}.png")
        yc, zt = sy * 45.0, (bz + 1.2) * MM
        _quad(stage, f"{root}/base/mark_{name}", [((-16) * MM, (yc + 14) * MM, zt), ((-16) * MM, (yc - 14) * MM, zt),
                                                  (16 * MM, (yc - 14) * MM, zt), (16 * MM, (yc + 14) * MM, zt)], m)
    handle = kit.rigid_body(stage, f"{root}/handle", (0, 0, bz * MM), 0.05)
    hx, hy, hh = HANDLE
    Frame().box(stage, f"{root}/handle/lever", (-hx, hx), (-hy, hy), (0.0, hh), mats["black"])
    revolute = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/joints/handle_joint")
    revolute.CreateBody0Rel().SetTargets([base.GetPath()])
    revolute.CreateBody1Rel().SetTargets([handle.GetPath()])
    revolute.CreateAxisAttr("X")
    revolute.CreateLocalPos0Attr(Gf.Vec3f(0, 0, bz * MM))
    revolute.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    revolute.CreateLowerLimitAttr(HANDLE_OFF_DEG)
    revolute.CreateUpperLimitAttr(HANDLE_ON_DEG)
    UsdPhysics.DriveAPI.Apply(revolute.GetPrim(), "angular")
    _thin(stage, root)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def _extra_mats(stage, root):
    out = {}
    for name in ("brass", "nickel", "kirk_red", "breaker", "door"):
        path = f"{root}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[name]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35 if name in ("brass", "nickel") else 0.6)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.6 if name in ("brass", "nickel") else 0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        out[name] = material
    return out


def make_door(out_path: Path, code: str) -> None:
    root = "/door_station"
    stage, _ = kit.new_stage(out_path, root)
    _station_root(stage, root)
    mats = kit.materials(stage, root, ("steel", "black", "grey", "dark", "enclosure"))
    mats.update(_extra_mats(stage, root))
    base = kit.rigid_body(stage, f"{root}/base", (0, 0, 0), 10.0)
    kit.fix_to_world(stage, root, base.GetPath())
    o = Frame()
    (ex0, ex1), (ey0, ey1), (ez0, ez1) = ENCLOSURE
    o.box(stage, f"{root}/base/enclosure", (ex0, ex1), (ey0, ey1), (ez0, ez1), mats["enclosure"])
    tex_plate = _tex_material(stage, root, "plate", f"./tex/plate_{code}_L-C.png")
    add_kirk_lock(stage, root, base.GetPath(), Frame((0.0, 0.0, DOOR_LOCK_Z), -90.0), code, mats, tex_plate)
    # Equipment label on the enclosure's top (reads from behind the robot: text up toward +x).
    top = _tex_material(stage, root, "top_label", f"./tex/label_{code}.png")
    zt = (ez1 + 0.3) * MM
    x0, x1, y0, y1 = DOOR_TOP_LABEL
    _quad(stage, f"{root}/base/top_label", [(x0 * MM, y1 * MM, zt), (x0 * MM, y0 * MM, zt), (x1 * MM, y0 * MM, zt),
                                            (x1 * MM, y1 * MM, zt)], top)
    door = kit.rigid_body(stage, f"{root}/door", (HINGE[0] * MM, HINGE[1] * MM, 0.0), DOOR_MASS)
    d = Frame((-HINGE[0], -HINGE[1], 0.0))
    d.box(stage, f"{root}/door/panel", DOOR_X, DOOR_Y, DOOR_Z, mats["door"])
    # Keeper: a plate (normal y) with a square hole for the bolt, on the door's lower free corner.
    kz = (BOLT_Z + DOOR_LOCK_Z - KEEPER_HOLE, BOLT_Z + DOOR_LOCK_Z + KEEPER_HOLE)
    kx = (-KEEPER_HOLE, KEEPER_HOLE)
    d.box(stage, f"{root}/door/keeper_in", (KEEPER_X[0], kx[0]), KEEPER_Y, KEEPER_Z, mats["steel"])
    d.box(stage, f"{root}/door/keeper_out", (kx[1], KEEPER_X[1]), KEEPER_Y, KEEPER_Z, mats["steel"])
    d.box(stage, f"{root}/door/keeper_lo", kx, KEEPER_Y, (KEEPER_Z[0], kz[0]), mats["steel"])
    d.box(stage, f"{root}/door/keeper_hi", kx, KEEPER_Y, (kz[1], KEEPER_Z[1]), mats["steel"])
    d.box(stage, f"{root}/door/keeper_foot", (kx[1], KEEPER_X[1]), (DOOR_Y[0], KEEPER_Y[1]), (KEEPER_Z[0], KEEPER_Z[0] + 3),
          mats["steel"])
    (hx0, hx1), (hy0, hy1), (hz0, hz1) = DOOR_HANDLE
    d.box(stage, f"{root}/door/handle", (hx0, hx1), (hy0, hy1), (hz0, hz1), mats["black"])
    # Door label (visual), facing the robot.
    lab = _tex_material(stage, root, "door_label", f"./tex/door_{code}.png")
    xq = (DOOR_X[0] - 0.3 - HINGE[0]) * MM
    yl, yr = (DOOR_Y[1] - 10 - HINGE[1]) * MM, (70.0 - HINGE[1]) * MM
    zb, zt = 70.0 * MM, 115.0 * MM
    _quad(stage, f"{root}/door/label", [(xq, yl, zb), (xq, yr, zb), (xq, yr, zt), (xq, yl, zt)], lab)
    _joint(stage, f"{root}/joints/door_joint", "revolute", base.GetPath(), door.GetPath(), "Z",
           (HINGE[0] * MM, HINGE[1] * MM, 0.0), 0.0, DOOR_OPEN_DEG, 0.0)
    _thin(stage, root)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_key(out_path: Path, code: str) -> None:
    root = "/key"
    stage, root_prim = kit.new_stage(out_path, root)
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(KEY_MASS)
    mats = _extra_mats(stage, root)
    k = Frame()
    hx, hy = CODES[code]["shank"]
    k.box(stage, f"{root}/shank", (-hx, hx), (-hy, hy), (0.0, SHANK_TOP), mats["nickel"])
    k.box(stage, f"{root}/bit", (-BIT_T, BIT_T), (hy, BIT_RMAX), BIT_Z, mats["nickel"])
    bw, bt, bh = BOW
    k.box(stage, f"{root}/bow", (-bw, bw), (-bt, bt), (SHANK_TOP, SHANK_TOP + bh), mats["nickel"])
    tex = _tex_material(stage, root, "stamp", f"./tex/key_{code}.png")
    for s in (-1, 1):
        y = s * (bt + 0.2) * MM
        _quad(stage, f"{root}/stamp{'np'[s > 0]}",
              [(-s * 10 * MM, y, (SHANK_TOP + 6) * MM), (s * 10 * MM, y, (SHANK_TOP + 6) * MM),
               (s * 10 * MM, y, (SHANK_TOP + 18) * MM), (-s * 10 * MM, y, (SHANK_TOP + 18) * MM)], tex)
    kit.friction(stage, root, [f"{root}/bow"], static=1.0, dynamic=0.9)
    _thin(stage, root)
    root_prim.AddAppliedSchema("PhysxRigidBodyAPI")
    root_prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    make_textures(out_dir / "tex", [("A1", "L-O"), ("A1", "L-C"), ("B1", "L-O"), ("B1", "L-C")])
    for code in ("A1", "B1"):
        make_breaker(out_dir / f"breaker_{code}.usda", code)
        make_door(out_dir / f"door_{code}.usda", code)
        make_key(out_dir / f"key_{code}.usda", code)

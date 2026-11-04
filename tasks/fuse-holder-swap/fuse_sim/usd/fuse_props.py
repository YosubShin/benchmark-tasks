"""Generate the props of the fuse-holder swap (the named flip-open DIN fuse holder, fuses, stand and tray) as USD.

Real parts (references in the README). (P) published, (D) measured on the
published drawing (Bussmann Technical Data 720147 p.3, side view with the carrier open dashed, ~14.3 px/mm at
600 dpi), (I) inferred, (TM) to measure on a bought part.

- Fuse holder: Eaton Bussmann CHM1DU, 1-pole modular holder for 10 x 38 mm midget fuses. (P) 17.5 mm wide,
  73.9 mm tall, 52.8 mm from the back to the carrier's front face, the grip tab reaching 61.8 mm; the body's
  front steps back to 42.6 mm (lower) and 43.2 mm (upper) from the back. (D) carrier hinge 29.5 mm above the
  bottom and 4 mm behind the front face; fully open about 72 deg (the 4 mm lockout hole in the grip tab sits on
  a circle about the hinge boss in both drawn positions). (I) the fuse's axis 18 mm behind the carrier face, its
  centre 40 mm above the bottom (chosen within the body so the YAM's fingers clear the open carrier); the carrier's internals (a U clip round the inner ferrule, open toward the body, a saddle under the outer
  ferrule, a bottom stop) are ours: Bussmann does not publish them.
  Each carrier is a rigid body on a revolute joint about y at the hinge (0 = closed, opening is negative about
  +y: its top swings toward the robot). The latch is a task rule (a spring pulling the carrier shut inside
  LATCH_DEG); the open stop is the joint limit; gravity swings an unlatched carrier open, as on the real part.
  Collision: the plate, the grip tab, the U clip, saddle and bottom stop; the body's blocks and side walls; an
  upper contact block behind the outer ferrule (the inner contact is visual, so the swinging inner end clears).
- Fuse: Bussmann KTK class 10 x 38 midget (P: 10.3 x 38.1 mm). Caps 10 mm long, tube 8.8 mm across (I, from the
  Schurter ASO 10.3x38 drawing). 9 g (P, Weidmueller 10x38). The rating is printed on the
  tube ("BUSSMANN KTK-6"), black on the grey-white body, along the axis, as on the KTK-10 photo.
  Collision: three 16-gon prisms.
- Spares stand, tray, back pan and the circuit legends: ours.

Holder frame: d = depth behind the carrier's front face (world +x), y across, z above the holder's bottom.
Usage: python -m fuse_sim.usd.fuse_props [<out dir>]   (default: fuse_sim/assets)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from fuse_sim.spec import (BASE_Z, CIRCUITS, FRONT_X, RATING_COLOURS, SLOT_Y, SPARE_AXIS_Z, SPARE_RATINGS, SPARE_X,
                           SPARE_Y, TRAY_XY)
from fuse_sim.usd import usd_kit as kit

MM = 0.001
# Holder (CHM1DU), holder frame, metres ------------------------------------------------------------------------------
W = 17.5 * MM
H = 73.9 * MM
DEPTH = 52.8 * MM
LOWER_FRONT_D = (52.8 - 42.6) * MM
UPPER_FRONT_D = (52.8 - 43.2) * MM
LOWER_TOP = 11.0 * MM
"""Top of the lower block (I): low enough that the swinging fuse and bottom stop clear it."""
UPPER_BOTTOM = 65.5 * MM
"""Underside of the upper block. (D) the body's upper front starts 61.4 mm up; inside it there must be room for the
fuse's upper end to swing out (the real body houses the upper clip there), so the block starts at 65.5 mm (I):
at 61.4 the swinging fuse caught on it and held the carrier 8.6 deg open."""
BACK_D = 31.0 * MM
"""Front of the back block (I): the cavity the carrier swings in."""
WALL_IN = 7.75 * MM
HINGE_D, HINGE_Z = 4.0 * MM, 29.5 * MM
OPEN_DEG = 72.0
PLATE = (0.0, 2.0 * MM)
PLATE_Z = (21.0 * MM, 52.0 * MM)
PLATE_HALF_Y = 7.4 * MM
TAB_D = (-9.0 * MM, 0.0)
"""Grip tab: 9 mm proud of the carrier face (P: 61.8 - 52.8)."""
TAB_Z = (52.0 * MM, 61.0 * MM)
TAB_HALF_Y = 8.6 * MM
CARRIER_MASS = 0.015
CARRIER_COM = (-8.0 * MM, 0.0, 14.0 * MM)
"""Carrier centre of mass relative to the hinge (I): toward the grip tab, so that past ~8 deg the carrier with its
fuse falls open by gravity (with the default uniform-density COM it crept from 32 to 45 deg in 12 s)."""

# Fuse (10 x 38 midget) --------------------------------------------------------------------------------------------
FUSE_R = 5.15 * MM
FUSE_L = 38.1 * MM
CAP_L = 10.0 * MM
TUBE_R = 4.4 * MM
FUSE_MASS = 0.009
AXIS_D = 18.0 * MM
FUSE_ZC = 40.0 * MM
FUSE_Z0 = FUSE_ZC - FUSE_L / 2
FUSE_Z1 = FUSE_ZC + FUSE_L / 2
GAP = 0.1 * MM
CLIP_CLR = 1.0 * MM
CLIP_T = 1.5 * MM
CLIP_LIP = -1.0 * MM
CLIP_Z = (FUSE_Z0, FUSE_Z0 + CAP_L)
"""The inner clip (I): a U round the inner ferrule, open toward the body (up when the carrier is open), its cheeks
ending 1 mm short of the ferrule's axis, 1 mm clearance, so a ferrule lowered up to 4 mm off centre slides in off
the cheek's edge (with the cheeks 1 mm past the axis and 0.8 mm clearance a 1 mm offset left it perched on a cheek); with the bottom stop and a saddle under the outer ferrule it holds
the fuse in the carrier against gravity at any opening, and the body's upper contact block closes it in when shut.
The fuse is laid in and lifted out across its axis. A closed sleeve that it slid into along its axis jammed the
new fuse short of its seat."""
STOP_Z = (FUSE_Z0 - 2.3 * MM, FUSE_Z0 - GAP)
SADDLE_HALF_Y = 2.5 * MM
SADDLE_Z = (FUSE_Z1 - 9.5 * MM, FUSE_Z1 - 3.0 * MM)
UPPER_CONTACT_D = (AXIS_D + FUSE_R + GAP, AXIS_D + FUSE_R + 1.2 * MM)

LATCH_DEG = 6.0
OPEN_DETENT_DEG = 8.0
SPARE_SUPPORT_X0 = 0.248
LEGEND_W, LEGEND_H = 0.030, 0.025
"""Legend plate (ours): 30 x 25 mm, the rating's colour field on its lower half."""
LEGEND_Z0 = H + 0.008
"""Height of the legend plate's lower edge above the holders' bottom."""
LEGEND_TILT_DEG = 45.0
"""The plate leans back 45 deg from vertical (face up toward the robot) on its bracket."""
CRADLE_LABEL = (0.022, 0.028)
"""Cradle rating label on the stand's foot (x depth, y width)."""
FUSE_FRICTION = 0.6

COLORS = {
    "body": (0.86, 0.86, 0.84),
    "carrier": (0.80, 0.80, 0.78),
    "red": (0.78, 0.10, 0.12),
    "rail": (0.70, 0.72, 0.74),
    "pan": (0.84, 0.85, 0.83),
    "stand": (0.40, 0.40, 0.42),
    "copper": (0.80, 0.45, 0.22),
    "steel": (0.62, 0.64, 0.67),
    "fuse_tube": (0.88, 0.87, 0.82),
    "fuse_cap": (0.78, 0.79, 0.81),
    "tray": (0.20, 0.35, 0.65),
    "dark": (0.15, 0.15, 0.16),
}
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _mats(stage, root, names):
    out = {}
    for name in names:
        path = f"{root}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[name]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        if name in ("fuse_cap", "steel", "rail"):
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.6)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        out[name] = material
    return out


def _tex(stage, root, name, file_rel):
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
    """A textured quad (visual). corners: image bottom-left, bottom-right, top-right, top-left."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _span(stage, path, xs, ys, zs, material, collide=True):
    return kit.box(stage, path, ((xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2, (zs[0] + zs[1]) / 2),
                   (abs(xs[1] - xs[0]), abs(ys[1] - ys[0]), abs(zs[1] - zs[0])), material, collide)


def _contacts(stage, offset=0.001):
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(offset)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def _phys_mat(stage, root, prim_path, friction, name):
    mat = UsdShade.Material.Define(stage, f"{root}/PhysicsMaterials/{name}")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(friction)
    api.CreateDynamicFrictionAttr(friction)
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prim_path)).Bind(mat, UsdShade.Tokens.weakerThanDescendants,
                                                                     "physics")


def ngon(r, n=16):
    return [(r * math.cos(2 * math.pi * (k + 0.5) / n), r * math.sin(2 * math.pi * (k + 0.5) / n)) for k in range(n)]


# Geometry helpers shared with the task and the reference ---------------------------------------------------------------
def holder_to_world(slot: int, d: float, y: float, z: float) -> tuple[float, float, float]:
    return (FRONT_X + d, SLOT_Y[slot] + y, BASE_Z + z)


def hinge_world(slot: int) -> tuple[float, float, float]:
    return holder_to_world(slot, HINGE_D, 0.0, HINGE_Z)


def seat_in_carrier() -> tuple[float, float, float]:
    """The seated fuse's centre in the carrier (hinge) frame; its axis is the carrier's +z."""
    return (AXIS_D - HINGE_D, 0.0, FUSE_ZC - HINGE_Z)


def fuse_seat_world(slot: int) -> tuple[float, float, float]:
    return holder_to_world(slot, AXIS_D, 0.0, FUSE_ZC)


def carrier_names():
    return [f"carrier{s}" for s in range(3)]


def joint_names():
    return [f"carrier{s}_joint" for s in range(3)]


def spare_pose(k: int) -> tuple:
    """A spare lying on the stand, axis along +x (local z -> world +x), centred between its ends."""
    x = 0.5 * (SPARE_X[0] + SPARE_X[1])
    s = math.sqrt(0.5)
    return (x, SPARE_Y[k], SPARE_AXIS_Z, 0.0, s, 0.0, s)


# Holders -------------------------------------------------------------------------------------------------------------
def make_holders(out_path: Path) -> None:
    root = "/holders"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim)
    rp = root_prim
    rp.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    rp.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    m = _mats(stage, root, ("body", "carrier", "red", "rail", "pan", "stand", "copper", "steel", "fuse_tube",
                            "fuse_cap", "dark"))
    base = kit.rigid_body(stage, f"{root}/base", (0, 0, 0), 2.0)
    kit.fix_to_world(stage, root, base.GetPath())
    b = f"{root}/base"

    def hb(name, slot, ds, ys, zs, mat, collide=True):
        x0 = holder_to_world(slot, ds[0], ys[0], zs[0])
        x1 = holder_to_world(slot, ds[1], ys[1], zs[1])
        _span(stage, f"{b}/{name}", (x0[0], x1[0]), (x0[1], x1[1]), (x0[2], x1[2]), mat, collide)

    rail_d = (DEPTH, DEPTH + 7.5 * MM)
    rail_z = (39.0 * MM - 17.5 * MM, 39.0 * MM + 17.5 * MM)
    hb("rail", 1, rail_d, (-0.10, 0.10), rail_z, m["rail"], collide=False)
    pan_d = (rail_d[1], rail_d[1] + 3.0 * MM)
    hb("pan", 1, pan_d, (-0.12, 0.12), (-BASE_Z + 0.02, 0.15), m["pan"])
    hb("pan_foot", 1, (pan_d[0], pan_d[0] + 0.10), (-0.10, 0.10), (-BASE_Z, -BASE_Z + 0.02), m["stand"])
    hb("pan_brace", 1, (pan_d[1], pan_d[1] + 0.05), (-0.01, 0.01), (-BASE_Z + 0.02, 0.12), m["stand"])
    hw = W / 2 - 0.05 * MM
    for s in range(3):
        hb(f"lower{s}", s, (LOWER_FRONT_D, DEPTH), (-hw, hw), (0.0, LOWER_TOP), m["body"])
        hb(f"back{s}", s, (BACK_D, DEPTH), (-hw, hw), (LOWER_TOP, UPPER_BOTTOM), m["body"])
        hb(f"upper{s}", s, (UPPER_FRONT_D, DEPTH), (-hw, hw), (UPPER_BOTTOM, H), m["body"])
        for k, sv in enumerate((-1, 1)):
            hb(f"wall{s}{k}", s, (0.0, BACK_D), tuple(sorted((sv * WALL_IN, sv * hw))), (LOWER_TOP, UPPER_BOTTOM),
               m["body"])
        # Contacts: the upper one is a rear stop for the outer ferrule; the lower one is visual (see docstring).
        hb(f"contact_hi{s}", s, UPPER_CONTACT_D, (-3.0 * MM, 3.0 * MM), (FUSE_Z1 - 9.0 * MM, FUSE_Z1 - 1.0 * MM),
           m["copper"])
        hb(f"contact_lo{s}", s, UPPER_CONTACT_D, (-3.0 * MM, 3.0 * MM), (FUSE_Z0 + 1.0 * MM, FUSE_Z0 + 9.0 * MM),
           m["copper"], collide=False)
        # Terminal screws (visual) on the lower and upper fronts.
        for nm, dz, z in (("screw_lo", LOWER_FRONT_D, 5.5 * MM), ("screw_hi", UPPER_FRONT_D, 67.5 * MM)):
            c = holder_to_world(s, dz - 0.6 * MM, 0.0, z)
            kit.cylinder(stage, f"{b}/{nm}{s}", c, 3.5 * MM, 1.2 * MM, m["steel"], axis="X")
    # Carriers.
    for s in range(3):
        name = f"carrier{s}"
        pivot = hinge_world(s)
        body = kit.rigid_body(stage, f"{root}/{name}", pivot, CARRIER_MASS)
        UsdPhysics.MassAPI(body.GetPrim()).CreateCenterOfMassAttr(Gf.Vec3f(*CARRIER_COM))
        p = f"{root}/{name}"

        def cb(nm, ds, ys, zs, mat, collide=True):
            _span(stage, f"{p}/{nm}", (ds[0] - HINGE_D, ds[1] - HINGE_D), ys, (zs[0] - HINGE_Z, zs[1] - HINGE_Z), mat,
                  collide)

        cb("plate", PLATE, (-PLATE_HALF_Y, PLATE_HALF_Y), PLATE_Z, m["carrier"])
        cb("tab", TAB_D, (-TAB_HALF_Y, TAB_HALF_Y), TAB_Z, m["carrier"])
        # Bottom stop: up to 1 mm short of the axis, so a fuse lowered a little too far in lands on its end face
        # rather than perching on the stop's top (perched 3 mm high on a stop that reached 2 mm past the axis).
        cb("stop", (PLATE[1], AXIS_D - 1.0 * MM), (-FUSE_R - 2.0 * MM, FUSE_R + 2.0 * MM), STOP_Z, m["carrier"])
        # The inner clip: a U open toward the body (up when the carrier is open) round the inner ferrule.
        r_in = FUSE_R + CLIP_CLR
        cb("clip_floor", (PLATE[1], AXIS_D - FUSE_R - GAP), (-r_in - CLIP_T, r_in + CLIP_T), CLIP_Z, m["carrier"])
        for k, sv in enumerate((-1, 1)):
            cb(f"clip_side{k}", (AXIS_D - r_in, AXIS_D + CLIP_LIP), tuple(sorted((sv * r_in, sv * (r_in + CLIP_T)))),
               CLIP_Z, m["carrier"])
        cb("saddle", (PLATE[1], AXIS_D - FUSE_R - GAP), (-SADDLE_HALF_Y, SADDLE_HALF_Y), SADDLE_Z, m["carrier"])
        # Window recess in the face (visual) and the hinge boss.
        cb("window", (-0.4 * MM, 0.0), (-5.5 * MM, 5.5 * MM), (30.0 * MM, 44.0 * MM), m["dark"], collide=False)
        kit.cylinder(stage, f"{p}/boss", (0.0, 0.0, 0.0), 2.5 * MM, 2 * TAB_HALF_Y + 0.4 * MM, m["carrier"], axis="Y")
        # Red Bussmann label on the grip tab's front and top.
        tm = _tex(stage, root, f"tab{s}", "./textures/tab.png")
        d0, d1 = TAB_D[0] - 0.2 * MM - HINGE_D, TAB_D[1] - HINGE_D
        z0, z1 = TAB_Z[0] - HINGE_Z, TAB_Z[1] - HINGE_Z + 0.2 * MM
        hy = TAB_HALF_Y - 0.3 * MM
        _quad(stage, f"{p}/label_front", [(d0, hy, z0 + 1 * MM), (d0, -hy, z0 + 1 * MM), (d0, -hy, z1 - 1 * MM),
                                          (d0, hy, z1 - 1 * MM)], tm)
        _quad(stage, f"{p}/label_top", [(d0, hy, z1), (d0, -hy, z1), (d1, -hy, z1), (d1, hy, z1)], tm)
        if s != 1:  # decoy holders keep their fuse (visual; the task only scores the named holder's fuses)
            c = (AXIS_D - HINGE_D, 0.0, FUSE_ZC - HINGE_Z)
            kit.cylinder(stage, f"{p}/fuse_tube", c, TUBE_R, FUSE_L - 2 * CAP_L, m["fuse_tube"])
            for e in (-1, 1):
                kit.cylinder(stage, f"{p}/fuse_cap{e + 1}", (c[0], 0.0, c[2] + e * (FUSE_L - CAP_L) / 2), FUSE_R,
                             CAP_L, m["fuse_cap"])
        rb = body.GetPrim()
        rb.AddAppliedSchema("PhysxRigidBodyAPI")
        rb.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float).Set(0.3)
        kit.revolute(stage, f"{root}/joints/{name}_joint", base.GetPath(), body.GetPath(), "Y", pivot, -OPEN_DEG, 0.0)
    # Circuit legends: a plate per holder on a bracket off the pan above it, tilted LEGEND_TILT_DEG up toward the
    # robot (a vertical plate is seen almost edge-on from the overhead camera).
    for s_, (tag, circuit, amps) in enumerate(CIRCUITS):
        tm = _tex(stage, root, f"legend{s_}", f"./textures/legend_{tag}.png")
        y0 = SLOT_Y[s_]
        x_back = FRONT_X + pan_d[0]
        zb = BASE_Z + LEGEND_Z0
        t = math.radians(LEGEND_TILT_DEG)
        bot = (x_back - LEGEND_H * math.sin(t), zb)          # lower front edge
        top = (x_back - 0.0005, zb + LEGEND_H * math.cos(t))  # upper back edge, at the pan
        w = LEGEND_W / 2
        _quad(stage, f"{b}/legend{s_}", [(bot[0], y0 + w, bot[1]), (bot[0], y0 - w, bot[1]),
                                         (top[0], y0 - w, top[1]), (top[0], y0 + w, top[1])], tm)
        # Its bracket to the pan (visual).
        hb(f"legend_bracket{s_}", s_, (pan_d[0] - 0.012, pan_d[0]), (-0.003, 0.003), (LEGEND_Z0, LEGEND_Z0 + 0.004),
           m["dark"], collide=False)
    _phys_mat(stage, root, root, 0.5, "pbt")
    _contacts(stage)
    stage.GetRootLayer().Save()
    _textures(out_path.parent / "textures")
    print(f"wrote {out_path}")


def _font(bold, size):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def _textures(tex: Path) -> None:
    tex.mkdir(parents=True, exist_ok=True)
    # Legends: white engraved-style plate, tag and circuit in black, the rating on a field of its colour code.
    for tag, circuit, amps in CIRCUITS:
        _, rgb = RATING_COLOURS[amps]
        col = tuple(int(255 * c) for c in rgb)
        img = Image.new("RGB", (300, 250), (250, 250, 244))
        d = ImageDraw.Draw(img)
        d.text((150, 42), tag, fill=(0, 0, 0), font=_font(True, 64), anchor="mm")
        d.text((150, 98), circuit, fill=(0, 0, 0), font=_font(True, 30 if len(circuit) < 12 else 25), anchor="mm")
        d.rectangle([0, 125, 300, 250], fill=col)
        d.text((150, 188), f"{amps} A", fill=(255, 255, 255), font=_font(True, 84), anchor="mm")
        d.rectangle([1, 1, 298, 248], outline=(20, 20, 20), width=3)
        img.save(tex / f"legend_{tag}.png")
    # Spare cradle labels: the rating on its colour.
    for amps in SPARE_RATINGS:
        _, rgb = RATING_COLOURS[amps]
        img = Image.new("RGB", (300, 200), tuple(int(255 * c) for c in rgb))
        d = ImageDraw.Draw(img)
        d.text((150, 100), f"{amps} A", fill=(255, 255, 255), font=_font(True, 120), anchor="mm")
        img.save(tex / f"cradle_{amps}.png")
    img = Image.new("RGB", (240, 120), (190, 25, 30))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 240, 40], fill=(225, 225, 225))
    d.text((120, 20), "Bussmann", fill=(190, 25, 30), font=_font(True, 28), anchor="mm")
    d.text((120, 66), "Midget 10x38", fill=(255, 255, 255), font=_font(True, 22), anchor="mm")
    d.text((120, 98), "690V  32A", fill=(255, 255, 255), font=_font(False, 20), anchor="mm")
    img.save(tex / "tab.png")
    for amps in (6, 10):
        for blown in ((False, True) if amps == 6 else (False,)):
            img = Image.new("RGB", (360, 120), (226, 224, 214))
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, 110, 120], fill=(15, 15, 15))
            d.text((55, 60), "BUSS", fill=(255, 255, 255), font=_font(True, 30), anchor="mm")
            d.text((235, 42), f"KTK-{amps}", fill=(0, 0, 0), font=_font(True, 50), anchor="mm")
            d.text((235, 95), "600Vac FAST", fill=(0, 0, 0), font=_font(False, 22), anchor="mm")
            if blown:  # a black marker X over the whole printed face, as an electrician marks a fuse found open
                for a, b in (((0, 0), (360, 120)), ((0, 120), (360, 0))):
                    d.line([a, b], fill=(5, 5, 5), width=26)
            img.save(tex / f"fuse_{amps}{'_blown' if blown else ''}.png")


# Fuse --------------------------------------------------------------------------------------------------------------
def prism(stage, path, poly, z0, z1, material, collide=True):
    """A convex prism from a counter-clockwise xy polygon, between z0 and z1 (convex-hull collision)."""
    n = len(poly)
    pts = [Gf.Vec3f(x, y, z0) for x, y in poly] + [Gf.Vec3f(x, y, z1) for x, y in poly]
    faces, counts = list(range(n - 1, -1, -1)), [n]
    faces += list(range(n, 2 * n))
    counts.append(n)
    for k in range(n):
        a, b = k, (k + 1) % n
        faces += [a, b, n + b, n + a]
        counts.append(4)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    return mesh


def make_fuse(out_path: Path, amps: int, blown: bool = False) -> None:
    tex = f"{amps}_blown" if blown else f"{amps}"
    root = "/fuse"
    stage, root_prim = kit.new_stage(out_path, root)
    m = _mats(stage, root, ("fuse_tube", "fuse_cap"))
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(FUSE_MASS)
    root_prim.AddAppliedSchema("PhysxRigidBodyAPI")
    root_prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    root_prim.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float).Set(0.3)
    h = FUSE_L / 2
    prism(stage, f"{root}/cap_lo", ngon(FUSE_R), -h, -h + CAP_L, m["fuse_cap"])
    prism(stage, f"{root}/tube", ngon(TUBE_R), -h + CAP_L, h - CAP_L, m["fuse_tube"])
    prism(stage, f"{root}/cap_hi", ngon(FUSE_R), h - CAP_L, h, m["fuse_cap"])
    # Printed rating on four faces of the tube (reads along the axis).
    tm = _tex(stage, root, "print", f"./textures/fuse_{tex}.png")
    r = TUBE_R + 0.15 * MM
    a, half = h - CAP_L - 0.5 * MM, 2.6 * MM
    for k in range(4):
        ang = k * math.pi / 2
        n = (math.cos(ang), math.sin(ang))
        t = (-math.sin(ang), math.cos(ang))
        c = (r * n[0], r * n[1])
        corners = [(c[0] + half * t[0], c[1] + half * t[1], -a), (c[0] + half * t[0], c[1] + half * t[1], a),
                   (c[0] - half * t[0], c[1] - half * t[1], a), (c[0] - half * t[0], c[1] - half * t[1], -a)]
        _quad(stage, f"{root}/print{k}", corners, tm)
    _phys_mat(stage, root, root, FUSE_FRICTION, "nickel")
    _contacts(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


# Spares stand and tray --------------------------------------------------------------------------------------------------
def make_stands(out_path: Path) -> None:
    root = "/stands"
    stage, _ = kit.new_stage(out_path, root)
    m = _mats(stage, root, ("stand", "tray", "dark"))
    bottom = SPARE_AXIS_Z - FUSE_R - 0.3 * MM
    x0, x1 = SPARE_SUPPORT_X0, SPARE_X[1] + 0.5 * MM
    for k, y in enumerate(SPARE_Y):
        _span(stage, f"{root}/post{k}", (x0, x1 + 3 * MM), (y - 7 * MM, y + 7 * MM), (0.0, bottom), m["stand"])
        for j, sv in enumerate((-1, 1)):
            _span(stage, f"{root}/lip{k}{j}", (x0, x1), tuple(sorted((y + sv * (FUSE_R + 0.6 * MM), y + sv * 7 * MM))),
                  (bottom, SPARE_AXIS_Z), m["stand"])
        _span(stage, f"{root}/end{k}", (x1, x1 + 3 * MM), (y - 7 * MM, y + 7 * MM), (bottom, SPARE_AXIS_Z + 2 * MM),
              m["stand"])
        # The cradle's rating label: a plate on the stand's foot in front of the post, facing up.
        amps = SPARE_RATINGS[k]
        lx0, lx1 = x0 - CRADLE_LABEL[0] - 0.002, x0 - 0.002
        _span(stage, f"{root}/label_foot{k}", (lx0, lx1), (y - CRADLE_LABEL[1] / 2, y + CRADLE_LABEL[1] / 2),
              (0.0, 0.004), m["dark"])
        tm = _tex(stage, root, f"cradle{k}", f"./textures/cradle_{amps}.png")
        z = 0.0042
        hw = CRADLE_LABEL[1] / 2 - 0.001
        _quad(stage, f"{root}/label{k}", [(lx0 + 0.001, y + hw, z), (lx0 + 0.001, y - hw, z), (lx1 - 0.001, y - hw, z),
                                          (lx1 - 0.001, y + hw, z)], tm)
    tx, ty = TRAY_XY
    a, h = 0.09, 0.025
    _span(stage, f"{root}/tray_floor", (tx - a / 2, tx + a / 2), (ty - a / 2, ty + a / 2), (0.0, 0.003), m["tray"])
    for k, s in enumerate((-1, 1)):
        _span(stage, f"{root}/tray_wx{k}", (tx + s * a / 2 - 0.0015, tx + s * a / 2 + 0.0015), (ty - a / 2, ty + a / 2),
              (0.0, h), m["tray"])
        _span(stage, f"{root}/tray_wy{k}", (tx - a / 2, tx + a / 2), (ty + s * a / 2 - 0.0015, ty + s * a / 2 + 0.0015),
              (0.0, h), m["tray"])
    _contacts(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_holders(out_dir / "holders.usda")
    make_fuse(out_dir / "fuse_6_blown.usda", 6, blown=True)
    make_fuse(out_dir / "fuse_6.usda", 6)
    make_fuse(out_dir / "fuse_10.usda", 10)
    make_stands(out_dir / "stands.usda")

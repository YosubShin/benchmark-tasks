"""Generate the props of the ball-valve lockout task (close the named ball valve and padlock its locking handle) as USD.

Real part: Milwaukee Valve BA-100 with the "LD" locking device, 1/2" (distributor MPN BA100 LDA 12). Apollo
70-100 / 77C "-27" latch-lock works the same way (references in README.md).

How the real LD lock works (Milwaukee options sheet BV-37 drawing, Apollo 76F latch-lock video frames): a stamped stop
plate under the stem nut is fixed to the body and has a notch at the full-open and full-closed positions. The lever
carries a sliding stainless sleeve with a hooked tongue. Slid toward the stem at a stop, the tongue drops into the
notch and the lever cannot turn. The padlock shackle goes through a hole in the lever just outboard of the sleeve, so
the sleeve cannot slide back out: the padlock holds the latch, the latch holds the lever.

Published (Milwaukee BA-100 submittal BV-1, 1/2"): end to end B 2.19 in (56 mm), stem axis to end C 1.10 in (28 mm),
stem top to pipe centre D 1.91 in (49 mm), standard lever E 3.81 in (97 mm); LD lever 4-5/16 x 5/8 in (110 x 16 mm,
part 8686-D-940, distributor listing); takes a standard 5/16 in padlock shackle (BV-37); closes clockwise seen from above
(IOM). Inferred (to measure): lever 2.5 mm thick and flat (the real one has a sloped leg up from the nut), lock hole
9.5 mm (3/8 in) square 40 mm from the stem, sleeve 12 mm long, 20 x 8 mm outside, 10 mm travel, stop plate 44 mm across, 2 mm
thick, notches 20 deg wide and 6.5 mm deep, stop plate 10 mm below the lever, tongue 3.5 x 3 mm.

Parts:
- `pipe_run.usda`: a fixed-base articulation. Root = the pipe run (pipe on two stands, both valve bodies, both stop
  plates with their notches, and an engraved valve tag behind each valve on a strap clamped to the pipe). Per valve: `lever_<k>` on a revolute joint about the stem (z), 0 = open
  (lever along the pipe, pointing -y), -90 deg = closed (pointing -x, at the robot); `sleeve_<k>` on a prismatic joint
  along the lever (+y in the lever frame at open = toward the stem), 0 = in (latched), -SLEEVE_TRAVEL = out. Self
  collisions are on, so the sleeve's tongue really meets the stop plate (PhysX filters jointed parent/child pairs).
- The robot's padlock: Master Lock 410 (410PRP), the box model of valve_lockout_sim/usd/padlock_props.py in the 410's
  real shape (round shackle and bend, rounded body, label, keyway) by valve_lockout_sim/usd/padlock_410.py.
- `pedestal.usda`: the column the padlock stands on, upright, and the parking shelf.

Usage: python -m valve_lockout_sim.usd.valve_props valve_lockout_sim/assets
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from valve_lockout_sim.spec import TAGS
from valve_lockout_sim.usd import padlock_410, padlock_props
from valve_lockout_sim.usd import usd_kit as kit

# Layout (world) -------------------------------------------------------------------------------------------
VALVE_X = 0.26
"""World x of both stems (the pipe runs along y)."""
VALVE_Y = (0.0, 0.25)
"""World y of valve slot 0 (in front of the robot, the named one in seed 0) and slot 1 (left)."""
LEVER_Z = 0.165
"""World z of the lever plate's mid-plane."""
PIPE_D = 0.049
"""Stem top to pipe centre (D, published)."""
PIPE_Z = LEVER_Z - PIPE_D + 0.004
PIPE_Y = (-0.12, 0.36)

# Valve body (1/2", published B, C) --------------------------------------------------------------------------
BODY_LEN = 0.056
BODY_R = 0.016
HEX_AF = 0.027
PIPE_R = 0.0107
"""1/2 in schedule 40 pipe, 21.3 mm OD."""

# Lever, sleeve and stop plate (lever frame: origin on the stem axis at the lever's mid-plane; at q = 0 the lever
# points -y; s = -y is the distance out from the stem) ------------------------------------------------------
LEVER_LEN = 0.110
LEVER_W = 0.016
LEVER_T = 0.0025
LEVER_TAIL = 0.010
HOLE_S = 0.040
HOLE = 0.0095
GRIP_S = (0.052, LEVER_LEN)
GRIP_W, GRIP_T = 0.019, 0.006
SLEEVE_IN_S = (HOLE_S - HOLE / 2 - 0.0003 - 0.012, HOLE_S - HOLE / 2 - 0.0003)
"""Sleeve span along the lever when in (latched): its outer end at the lock hole's edge, so a shackle leg in the hole
leaves it about 1.6 mm of travel (the hole's clearance)."""
SLEEVE_W, SLEEVE_H, SLEEVE_WALL = 0.020, 0.008, 0.0012
SLEEVE_TRAVEL = 0.010
TONGUE_S = (0.016, 0.0195)
"""Tongue span along the lever when in."""
TONGUE_W = 0.0035
STOP_DZ = -0.010
"""Stop plate's top face below the lever's mid-plane."""
STOP_T = 0.0025
STOP_R = 0.022
STOP_R_IN = 0.013
STOP_CORE = 0.015
"""Half-size of the plate's solid centre square (the notch bottom)."""
STOP_SEGMENTS = 36
NOTCH_HALF_DEG = 10.0
NOTCH_ANGLES_DEG = (-90.0, 180.0)
"""Open (lever along -y) and closed (lever along -x), as directions from the stem."""
TONGUE_BOTTOM = STOP_DZ - STOP_T - 0.0015
"""In the notch the tongue reaches 1.5 mm below the plate, so the whole plate thickness bears on it."""
CLOSED_DEG = -90.0

LEVER_MASS = 0.06
SLEEVE_MASS = 0.01
CONTACT_OFFSET = 0.001
SOLVER_ITERATIONS = 32

# Valve tags (engraved two-ply plastic, 4 x 2 in) ---------------------------------------------------------------
TAG_W, TAG_H, TAG_T = 0.102, 0.051, 0.0016
TAG_DY = -0.015
"""Tag centre to the right of the stem (clear of the left gripper at the ready pose in the top camera)."""
TAG_DX = 0.068
"""Tag centre behind the stem (beyond the pipe, away from the robot)."""
TAG_Z = 0.150
TAG_TILT_DEG = 60.0
"""The tag face is tilted back 60 deg from vertical: its normal points toward the robot and 60 deg up, at the station's
top camera (which looks forward and 60 deg down from behind the arms)."""
TAG_STRAP_DY = 0.033

# Padlock pedestal ---------------------------------------------------------------------------------------------
PEDESTAL_XY = (0.17, 0.15)
PEDESTAL_TOP = 0.113
PARK_X, PARK_Z = 0.30, -0.30

COLORS = dict(kit.COLORS)
COLORS.update({
    "bronze": (0.72, 0.52, 0.25),
    "zinc": (0.70, 0.70, 0.66),
    "ss": (0.80, 0.81, 0.82),
    "vinyl": (0.95, 0.80, 0.08),
    "pipe": (0.30, 0.31, 0.33),
    "tag_plastic": (0.12, 0.12, 0.12),
})


def hole_world(slot: int, closed: bool = True) -> tuple[float, float, float]:
    """World centre of a valve's lever lock hole."""
    if closed:
        return (VALVE_X - HOLE_S, VALVE_Y[slot], LEVER_Z)
    return (VALVE_X, VALVE_Y[slot] - HOLE_S, LEVER_Z)


def _zbox(stage, path, center, size, material, yaw_deg, collide=True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddRotateZOp().Set(yaw_deg)
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _mats(stage, root):
    out = {}
    for name, rgb in COLORS.items():
        path = f"{root}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
        if name in ("bronze", "ss", "zinc"):
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.6)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        out[name] = material
    return out


def write_tag_textures(out_dir: Path) -> None:
    """Engraved two-ply plastic valve tags (white face, black core): the valve number in 32 mm letters."""
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 150)
    w, h = 400, round(400 * TAG_H / TAG_W)
    for k, tag in enumerate(TAGS):
        img = Image.new("RGB", (w, h), (238, 238, 232))
        d = ImageDraw.Draw(img)
        d.rectangle([6, 6, w - 7, h - 7], outline=(20, 20, 20), width=8)
        d.ellipse([16, h // 2 - 9, 34, h // 2 + 9], fill=(60, 60, 60))
        d.text((w / 2 + 12, h / 2 + 4), tag, fill=(15, 15, 15), font=font, anchor="mm")
        img.save(out_dir / f"valve_tag_{k}.png")


def _tex_material(stage, root, name, file_rel):
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


def tag_frame(slot: int):
    """(centre, image-up direction, image-right direction, face normal) of a valve's tag, world frame."""
    t = math.radians(TAG_TILT_DEG)
    centre = (VALVE_X + TAG_DX, VALVE_Y[slot] + TAG_DY, TAG_Z)
    up = (math.sin(t), 0.0, math.cos(t))
    right = (0.0, -1.0, 0.0)
    normal = (-math.cos(t), 0.0, math.sin(t))
    return centre, up, right, normal


def _valve_tag(stage, b, slot, plastic, face, strap):
    """The engraved tag on its strap: the strap is clamped round the pipe beside the valve body, runs back over the
    pipe and holds the tag behind the valve, its face tilted toward the robot (and the station's top camera)."""
    c, up, right, n = tag_frame(slot)
    t = TAG_TILT_DEG
    cube = UsdGeom.Cube.Define(stage, f"{b}/tag{slot}")
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*c))
    xf.AddRotateYOp().Set(t - 90.0)
    xf.AddScaleOp().Set(Gf.Vec3f(TAG_H, TAG_W, TAG_T))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(plastic)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    off = TAG_T / 2 + 0.0003
    f = [c[i] + off * n[i] for i in range(3)]

    def corner(a, b_):
        return tuple(f[i] + a * right[i] * TAG_W / 2 + b_ * up[i] * TAG_H / 2 for i in range(3))

    _quad(stage, f"{b}/tag{slot}_face", [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)], face)
    # Strap: from the top of the pipe, 3.3 cm left of the stem, back to the tag's lower edge.
    sy = VALVE_Y[slot] + TAG_STRAP_DY
    low = [c[i] - up[i] * TAG_H / 2 for i in range(3)]
    x0, z0 = VALVE_X, PIPE_Z + PIPE_R + 0.001
    length = math.hypot(low[0] - x0, low[2] - z0)
    cube = UsdGeom.Cube.Define(stage, f"{b}/tag_strap{slot}")
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d((x0 + low[0]) / 2, sy, (z0 + low[2]) / 2))
    xf.AddRotateYOp().Set(-math.degrees(math.atan2(low[2] - z0, low[0] - x0)))
    xf.AddScaleOp().Set(Gf.Vec3f(length + 0.004, 0.008, 0.002))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(strap)
    kit.cylinder(stage, f"{b}/tag_clamp{slot}", (VALVE_X, sy, PIPE_Z), PIPE_R + 0.0015, 0.008, strap, axis="Y")


def _stop_plate(stage, path, cx, cy, z_top, mat):
    """Stop plate: a solid centre square, a rim of segments from STOP_R_IN to STOP_R with two notches."""
    zc = z_top - STOP_T / 2
    kit.box(stage, f"{path}_core", (cx, cy, zc), (2 * STOP_CORE, 2 * STOP_CORE, STOP_T), mat)
    seg = 360.0 / STOP_SEGMENTS
    rc = (STOP_R + STOP_R_IN) / 2
    chord = 2 * STOP_R * math.tan(math.radians(seg / 2)) + 0.0004
    k = 0
    for i in range(STOP_SEGMENTS):
        a = i * seg + seg / 2 - 180.0
        if any(abs((a - n + 180) % 360 - 180) < NOTCH_HALF_DEG for n in NOTCH_ANGLES_DEG):
            continue
        ar = math.radians(a)
        _zbox(stage, f"{path}_rim{k}", (cx + rc * math.cos(ar), cy + rc * math.sin(ar), zc),
              (STOP_R - STOP_R_IN, chord, STOP_T), mat, a)
        k += 1


def _thin_contacts(stage, prefix):
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def make_pipe_run(out_path: Path) -> None:
    root = "/pipe_run"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=True)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
        SOLVER_ITERATIONS)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    m = _mats(stage, root)
    tags = [_tex_material(stage, root, f"tag_{k}", f"./valve_tag_{k}.png") for k in range(len(TAGS))]
    base = kit.rigid_body(stage, f"{root}/base", (0, 0, 0), 5.0)
    kit.fix_to_world(stage, root, base.GetPath())
    b = f"{root}/base"
    # Pipe (collision box, round visual) and its two stands.
    y0, y1 = PIPE_Y
    kit.box(stage, f"{b}/pipe_col", (VALVE_X, (y0 + y1) / 2, PIPE_Z), (2 * PIPE_R, y1 - y0, 2 * PIPE_R), m["pipe"],
            collide=True)
    UsdGeom.Imageable(stage.GetPrimAtPath(f"{b}/pipe_col")).MakeInvisible()
    kit.cylinder(stage, f"{b}/pipe", (VALVE_X, (y0 + y1) / 2, PIPE_Z), PIPE_R, y1 - y0, m["pipe"], axis="Y")
    for k, ys in enumerate((y0 + 0.03, y1 - 0.03)):
        kit.box(stage, f"{b}/stand{k}", (VALVE_X, ys, (PIPE_Z - PIPE_R) / 2), (0.03, 0.02, PIPE_Z - PIPE_R), m["stand"])
        kit.box(stage, f"{b}/stand_foot{k}", (VALVE_X, ys, 0.004), (0.08, 0.05, 0.008), m["stand"])
    for slot, vy in enumerate(VALVE_Y):
        # Valve body: bronze body and hex ends (collision: one box), bonnet up to the stop plate.
        kit.box(stage, f"{b}/body{slot}_col", (VALVE_X, vy, PIPE_Z), (2 * BODY_R, BODY_LEN, 2 * BODY_R), m["bronze"])
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{b}/body{slot}_col")).MakeInvisible()
        kit.cylinder(stage, f"{b}/body{slot}", (VALVE_X, vy, PIPE_Z), BODY_R, BODY_LEN - 0.02, m["bronze"], axis="Y")
        for e, sgn in enumerate((-1, 1)):
            kit.cylinder(stage, f"{b}/end{slot}_{e}", (VALVE_X, vy + sgn * (BODY_LEN / 2 - 0.008), PIPE_Z),
                         HEX_AF / math.sqrt(3), 0.016, m["bronze"], axis="Y")
        stop_top = LEVER_Z + STOP_DZ
        neck_h = stop_top - STOP_T - PIPE_Z
        kit.box(stage, f"{b}/bonnet{slot}", (VALVE_X, vy, PIPE_Z + neck_h / 2), (0.020, 0.020, neck_h), m["bronze"])
        _stop_plate(stage, f"{b}/stop{slot}", VALVE_X, vy, stop_top, m["zinc"])
        kit.cylinder(stage, f"{b}/packing_nut{slot}", (VALVE_X, vy, stop_top + 0.003), 0.007, 0.006, m["bronze"])
        _valve_tag(stage, b, slot, m["tag_plastic"], tags[slot], m["zinc"])

        # Lever: body origin on the stem axis at the lever's mid-plane.
        lv = f"{root}/lever_{slot}"
        kit.rigid_body(stage, lv, (VALVE_X, vy, LEVER_Z), LEVER_MASS)
        lo, hi = HOLE_S - HOLE / 2, HOLE_S + HOLE / 2
        kit.box(stage, f"{lv}/arm_inner", (0, -(lo - LEVER_TAIL) / 2, 0),
                (LEVER_W, lo + LEVER_TAIL, LEVER_T), m["zinc"])
        side = (LEVER_W - HOLE) / 2
        for k, sgn in enumerate((-1, 1)):
            kit.box(stage, f"{lv}/hole_side{k}", (sgn * (HOLE + side) / 2, -HOLE_S, 0), (side, HOLE, LEVER_T), m["zinc"])
        kit.box(stage, f"{lv}/arm_outer", (0, -(hi + LEVER_LEN) / 2, 0), (LEVER_W, LEVER_LEN - hi, LEVER_T), m["zinc"])
        kit.box(stage, f"{lv}/grip", (0, -(GRIP_S[0] + GRIP_S[1]) / 2, 0), (GRIP_W, GRIP_S[1] - GRIP_S[0], GRIP_T),
                m["vinyl"])
        kit.box(stage, f"{lv}/hub", (0, 0, -0.003), (0.016, 0.016, 0.004), m["zinc"], collide=False)
        kit.cylinder(stage, f"{lv}/stem_nut", (0, 0, 0.004), 0.0055, 0.006, m["zinc"])
        kit.revolute(stage, f"{root}/joints/lever_{slot}_joint", base.GetPath(), lv, "Z", (VALVE_X, vy, LEVER_Z),
                     CLOSED_DEG, 0.0)
        # Sleeve: body origin at its latched (in) position, in the lever frame.
        sl = f"{root}/sleeve_{slot}"
        kit.rigid_body(stage, sl, (VALVE_X, vy, LEVER_Z), SLEEVE_MASS)
        s0, s1 = SLEEVE_IN_S
        sc, sl_len = -(s0 + s1) / 2, s1 - s0
        w = SLEEVE_WALL
        half_in = LEVER_T / 2 + 0.0003
        kit.box(stage, f"{sl}/top", (0, sc, half_in + w / 2 + 0.0015), (SLEEVE_W, sl_len, w + 0.003), m["ss"])
        kit.box(stage, f"{sl}/bottom", (0, sc, -half_in - w / 2), (SLEEVE_W, sl_len, w), m["ss"])
        for k, sgn in enumerate((-1, 1)):
            kit.box(stage, f"{sl}/side{k}", (sgn * (SLEEVE_W / 2 - w / 2), sc, 0), (w, sl_len, 2 * half_in), m["ss"])
        t0, t1 = TONGUE_S
        arm_z = -half_in - w / 2
        kit.box(stage, f"{sl}/tongue_arm", (0, -(t0 + s0) / 2, arm_z), (TONGUE_W, s0 - t0, w), m["ss"])
        t_top = arm_z
        kit.box(stage, f"{sl}/tongue", (0, -(t0 + t1) / 2, (t_top + TONGUE_BOTTOM) / 2),
                (TONGUE_W, t1 - t0, t_top - TONGUE_BOTTOM), m["ss"])
        kit.prismatic(stage, f"{root}/joints/sleeve_{slot}_joint", lv, sl, "Y", (0, 0, 0), -SLEEVE_TRAVEL, 0.0)
    _thin_contacts(stage, root)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_pedestal(out_path: Path) -> None:
    root = "/pedestal"
    stage, _ = kit.new_stage(out_path, root)
    mats = kit.materials(stage, root, ("stand",))
    x, y = PEDESTAL_XY
    kit.box(stage, f"{root}/foot", (x, y, 0.005), (0.05, 0.05, 0.01), mats["stand"])
    kit.box(stage, f"{root}/column", (x, y, PEDESTAL_TOP / 2), (0.010, 0.012, PEDESTAL_TOP), mats["stand"])
    kit.box(stage, f"{root}/park_shelf", (PARK_X, 0.0, PARK_Z - 0.005), (0.12, 0.60, 0.01), mats["stand"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tag_textures(out_dir)
    make_pipe_run(out_dir / "pipe_run.usda")
    make_pedestal(out_dir / "pedestal.usda")
    padlock_410.make_padlock_410(out_dir / "padlock_ml410.usda", padlock_props.PADLOCKS["ml410"])

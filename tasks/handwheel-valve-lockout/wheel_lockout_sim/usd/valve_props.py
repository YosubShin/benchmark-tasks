"""Generate the props of the handwheel valve lockout task as USD: the pipe run with two handwheel gate valves and their
colour-coded valve tags, and the stands (the lockout cover's pedestal and the padlock's column).

Valve: Milwaukee Valve UP148 1/2 in bronze gate valve (rising stem, solid wedge, threaded bonnet, malleable iron
handwheel). Published (UP148 data sheet, 1/2"): end to end A 2.00 in (50.8 mm), pipe centre to the top of the wheel
nut, open, B 4.38 in (111 mm), handwheel diameter C 2.00 in (50.8 mm). Closes clockwise seen from the wheel.
Inferred: rim section 6 mm, four spokes, wheel nut 11 mm, bonnet and stuffing nut sizes. The stem's rise while the
wheel turns (a few millimetres per turn) is not modelled: the wheel turns in place, and its closed stop (the wedge
seating) is a joint limit.

Parts:
- `pipe_run.usda`: a fixed-base articulation. Root = the pipe run (pipe on two stands, both valve bodies and bonnets,
  and an engraved valve tag behind each valve on a strap clamped to the pipe). Per valve: `wheel_<k>` on a revolute
  joint about the stem (z), q = WHEEL_START at the start (one turn open), q = WHEEL_SEAT closed (clockwise from above
  = decreasing q).
- `stands.usda`: the lockout cover's pedestal and the padlock's column.
- The lockout cover: wheel_lockout_sim/usd/cover_props.py. The padlock: Master Lock 410 (padlock_410.py).

Usage: python -m wheel_lockout_sim.usd.valve_props wheel_lockout_sim/assets
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from wheel_lockout_sim.spec import TAG_COLOURS, TAGS
from wheel_lockout_sim.usd import padlock_410, padlock_props
from wheel_lockout_sim.usd import usd_kit as kit

# Layout (world) -------------------------------------------------------------------------------------------
VALVE_X = 0.26
"""World x of both stems (the pipe runs along y)."""
VALVE_Y = (0.0, 0.25)
"""World y of valve slot 0 (in front of the robot, the named one) and slot 1 (25 cm left)."""
WHEEL_Z = 0.150
"""World z of the handwheel rim's mid-plane."""
NUT_TOP = WHEEL_Z + 0.010
PIPE_Z = NUT_TOP - 0.105
"""Pipe centre: 105 mm below the wheel nut's top (published 111 mm fully open; the valve is one turn from closed)."""
PIPE_Y = (-0.12, 0.36)
PIPE_R = 0.0107
"""1/2 in schedule 40 pipe, 21.3 mm OD."""

# Valve body (UP148 1/2") -----------------------------------------------------------------------------------
BODY_LEN = 0.0508
BODY_R = 0.015
BONNET_TOP = WHEEL_Z - 0.040
"""Bonnet hex top; the stuffing nut sits on it, its top 30 mm under the rim, clear of the fitted cover."""
STUFF_NUT_TOP = WHEEL_Z - 0.033
STEM_R = 0.0042

# Handwheel (wheel frame: origin on the stem axis at the rim's mid-plane) -------------------------------------
WHEEL_D = 0.0508
RIM_T = 0.006
"""Rim section, 6 mm round (box collider 6 x 6 mm)."""
RIM_SEGMENTS = 16
SPOKES = 4
HUB_R = 0.007
NUT_AF = 0.011
WHEEL_MASS = 0.09
WHEEL_START_DEG = 170.0
WHEEL_SEAT_DEG = -190.0
"""One full turn (360 deg) from the start to the seat."""
WHEEL_OPEN_LIMIT_DEG = 350.0
"""Opening is stopped half a turn past the start (simplification; the real valve opens about five turns)."""

CONTACT_OFFSET = 0.001
SOLVER_ITERATIONS = 32

# Valve tags (engraved two-ply plastic, 4 x 2 in, colour-coded by service) -------------------------------------
TAG_W, TAG_H, TAG_T = 0.102, 0.051, 0.0016
TAG_DY = 0.020
TAG_DX = 0.095
"""Tag centre behind the stem (beyond the pipe and the fitted cover)."""
TAG_Z = 0.140
TAG_TILT_DEG = 60.0
"""The tag face is tilted back 60 deg from vertical, toward the robot and the station's top camera."""
TAG_STRAP_DY = 0.075

# Stands ----------------------------------------------------------------------------------------------------
COVER_PED_XY = (0.13, -0.13)
"""The lockout cover's pedestal (under the cover's axis)."""
COVER_PED_TOP = 0.080
PEDESTAL_XY = (0.14, 0.14)
"""The padlock's column (the lock stands on it upright and open)."""
PEDESTAL_TOP = 0.113

COLORS = dict(kit.COLORS)
COLORS.update({
    "bronze": (0.72, 0.52, 0.25),
    "zinc": (0.70, 0.70, 0.66),
    "wheel": (0.22, 0.23, 0.25),
    "pipe": (0.30, 0.31, 0.33),
    "tag_plastic": (0.12, 0.12, 0.12),
})


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


TAG_FACE_RGB = {"yellow": ((250, 204, 20), (12, 12, 12)), "green": ((20, 130, 60), (245, 245, 240))}
"""Two-ply engraving plastic: face colour and the core colour the letters are engraved down to (ANSI A13.1 style:
yellow with black letters, green with white letters)."""


def write_tag_textures(out_dir: Path) -> None:
    """Engraved two-ply plastic valve tags, colour-coded: the valve number in 28 mm letters."""
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 150)
    w, h = 400, round(400 * TAG_H / TAG_W)
    for tag, colour in zip(TAGS, TAG_COLOURS):
        face, core = TAG_FACE_RGB[colour]
        img = Image.new("RGB", (w, h), face)
        d = ImageDraw.Draw(img)
        d.rectangle([6, 6, w - 7, h - 7], outline=core, width=8)
        d.ellipse([16, h // 2 - 9, 34, h // 2 + 9], fill=(60, 60, 60))
        d.text((w / 2 + 12, h / 2 + 4), tag, fill=core, font=font, anchor="mm")
        img.save(out_dir / f"wheelvalve_tag_{tag.replace('-', '')}.png")


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
    # Strap: from the top of the pipe, 7.5 cm left of the stem, back to the tag's lower edge.
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


def _thin_contacts(stage, prefix):
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def _wheel(stage, path, m):
    """The handwheel: rim (16 box segments), four spokes, hub and wheel nut. Body origin at the rim's mid-plane."""
    r_mid = WHEEL_D / 2 - RIM_T / 2
    seg_len = 2 * (r_mid + RIM_T / 2) * math.tan(math.pi / RIM_SEGMENTS) + 0.0005
    for k in range(RIM_SEGMENTS):
        a = 360.0 * k / RIM_SEGMENTS
        c = (r_mid * math.cos(math.radians(a)), r_mid * math.sin(math.radians(a)), 0.0)
        _zbox(stage, f"{path}/rim{k}", c, (RIM_T, seg_len, RIM_T), m["wheel"], a, collide=True)
    for k in range(SPOKES):
        a = 360.0 * k / SPOKES + 45.0
        rr = (HUB_R + r_mid) / 2
        c = (rr * math.cos(math.radians(a)), rr * math.sin(math.radians(a)), 0.0)
        _zbox(stage, f"{path}/spoke{k}", c, (r_mid - HUB_R + 0.002, 0.005, 0.004), m["wheel"], a, collide=True)
    kit.box(stage, f"{path}/hub", (0, 0, 0.0005), (2 * HUB_R, 2 * HUB_R, 0.009), m["wheel"])
    nut_h = NUT_TOP - WHEEL_Z - 0.005
    kit.box(stage, f"{path}/nut_col", (0, 0, 0.005 + nut_h / 2), (NUT_AF, NUT_AF, nut_h), m["zinc"])
    UsdGeom.Imageable(stage.GetPrimAtPath(f"{path}/nut_col")).MakeInvisible()
    kit.cylinder(stage, f"{path}/nut", (0, 0, 0.005 + nut_h / 2), NUT_AF / math.sqrt(3), nut_h, m["zinc"])
    # Visual: a round rim over the box segments.
    torus_pts, faces, counts = [], [], []
    nu, nv = 32, 8
    for i in range(nu):
        u = 2 * math.pi * i / nu
        for j in range(nv):
            v = 2 * math.pi * j / nv
            rr = r_mid + RIM_T / 2 * math.cos(v)
            torus_pts.append(Gf.Vec3f(rr * math.cos(u), rr * math.sin(u), RIM_T / 2 * math.sin(v)))
    for i in range(nu):
        for j in range(nv):
            a, b = i * nv + j, ((i + 1) % nu) * nv + j
            c, d = ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv
            counts.append(4)
            faces.extend([a, b, c, d])
    mesh = UsdGeom.Mesh.Define(stage, f"{path}/rim_visual")
    mesh.CreatePointsAttr(torus_pts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(faces)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(m["wheel"])
    for k in range(RIM_SEGMENTS):
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{path}/rim{k}")).MakeInvisible()


def make_pipe_run(out_path: Path) -> None:
    root = "/pipe_run"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
        SOLVER_ITERATIONS)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    m = _mats(stage, root)
    tags = [_tex_material(stage, root, f"wheelvalve_tag_{t.replace('-', '')}", f"./wheelvalve_tag_{t.replace('-', '')}.png")
            for t in TAGS]
    base = kit.rigid_body(stage, f"{root}/base", (0, 0, 0), 5.0)
    kit.fix_to_world(stage, root, base.GetPath())
    b = f"{root}/base"
    y0, y1 = PIPE_Y
    kit.box(stage, f"{b}/pipe_col", (VALVE_X, (y0 + y1) / 2, PIPE_Z), (2 * PIPE_R, y1 - y0, 2 * PIPE_R), m["pipe"])
    UsdGeom.Imageable(stage.GetPrimAtPath(f"{b}/pipe_col")).MakeInvisible()
    kit.cylinder(stage, f"{b}/pipe", (VALVE_X, (y0 + y1) / 2, PIPE_Z), PIPE_R, y1 - y0, m["pipe"], axis="Y")
    for k, ys in enumerate((y0 + 0.03, y1 - 0.03)):
        kit.box(stage, f"{b}/stand{k}", (VALVE_X, ys, (PIPE_Z - PIPE_R) / 2), (0.03, 0.02, PIPE_Z - PIPE_R), m["stand"])
        kit.box(stage, f"{b}/stand_foot{k}", (VALVE_X, ys, 0.004), (0.08, 0.05, 0.008), m["stand"])
    for slot, vy in enumerate(VALVE_Y):
        # Body (collision: one box), hex ends, bonnet, stuffing nut, stem.
        kit.box(stage, f"{b}/body{slot}_col", (VALVE_X, vy, PIPE_Z + 0.004), (2 * BODY_R, BODY_LEN, 2 * BODY_R + 0.008),
                m["bronze"])
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{b}/body{slot}_col")).MakeInvisible()
        kit.cylinder(stage, f"{b}/body{slot}", (VALVE_X, vy, PIPE_Z), BODY_R, BODY_LEN - 0.016, m["bronze"], axis="Y")
        kit.cylinder(stage, f"{b}/dome{slot}", (VALVE_X, vy, PIPE_Z + 0.008), 0.017, 0.022, m["bronze"])
        for e, sgn in enumerate((-1, 1)):
            kit.cylinder(stage, f"{b}/end{slot}_{e}", (VALVE_X, vy + sgn * (BODY_LEN / 2 - 0.007), PIPE_Z),
                         0.0155, 0.014, m["bronze"], axis="Y")
        bonnet_bottom = PIPE_Z + 0.019
        kit.box(stage, f"{b}/bonnet{slot}", (VALVE_X, vy, (bonnet_bottom + BONNET_TOP) / 2),
                (0.022, 0.022, BONNET_TOP - bonnet_bottom), m["bronze"])
        kit.box(stage, f"{b}/stuff_nut{slot}", (VALVE_X, vy, (BONNET_TOP + STUFF_NUT_TOP) / 2),
                (0.016, 0.016, STUFF_NUT_TOP - BONNET_TOP), m["bronze"])
        stem_h = WHEEL_Z - STUFF_NUT_TOP
        kit.box(stage, f"{b}/stem{slot}_col", (VALVE_X, vy, STUFF_NUT_TOP + stem_h / 2), (2 * STEM_R, 2 * STEM_R, stem_h),
                m["bronze"])
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{b}/stem{slot}_col")).MakeInvisible()
        kit.cylinder(stage, f"{b}/stem{slot}", (VALVE_X, vy, STUFF_NUT_TOP + stem_h / 2), STEM_R, stem_h, m["bronze"])
        _valve_tag(stage, b, slot, m["tag_plastic"], tags[slot], m["zinc"])

        wv = f"{root}/wheel_{slot}"
        kit.rigid_body(stage, wv, (VALVE_X, vy, WHEEL_Z), WHEEL_MASS)
        _wheel(stage, wv, m)
        kit.revolute(stage, f"{root}/joints/wheel_{slot}_joint", base.GetPath(), wv, "Z", (VALVE_X, vy, WHEEL_Z),
                     WHEEL_SEAT_DEG, WHEEL_OPEN_LIMIT_DEG)
    kit.friction(stage, root, [f"{root}/wheel_{s}" for s in range(len(VALVE_Y))], static=0.9, dynamic=0.8,
                 name="painted_iron", combine="max")
    _thin_contacts(stage, root)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_stands(out_path: Path) -> None:
    root = "/stands"
    stage, _ = kit.new_stage(out_path, root)
    mats = kit.materials(stage, root, ("stand",))
    x, y = PEDESTAL_XY
    kit.box(stage, f"{root}/lock_foot", (x, y, 0.005), (0.05, 0.05, 0.01), mats["stand"])
    kit.box(stage, f"{root}/lock_column", (x, y, PEDESTAL_TOP / 2), (0.010, 0.012, PEDESTAL_TOP), mats["stand"])
    px, py = COVER_PED_XY
    kit.box(stage, f"{root}/cover_pedestal", (px, py, COVER_PED_TOP / 2), (0.045, 0.045, COVER_PED_TOP), mats["stand"])
    kit.box(stage, f"{root}/cover_pedestal_foot", (px, py, 0.004), (0.07, 0.07, 0.008), mats["stand"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tag_textures(out_dir)
    make_pipe_run(out_dir / "pipe_run.usda")
    from wheel_lockout_sim.usd import cover_props
    cover_props.make_cover(out_dir / "ml480.usda")
    make_stands(out_dir / "stands.usda")
    padlock_410.make_padlock_410(out_dir / "padlock_ml410.usda", padlock_props.PADLOCKS["ml410"])

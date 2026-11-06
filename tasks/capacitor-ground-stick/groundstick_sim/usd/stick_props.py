"""Generate the props of the capacitor ground-stick task as USD: the bench (two capacitors, earth bar, clip tab, stick
cradle, WAIT lamp) and the discharge stick with its lead and ground clip (one articulation).

Real parts (vendor pages in the README's References):

- Capacitor: Chicago Condenser CQ72E1FM105K3 (1.0 uF, 4 kV DC, oil-filled rectangular can). Published
  (capacitorindustries.com listing): case 82 x 96 x 122 mm (W x T x H). From the listing photos (scaled
  on the 82 mm width; inferred): two cream porcelain bushings 40 mm apart along the 82 mm width, about
  18 mm across at the base, 32 mm tall, a hex nut, a 1/4 in stud about 9 mm proud, solder-lug tabs. The
  case is light blue-grey. Two of them, C1 and C2, each with an equipment ID label on the face toward the robot,
  on the back face and on the lid in front of the bushings. The labels are coloured per unit (C1 blue with white
  print, C2 orange with black print) so that the named can is found in the 320 x 176 overhead camera, where the
  printed ID is only a few pixels tall.
- Earth bar: a copper ground bar 1/4 x 2 x 12 in (6.35 x 51 x 305 mm) on two insulators (nVent ERICO
  EGB-class size; the bench mounting is our layout). It stands on edge, faces normal to y.
- Discharge stick: Salisbury 22629 class "static discharge stick with universal 1 ft length" (published
  1 ft, 0.17 kg; Salisbury sheet; yellow fiberglass), with a universal head carrying a bent-rod probe like
  the one in EFCOG BP261 Fig. 9 (rod across both terminals, tip turned down). Rod 1/4 in (Cementex CPCD
  hook size). The tube is 32 mm across (Cementex CPCD 1-1/4 in OD; the 22629's is not published) and round,
  collider included, where the gripper holds it; a square anti-roll end cap (our addition) sits on the butt.
- Ground lead: 6 ft (1.83 m) of cable on the real tool; 0.90 m here (18 links of 50 mm on alternating hinges,
  a declared shortening), 10 mm across, 0.11 kg (lighter than the real cable; inferred).
- Ground clip: Mueller BU-21C solid copper clip (published: 104 mm long, 19 mm wide, 41 mm tall, 27 mm
  maximum jaw spread; Mueller datasheet). Pivot position, pad length, spring torque inferred: the spring
  (the task's joint drive) needs about 15 N at the handles to open the jaws 22 mm, and closes them on a
  6.35 mm bar with about 10 N at the pad edge. The real 22629 ships with a Salisbury 1814 C-type screw
  clamp; the spring clip is our choice (no screw turning) and a real alternative EFCOG Fig. 3
  names ("clip, eyelet, or clamp").
- Clip parking tab: an insulating tab on a post, the clip clipped to it at the start (our stand-in for
  "the lead lies on the bench").
- WAIT lamp: a Schneider Electric Harmony XB7 22 mm amber LED pilot light (XB7EV05BP, 24 V) in a grey one-hole
  control station on a post, with a "WAIT" plate above it. On the real bench it is powered from a 24 V battery pack
  through the break contact of an on-delay timer (Omron H3Y-2 DC24, 30 s range, set to 20 s) that the operator
  starts with the trial: lit for 20 s, then dark. The task's rule copies that timer.

Frames. Stick (articulation root body `stick`): origin at the grip centre, +x toward the head, the probe's
end turned down along -z. Clip: origin on the pivot, +x toward the jaw tips, the jaws part along y (pad A
inner face at y = 0, the fixed half on -y), pivot axis z. Lead links: origin at the link's start, along +x.

Usage: python -m groundstick_sim.usd.stick_props [<out dir>]   (default: groundstick_sim/assets)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from groundstick_sim.spec import CAP_COLORS, CAP_LABELS, CAP_X, CAP_Y, CLIP_TAB_XY, EARTH_BAR_Y, STICK_CRADLE_XY
from groundstick_sim.usd import usd_kit as kit

# Capacitor (Chicago Condenser CQ72E1FM105K3) --------------------------------------------------------------
CASE = (0.096, 0.082, 0.122)
"""x (thickness, toward the robot), y (width, along the terminal row), z (height): published 96 x 82 x 122 mm."""
TERMINAL_DY = 0.020
"""Each terminal's offset from the case centre along y (40 mm apart; inferred from the photo)."""
BUSH_R = (0.009, 0.007)
BUSH_H = (0.016, 0.016)
NUT_R, NUT_H = 0.0055, 0.005
STUD_R, STUD_H = 0.0032, 0.009
STUD_TOP = CASE[2] + sum(BUSH_H) + NUT_H + STUD_H
"""World z of a stud's top: 0.168 m."""
NUT_BOTTOM = CASE[2] + sum(BUSH_H)

# Earth bar -------------------------------------------------------------------------------------------------
BAR_T = 0.00635
BAR_X = (0.105, 0.41)
BAR_Z = (0.045, 0.096)
INSULATOR_X = (0.14, 0.375)

# Clip parking tab ------------------------------------------------------------------------------------------
TAB_TOP = 0.09
TAB_X_LEN = 0.03

# Stick (Salisbury 22629 class) -----------------------------------------------------------------------------
TUBE_R = 0.016
TUBE_X = (-0.085, 0.215)
"""1 ft of fiberglass (published length) from the butt to the head."""
GUARD_X = 0.07
"""Hand guard position (stick frame)."""
CAP_END_X = -0.03
"""The square anti-roll end cap runs from the butt (TUBE_X[0]) to here."""
GRIP_END_X = 0.064
"""The black handgrip sleeve runs from the end cap to here, just behind the hand guard."""
HEAD_R = 0.0125
HEAD_X = (0.215, 0.25)
ROD_R = 0.0032
PROBE_END_X = 0.36
"""The probe runs straight out of the head along the axis to here, then turns down."""
CURL_DROP = 0.045
RAIL_X = (0.27, 0.35)
RAIL_Y = 0.0078
RAIL_R = 0.002
RAIL_DROP = 0.008
"""Side rails (stick frame): x range, offset either side of the rod, radius, depth of their axis below the rod's."""
STICK_MASS = 0.17
LEAD_ATTACH = (0.232, 0.0, -HEAD_R)
"""Where the lead leaves the head (stick frame)."""

# Lead ---------------------------------------------------------------------------------------------------
LINK_N = 18
LINK_L = 0.05
LINK_R = 0.005
LINK_MASS = 0.006
LINK_LIMIT_DEG = 100.0

# Clip (Mueller BU-21C) -----------------------------------------------------------------------------------
PAD_X = (0.030, 0.056)
"""The toothed pads, from the pivot toward the tips (the 104 mm clip: 56 mm of jaw, 50 mm of handle)."""
HANDLE_X = (-0.050, -0.008)
HANDLE_Y = (0.012, 0.016)
"""Handle plate's inner and outer face (|y|) when closed: 32 mm across the handles."""
THROAT_Y = 0.006
CLIP_HALF_Z = 0.0095
"""19 mm wide (published)."""
JAW_MAX_DEG = 29.0
"""asin(27 mm / 56 mm): the published 27 mm maximum spread at the tips."""
CLIP_MASS = (0.035, 0.025)
LUG = (HANDLE_X[0] + 0.006, -0.014, -(CLIP_HALF_Z + 0.006))
"""Where the lead's end sits (clip frame): on a lug off the fixed half's side face near its rear. The lead leaves
along -z (world -x, toward the robot), past the gripper's fingers, which straddle the handles in y."""
BITE = (0.038, 0.003, 0.0)
"""A point between the pads (clip frame); inside the bar when the clip is clamped on it."""
THROAT_DEPTH = 0.024
"""How far up the throat (pivot to the bar's top edge) the bar sits when clamped."""


def clamp_angle(thickness: float) -> float:
    """Jaw angle (rad) at which pad B's inner edge (x = PAD_X[0]) rests on a plate of this thickness."""
    return math.asin(thickness / PAD_X[0])


CLIP_QUAT = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
"""Clip hanging jaws down: +90 deg about y (local x -> world -z, local y -> world y, local z -> world x)."""
STICK_QUAT = (0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5))
"""Stick lying along -y (local x -> world -y), probe end down."""


def clip_pose_on(plate_mid_y: float, plate_top: float, x: float) -> tuple:
    """Clip pose clamped on a plate standing on edge (faces normal to y): pad A against the plate's -y face."""
    return (x, plate_mid_y - BAR_T / 2, plate_top + THROAT_DEPTH) + CLIP_QUAT


def stick_rest_pose() -> tuple:
    return (STICK_CRADLE_XY[0], STICK_CRADLE_XY[1], 0.0 + TUBE_R + 0.041) + STICK_QUAT


def terminal_xyz(cap: int, k: int) -> tuple:
    """Stud top centre of terminal k (0: +y, nearer the stick's approach; 1: -y) of capacitor cap."""
    return (CAP_X[cap], CAP_Y + (TERMINAL_DY if k == 0 else -TERMINAL_DY), STUD_TOP)


# Rotation maths (numpy, xyzw) -----------------------------------------------------------------------------
def _qmat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _mat_q(m):
    from scipy.spatial.transform import Rotation

    return tuple(float(v) for v in Rotation.from_matrix(m).as_quat())


def _rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == "Z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


ATTACH_ROT = _rot("Y", math.pi / 2)
CLIP_LUG_ROT = _rot("Y", math.pi / 2)
"""Clip frame relative to the last link: the link's +x (toward the lug) is the clip's +z."""
"""The lead leaves the head pointing down (stick -z)."""


def link_axis(k: int) -> str:
    return "Z" if k % 2 == 0 else "Y"


def lead_fk(stick_pos, stick_rot, q):
    """Link frames (pos, rot) in the world for joint angles q; returns links and the clip frame."""
    p = np.asarray(stick_pos) + stick_rot @ np.asarray(LEAD_ATTACH)
    r = stick_rot @ ATTACH_ROT
    links = []
    for k in range(LINK_N):
        r = r @ _rot(link_axis(k), q[k])
        links.append((p.copy(), r.copy()))
        p = p + r @ np.array([LINK_L, 0, 0])
    clip_rot = r @ CLIP_LUG_ROT
    clip_pos = p - clip_rot @ np.asarray(LUG)
    return links, (clip_pos, clip_rot)


def solve_lead(stick_pose, clip_pose):
    """Joint angles that route the lead from the stick's head to the clip, kept above the desk."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    sp, sr = np.asarray(stick_pose[:3]), _qmat(stick_pose[3:])
    cp, cr = np.asarray(clip_pose[:3]), _qmat(clip_pose[3:])

    def residual(q):
        links, (pos, rot) = lead_fk(sp, sr, q)
        err_p = (pos - cp) * 200.0
        err_r = Rotation.from_matrix(cr.T @ rot).as_rotvec() * 30.0
        ends = [p + r @ np.array([LINK_L, 0, 0]) for p, r in links]
        lows = [max(0.0, 0.02 - e[2]) * 200.0 for e in ends]
        # Keep the lead in front of the capacitors (x < 0.250) and let it lie low, as a cable would.
        clear = [max(0.0, e[0] - 0.250) * 200.0 * (-0.10 < e[1] < 0.04) for e in ends]
        low = [4.0 * e[2] for e in ends]
        # ...and off the clip tab's post and the stick cradle's supports.
        tx, ty = CLIP_TAB_XY
        clear += [max(0.0, 0.035 - math.hypot(e[0] - tx, e[1] - ty)) * 200.0 * (e[2] < 0.10) for e in ends]
        # ...and out of the space above the clip, where the gripper comes down to squeeze it (all but the last link,
        # which rises straight out of the lug).
        clear += [max(0.0, 0.05 - math.hypot(e[0] - cp[0], e[1] - cp[1])) * 200.0 * (e[2] > 0.10) for e in ends[:-1]]
        # ...on the desk in front of the robot, clear of the arm bases.
        clear += [max(0.0, 0.03 - e[0]) * 200.0 for e in ends]
        clear += [max(0.0, 0.16 - math.hypot(e[0], abs(e[1]) - 0.305)) * 200.0 for e in ends]
        sx, sy = STICK_CRADLE_XY
        clear += [max(0.0, 0.04 - math.hypot(e[0] - sx, e[1] - (sy - off))) * 200.0 * (e[2] < 0.06)
                  for e in ends for off in (-0.05, 0.14)]
        return np.concatenate([err_p, err_r, 0.3 * q, lows, clear, low])

    lim = math.radians(LINK_LIMIT_DEG) * 0.9
    best = None
    for trial in range(8):
        q0 = np.random.default_rng(trial).uniform(-0.6, 0.6, LINK_N)
        sol = least_squares(residual, q0, bounds=(-lim, lim))
        if best is None or sol.cost < best.cost:
            best = sol
    links, (pos, rot) = lead_fk(sp, sr, best.x)
    print(f"lead solve: clip pos error {np.linalg.norm(pos - cp) * 1000:.2f} mm, rot error "
          f"{np.degrees(np.linalg.norm(Rotation.from_matrix(cr.T @ rot).as_rotvec())):.2f} deg, "
          f"lowest link z {min(p[2] for p, _ in links):.3f}, max x {max(p[0] for p, _ in links):.3f}")
    return best.x


# USD helpers (copied from make_c20_estop_props.py; not imported so that task can change freely) -------------
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


def _quad_x(stage, path, x, y_range, z_range, material, facing=-1, u_range=(0.0, 1.0)):
    """A textured quad in a plane of constant x, facing -x (facing=-1) or +x, showing u_range of the texture."""
    y0, y1 = y_range
    z0, z1 = z_range
    if facing < 0:
        pts = [(x, y1, z0), (x, y0, z0), (x, y0, z1), (x, y1, z1)]
    else:
        pts = [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateSubdivisionSchemeAttr("none")
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    u0, u1 = u_range
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(u0, 0), Gf.Vec2f(u1, 0), Gf.Vec2f(u1, 1), Gf.Vec2f(u0, 1)]))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


LABEL_RGB = {"blue": ((25, 70, 200), (250, 250, 250)), "orange": ((245, 120, 15), (10, 10, 10))}
"""Label background and print colours."""


def _quad_z(stage, path, z, x_range, y_range, material, u_range=(0.0, 1.0)):
    """A textured quad in a plane of constant z, facing up; the texture's top edge toward +x (read from the robot)."""
    x0, x1 = x_range
    y0, y1 = y_range
    pts = [(x0, y1, z), (x0, y0, z), (x1, y0, z), (x1, y1, z)]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateSubdivisionSchemeAttr("none")
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    u0, u1 = u_range
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(u0, 0), Gf.Vec2f(u1, 0), Gf.Vec2f(u1, 1), Gf.Vec2f(u0, 1)]))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _labels_png(out_png: Path, labels, colors) -> None:
    """One texture with every label side by side (separate label textures were swapped by the renderer)."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = 480, 280
    img = Image.new("RGB", (w * len(labels), h), (246, 246, 240))
    d = ImageDraw.Draw(img)
    bold = _font()
    for i, (label, color) in enumerate(zip(labels, colors)):
        x0 = i * w
        bg, fg = LABEL_RGB[color]
        d.rectangle([x0, 0, x0 + w - 1, h - 1], fill=bg)
        d.rectangle([x0 + 4, 4, x0 + w - 5, h - 5], outline=fg, width=6)
        d.text((x0 + w / 2, 112), label, font=ImageFont.truetype(bold, 150), fill=fg, anchor="mm")
        d.text((x0 + w / 2, 232), "1.0 MFD  4 KVDC", font=ImageFont.truetype(bold, 40), fill=fg, anchor="mm")
    img.save(out_png)


def _font() -> str:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"):
        if Path(path).exists() or "/" not in path:
            return path
    return path


def _wait_png(out_png: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (320, 120), (20, 20, 20))
    d = ImageDraw.Draw(img)
    d.text((160, 60), "WAIT", font=ImageFont.truetype(_font(), 80),
           fill=(250, 200, 30), anchor="mm")
    img.save(out_png)


def _cyl(stage, path, center, radius, height, material, axis="Z", collide=True):
    cyl = kit.cylinder(stage, path, center, radius, height, material, axis=axis)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    return cyl


def _capsule(stage, path, p0, p1, radius, material):
    """A collision capsule from p0 to p1 (axis-aligned segments only)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    axis = "XYZ"[int(np.argmax(np.abs(d)))]
    cap = UsdGeom.Capsule.Define(stage, path)
    cap.CreateAxisAttr(axis)
    cap.CreateRadiusAttr(radius)
    cap.CreateHeightAttr(float(np.linalg.norm(d)))
    UsdGeom.Xformable(cap).AddTranslateOp().Set(Gf.Vec3d(*((p0 + p1) / 2)))
    UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    return cap


def _body(stage, path, pos, quat, mass):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    x, y, z, w = quat
    xform.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(mass)
    return xform


def _joint(stage, path, body0, body1, axis, pos0, rot0, pos1, lower_deg, upper_deg):
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(str(body0))])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(str(body1))])
    joint.CreateAxisAttr(axis)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    x, y, z, w = rot0
    joint.CreateLocalRot0Attr(Gf.Quatf(float(w), float(x), float(y), float(z)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*pos1))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLowerLimitAttr(lower_deg)
    joint.CreateUpperLimitAttr(upper_deg)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    return joint


def _materials(stage, root):
    colors = {
        "case": (0.66, 0.71, 0.74), "case_top": (0.30, 0.33, 0.35), "porcelain": (0.93, 0.90, 0.78),
        "steel": (0.72, 0.74, 0.77), "copper": (0.80, 0.45, 0.20), "insulator": (0.12, 0.12, 0.13),
        "yellow": (0.98, 0.80, 0.08), "black": (0.05, 0.05, 0.05), "alu": (0.78, 0.79, 0.80),
        "cable": (0.96, 0.82, 0.10), "red": (0.80, 0.06, 0.05), "stand": (0.40, 0.40, 0.42),
        "lamp_off": (0.30, 0.20, 0.08), "green": (0.10, 0.55, 0.20), "box_grey": (0.62, 0.63, 0.64),
    }
    out = {}
    for name, c in colors.items():
        m = UsdShade.Material.Define(stage, f"{root}/Looks/{name}")
        sh = UsdShade.Shader.Define(stage, f"{root}/Looks/{name}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*c))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
        if name in ("copper", "steel", "alu"):
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.6)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        out[name] = m
    lamp = UsdShade.Material.Define(stage, f"{root}/Looks/lamp_on")
    sh = UsdShade.Shader.Define(stage, f"{root}/Looks/lamp_on/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.75, 0.1))
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.7, 0.05))
    lamp.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    out["lamp_on"] = lamp
    return out


# Props ----------------------------------------------------------------------------------------------------
LAMP_POS = (0.50, 0.14, 0.10)
"""Front face of the lamp's control station (x), its centre line (y), the top of its post (z)."""


def make_bench(out_dir: Path) -> None:
    """Static scenery in world coordinates: two capacitors, the earth bar, the clip tab, the stick cradle,
    the WAIT lamp."""
    root = "/bench"
    stage, _ = kit.new_stage(out_dir / "bench.usda", root)
    m = _materials(stage, root)
    _labels_png(out_dir / "labels.png", CAP_LABELS, CAP_COLORS)
    lab = _tex_material(stage, root, "labels", "labels.png")
    n_lab = len(CAP_LABELS)
    for i, (x, label) in enumerate(zip(CAP_X, CAP_LABELS)):
        c = f"{root}/cap_{i}"
        UsdGeom.Xform.Define(stage, c)
        kit.box(stage, f"{c}/case", (x, CAP_Y, CASE[2] / 2), CASE, m["case"])
        kit.box(stage, f"{c}/top", (x, CAP_Y, CASE[2] + 0.0005), (CASE[0] - 0.008, CASE[1] - 0.008, 0.001),
                m["case_top"], collide=False)
        for face in (-1, 1):
            _quad_x(stage, f"{c}/label_{'front' if face < 0 else 'back'}", x + face * (CASE[0] / 2 + 0.0008),
                    (CAP_Y - 0.032, CAP_Y + 0.032), (0.045, 0.083), lab, facing=face, u_range=(i / n_lab, (i + 1) / n_lab))
        # Lid label in front of the bushings (seen by the overhead camera), print toward the robot.
        _quad_z(stage, f"{c}/label_lid", CASE[2] + 0.0012, (x - CASE[0] / 2 + 0.006, x - 0.012),
                (CAP_Y - 0.032, CAP_Y + 0.032), lab, u_range=(i / n_lab, (i + 1) / n_lab))
        for k in (0, 1):
            tx, ty, _ = terminal_xyz(i, k)
            z = CASE[2]
            _cyl(stage, f"{c}/bushing{k}_lo", (tx, ty, z + BUSH_H[0] / 2), BUSH_R[0], BUSH_H[0], m["porcelain"])
            _cyl(stage, f"{c}/bushing{k}_hi", (tx, ty, z + BUSH_H[0] + BUSH_H[1] / 2), BUSH_R[1], BUSH_H[1],
                 m["porcelain"])
            for r in range(3):
                _cyl(stage, f"{c}/rib{k}_{r}", (tx, ty, z + 0.006 + 0.009 * r), BUSH_R[0] + 0.0012 - 0.0008 * r,
                     0.003, m["porcelain"], collide=False)
            _cyl(stage, f"{c}/nut{k}", (tx, ty, NUT_BOTTOM + NUT_H / 2), NUT_R, NUT_H, m["steel"])
            _cyl(stage, f"{c}/stud{k}", (tx, ty, NUT_BOTTOM + NUT_H + STUD_H / 2), STUD_R, STUD_H, m["steel"])
            side = 1 if k == 0 else -1
            kit.box(stage, f"{c}/lug{k}", (tx, ty + side * 0.011, NUT_BOTTOM + 0.002), (0.006, 0.012, 0.001),
                    m["steel"], collide=False)
    # Earth bar on two insulators, with the earth symbol's green/yellow band.
    bx = (BAR_X[0] + BAR_X[1]) / 2
    kit.box(stage, f"{root}/earth_bar", (bx, EARTH_BAR_Y, (BAR_Z[0] + BAR_Z[1]) / 2),
            (BAR_X[1] - BAR_X[0], BAR_T, BAR_Z[1] - BAR_Z[0]), m["copper"])
    for j, ix in enumerate(INSULATOR_X):
        _cyl(stage, f"{root}/insulator_{j}", (ix, EARTH_BAR_Y, BAR_Z[0] / 2), 0.012, BAR_Z[0], m["insulator"])
    for j in range(6):
        kit.box(stage, f"{root}/earth_band_{j}", (BAR_X[0] + 0.006 + 0.008 * j, EARTH_BAR_Y - BAR_T / 2 - 0.0004,
                                                   (BAR_Z[0] + BAR_Z[1]) / 2), (0.008, 0.0008, 0.03),
                m["green"] if j % 2 == 0 else m["yellow"], collide=False)
    # Clip parking tab (insulating) on a post.
    tx, ty = CLIP_TAB_XY
    kit.box(stage, f"{root}/tab", (tx, ty, (0.05 + TAB_TOP) / 2), (TAB_X_LEN, BAR_T, TAB_TOP - 0.05), m["insulator"])
    kit.box(stage, f"{root}/tab_post", (tx, ty, 0.025), (0.03, 0.03, 0.05), m["stand"])
    # Stick cradle: two supports under the tube.
    sx, sy = STICK_CRADLE_XY
    for j, off in enumerate((-0.05, 0.14)):
        kit.box(stage, f"{root}/cradle_{j}", (sx, sy - off, 0.0205), (0.05, 0.02, 0.041), m["stand"])
    # WAIT lamp: an XB7 amber pilot light in a grey one-hole control station on a post, a WAIT plate above it.
    lx, ly, lz = LAMP_POS
    _cyl(stage, f"{root}/lamp_post", (lx + 0.01, ly, lz / 2), 0.008, lz, m["black"])
    kit.box(stage, f"{root}/lamp_box", (lx + 0.03, ly, lz + 0.034), (0.07, 0.068, 0.068), m["box_grey"])
    _cyl(stage, f"{root}/lamp_bezel", (lx - 0.006, ly, lz + 0.034), 0.0145, 0.006, m["black"], axis="X",
         collide=False)
    _wait_png(out_dir / "wait.png")
    wait = _tex_material(stage, root, "wait_plate", "wait.png")
    _quad_x(stage, f"{root}/lamp_plate", lx - 0.0055, (ly - 0.032, ly + 0.032), (lz + 0.071, lz + 0.095), wait)
    kit.box(stage, f"{root}/lamp_plate_back", (lx - 0.002, ly, lz + 0.083), (0.006, 0.068, 0.026), m["black"],
            collide=False)
    for name, mat in (("lamp_on", m["lamp_on"]), ("lamp_off", m["lamp_off"])):
        _cyl(stage, f"{root}/{name}", (lx - 0.0105, ly, lz + 0.034), 0.0125, 0.004, mat, axis="X", collide=False)
    kit.hide(stage, f"{root}/lamp_off")
    stage.GetRootLayer().Save()
    print(f"wrote {out_dir / 'bench.usda'}")


def _clip_half(stage, path, s, m):
    """One half of the clip; s = -1 for the fixed half (A, on -y), +1 for the moving half (B)."""
    z = 2 * CLIP_HALF_Z
    kit.box(stage, f"{path}/pad", ((PAD_X[0] + PAD_X[1]) / 2, s * 0.0015, 0), (PAD_X[1] - PAD_X[0], 0.003, z),
            m["copper"])
    for t in range(4):  # teeth (visual)
        kit.box(stage, f"{path}/tooth{t}", (PAD_X[0] + 0.004 + 0.006 * t, s * 0.0002, 0), (0.002, 0.0006, z),
                m["copper"], collide=False)
    kit.box(stage, f"{path}/step", (PAD_X[0] - 0.0015, s * (THROAT_Y + 0.003) / 2, 0), (0.003, THROAT_Y + 0.003, z),
            m["copper"])
    kit.box(stage, f"{path}/throat", (0.015, s * (THROAT_Y + 0.0015), 0), (0.029, 0.003, z), m["copper"])
    kit.box(stage, f"{path}/web", (-0.002, s * (THROAT_Y + HANDLE_Y[1]) / 2, 0), (0.014, HANDLE_Y[1] - THROAT_Y, z),
            m["copper"])
    kit.box(stage, f"{path}/handle", ((HANDLE_X[0] + HANDLE_X[1]) / 2, s * (HANDLE_Y[0] + HANDLE_Y[1]) / 2, 0),
            (HANDLE_X[1] - HANDLE_X[0], HANDLE_Y[1] - HANDLE_Y[0], z), m["copper"])


def make_stick(out_dir: Path) -> dict:
    """The discharge stick, its lead and the ground clip as one floating articulation (root body: the stick).
    Bodies are placed at the start pose (stick in its cradle, clip on its tab), relative to the stick."""
    root = "/stick"
    stage, _ = kit.new_stage(out_dir / "stick.usda", root)
    m = _materials(stage, root)
    stick = _body(stage, f"{root}/stick", (0, 0, 0), (0, 0, 0, 1), STICK_MASS)
    sp = stick.GetPrim()
    kit.articulation_root(stage, sp)
    sp.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    sp.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    s = f"{root}/stick"
    # The tube, round (visual and collider) from the end cap to the head, with the black handgrip sleeve up to the
    # hand guard, where the gripper holds it. At the butt, a square anti-roll end cap (32 mm across flats, 55 mm long):
    # with a round butt the hanging stick rolled about its axis on the table under the after-release pull and its
    # turned-down end came off the far stud (tested).
    kit.box(stage, f"{s}/end_cap", ((TUBE_X[0] + CAP_END_X) / 2, 0, 0), (CAP_END_X - TUBE_X[0], 2 * TUBE_R, 2 * TUBE_R),
            m["black"])
    _cyl(stage, f"{s}/tube", ((CAP_END_X + TUBE_X[1]) / 2, 0, 0), TUBE_R, TUBE_X[1] - CAP_END_X, m["yellow"], axis="X")
    _cyl(stage, f"{s}/handgrip", ((CAP_END_X + GRIP_END_X) / 2, 0, 0), TUBE_R + 0.0008, GRIP_END_X - CAP_END_X,
         m["black"], axis="X", collide=False)
    _cyl(stage, f"{s}/handguard", (GUARD_X, 0, 0), TUBE_R + 0.006, 0.006, m["red"], axis="X", collide=False)
    _cyl(stage, f"{s}/head", ((HEAD_X[0] + HEAD_X[1]) / 2, 0, 0), HEAD_R, HEAD_X[1] - HEAD_X[0], m["alu"], axis="X")
    _capsule(stage, f"{s}/probe", (HEAD_X[1] - 0.005, 0, 0), (PROBE_END_X, 0, 0), ROD_R, m["copper"])
    _capsule(stage, f"{s}/curl", (PROBE_END_X, 0, 0), (PROBE_END_X, 0, -CURL_DROP), ROD_R, m["copper"])
    # Two short side rails under the rod: they straddle the near stud, so the hanging stick cannot roll off the
    # stud tops sideways (it did, and came to rest beside the studs with the end on the case lid).
    for k, side in enumerate((-1, 1)):
        y = side * RAIL_Y
        _capsule(stage, f"{s}/rail_{k}", (RAIL_X[0], y, -RAIL_DROP), (RAIL_X[1], y, -RAIL_DROP), RAIL_R, m["copper"])
        for j, x in enumerate(RAIL_X):
            _capsule(stage, f"{s}/rail_{k}_post{j}", (x, y, 0.0), (x, y, -RAIL_DROP), RAIL_R, m["copper"])
    kit.friction(stage, root, [f"{s}/tube", f"{s}/end_cap"], static=1.2, dynamic=1.0, name="grip")

    stick_pose = stick_rest_pose()
    clip_pose = clip_pose_on(CLIP_TAB_XY[1], TAB_TOP, CLIP_TAB_XY[0])
    q = solve_lead(stick_pose, clip_pose)
    sp_w, sr_w = np.asarray(stick_pose[:3]), _qmat(stick_pose[3:])
    links, (cpos, crot) = lead_fk(sp_w, sr_w, q)

    def rel(pos, rot):  # world -> stick frame
        return sr_w.T @ (pos - sp_w), _mat_q(sr_w.T @ rot)

    prev = stick.GetPath()
    for k, (p, r) in enumerate(links):
        lp, lq = rel(p, r)
        body = _body(stage, f"{root}/lead_{k:02d}", lp, lq, LINK_MASS)
        _capsule(stage, f"{root}/lead_{k:02d}/cable", (0.0, 0, 0), (LINK_L, 0, 0), LINK_R, m["cable"])
        if k == 0:
            _joint(stage, f"{root}/joints/lead_j{k:02d}", prev, body.GetPath(), link_axis(k), LEAD_ATTACH,
                   _mat_q(ATTACH_ROT), (0, 0, 0), -LINK_LIMIT_DEG, LINK_LIMIT_DEG)
        else:
            _joint(stage, f"{root}/joints/lead_j{k:02d}", prev, body.GetPath(), link_axis(k), (LINK_L, 0, 0),
                   (0, 0, 0, 1), (0, 0, 0), -LINK_LIMIT_DEG, LINK_LIMIT_DEG)
        prev = body.GetPath()
    # A jacketed cable slides on a bench: low friction on the lead links.
    kit.friction(stage, root, [f"{root}/lead_{k:02d}/cable" for k in range(LINK_N)], static=0.3, dynamic=0.25,
                 name="jacket", combine="min")
    lp, lq = rel(cpos, crot)
    clip = _body(stage, f"{root}/clip", lp, lq, CLIP_MASS[0])
    fixed = UsdPhysics.FixedJoint.Define(stage, f"{root}/joints/lug_fixed")
    fixed.CreateBody0Rel().SetTargets([prev])
    fixed.CreateBody1Rel().SetTargets([clip.GetPath()])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(LINK_L, 0, 0))
    x, y, z, w = _mat_q(CLIP_LUG_ROT)
    fixed.CreateLocalRot0Attr(Gf.Quatf(float(w), float(x), float(y), float(z)))
    fixed.CreateLocalPos1Attr(Gf.Vec3f(*LUG))
    fixed.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    jaw_q = clamp_angle(BAR_T)
    jq = _mat_q(crot @ _rot("Z", jaw_q))
    _, jq = rel(cpos, crot @ _rot("Z", jaw_q))
    jaw = _body(stage, f"{root}/clip_jaw", lp, jq, CLIP_MASS[1])
    _clip_half(stage, f"{root}/clip", -1, m)
    _clip_half(stage, f"{root}/clip_jaw", 1, m)
    kit.box(stage, f"{root}/clip/lug", (LUG[0], LUG[1], (LUG[2] - CLIP_HALF_Z) / 2), (0.008, 0.006, -LUG[2] - CLIP_HALF_Z),
            m["copper"], collide=False)
    _cyl(stage, f"{root}/clip/rivet", (0, 0, 0), 0.003, 2 * CLIP_HALF_Z + 0.004, m["steel"], collide=False)
    _joint(stage, f"{root}/joints/jaw_joint", clip.GetPath(), jaw.GetPath(), "Z", (0, 0, 0), (0, 0, 0, 1), (0, 0, 0),
           0.0, JAW_MAX_DEG)
    kit.friction(stage, root, [f"{root}/clip/pad", f"{root}/clip_jaw/pad"], static=1.0, dynamic=0.9, name="teeth")
    kit.friction(stage, root, [f"{root}/clip/handle", f"{root}/clip_jaw/handle"], static=1.2, dynamic=1.0,
                 name="handles")
    stage.GetRootLayer().Save()
    print(f"wrote {out_dir / 'stick.usda'}")
    init = {f"lead_j{k:02d}": float(v) for k, v in enumerate(q)}
    init["jaw_joint"] = float(jaw_q)
    return init


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets"
    out.mkdir(parents=True, exist_ok=True)
    make_bench(out)
    init = make_stick(out)
    (out / "init_joints.json").write_text(json.dumps(init, indent=1))
    print(f"wrote {out / 'init_joints.json'}")

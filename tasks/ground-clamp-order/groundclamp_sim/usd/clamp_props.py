"""Generate the props of the ground clamp task (a grounding set on a bench bus mock) as USD.

Real parts (vendor pages in README.md):

- Grounding clamp: CHANCE PSC6003412 class C-type clamp (aluminium frame, serrated jaw, bronze eye screw, ferrule
  boss behind the jaw; 0.162-2.0 in jaw range, 1.7 lb = 0.7 kg, 250 in-lb = 28 N*m eye-screw installation
  torque; Hubbell spec sheet and sales drawing SA6003412). Overall dimensions are published only for the PST6003426
  (12.41 x 7 x 2.48 in); jaw depth, screw diameter and thread are not published. Inferred here: throat 22 mm from
  the spine to the screw axis, 34 mm throat height (a 1/4 in bar needs no more), frame 24 mm thick, 5/8 in screw,
  Acme thread of 1/8 in pitch (Salisbury: "Acme thread"; pitch inferred). Used upside down on the bench: hook jaw
  under the bar, eye screw on top pointing down, so the stick comes from above (the fingers-down YAM).
- Clamp stick: CHANCE Grip-All C4030291 class (J-hook head that pulls the eye into the head; 1-1/4 in pole; 4 ft
  class, 1.45 m overall, 2.6 kg). Shortened here to the head plus a 55 mm square handle (the YAM's fingers-down
  band ends at a grasp height of about 0.22 m). The hook has caught the eye: the stick is fixed to the eye screw
  (release not modelled).
- Ground cable: 2/0 clear-jacket copper (0.73 in OD; Hubbell S6450) on the real set, yellow jacket in the Hubbell
  video. Here 16 mm across, 0.9 m (18 links of 50 mm on alternating hinges), 0.22 kg (lighter than the real ~0.6
  kg/m; inferred).
- Bars: copper 1/4 x 2 in (Burndy BBB14212P size), cut to bench-mock lengths; insulating back board (our layout).

Frames. Clamp frame (bodies `g_frame`, `l_frame`): origin on the screw axis at the jaw's top face, +z up along the
screw, +x toward the throat opening (the bar), the spine on -x. Pad carrier (`*_pad`): prismatic along the frame's
z, origin at the pad's bottom face (joint position p = pad height above the jaw). Screw (`*_screw`, with the stick):
revolute about z at the pad. Cable links: origin at the link's start, along +x. The articulation root is g_frame.

Usage: python -m groundclamp_sim.usd.clamp_props [out dir]   (default: groundclamp_sim/assets)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from groundclamp_sim.spec import GROUND_LEN, GROUND_Y, PARK_LEN, PARK_Y, PHASE_LEN, PHASE_Y, PHASES, TAGGED
from groundclamp_sim.usd import usd_kit as kit

# Rotation maths and USD helpers (copied from our capacitor-discharge prop generator) ------------------------------
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



# Bench bus mock ------------------------------------------------------------------------------------------------
X_EDGE = 0.24
"""World x of every bar's front edge (toward the robot)."""
BAR_W = 0.0508
BAR_T = 0.00635
BAR_TOP = 0.060
BOARD_X = (X_EDGE + BAR_W, X_EDGE + BAR_W + 0.018)
BOARD_Y = (-0.34, 0.43)
BOARD_Z = (0.0, 0.12)

# Clamp (CHANCE PSC6003412 class, inferred proportions) -----------------------------------------------------------
HALF_W = 0.012
"""Frame thickness 24 mm (y)."""
SPINE_X = (-0.034, -0.022)
THROAT_H = 0.034
ARM_T = 0.012
UPPER_T = 0.016
JAW_TIP_X = 0.016
BOSS_TIP_X = 0.014
FERRULE = (-0.028, 0.018)
"""Ferrule boss (x, length below the jaw's underside); the cable leaves straight down, parallel to the screw."""
LUG = (FERRULE[0], 0.0, -ARM_T - FERRULE[1])
PAD = (0.024, 0.022, 0.005)
SCREW_R = 0.0079
SCREW_L = 0.060
EYE_R = 0.012
HEAD_L = 0.030
POLE_L = 0.008
HANDLE = (0.026, 0.055)
"""Square handle side, length (collider; the gripper holds it)."""
GAP = 0.003
"""Total slack between the jaws and the bar at the start (pad 3 mm above the bar when the jaw is on it)."""
P0 = BAR_T + GAP
"""Pad height above the jaw at the start (prismatic joint position)."""
P_LIMITS = (0.0, THROAT_H - PAD[2] - 0.001)
FRAME_MASS, PAD_MASS, SCREW_MASS = 0.62, 0.03, 0.12

PITCH = 0.003175
"""1/8 in per turn (Acme, inferred)."""
SPINE_CLEAR = 0.0005
"""Spine inner face this far in front of the bar edge when the clamp is home."""

# Cable --------------------------------------------------------------------------------------------------------
LINK_N = 18
LINK_L = 0.05
LINK_R = 0.008
LINK_MASS = 0.012
LINK_LIMIT_DEG = 100.0


def screw_z(p: float) -> float:
    """Screw-body origin above the frame origin for pad height p (the pad's bottom face)."""
    return p


def handle_center_local() -> float:
    """Handle centre above the screw-body origin."""
    return PAD[2] + SCREW_L + 2 * EYE_R - 0.006 + HEAD_L + POLE_L + HANDLE[1] / 2


def frame_home(bar_y: float) -> tuple:
    """Frame position (world) for a clamp hung on the bar or stub centred at bar_y (pad on the bar's top)."""
    return (X_EDGE + (-SPINE_X[1]) - SPINE_CLEAR, bar_y, BAR_TOP - P0)


def bars() -> list[tuple]:
    """(name, y centre, length, kind) of every horizontal bar or stub: kind 'ground', 'phase' or 'park'."""
    out = [("GND", GROUND_Y, GROUND_LEN, "ground")]
    out += [(p, y, PHASE_LEN, "phase") for p, y in zip(PHASES, PHASE_Y)]
    out += [("PARK_L", PARK_Y[1], PARK_LEN, "park")]
    if GROUND_ON_PARK:
        out += [("PARK_G", PARK_Y[0], PARK_LEN, "park")]
    return out


GROUND_ON_PARK = True
"""The ground clamp starts on its own parking plate right of the ground bar and the robot picks it up (False: it starts
hung, loose, on the ground bar, as in our first prototype, where the pick-up was unreliable)."""
LABEL_Z = (0.088, 0.112)
LEDGE_X = (BOARD_X[1] - 0.018, BOARD_X[1] + 0.072)
LEDGE_T = 0.006
"""A label ledge on top of the back board (9 cm deep, reaching back away from the robot): the top camera looks almost
straight down, so the bar letters and the TESTED tag lie flat on it, behind their bars."""
LEDGE_LETTER_X = (LEDGE_X[0] + 0.004, LEDGE_X[0] + 0.034)
TAG_X = (LEDGE_X[0] + 0.040, LEDGE_X[0] + 0.084)
TAG_HALF_W = 0.036
"""The TESTED tag: a yellow card 7.2 x 4.4 cm lying on the ledge behind the tagged bar (behind its letter)."""

PARK_T = 0.003
"""The parking stubs are thin insulating plates, so a clamp hung on one has 6 mm under its jaw: lifting it off does
not clamp the stub (with a bar-thick stub the left arm's pick-up clamped it)."""


def bar_box(i: int) -> tuple:
    _, y, length, kind = bars()[i]
    t = PARK_T if kind == "park" else BAR_T
    return (X_EDGE, y - length / 2, BAR_TOP - t), (X_EDGE + BAR_W, y + length / 2, BAR_TOP)


def _materials(stage, root):
    colors = {
        "copper": (0.80, 0.45, 0.20), "board": (0.20, 0.21, 0.22), "stub": (0.10, 0.10, 0.11),
        "alu": (0.70, 0.71, 0.70), "bronze": (0.72, 0.52, 0.30), "black": (0.04, 0.04, 0.04),
        "red": (0.80, 0.10, 0.06), "cable": (0.96, 0.82, 0.10), "ferrule": (0.82, 0.83, 0.85),
        "green": (0.10, 0.55, 0.20), "yellow": (0.98, 0.80, 0.08), "dark": (0.15, 0.15, 0.15),
        "ledge": (0.42, 0.44, 0.46),
    }
    out = {}
    for name, c in colors.items():
        m = UsdShade.Material.Define(stage, f"{root}/Looks/{name}")
        sh = UsdShade.Shader.Define(stage, f"{root}/Looks/{name}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*c))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
        if name in ("copper", "alu", "bronze", "ferrule"):
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.6)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        out[name] = m
    return out


def _quad_z(stage, path, z, x_range, y_range, material, u_range=(0.0, 1.0)):
    """A textured quad lying flat at height z, facing up, its text upright for a viewer behind the robot looking
    forward (text up = +x, text right = -y)."""
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


def _labels_png(out_png: Path) -> list[str]:
    """One texture strip: bar labels A, B, C, GND, PARK, PARK and the TESTED tag."""
    from PIL import Image, ImageDraw, ImageFont

    names = list(PHASES) + ["GND", "PARK", "TESTED"]
    w, h = 300, 160
    img = Image.new("RGB", (w * len(names), h), (240, 240, 235))
    d = ImageDraw.Draw(img)
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    for i, name in enumerate(names):
        x0 = i * w
        if name == "TESTED":
            d.rectangle([x0, 0, x0 + w - 1, h - 1], fill=(250, 205, 20))
            d.ellipse([x0 + w / 2 - 9, 6, x0 + w / 2 + 9, 24], fill=(40, 40, 40))
            d.text((x0 + w / 2, 92), "TESTED", font=ImageFont.truetype(bold, 74), fill=(10, 10, 10), anchor="mm")
        else:
            d.rectangle([x0 + 4, 4, x0 + w - 5, h - 5], outline=(20, 20, 20), width=6)
            size = 100 if len(name) == 1 else 70
            d.text((x0 + w / 2, h / 2), name, font=ImageFont.truetype(bold, size), fill=(10, 10, 10), anchor="mm")
    img.save(out_png)
    return names


def make_bench(out_dir: Path) -> None:
    """Static scenery in world coordinates: back board, bars, parking stubs, labels and the TESTED tag."""
    root = "/bench"
    stage, _ = kit.new_stage(out_dir / "bench.usda", root)
    m = _materials(stage, root)
    names = _labels_png(out_dir / "labels.png")
    lab = _tex_material(stage, root, "labels", "labels.png")
    n_lab = len(names)

    def label(path, y, z_range, name, half_w):
        i = names.index(name)
        _quad_x(stage, path, BOARD_X[0] - 0.0008, (y - half_w, y + half_w), z_range, lab, facing=-1,
                u_range=(i / n_lab, (i + 1) / n_lab))

    def flat(path, x_range, y, name, half_w):
        i = names.index(name)
        _quad_z(stage, path, BOARD_Z[1] + LEDGE_T + 0.0006, x_range, (y - half_w, y + half_w), lab,
                u_range=(i / n_lab, (i + 1) / n_lab))

    kit.box(stage, f"{root}/ledge", ((LEDGE_X[0] + LEDGE_X[1]) / 2, (BOARD_Y[0] + BOARD_Y[1]) / 2,
                                     BOARD_Z[1] + LEDGE_T / 2),
            (LEDGE_X[1] - LEDGE_X[0], BOARD_Y[1] - BOARD_Y[0], LEDGE_T), m["ledge"])
    kit.box(stage, f"{root}/board", ((BOARD_X[0] + BOARD_X[1]) / 2, (BOARD_Y[0] + BOARD_Y[1]) / 2,
                                     (BOARD_Z[0] + BOARD_Z[1]) / 2),
            (BOARD_X[1] - BOARD_X[0], BOARD_Y[1] - BOARD_Y[0], BOARD_Z[1] - BOARD_Z[0]), m["board"])
    for i, (name, y, length, kind) in enumerate(bars()):
        lo, hi = bar_box(i)
        mat = m["stub"] if kind == "park" else m["copper"]
        kit.box(stage, f"{root}/bar_{i}", tuple((a + b) / 2 for a, b in zip(lo, hi)),
                tuple(b - a for a, b in zip(lo, hi)), mat)
        if kind == "ground":
            for j in range(8):  # green/yellow band on the top face, at the back (visual)
                kit.box(stage, f"{root}/gnd_band_{j}", (X_EDGE + BAR_W - 0.008, y - 0.028 + 0.008 * j, BAR_TOP + 0.0003),
                        (0.012, 0.008, 0.0006), m["green"] if j % 2 == 0 else m["yellow"], collide=False)
        # Bolts into the board (visual).
        for k in (-1, 1):
            kit.cylinder(stage, f"{root}/bolt_{i}_{k + 1}", (X_EDGE + BAR_W - 0.012, y + k * length * 0.3, BAR_TOP + 0.002),
                         0.005, 0.004, m["alu"])
        lab_name = "PARK" if kind == "park" else name
        label(f"{root}/label_{i}", y, LABEL_Z, lab_name, 0.022)
        if kind != "park":
            flat(f"{root}/ledge_label_{i}", LEDGE_LETTER_X, y, lab_name, 0.022 if kind == "phase" else 0.030)
    # The TESTED tag: a yellow card lying on the ledge behind the tagged bar.
    flat(f"{root}/tag", TAG_X, PHASE_Y[TAGGED], "TESTED", TAG_HALF_W)
    stage.GetRootLayer().Save()
    print(f"wrote {out_dir / 'bench.usda'}")


def _clamp_visuals(stage, frame, pad, screw, m):
    """Collision boxes and visuals of one clamp: frame (C body), pad carrier, screw with the stick."""
    w = 2 * HALF_W
    # Frame: spine, hook jaw (under the bar), upper arm with the screw boss, ferrule boss.
    kit.box(stage, f"{frame}/spine", ((SPINE_X[0] + SPINE_X[1]) / 2, 0, (-ARM_T + THROAT_H + UPPER_T) / 2),
            (SPINE_X[1] - SPINE_X[0], w, THROAT_H + UPPER_T + ARM_T), m["alu"])
    kit.box(stage, f"{frame}/jaw", ((SPINE_X[1] + JAW_TIP_X) / 2, 0, -ARM_T / 2), (JAW_TIP_X - SPINE_X[1], w, ARM_T),
            m["alu"])
    kit.box(stage, f"{frame}/jaw_lip", (JAW_TIP_X - 0.003, 0, 0.001), (0.006, w, 0.002), m["alu"], collide=False)
    for t in range(6):  # serrations (visual)
        kit.box(stage, f"{frame}/tooth{t}", (SPINE_X[1] + 0.004 + 0.006 * t, 0, 0.0003), (0.002, w, 0.0006),
                m["alu"], collide=False)
    kit.box(stage, f"{frame}/upper", ((SPINE_X[1] + BOSS_TIP_X) / 2, 0, THROAT_H + UPPER_T / 2),
            (BOSS_TIP_X - SPINE_X[1], w, UPPER_T), m["alu"])
    kit.cylinder(stage, f"{frame}/boss", (0, 0, THROAT_H + UPPER_T / 2), 0.014, UPPER_T, m["alu"])
    kit.cylinder(stage, f"{frame}/ferrule_boss", (FERRULE[0], 0, -ARM_T - FERRULE[1] / 2 + 0.004), 0.009,
                 FERRULE[1] - 0.008, m["alu"])
    kit.cylinder(stage, f"{frame}/ferrule", (FERRULE[0], 0, -ARM_T - FERRULE[1] + 0.003), 0.0095, 0.008,
                 m["ferrule"])
    # Pad (swivel pad on the screw's end).
    kit.box(stage, f"{pad}/pad", (0, 0, PAD[2] / 2), PAD, m["alu"])
    # Screw, eye, and the stick (head with its J-hook round the eye, short pole, square handle).
    z = PAD[2]
    kit.cylinder(stage, f"{screw}/shaft", (0, 0, z + SCREW_L / 2), SCREW_R, SCREW_L, m["bronze"])
    for k in range(5):  # thread crests (visual)
        kit.cylinder(stage, f"{screw}/crest{k}", (0, 0, z + 0.006 + 0.006 * k), SCREW_R + 0.0008, 0.0015, m["bronze"])
    ez = z + SCREW_L + EYE_R
    kit.cylinder(stage, f"{screw}/eye", (0, 0, ez), EYE_R, 0.006, m["bronze"], axis="Y")
    kit.cylinder(stage, f"{screw}/eye_hole", (0, 0, ez), EYE_R * 0.55, 0.0065, m["dark"], axis="Y")
    hz = ez + EYE_R - 0.006
    kit.cylinder(stage, f"{screw}/head", (0, 0, hz + HEAD_L / 2), 0.017, HEAD_L, m["black"])
    kit.box(stage, f"{screw}/hook", (0.0, 0.0, ez + 0.002), (0.004, 0.012, EYE_R * 1.1), m["dark"], collide=False)
    pz = hz + HEAD_L
    kit.cylinder(stage, f"{screw}/pole", (0, 0, pz + POLE_L / 2), 0.016, POLE_L, m["red"])
    cz = pz + POLE_L + HANDLE[1] / 2
    kit.box(stage, f"{screw}/handle", (0, 0, cz), (HANDLE[0], HANDLE[0], HANDLE[1]), m["black"])
    kit.cylinder(stage, f"{screw}/cap", (0, 0, cz + HANDLE[1] / 2 + 0.002), 0.012, 0.004, m["red"])
    assert abs(cz - handle_center_local()) < 1e-9


ATTACH_ROT = _rot("Y", math.pi / 2)
"""First link relative to the ground clamp's frame: the cable leaves the ferrule straight down (link +x = frame -z)."""
LUG_ROT = _rot("Y", math.pi / 2)
"""Line clamp frame relative to the last link at zero ferrule angle: link +x (into the ferrule) = frame +z. The
ferrule joint (a hinge about the clamp's x) lets the cable leave sideways, as the first joint does at the ground
clamp."""


def link_axis(k: int) -> str:
    return "Z" if k % 2 == 0 else "Y"


def cable_fk(g_pos, g_rot, q):
    p = np.asarray(g_pos) + g_rot @ np.asarray(LUG)
    r = g_rot @ ATTACH_ROT
    links = []
    for k in range(LINK_N):
        r = r @ _rot(link_axis(k), q[k])
        links.append((p.copy(), r.copy()))
        p = p + r @ np.array([LINK_L, 0, 0])
    l_rot = r @ LUG_ROT @ _rot("X", q[LINK_N])
    l_pos = p - l_rot @ np.asarray(LUG)
    return links, (l_pos, l_rot)


def solve_cable(g_pose, l_pose):
    """Joint angles routing the cable from the ground clamp's ferrule to the line clamp's, lying on the desk in front
    of the bars and clear of the arm bases."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    gp, gr = np.asarray(g_pose[:3]), _qmat(g_pose[3:])
    lp, lr = np.asarray(l_pose[:3]), _qmat(l_pose[3:])

    def residual(q):
        links, (pos, rot) = cable_fk(gp, gr, q)
        err_p = (pos - lp) * 300.0
        err_r = Rotation.from_matrix(lr.T @ rot).as_rotvec() * 30.0
        ends = [p + r @ np.array([LINK_L, 0, 0]) for p, r in links]
        pen = [max(0.0, LINK_R + 0.001 - e[2]) * 300.0 for e in ends]
        pen += [max(0.0, e[0] - (X_EDGE - 0.03)) * 300.0 for e in ends[1:-2]]
        pen += [max(0.0, 0.15 - math.hypot(e[0], abs(e[1]) - 0.305)) * 300.0 for e in ends]
        low = [20.0 * e[2] for e in ends]
        return np.concatenate([err_p, err_r, 0.3 * q, pen, low])

    lim = math.radians(LINK_LIMIT_DEG) * 0.9
    best = None
    for trial in range(10):
        q0 = np.random.default_rng(trial).uniform(-0.6, 0.6, LINK_N + 1)
        sol = least_squares(residual, q0, bounds=(-lim, lim))
        if best is None or sol.cost < best.cost:
            best = sol
    links, (pos, rot) = cable_fk(gp, gr, best.x)
    print(f"cable solve: line clamp pos error {np.linalg.norm(pos - lp) * 1000:.2f} mm, rot error "
          f"{np.degrees(np.linalg.norm(Rotation.from_matrix(lr.T @ rot).as_rotvec())):.2f} deg, "
          f"lowest link z {min(p[2] for p, _ in links):.3f}, max x {max(p[0] for p, _ in links[1:]):.3f}")
    return best.x


def park_pose(which: int) -> tuple:
    """Start pose of a clamp's frame: 0 the ground clamp, 1 the line clamp, each hung (loose, pad on the plate) on its
    insulating parking plate (with GROUND_ON_PARK False the ground clamp starts hung, loose, on the ground bar)."""
    if which == 0:
        return frame_home(PARK_Y[0] if GROUND_ON_PARK else GROUND_Y) + (0.0, 0.0, 0.0, 1.0)
    return frame_home(PARK_Y[1]) + (0.0, 0.0, 0.0, 1.0)


def make_set(out_dir: Path) -> dict:
    """The grounding set (two clamps with their sticks, and the cable) as one floating articulation rooted at the
    ground clamp's frame; bodies placed at the start pose (both clamps on their parking stubs)."""
    root = "/ground_set"
    stage, _ = kit.new_stage(out_dir / "set.usda", root)
    m = _materials(stage, root)
    g_pose, l_pose = park_pose(0), park_pose(1)
    gp, gr = np.asarray(g_pose[:3]), _qmat(g_pose[3:])

    def rel(pos, rot):  # world -> ground-frame coordinates
        return gr.T @ (np.asarray(pos) - gp), _mat_q(gr.T @ rot)

    def clamp(prefix, f_pos, f_quat):
        frame = _body(stage, f"{root}/{prefix}_frame", f_pos, f_quat, FRAME_MASS)
        fr = _qmat(f_quat)
        pad = _body(stage, f"{root}/{prefix}_pad", tuple(np.asarray(f_pos) + fr @ np.array([0, 0, P0])), f_quat,
                    PAD_MASS)
        screw = _body(stage, f"{root}/{prefix}_screw", tuple(np.asarray(f_pos) + fr @ np.array([0, 0, P0])), f_quat,
                      SCREW_MASS)
        _clamp_visuals(stage, frame.GetPath(), pad.GetPath(), screw.GetPath(), m)
        kit.prismatic(stage, f"{root}/joints/{prefix}_slide", frame.GetPath(), pad.GetPath(), "Z", (0, 0, 0),
                      P_LIMITS[0], P_LIMITS[1])
        kit.revolute(stage, f"{root}/joints/{prefix}_turn", pad.GetPath(), screw.GetPath(), "Z", (0, 0, 0), -1440.0,
                     1440.0)
        kit.friction(stage, root, [f"{frame.GetPath()}/jaw", f"{frame.GetPath()}/spine", f"{pad.GetPath()}/pad"],
                     static=1.0, dynamic=0.8, name=f"{prefix}_serrated")
        kit.friction(stage, root, [f"{screw.GetPath()}/handle"], static=1.2, dynamic=1.0, name=f"{prefix}_grip")
        return frame

    g_frame = clamp("g", (0, 0, 0), (0, 0, 0, 1))
    kit.articulation_root(stage, g_frame.GetPrim())
    g_frame.GetPrim().CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    g_frame.GetPrim().CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)

    q = solve_cable(g_pose, l_pose)
    links, (lpos, lrot) = cable_fk(gp, gr, q)
    prev = g_frame.GetPath()
    for k, (p, r) in enumerate(links):
        lp_, lq = rel(p, r)
        body = _body(stage, f"{root}/cable_{k:02d}", lp_, lq, LINK_MASS)
        _capsule(stage, f"{root}/cable_{k:02d}/jacket", (0.0, 0, 0), (LINK_L, 0, 0), LINK_R, m["cable"])
        if k == 0:
            _joint(stage, f"{root}/joints/cable_j{k:02d}", prev, body.GetPath(), link_axis(k), LUG,
                   _mat_q(ATTACH_ROT), (0, 0, 0), -LINK_LIMIT_DEG, LINK_LIMIT_DEG)
        else:
            _joint(stage, f"{root}/joints/cable_j{k:02d}", prev, body.GetPath(), link_axis(k), (LINK_L, 0, 0),
                   (0, 0, 0, 1), (0, 0, 0), -LINK_LIMIT_DEG, LINK_LIMIT_DEG)
        prev = body.GetPath()
    kit.friction(stage, root, [f"{root}/cable_{k:02d}/jacket" for k in range(LINK_N)], static=0.4, dynamic=0.3,
                 name="jacket", combine="min")
    lp_, lq = rel(lpos, lrot)
    l_frame = clamp("l", lp_, lq)
    ferrule = _joint(stage, f"{root}/joints/cable_jend", prev, l_frame.GetPath(), "X", (LINK_L, 0, 0), _mat_q(LUG_ROT),
                     LUG, -LINK_LIMIT_DEG, LINK_LIMIT_DEG)
    del ferrule
    stage.GetRootLayer().Save()
    print(f"wrote {out_dir / 'set.usda'}; line clamp at {np.round(lpos, 4)} (park {l_pose[:3]})")
    init = {f"cable_j{k:02d}": float(v) for k, v in enumerate(q[:LINK_N])}
    init["cable_jend"] = float(q[LINK_N])
    init.update(g_slide=P0, l_slide=P0, g_turn=0.0, l_turn=0.0)
    return init


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets"
    out.mkdir(parents=True, exist_ok=True)
    make_bench(out)
    init = make_set(out)
    (out / "init_joints.json").write_text(json.dumps(init, indent=1))
    print(f"wrote {out / 'init_joints.json'}")

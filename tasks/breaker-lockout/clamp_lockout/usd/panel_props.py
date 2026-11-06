"""Generate the props as USD: a small Square D QO load centre (deadfront, six QO120 breakers with real handle throws,
colour-coded circuit labels), the Master Lock 493B Grip Tight clamp-on lockout as a four-link articulation, the Master
Lock 410 padlock (usd/padlock_410.py) and the two stands.

Values: (P) published, (I) inferred from photos/video, (TM) to measure on a bought part. Collision shapes are boxes,
except the knob (a cylinder, so a fingertip can roll it).

Frames (desk frame: x away from the robot, y to the robot's left, z up; desk top z = 0)
- panel frame: origin on the deadfront's front face at the panel's centre line and the middle row's height; axes as
  the desk frame. Handles stand out along -x.
- lockout frame ("D"): origin on the panel-side plane of the lockout (its rear edges, which rest on the deadfront),
  at the fixed jaw's face, mid-height; x into the panel, y along the channel toward the knob end (toward the panel
  centre when installed on the right column), z up. Installed, D's axes equal the desk frame's.
  Links: body (root: the steel channel, its end loops with the padlock holes, the hinge block and the fixed jaw),
  jaw (prismatic along y: the screw-driven serrated jaw; q = 0 open, q < 0 toward the handle), knob (revolute about
  y: the thumbscrew's knurled knob) and lever (revolute about z at the hinge: 0 closed over the channel, > 0 open).

Usage: python -m clamp_lockout.usd.panel_props <out dir>
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from clamp_lockout import spec as S
from clamp_lockout.usd import padlock_410
from clamp_lockout.usd import padlock_props
from clamp_lockout.usd import usd_helpers as dr

MM = 0.001

# --- Square D QO120 (Schneider product data sheet, dimension drawing p.3) --------------------------------------------
POLE = 19.0 * MM
"""(P) one space, 3/4 in."""
BREAKER_LEN = 76.0 * MM
"""(P) 3.00 in along the handle's throw."""
STRIP_W = 40.0 * MM
"""Centre strip between the columns, carrying the circuit numbers (I, from panel photos; on the wide side so the
labels can be large enough for the top camera)."""
HANDLE_W = 10.0 * MM
"""(P) 0.38 in across the breaker (along z in the panel)."""
HANDLE_T = 7.0 * MM
"""(I) thickness along the throw, scaled from the drawing."""
HANDLE_L = 19.0 * MM
"""(I) pivot to tip, scaled from the drawing."""
PIVOT_IN = 7.0 * MM
"""(I) pivot behind the deadfront's face: the handle then stands 10.7-11 mm proud (drawing: ~11-13 mm)."""
PIVOT_FROM_STRIP = 24.0 * MM
"""(P) handle pivot 24 mm from the breaker's end face (the bus end, at the centre strip)."""
ON_DEG, OFF_DEG = 22.0, 21.0
"""(P) handle at ON 22 deg toward the panel centre, OFF 21 deg toward the outside, from the panel normal."""
LIMIT_OUT_DEG = 26.0
"""Past OFF (the drawing's reset travel is 26 deg further; the sim stops at 26 deg from the normal)."""
LIMIT_IN_DEG = 24.0
DEADFRONT_T = 4.0 * MM
ENCLOSURE = (80.0 * MM, 260.0 * MM, 200.0 * MM)
"""Depth, width, height of the load-centre tub behind the deadfront (I: a small 6-space QO tub, shortened)."""
LABEL = (18.0 * MM, 17.0 * MM)
"""Colour-coded circuit label on the centre strip, width x height (I: Brady/Panduit breaker ID labels class)."""

# --- Master Lock 493B (technical data sheet DOC-000705 rev A: 74.0 x 19.8 x 18.9 mm, slot 11.4 deep, 8.3 mm hole) ---
LEN_HINGE = -24.0 * MM
"""Channel's hinge end (outboard). The jaw gap sits about 30 mm from this end and 44 mm from the loop end (P, drawing)."""
LEN_LOOP = 51.0 * MM
DEPTH = 18.3 * MM
"""(P) side plate height out of the panel (the lever adds 1.5 mm: 19.8 overall)."""
HALF_W = 9.45 * MM
"""(P) 18.9 mm across (z)."""
PLATE_T = 1.5 * MM
"""(I) powder-coated steel side plates."""
HINGE_BLOCK = (LEN_HINGE, -19.5 * MM)
FIXED_JAW_T = 3.0 * MM
JAW_X = (-12.0 * MM, -7.0 * MM)
"""Jaw faces span 7-12 mm in front of the deadfront: they grip the handle's outer end (it stands ~11 mm proud), so
the squeeze does not twist the tilted handle much."""
JAW_HALF_Z = 7.5 * MM
JAW_T = 3.0 * MM
NUT_WALL = (26.5 * MM, 28.5 * MM)
NUT_WALL_X = (-8.0 * MM, -3.0 * MM)
"""Thread boss behind the knob (I), clear of a fingertip rolling the knob from the front."""
KNOB_Y = (29.5 * MM, 37.5 * MM)
CHANNEL_END = 28.5 * MM
"""The side plates end here: past it the knob is exposed on top, bottom and end (493B photo, video 0:14-0:23)."""
SPINE_X = (-2.5 * MM, 0.0)
"""A narrow strap along the panel side joins the bottom plate to the end loop, behind the knob."""
LOOP_Y0 = 38.5 * MM
COVER_END = (38.2 * MM, 39.7 * MM)
"""The lever's end wall, just past the knob's end, when shut."""
TOP_SLOT = ((-6.0 * MM, 0.0), (-3.0 * MM, 14.0 * MM))
"""(x, y) of the slot cut in the top plate's panel-side edge over the handle (the real slot is 11.4 mm deep; 6 mm
here so the plate still rests on the handle's top before it is clamped)."""
KNOB_R = 7.5 * MM
KNOB_X = -10.5 * MM
"""Knob axis: its front rim is 0.3 mm behind the side plates' front edges (I: the thumb rolls it from the front)."""
HOLE = 8.3 * MM
"""(P) padlock hole; square in the sim (a 6.35 mm leg has 0.97 mm play each side)."""
HOLE_C = (-10.5 * MM, 44.5 * MM)
"""(I) hole centre (x, y) in the end loops."""
LEVER_PIVOT = (-19.8 * MM, -22.8 * MM)
LEVER_T = 3.0 * MM
TAB_Y0 = 39.7 * MM
TAB_Z = (-7.9 * MM, -3.9 * MM)
"""The lever's tip tab lies just above the channel's loop (the bottom plate's extension), holes in line (I, from the
photos: the body loop and the lever's loop are two flat tabs, one over the other). 4 mm thick (I; polymer)."""
LOOP_Z = (-11.0 * MM, -7.95 * MM)
"""The channel's loop with the padlock hole: the bottom plate's extension past the knob, thickened to 3 mm (I) so a
shackle cannot tunnel through it under the after-release check."""
TAB_X1 = -2.0 * MM
LEVER_OPEN_DEG = 115.0
LEVER_LIMIT_DEG = 178.0
PITCH = 1.25 * MM
"""(I) thumbscrew M8 coarse pitch; turns not published (the video shows a few seconds of rolling)."""
JAW_SLACK = 0.3 * MM
"""Open jaw 0.3 mm off the handle's centre-side face when the lockout sits on it (I)."""
FIXED_CLEAR = 0.2 * MM
BODY_MASS, JAW_MASS, KNOB_MASS, LEVER_MASS = 0.030, 0.008, 0.008, 0.010
"""53 g in all (P: 53 g); the small links are heavier than real for GPU solver stability."""

# --- stands (desk frame) -----------------------------------------------------------------------------------------------
LOCKOUT_START_D = (0.236, -0.140, S.ROW_Z)
"""The lockout frame's origin on its stand, in the installed orientation (a post under its bottom plate)."""
PADLOCK_XY = (0.262, 0.165)
PADLOCK_YAW = -90.0
"""Padlock legs along the desk's y, short leg toward -y (toward the lockout's loop when installed)."""
PADLOCK_PEDESTAL_TOP = 0.110

COLORS = dict(dr.COLORS, black=(0.06, 0.06, 0.07), steel=(0.72, 0.74, 0.77), lock_purple=(0.45, 0.20, 0.60),
              deadfront=(0.78, 0.78, 0.76), tub=(0.70, 0.70, 0.69), plywood=(0.72, 0.58, 0.40),
              breaker_black=(0.10, 0.10, 0.11), handle_black=(0.05, 0.05, 0.05), lockout_red=(0.78, 0.07, 0.05),
              zinc=(0.86, 0.87, 0.88), knob=(0.04, 0.04, 0.04), window=(0.20, 0.20, 0.21))
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LABEL_RGB = {"white": (240, 240, 236), "yellow": (235, 190, 0), "green": (0, 120, 40), "blue": (0, 60, 230),
             "orange": (230, 90, 0)}


# --------------------------------------------------------------------------------------------------------------------
# Derived geometry (used by the task and the demo)
# --------------------------------------------------------------------------------------------------------------------

def panel_origin() -> tuple[float, float, float]:
    return (S.PANEL_X, S.PANEL_Y, S.ROW_Z)


def pivot_panel(k: int) -> tuple[float, float, float]:
    """Handle k's pivot in the panel frame."""
    return (PIVOT_IN, S.column(k) * (STRIP_W / 2 + PIVOT_FROM_STRIP), (1 - S.row(k)) * POLE)


def pivot_world(k: int) -> tuple[float, float, float]:
    px, py, pz = pivot_panel(k)
    ox, oy, oz = panel_origin()
    return (ox + px, oy + py, oz + pz)


def on_sign(k: int) -> float:
    """Joint angle (about +z) sign that turns handle k toward ON (toward the panel centre)."""
    return 1.0 if S.column(k) > 0 else -1.0


def on_angle(k: int) -> float:
    return on_sign(k) * math.radians(ON_DEG)


def off_angle(k: int) -> float:
    return -on_sign(k) * math.radians(OFF_DEG)


def handle_dir(angle: float) -> tuple[float, float]:
    """(x, y) unit vector from the pivot toward the tip at joint angle `angle` about +z."""
    return (-math.cos(angle), -math.sin(angle))


def _handle_corners(k: int, angle: float):
    """The handle's outline corners (x, y) in the panel frame in front of the deadfront."""
    px, py, _ = pivot_panel(k)
    dx, dy = handle_dir(angle)
    nx, ny = -dy, dx
    pts = []
    for s in (PIVOT_IN / math.cos(angle), HANDLE_L):
        for t in (-HANDLE_T / 2, HANDLE_T / 2):
            pts.append((px + s * dx + t * nx, py + s * dy + t * ny))
    return pts


def lockout_seat_panel(k: int) -> tuple[float, float, float]:
    """The lockout frame's origin (panel frame) when it sits on handle k at OFF (right column only)."""
    assert S.column(k) < 0, "the lockout is modelled for the right column (knob end toward the centre, +y)"
    pts = _handle_corners(k, off_angle(k))
    y_min = min(y for x, y in pts if x < -0.5 * MM)
    return (0.0, y_min - FIXED_CLEAR, pivot_panel(k)[2])


def lockout_seat_world(k: int) -> tuple[float, float, float]:
    sx, sy, sz = lockout_seat_panel(k)
    ox, oy, oz = panel_origin()
    return (ox + sx, oy + sy, oz + sz)


def jaw_open_y(k: int) -> float:
    """Moving jaw face (lockout frame y) when open: JAW_SLACK off the handle's centre-side face at the jaw's rear end."""
    px, py, _ = pivot_panel(k)
    a = off_angle(k)
    dx, dy = handle_dir(a)
    nx, ny = -dy, dx
    # centre-side face (+y side) at x = JAW_X[1]: point on the centre line at that x, plus the half thickness along n
    s = (px - JAW_X[1]) / -dx
    y_face = max(y for x, y in [(px + s * dx + t * nx, py + s * dy + t * ny) for t in (-HANDLE_T / 2, HANDLE_T / 2)])
    return y_face - lockout_seat_panel(k)[1] + JAW_SLACK


def jaw_boxes(k: int):
    """((centre, yaw deg, length) of the fixed jaw, (centre, ...) of the moving jaw when open), lockout frame. Each
    jaw's face is parallel to the handle's flank at OFF: the fixed jaw FIXED_CLEAR outside the outer flank, the open
    moving jaw JAW_SLACK outside the inner flank (both measured along y)."""
    px, py, _ = pivot_panel(k)
    ys = lockout_seat_panel(k)[1]
    a = off_angle(k)
    dx, dy = handle_dir(a)
    nx, ny = -dy, dx
    if ny < 0:
        nx, ny = -nx, -ny  # normal of the inner (+y) flank
    s0, s1 = (PIVOT_IN - (JAW_X[1] + 0.5 * MM)) / abs(dx), (PIVOT_IN - (JAW_X[0] - 1.5 * MM)) / abs(dx)
    sm, length = (s0 + s1) / 2, abs(s1 - s0)
    cx, cy = px + sm * dx, py - ys + sm * dy
    yaw = math.degrees(math.atan2(dy, dx))
    fixed = (cx - (HANDLE_T / 2 + FIXED_JAW_T / 2) * nx, cy - (HANDLE_T / 2 + FIXED_JAW_T / 2) * ny - FIXED_CLEAR, 0.0)
    moving = (cx + (HANDLE_T / 2 + JAW_T / 2) * nx, cy + (HANDLE_T / 2 + JAW_T / 2) * ny + JAW_SLACK, 0.0)
    return (fixed, yaw, length), (moving, yaw, length)


def hole_d() -> tuple[float, float, float]:
    return (HOLE_C[0], HOLE_C[1], 0.0)


def lockout_start_world() -> tuple[float, ...]:
    return tuple(LOCKOUT_START_D) + (0.0, 0.0, 0.0, 1.0)


def lockout_post():
    """(x range, y range, top z) of the post the lockout rests on (under its bottom plate, clear of the grasp)."""
    x, y, z = LOCKOUT_START_D
    return ((x - 0.017, x - 0.002), (y - 0.022, y - 0.004), z - HALF_W)


def lever_rest():
    """(x range, y range, top z) of the small post under the open lever's far half (it would tip the lockout)."""
    x, y, z = LOCKOUT_START_D
    px, py = LEVER_PIVOT
    a = math.radians(LEVER_OPEN_DEG)
    cx, cy = x + px - 0.050 * math.sin(a), y + py + 0.050 * math.cos(a)
    return ((cx - 0.008, cx + 0.008), (cy - 0.008, cy + 0.008), z - HALF_W)


def padlock_start_world() -> tuple[float, ...]:
    v = padlock_props.PADLOCKS["ml410"]
    x, y = PADLOCK_XY
    h = math.radians(PADLOCK_YAW) / 2
    return (x, y, PADLOCK_PEDESTAL_TOP + v.body[2] / 2 + 0.001, 0.0, 0.0, math.sin(h), math.cos(h))


# --------------------------------------------------------------------------------------------------------------------
# USD helpers
# --------------------------------------------------------------------------------------------------------------------

def _mats(stage, root):
    class M(dict):
        def __missing__(self, key):
            self[key] = _material(stage, root, key)
            return self[key]
    return M()


def _material(stage, root, name):
    path = f"{root}/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[name]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


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


def _cyl(stage, path, center, radius, height, material, axis="Y", collide=False):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(axis)
    cyl.CreateHeightAttr(height)
    cyl.CreateRadiusAttr(radius)
    UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    return cyl


def _obox(stage, path, center, size, yaw_deg, material, collide=True):
    """A box turned yaw_deg about z (collider unless collide=False)."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(cube)
    x.AddTranslateOp().Set(Gf.Vec3d(*center))
    x.AddRotateZOp().Set(float(yaw_deg))
    x.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _joint(stage, path, kind, body0, body1, axis, pos0, lower, upper, drive):
    joint = (UsdPhysics.RevoluteJoint if kind == "rev" else UsdPhysics.PrismaticJoint).Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0.GetPath()])
    joint.CreateBody1Rel().SetTargets([body1.GetPath()])
    joint.CreateAxisAttr(axis)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), drive)
    return joint


def make_textures(tex: Path) -> None:
    tex.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT_B, 92)
    for c in S.CIRCUITS:
        bg = LABEL_RGB[S.LABEL_COLORS[c]]
        fg = (20, 20, 20) if sum(bg) > 400 else (250, 250, 250)
        img = Image.new("RGB", (160, 100), bg)
        d = ImageDraw.Draw(img)
        d.rectangle([3, 3, 156, 96], outline=fg, width=4)
        d.text((80, 52), str(c), fill=fg, font=font, anchor="mm")
        img.save(tex / f"circuit_{c}.png")
    small = ImageFont.truetype(FONT_B, 30)
    for name, text in (("on", "ON"), ("off", "OFF")):
        img = Image.new("RGB", (96, 48), (16, 16, 18))
        ImageDraw.Draw(img).text((48, 24), text, fill=(235, 235, 235), font=small, anchor="mm")
        img.save(tex / f"brk_{name}.png")


# --------------------------------------------------------------------------------------------------------------------
# Props
# --------------------------------------------------------------------------------------------------------------------

def make_panel(out_path: Path) -> None:
    """Fixed-base articulation: backboard, tub, deadfront with the breakers' faces (collision flush with the deadfront),
    labels; one handle link per breaker on a revolute joint about z at its pivot."""
    stage, root_prim, _ = dr._new_stage(out_path, "/panel")
    m = _mats(stage, "/panel")
    r = "/panel"
    base = dr._fixed_articulation(stage, root_prim, r)
    b = f"{r}/base"
    d, w, h = ENCLOSURE
    # Backboard (plywood) standing on the table, the tub on it, the deadfront in front.
    dr._boxspan(stage, f"{b}/backboard", (d, d + 0.018), (-0.20, 0.20), (-S.ROW_Z, 0.13), m["plywood"])
    dr._boxspan(stage, f"{b}/tub", (DEADFRONT_T, d), (-w / 2, w / 2), (-h / 2, h / 2), m["tub"])
    dr._boxspan(stage, f"{b}/deadfront", (0.0, DEADFRONT_T), (-w / 2 + 0.005, w / 2 - 0.005),
                (-h / 2 + 0.005, h / 2 - 0.005), m["deadfront"])
    rel = "./tex"
    for k, c in enumerate(S.CIRCUITS):
        col, z = S.column(k), (1 - S.row(k)) * POLE
        y0, y1 = sorted((col * STRIP_W / 2, col * (STRIP_W / 2 + BREAKER_LEN)))
        # The breaker's face in the deadfront window (visual), its handle collar, its ON/OFF marks.
        dr._boxspan(stage, f"{b}/breaker_{k}", (-0.0004, 0.0), (y0, y1), (z - POLE / 2 + 0.0004, z + POLE / 2 - 0.0004),
                    m["breaker_black"], collide=False)
        py = pivot_panel(k)[1]
        dr._boxspan(stage, f"{b}/collar_{k}", (-0.0012, -0.0004), (py - 0.009, py + 0.009), (z - 0.0075, z + 0.0075),
                    m["window"], collide=False)
        for name, sgn in (("on", 1.0), ("off", -1.0)):
            yc = py - col * sgn * 0.0155  # beside the collar, ON toward the centre
            mat = _tex_material(stage, r, f"brk_{name}", f"{rel}/brk_{name}.png")
            _quad(stage, f"{b}/mark_{name}_{k}", [(-0.0006, yc - 0.006, z - 0.003), (-0.0006, yc + 0.006, z - 0.003),
                                                  (-0.0006, yc + 0.006, z + 0.003), (-0.0006, yc - 0.006, z + 0.003)],
                  mat)
        # Circuit label on the centre strip, beside its breaker (seen from the front: image left = robot's left = +y).
        lw, lh = LABEL
        yl = col * (lw / 2 + 0.001)
        mat = _tex_material(stage, r, f"circuit_{c}", f"{rel}/circuit_{c}.png")
        half = lw / 2
        _quad(stage, f"{b}/label_{k}", [(-0.0006, yl + half, z - lh / 2), (-0.0006, yl - half, z - lh / 2),
                                        (-0.0006, yl - half, z + lh / 2), (-0.0006, yl + half, z + lh / 2)], mat)
        # Handle link.
        pivot = pivot_panel(k)
        hl = dr._rigid(stage, f"{r}/handle_{k}", 0.006, pivot)
        dr._boxspan(stage, f"{r}/handle_{k}/lever", (-HANDLE_L, -PIVOT_IN + 0.0005), (-HANDLE_T / 2, HANDLE_T / 2),
                    (-HANDLE_W / 2, HANDLE_W / 2), m["handle_black"])
        lo, hi = sorted((on_sign(k) * LIMIT_IN_DEG, -on_sign(k) * LIMIT_OUT_DEG))
        _joint(stage, f"{r}/joints/handle_{k}_joint", "rev", base, hl, "Z", pivot, lo, hi, "angular")
    dr._friction(stage, r, [b], static=0.6, dynamic=0.5, name="steel")
    stage.GetRootLayer().Save()


def _hole_plate_xy(stage, path, mat, xs, ys, zs, hole_c, hole):
    """A plate spanning xs x ys (thickness zs) with a square hole of side `hole` centred at hole_c (x, y)."""
    hx0, hx1 = hole_c[0] - hole / 2, hole_c[0] + hole / 2
    hy0, hy1 = hole_c[1] - hole / 2, hole_c[1] + hole / 2
    dr._boxspan(stage, f"{path}_a", (xs[0], hx0), ys, zs, mat)
    dr._boxspan(stage, f"{path}_b", (hx1, xs[1]), ys, zs, mat)
    dr._boxspan(stage, f"{path}_c", (hx0, hx1), (ys[0], hy0), zs, mat)
    dr._boxspan(stage, f"{path}_d", (hx0, hx1), (hy1, ys[1]), zs, mat)


def make_lockout(out_path: Path, k: int) -> None:
    """The 493B: body (root), jaw (prismatic y), knob (revolute y), lever (revolute z)."""
    stage, root_prim, _ = dr._new_stage(out_path, "/lockout")
    m = _mats(stage, "/lockout")
    r = "/lockout"
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    root_prim.AddAppliedSchema("PhysxArticulationAPI")
    root_prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    body = dr._rigid(stage, f"{r}/body", BODY_MASS)
    b = f"{r}/body"
    red, zinc = m["lockout_red"], m["zinc"]
    zin, zout = HALF_W - PLATE_T, HALF_W
    (sx0, sx1), (sy0, sy1) = TOP_SLOT
    # Top plate, with a slot over the handle at its panel-side edge (the handle shows through it from above).
    dr._boxspan(stage, f"{b}/plate_top_a", (-DEPTH, 0.0), (LEN_HINGE, sy0), (zin, zout), red)
    dr._boxspan(stage, f"{b}/plate_top_b", (-DEPTH, sx0), (sy0, sy1), (zin, zout), red)
    dr._boxspan(stage, f"{b}/plate_top_c", (-DEPTH, 0.0), (sy1, CHANNEL_END), (zin, zout), red)
    dr._boxspan(stage, f"{b}/plate_bot", (-DEPTH, 0.0), (LEN_HINGE, CHANNEL_END), (-zout, -zin), red)
    # Strap behind the knob to the end loop, and the loop with the padlock hole.
    dr._boxspan(stage, f"{b}/spine", SPINE_X, (CHANNEL_END - 1.0 * MM, LOOP_Y0 + 1.0 * MM), LOOP_Z, red)
    _hole_plate_xy(stage, f"{b}/loop", red, (-DEPTH, 0.0), (LOOP_Y0, LEN_LOOP), LOOP_Z, HOLE_C, HOLE)
    dr._boxspan(stage, f"{b}/hinge_block", (-DEPTH, 0.0), HINGE_BLOCK, (-zin, zin), red)
    # Jaws: faces parallel to the handle's flanks at OFF (the serrated jaws bear on the flanks, not on a corner).
    (fc, fyaw, flen), (mc, _, _) = jaw_boxes(k)
    _obox(stage, f"{b}/fixed_jaw", fc, (flen, FIXED_JAW_T, 2 * zin), fyaw, zinc)
    dr._boxspan(stage, f"{b}/jaw_back", (-DEPTH, JAW_X[0]), (HINGE_BLOCK[1], 0.0), (-zin, zin), red)
    dr._boxspan(stage, f"{b}/nut_wall", NUT_WALL_X, NUT_WALL, (-zin, zin), red)
    # Serrations (visual) on the fixed jaw's face.
    dr._friction(stage, r, [b], static=1.5, dynamic=1.2, name="serrated")

    jaw0 = jaw_open_y(k)
    jaw = dr._rigid(stage, f"{r}/jaw", JAW_MASS, (0.0, jaw0, 0.0))
    _obox(stage, f"{r}/jaw/block", (mc[0], mc[1] - jaw0, mc[2]), (flen, JAW_T, 2 * JAW_HALF_Z), fyaw, zinc)
    # The jaw's carrier back to the screw (visual).
    dr._boxspan(stage, f"{r}/jaw/carrier", (-DEPTH + 1.0 * MM, JAW_X[0]), (0.0, JAW_T), (-JAW_HALF_Z, JAW_HALF_Z), zinc,
                collide=False)
    screw_len = KNOB_Y[0] - jaw0 - JAW_T
    _cyl(stage, f"{r}/jaw/screw", (KNOB_X, JAW_T + screw_len / 2, 0.0), 3.0 * MM, screw_len, m["steel"])
    dr._friction(stage, r, [f"{r}/jaw"], static=1.5, dynamic=1.2, name="serrated_jaw")
    _joint(stage, f"{r}/joints/jaw_joint", "pri", body, jaw, "Y", (0.0, jaw0, 0.0), -(jaw0 - 1.0 * MM), 0.002,
           "linear")

    kc = (KNOB_X, (KNOB_Y[0] + KNOB_Y[1]) / 2, 0.0)
    knob = dr._rigid(stage, f"{r}/knob", KNOB_MASS, kc)
    _cyl(stage, f"{r}/knob/wheel", (0, 0, 0), KNOB_R, KNOB_Y[1] - KNOB_Y[0], m["knob"], collide=True)
    for i in range(6):  # knurl marks (visual)
        a = i * math.pi / 3
        dr._box(stage, f"{r}/knob/knurl_{i}", (KNOB_R * math.cos(a), 0, KNOB_R * math.sin(a)),
                (0.0012, KNOB_Y[1] - KNOB_Y[0] - 0.001, 0.0012), m["window"], collide=False)
    dr._friction(stage, r, [f"{r}/knob"], static=1.5, dynamic=1.2, name="knurl")
    _joint(stage, f"{r}/joints/knob_joint", "rev", body, knob, "Y", kc, -1.0e6, 1.0e6, "angular")

    px, py = LEVER_PIVOT
    lever = dr._rigid(stage, f"{r}/lever", LEVER_MASS, (px, py, 0.0))
    lv = f"{r}/lever"
    # Lever plate (local frame = D shifted to the pivot), closed over the channel's front.
    dr._boxspan(stage, f"{lv}/plate", (-LEVER_T / 2, LEVER_T / 2), (LEN_HINGE - py, LEN_LOOP - py), (-zout, zout), red)
    hx, hy = HOLE_C[0] - px, HOLE_C[1] - py
    _hole_plate_xy(stage, f"{lv}/tab", red, (LEVER_T / 2, TAB_X1 - px), (TAB_Y0 - py, LEN_LOOP - py),
                   TAB_Z, (hx, hy), HOLE)
    # The cover round the knob: top, bottom and end walls (shut, nothing can reach the knob).
    cx = (LEVER_T / 2, -3.0 * MM - px)
    dr._boxspan(stage, f"{lv}/cover_top", cx, (CHANNEL_END - py, COVER_END[1] - py), (zin, zout), red)
    dr._boxspan(stage, f"{lv}/cover_bot", cx, (CHANNEL_END - py, COVER_END[0] - py), (-zout, -zin), red)
    dr._boxspan(stage, f"{lv}/cover_end", cx, (COVER_END[0] - py, COVER_END[1] - py), (TAB_Z[0], zin), red)
    # Embossed brand band (visual, darker red).
    dr._boxspan(stage, f"{lv}/emboss", (-LEVER_T / 2 - 0.0003, -LEVER_T / 2), (5 * MM, 50 * MM), (-3 * MM, 3 * MM),
                m["lockout_red2"], collide=False)
    _joint(stage, f"{r}/joints/lever_joint", "rev", body, lever, "Z", (px, py, 0.0), 0.0, LEVER_LIMIT_DEG, "angular")
    stage.GetRootLayer().Save()


def make_stands(out_path: Path) -> None:
    stage, _, _ = dr._new_stage(out_path, "/stands")
    m = _mats(stage, "/stands")
    r = "/stands"
    xs, ys, top = lockout_post()
    cx, cy = (xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2
    dr._box(stage, f"{r}/lockout_foot", (cx, cy, 0.005), (0.06, 0.05, 0.01), m["stand"])
    dr._boxspan(stage, f"{r}/lockout_post", xs, ys, (0.0, top), m["stand"])
    xs, ys, top = lever_rest()
    dr._boxspan(stage, f"{r}/lever_rest", xs, ys, (0.0, top), m["stand"])
    x0, y0, _ = LOCKOUT_START_D
    dr._boxspan(stage, f"{r}/mid_rest", (x0 - 0.017, x0 - 0.002), (y0 + 0.004, y0 + 0.022), (0.0, top), m["stand"])
    x, y = PADLOCK_XY
    dr._box(stage, f"{r}/padlock_foot", (x, y, 0.005), (0.05, 0.06, 0.01), m["stand"])
    dr._box(stage, f"{r}/padlock_column", (x, y, PADLOCK_PEDESTAL_TOP / 2), (0.009, 0.030, PADLOCK_PEDESTAL_TOP),
            m["stand"])
    stage.GetRootLayer().Save()


def _robust(out_path: Path, root: str) -> None:
    """More solver iterations and a small explicit contact offset (as in the other lockout props)."""
    stage = Usd.Stage.Open(str(out_path))
    rp = stage.GetPrimAtPath(root)
    if not rp.HasAPI(UsdPhysics.ArticulationRootAPI):
        rp.AddAppliedSchema("PhysxArticulationAPI")
    rp.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(64)
    rp.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(8)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.002)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    stage.GetRootLayer().Save()


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    k = S.CIRCUITS.index(S.NAMED_CIRCUIT)
    make_textures(out_dir / "tex")
    make_panel(out_dir / "panel.usda")
    make_lockout(out_dir / "lockout.usda", k)
    padlock_410.make_padlock_410(out_dir / "padlock_ml410.usda")
    make_stands(out_dir / "stands.usda")
    for name, root in (("panel", "/panel"), ("lockout", "/lockout"), ("padlock_ml410", "/padlock")):
        _robust(out_dir / f"{name}.usda", root)
    print(f"wrote breaker-lockout props to {out_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

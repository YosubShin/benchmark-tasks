"""Generate the props of the drawout relay withdrawal task as USD (panel.usda, unit.usda, label_*.png).

Real part: ABB Relion 615 series (REF615) plug-in unit in its flush-mounted case. Published, from the 615
series installation manual 1MRS756375 G (library.e.abb.com/public/71f77e79135977bbc1257b130056cac1):
- frame (bezel) 177 x 177 mm (4U); case 164 mm wide x 160 mm high; depth 201 mm = 153 mm behind the panel
  + 48 mm in front of it (frame, HMI and handle); panel cut-out 165.5 x 161.5 mm (fig. 5, fig. 41, table 5);
- plug-in unit only 2.1 kg, complete IED 4.1 kg (table 5);
- 4.2.1: "Lift the handle to 90 degrees to release the latching mechanism. The plug-in unit is pushed about
  7 mm out of the case and the connectors are separated. Pull the unit out of the case." Fig. 2 and 3 show
  the handle: a bail pivoted on both sides of the frame about a third of the way down, its cross-bar along
  the bottom of the front at rest and standing straight out at 90 degrees. No end stop is shown or
  mentioned: the unit comes all the way out.
Inferred (not published, to measure on a real unit): the body inside the case 155 x 151 x 147 mm (2 mm
clearance a side), bezel 24 mm thick, pivot 50 mm below the frame top and 12 mm in front of the panel,
bail arm 122 mm (pivot to cross-bar), cross-bar 12 mm round and 12 mm in front of the HMI, handle 0.1 kg.

Mechanism (geometry; see relay_sim/task.py for the connector rule):
- The handle is a revolute joint about y at the pivots, 0 (down) to 92 deg (lifted).
- Latch: on each hub a hook points forward with a lug hanging from its end. Fixed beside it is a catch
  wall just in front of the lug: pulling the unit drives the lug into the catch below the pivot, which
  presses the handle down onto its stop, so the pull cannot open it (self-locking, 5 mm of play). The
  catch lies just outside the lug's swing circle, so lifting the handle ~25 deg raises the lug over it.
- Cam: a tooth on the hub points up at rest; it rides just outside the case cut-out, in front of the panel plate.
  Lifting the handle swings the tooth back; from ~28 deg it bears on the panel's front face and pushes the unit
  out by TOOTH_R - CAM_GAP = 7 mm at 90 deg. (A separate thin cam keeper in front of the panel did not work in
  PhysX: the tooth sank through it to the plate and then caught on the keeper's front face on the way out.)
- The arm and cam tooth share one 6 mm band beside the bezel; the hook, lug and catch the next band out;
  the catch's strut to the panel the band after that (so the arm never sweeps through the catch).
- The real latch and cam are inside the frame; here they sit outside the bezel's sides so they can be
  boxes. The panel, cases, keepers, designation labels and the bench mat are one static asset.

Labels: an engraved two-layer laminate plate (white face, black letters cut through) under each relay, with a
round colour marker on its right end (seen from the robot), the way panel shops mark circuits. At the top camera's 320x176 the letters are
not legible (about 3 px tall), the marker is the distinguishing feature there; the wrist cameras read the letters.

Frames: the unit's origin is at the front edge of its body's bottom face (x into the case). The panel
asset is in world coordinates.

Usage: python -m relay_sim.usd.relay_props [<out dir>]   (default: relay_sim/assets)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from relay_sim.spec import BODY_Z, LABELS, MARKER_RGB, PANEL_X, SLOT_Y, UNIT_MASS_KG
from relay_sim.usd import usd_kit as kit

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Plug-in unit (unit frame) -----------------------------------------------------------------------------
BODY = (0.147, 0.155, 0.151)
"""Depth (x), width, height of the body that slides in the case (inferred from the 164 x 160 case)."""
BEZEL = (0.024, 0.177, 0.177)
BEZEL_Z0 = -0.013
"""Bezel bottom relative to the body's bottom (the 177 mm frame centred on the 151 mm body)."""
HANDLE_MASS = 0.10
PIVOT = (-0.012, BEZEL_Z0 + BEZEL[2] - 0.050)
"""(x, z) of the handle pivots in the unit frame."""
HUB_Y = BEZEL[1] / 2 + 0.003
"""The hubs, teeth and bail arms ride just outside the bezel's sides (6 mm thick)."""
HUB_T = 0.006
BAR_LOCAL = (-0.030, -0.122)
"""(x, z) of the cross-bar's centre from the pivot, handle down."""
BAR = 0.012
BAR_SPAN = 2 * (HUB_Y + HUB_T / 2)
ARM_W = 0.010
TOOTH_R = 0.0185
"""Tooth length from the pivot (points +z with the handle down)."""
TOOTH_W = 0.006
CAM_GAP = 0.0115
"""The cam face (the panel's front face) behind the pivot (x), unit seated: the eject stroke is TOOTH_R - CAM_GAP."""
HANDLE_UP_DEG = 92.0

# Case, panel, keepers (world) ----------------------------------------------------------------------------
SEAT_GAP = 0.0005
CASE_DEPTH = 0.153
CASE_W, CASE_H = 0.164, 0.160
WALL = 0.0025
FLOOR_T = 0.003
PANEL_T = 0.003
PANEL_Y = (-0.12, 0.42)
PANEL_TOP = 0.34
CUTOUT = (0.1655, 0.1615)
HOOK_A = 0.007
"""Hook length forward of the pivot, to the lug's centre."""
LUG = (0.005, (-0.010, 0.002))
"""Lug width (x) and z range (handle frame)."""
CATCH_X = (-0.024, -0.0145)
CATCH_Z = (-0.017, -0.0075)
"""Catch wall, relative to the pivot (unit seated). Its nearest corner is 16.3 mm from the pivot; the lug's farthest
corner swings at 13.8 mm (2.5 mm margin, so the handle still lifts with the unit up to ~2 mm proud). Latched play 5 mm;
lug engagement 2.5 mm. (Raising lug and catch by 4 mm for more margin put the catch's top under the bail arm's lower
edge as the arm passes over it during the withdrawal, and the unit stalled ~8 mm out.)"""
STRUT_Z = (-0.017, -0.011)
KEEPER_CLEAR = 0.002
"""Side gap between each fixed keeper and the handle parts it must not touch (the bail arm passes beside the catch)."""
LABEL_SIZE = (0.160, 0.040)
"""Designation label (engraved laminate, 1.6 mm), centred above each relay on the panel face."""
LABEL_Z = 0.258
"""Label centre height: 8 mm above the bezel's top (0.2295), on the panel above the cut-out."""

MAT_TOP = BODY_Z - 0.020
"""The bench mat's top: the withdrawn unit is lowered 2 cm onto it."""
WITHDRAWN_S = 0.170
"""Withdrawal (from seated) at which the unit is set down: body 2 cm clear of the case."""
S_SUPPORT = 0.12
"""The reference pulls the unit this far by the handle alone before the second arm goes under the body."""
B_FROM_FRONT = 0.035
"""The second arm's fingers sit this far behind the body's front face (x)."""


def mat_strips(slot: int = 0) -> tuple[tuple[float, float], tuple[float, float]]:
    """x ranges of the mat's two strips: the front one ahead of where the supporting fingers end up, the back
    one behind where they went in (the fingers never pass over a strip and end in the gap between them)."""
    x0 = seated_origin(slot)[0]
    b_start = x0 - S_SUPPORT + B_FROM_FRONT
    b_end = x0 - WITHDRAWN_S + B_FROM_FRONT
    front = (x0 - WITHDRAWN_S + 0.002, b_end - 0.014)
    back = (b_start + 0.016, x0 - WITHDRAWN_S + BODY[0] - 0.004)
    return front, back

UNIT_FRICTION = (0.30, 0.25)
"""Painted sheet steel on the case floor (inferred)."""

COLORS = dict(kit.COLORS)
COLORS.update({
    "abb_grey": (0.86, 0.87, 0.86),
    "abb_bezel": (0.80, 0.81, 0.81),
    "case_steel": (0.62, 0.64, 0.66),
    "panel": (0.74, 0.75, 0.73),
    "lcd": (0.16, 0.22, 0.20),
    "key_blue": (0.25, 0.50, 0.85),
    "abb_red": (0.85, 0.05, 0.08),
    "mat": (0.08, 0.08, 0.09),
})


def seated_origin(slot: int) -> tuple[float, float, float]:
    """World position of a seated unit's origin."""
    return (PANEL_X + SEAT_GAP, SLOT_Y[slot], BODY_Z + SEAT_GAP)


def pivot_world(slot: int) -> tuple[float, float]:
    """(x, z) of a seated unit's pivots."""
    x, _, z = seated_origin(slot)
    return (x + PIVOT[0], z + PIVOT[1])


def bar_local(phi: float) -> tuple[float, float]:
    """(x, z) of the cross-bar centre from the pivot at handle angle phi (rad; rotation about +y)."""
    x, z = BAR_LOCAL
    return (x * math.cos(phi) + z * math.sin(phi), -x * math.sin(phi) + z * math.cos(phi))


def mat_center(slot: int) -> tuple[float, float]:
    x, y, _ = seated_origin(slot)
    return (x - WITHDRAWN_S + BODY[0] / 2, y)


def _materials(stage, root, names):
    out = {}
    for name in names:
        path = f"{root}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[name]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        out[name] = material
    return out


def _rbox(stage, path, center, size, material, rot_y_deg=0.0, collide=True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddRotateYOp().Set(rot_y_deg)
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _span(stage, path, x0, x1, y0, y1, z0, z1, mat, collide=True):
    return kit.box(stage, path, ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), mat,
                   collide=collide)


def _contacts(stage, prefix, offset=0.002):
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(offset)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def make_unit(out_path: Path) -> None:
    """The plug-in unit: body (root, free) and the bail handle on a revolute joint."""
    root = "/unit"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    mats = _materials(stage, root, ("abb_grey", "abb_bezel", "case_steel", "lcd", "key_blue", "abb_red", "green",
                                     "red", "yellow", "dark"))
    body = kit.rigid_body(stage, f"{root}/body", (0, 0, 0), UNIT_MASS_KG - HANDLE_MASS)
    d, w, h = BODY
    kit.box(stage, f"{root}/body/box", (d / 2, 0, h / 2), (d, w, h), mats["case_steel"])
    bt, bw, bh = BEZEL
    fx = -bt / 2
    kit.box(stage, f"{root}/body/bezel", (fx, 0, BEZEL_Z0 + bh / 2), (bt, bw, bh), mats["abb_bezel"])
    # HMI (visual), as on the REF615 front: LCD top left, LED column, keypad below, ABB logo top left.
    face = -bt - 0.0006
    zc = BEZEL_Z0
    kit.box(stage, f"{root}/body/hmi", (face + 0.0003, 0.0, zc + bh / 2), (0.0006, bw - 0.012, bh - 0.012),
            mats["abb_grey"], collide=False)
    kit.box(stage, f"{root}/body/lcd", (face, 0.012, zc + 0.118), (0.001, 0.085, 0.060), mats["lcd"], collide=False)
    kit.box(stage, f"{root}/body/logo", (face, 0.060, zc + 0.160), (0.001, 0.022, 0.008), mats["abb_red"],
            collide=False)
    for k in range(10):
        kit.box(stage, f"{root}/body/led_{k}", (face, -0.045, zc + 0.145 - 0.0065 * k), (0.001, 0.004, 0.004),
                mats["red" if k < 2 else ("yellow" if k < 4 else "green")] if k else mats["green"], collide=False)
    for k, (dy, dz) in enumerate([(0.03, 0.06), (-0.0, 0.06), (-0.03, 0.06), (0.015, 0.042), (-0.015, 0.042),
                                  (0.03, 0.024), (0.0, 0.024), (-0.03, 0.024)]):
        kit.box(stage, f"{root}/body/key_{k}", (face, dy, zc + dz), (0.001, 0.012, 0.012), mats["key_blue"],
                collide=False)
    kit.box(stage, f"{root}/body/btn_close", (face, 0.065, zc + 0.062), (0.001, 0.014, 0.012), mats["red"],
            collide=False)
    kit.box(stage, f"{root}/body/btn_open", (face, 0.065, zc + 0.028), (0.001, 0.014, 0.012), mats["green"],
            collide=False)

    # Bail handle, body origin at the pivot axis (y = 0).
    px, pz = PIVOT
    handle = kit.rigid_body(stage, f"{root}/handle", (px, 0, pz), HANDLE_MASS)
    bx, bz = BAR_LOCAL
    arm_len = math.hypot(bx, bz)
    for k, s in enumerate((-1, 1)):
        y = s * HUB_Y
        # Arm from the pivot to the bar's end: a box along z, turned about y to point at the bar.
        ang = math.degrees(math.atan2(bx, bz))
        _rbox(stage, f"{root}/handle/arm_{k}", (bx / 2, y, bz / 2), (ARM_W, HUB_T, arm_len), mats["abb_grey"],
              rot_y_deg=ang)
        kit.box(stage, f"{root}/handle/tooth_{k}", (0, y, TOOTH_R / 2), (TOOTH_W, HUB_T, TOOTH_R), mats["dark"])
        y2 = s * (HUB_Y + HUB_T)
        kit.box(stage, f"{root}/handle/hook_{k}", (-HOOK_A / 2, y2, -0.001), (HOOK_A, HUB_T, 0.006), mats["dark"])
        kit.box(stage, f"{root}/handle/lug_{k}", (-HOOK_A, y2, sum(LUG[1]) / 2), (LUG[0], HUB_T, LUG[1][1] - LUG[1][0]),
                mats["dark"])
        kit.cylinder(stage, f"{root}/handle/hub_{k}", (0, y + s * HUB_T / 2, 0), 0.008, 2 * HUB_T + 0.001,
                     mats["abb_grey"], axis="Y")
    # Round cross-bar (collision cylinder): it can turn in a closed gripper, so the gripper need not turn the
    # full 90 deg with the handle.
    bar = kit.cylinder(stage, f"{root}/handle/bar", (bx, 0, bz), BAR / 2, BAR_SPAN, mats["abb_grey"], axis="Y")
    UsdPhysics.CollisionAPI.Apply(bar.GetPrim())
    kit.revolute(stage, f"{root}/joints/handle_joint", body.GetPath(), handle.GetPath(), "Y", (px, 0, pz), 0.0,
                 HANDLE_UP_DEG)
    kit.friction(stage, root, [f"{root}/body"], static=UNIT_FRICTION[0], dynamic=UNIT_FRICTION[1], name="steel",
                 combine="min")
    # Only the cross-bar gets the grip friction (it is held in the fingers); the hubs, teeth, hooks and lugs are
    # plain steel, so the latch and cam parts slide on their keepers.
    kit.friction(stage, root, [f"{root}/handle"], static=0.20, dynamic=0.15, name="latch_steel", combine="min")
    kit.friction(stage, root, [f"{root}/handle/bar", f"{root}/handle/arm_0", f"{root}/handle/arm_1"], static=1.0,
                 dynamic=0.9, name="grip", combine="max")
    _contacts(stage, f"{root}/handle")
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_panel(out_path: Path) -> None:
    """Panel with two cut-outs, the two cases behind it, the keepers, tags, the bench mat (static)."""
    root = "/panel"
    stage, _ = kit.new_stage(out_path, root)
    mats = _materials(stage, root, ("panel", "case_steel", "dark", "mat", "stand"))
    labels = [_tex_material(stage, root, f"label_{s}", f"label_{s}.png") for s in (0, 1)]
    X = PANEL_X
    # Panel plate around the cut-outs.
    cut = [(SLOT_Y[s], BODY_Z - FLOOR_T + CASE_H / 2) for s in (0, 1)]
    cw, ch = CUTOUT
    ys = sorted([PANEL_Y[0], PANEL_Y[1]] + [c[0] + d * cw / 2 for c in cut for d in (-1, 1)])
    z_lo = min(c[1] for c in cut) - ch / 2
    z_hi = max(c[1] for c in cut) + ch / 2
    _span(stage, f"{root}/plate_bottom", X, X + PANEL_T, PANEL_Y[0], PANEL_Y[1], 0.0, z_lo, mats["panel"])
    _span(stage, f"{root}/plate_top", X, X + PANEL_T, PANEL_Y[0], PANEL_Y[1], z_hi, PANEL_TOP, mats["panel"])
    for k in range(0, len(ys), 2):
        _span(stage, f"{root}/plate_post{k}", X, X + PANEL_T, ys[k], ys[k + 1], z_lo, z_hi, mats["panel"])
    for s in (0, 1):
        y0 = SLOT_Y[s]
        p = f"{root}/case{s}"
        z0 = BODY_Z - FLOOR_T
        iw = CASE_W / 2 - WALL
        _span(stage, f"{p}_floor", X, X + CASE_DEPTH, y0 - CASE_W / 2, y0 + CASE_W / 2, z0, BODY_Z, mats["case_steel"])
        _span(stage, f"{p}_roof", X, X + CASE_DEPTH, y0 - CASE_W / 2, y0 + CASE_W / 2, z0 + CASE_H - WALL, z0 + CASE_H,
              mats["case_steel"])
        for k, (ya, yb) in enumerate(((y0 - iw - WALL, y0 - iw), (y0 + iw, y0 + iw + WALL))):
            _span(stage, f"{p}_wall{k}", X, X + CASE_DEPTH, ya, yb, z0, z0 + CASE_H, mats["case_steel"])
        _span(stage, f"{p}_back", X + CASE_DEPTH - WALL, X + CASE_DEPTH, y0 - CASE_W / 2, y0 + CASE_W / 2, z0, z0 + CASE_H,
              mats["dark"])
        _span(stage, f"{p}_stand", X + 0.02, X + CASE_DEPTH - 0.01, y0 - 0.05, y0 + 0.05, 0.0, z0, mats["stand"])
        # Keepers beside the hubs (unit seated): catch and strut (the cam face is the panel).
        ppx, ppz = pivot_world(s)
        for k, sg in enumerate((-1, 1)):
            def span_y(r_in, r_out):
                """y range of the band between distances r_in and r_out from the relay's centre line, this side."""
                return tuple(sorted((y0 + sg * r_in, y0 + sg * r_out)))

            # Handle bands (distance from the centre line): 0 = arm and tooth, 1 = hook and lug, 2 = free.
            edge = [HUB_Y - HUB_T / 2 + j * HUB_T for j in range(4)]
            q = f"{p}_keeper{k}"
            # Every keeper stays KEEPER_CLEAR clear of the handle bands it must not touch and of the bezel's side.
            # (With the faces coplanar, PhysX stopped the bail arm on the catch at ~19 deg, and a unit left slightly
            # yawed by a hard pull caught its bezel on the cam keeper and stuck ~1.7 mm proud.)
            ya, yb = span_y(edge[1] + KEEPER_CLEAR, edge[3])
            _span(stage, f"{q}_catch", ppx + CATCH_X[0], ppx + CATCH_X[1], ya, yb,
                  ppz + CATCH_Z[0], ppz + CATCH_Z[1], mats["dark"])
            ya, yb = span_y(edge[2] + KEEPER_CLEAR, edge[3])
            _span(stage, f"{q}_strut", ppx + CATCH_X[1], X, ya, yb, ppz + STRUT_Z[0], ppz + STRUT_Z[1],
                  mats["dark"])
        # Designation label above the relay (engraved laminate, colour marker on its right end).
        _quad_x(stage, f"{root}/label{s}", X - 0.0017, y0, LABEL_Z, LABEL_SIZE, labels[s])
        _span(stage, f"{root}/label{s}_edge", X - 0.0016, X, y0 - LABEL_SIZE[0] / 2, y0 + LABEL_SIZE[0] / 2,
              LABEL_Z - LABEL_SIZE[1] / 2, LABEL_Z + LABEL_SIZE[1] / 2, mats["dark"], collide=False)
    # Bench mat in front of slot 0: two strips with a gap for the supporting fingers, on a stand.
    _, my = mat_center(0)
    for k, (xa, xb) in enumerate(mat_strips(0)):
        _span(stage, f"{root}/mat{k}", xa, xb, my - 0.075, my + 0.075, MAT_TOP - 0.006, MAT_TOP, mats["mat"])
        _span(stage, f"{root}/mat{k}_stand", xa + 0.003, xb - 0.003, my - 0.06, my + 0.06, 0.0, MAT_TOP - 0.006,
              mats["stand"])
    kit.friction(stage, root, [f"{root}/mat0", f"{root}/mat1"], static=0.9, dynamic=0.8, name="rubber", combine="max")
    _contacts(stage, f"{root}/case")
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def write_labels(out_dir: Path) -> None:
    """Label textures: white laminate face, black engraved letters (two lines), a round colour marker at the right
    end (the left arm's gripper at the ready pose hides the left end of the far relay's label from the top camera)."""
    from PIL import Image, ImageDraw, ImageFont

    w = 1200
    h = round(w * LABEL_SIZE[1] / LABEL_SIZE[0])
    to8 = lambda c: tuple(int(round(v * 255)) for v in c)  # noqa: E731
    for s, (big, small) in enumerate(LABELS):
        img = Image.new("RGB", (w, h), (243, 243, 238))
        draw = ImageDraw.Draw(img)
        draw.rectangle([6, 6, w - 7, h - 7], outline=(20, 20, 20), width=6)
        r = h * 0.43
        cx, cy = w - h * 0.52, h / 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=to8(MARKER_RGB[s]), outline=(20, 20, 20), width=4)
        x0, x1 = 0, w - h * 1.0
        draw.text(((x0 + x1) / 2, h * 0.36), big, font=ImageFont.truetype(FONT, int(h * 0.42)), fill=(15, 15, 15),
                  anchor="mm")
        draw.text(((x0 + x1) / 2, h * 0.78), small, font=ImageFont.truetype(FONT, int(h * 0.22)), fill=(15, 15, 15),
                  anchor="mm")
        img.save(out_dir / f"label_{s}.png")


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


def _quad_x(stage, path, x, yc, zc, size, material):
    """A textured quad in the plane x = const, facing -x (readable from the robot's side)."""
    hy, hz = size[0] / 2, size[1] / 2
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = [(x, yc + hy, zc - hz), (x, yc - hy, zc - hz), (x, yc - hy, zc + hz), (x, yc + hy, zc + hz)]
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_unit(out_dir / "unit.usda")
    write_labels(out_dir)
    make_panel(out_dir / "panel.usda")

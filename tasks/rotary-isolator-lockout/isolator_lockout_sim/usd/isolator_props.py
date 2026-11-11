"""Generate the props of the rotary isolator lockout task as USD.

Real parts: ABB OT16 3-pole switch-disconnector (operating torque 1 N m) behind an ABB OHB45J6 pistol handle (black,
45 mm handle, padlockable in the OFF position with up to three padlocks, bail 5-8 mm per ABB's OT catalogue, 5-10 mm
per the handle's leaflet). Each isolator sits in a small grey enclosure (100 x 150 x 90 mm, our size) with the handle
on its front, as local isolators are mounted beside machines. The handle's outline (visual) is ABB's own CAD model of
the OHB45J6 (``ohb45j6_mesh.npz``, from ABB's published STEP file, in the handle frame below).

How the real handle locks (ABB OH_ handle leaflet, steps 1-3; product photos): turned to OFF, a red padlock slide in
the grip's lower edge is pulled out; it only comes out at OFF, and with a shackle through its hole it cannot be pushed
back, so the handle cannot be turned to ON. The slide's internal shape is not published. Modelled (inferred): the
slide (a stripe along the grip's lower edge ending in a tab at the tip) moves straight out of the grip's lower edge;
a tail behind the tab reaches back to a fixed stop behind the handle (an invisible annulus sector, r 45-65 mm) with a
straight slot only at the OFF position. Pulled out at OFF, the tail runs down the slot and the handle cannot turn; at
any other angle the stop keeps the slide in. A shackle in the tab's hole meets the grip's lower edge if the slide is
pushed back, before the tail leaves the slot.

Handle frame (per isolator): origin on the handle's axis at the enclosure's front face; +x into the enclosure (world
+x); at OFF (joint angle 0) the grip points +y (left, as the robot sees it) and the slide comes out along -z (down).
The joint angle q is about +x: ON is +90 deg (grip pointing up), a clockwise turn as you face the handle.
Depth d in front of the enclosure is x = -d.

Parts:
- `isolators.usda`: a fixed-base articulation. Root = the backboard, both enclosures, their nameplates and the fixed
  fixed stops. Per isolator: `handle_<k>` on a revolute joint about x (0 = OFF, 90 deg = ON; the end stops are joint
  limits), `slide_<k>` on a prismatic joint along the handle's z (0 = in, -SLIDE_TRAVEL = out).
- `pedestal.usda`: the block the open padlock lies on, and the parking shelf.
- `padlock_ml410.usda`: the Master Lock 410 (same model as the other lockout tasks).

Usage: python -m isolator_lockout_sim.usd.isolator_props isolator_lockout_sim/assets
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from isolator_lockout_sim.spec import LOADS, PLATE_COLORS, TAGS
from isolator_lockout_sim.usd import padlock_410, padlock_props
from isolator_lockout_sim.usd import usd_kit as kit

# Layout (world, desk frame) -----------------------------------------------------------------------------------
FACE_X = 0.31
"""World x of both enclosures' front faces."""
SLOT_Y = (0.0, -0.24)
"""World y of the handle axes: slot 0 (Q1, on the midline) and slot 1 (Q2, 24 cm right)."""
AXIS_Z = 0.205
"""World z of both handle axes."""
ENC_W, ENC_H, ENC_D = 0.100, 0.150, 0.090
BOARD_X = FACE_X + ENC_D
BOARD_T = 0.018
BOARD_Y = (-0.32, 0.08)
BOARD_H = 0.33

# Handle (handle frame, metres) ---------------------------------------------------------------------------------
HUB_R = 0.0285
"""Hub collider (octagon apothem); the CAD hub cone is 30-33 mm in radius."""
HUB_D = (0.012, 0.030)
GRIP_Y = (-0.020, 0.044)
"""Grip collider along the grip (+y at OFF): 44 mm from the axis to the tip (ABB: 45 mm handle)."""
GRIP_HW = 0.019
GRIP_D = (0.030, 0.045)
"""The grip stands 30-45 mm in front of the enclosure (ABB CAD: 45 mm overall depth)."""
HANDLE_MASS = 0.10
"""ABB catalogue: 0.1 kg."""

# Padlock slide (slide frame = handle frame with the slide in; r = -z below the axis, depth d = -x) ---------------
# As on the real handle, the red slide lies in the grip's lower edge, flush, from the hub to the grip's tip (a red
# stripe), and comes out at the tip: its outer end is a flat tab, hidden inside the grip except for a 16 mm finger tab
# below the tip (enlarged so the robot's fingers can take it). Pulled out at OFF, the whole slide drops SLIDE_TRAVEL and
# the tab's padlock hole shows below the grip's outer half.
SLIDE_TRAVEL = 0.032
BAR_Y = (0.020, 0.042)
BAR_R = (0.013, 0.0195)
BAR_D = (0.031, 0.044)
TAB_Y = (0.036, 0.042)
"""The tab (and the hole's plane) near the grip's tip; its hole's axis runs along the grip."""
TAB_YC = (TAB_Y[0] + TAB_Y[1]) / 2
SLIDE_T = TAB_Y[1] - TAB_Y[0]
TAB_R = (-0.015, 0.035)
"""Radial span of the tab with the slide in: from 15 mm above the axis (inside the grip) to 16 mm below the grip."""
TAB_D = (0.029, 0.046)
HOLE = 0.0085
HOLE_D = 0.0375
HOLE_R = 0.008
"""Hole centre with the slide in (hidden in the grip); out, it is at r = 40 mm, its top 17 mm below the grip."""
TAIL_D = (0.004, 0.029)
TAIL_R = (0.004, 0.012)
TAIL_Y = (0.036, 0.040)
"""Narrower than the tab, so that pushed in it clears the fixed stop by 3 mm as the handle turns."""
SLIDE_MASS = 0.010

# Fixed stop behind the grip (base, handle frame; invisible: it stands for the handle's internal stop) -----------------
# An annulus sector (r 45-65 mm, depth 0-11 mm, from 100 deg below to 70 deg above the grip's OFF direction) with a
# straight slot cut where the slide's tail comes down at OFF. Pulled out at OFF, the tail runs down the slot and the
# handle cannot turn (about 3 deg of play); at any other angle the annulus stops the tail, so the slide stays in.
STOP_R = (0.045, 0.065)
STOP_DR = 0.005
STOP_D = (0.0, 0.011)
STOP_ANGLES_DEG = (-100.0, 70.0)
STOP_DA_DEG = 4.0
STOP_SLOT_Y = (0.033, 0.045)
STOP_SLOT_TOP = -0.010
"""The slot (handle frame at OFF): 12 mm wide around the tail, from 10 mm below the axis down."""

CONTACT_OFFSET = 0.001
SOLVER_ITERATIONS = 32

# Nameplates (engraved laminate, ours) ----------------------------------------------------------------------------
PLATE_W, PLATE_H, PLATE_T = 0.086, 0.030, 0.0016
PLATE_DZ = 0.054

# Padlock stand -----------------------------------------------------------------------------------------------------
LOCK_START = (0.25, 0.17)
"""World (x, y) of the open padlock's centre on its stand."""
LOCK_START_Z = AXIS_Z - 0.040
STAND_TOP = LOCK_START_Z - 0.0095
PARK_X, PARK_Z = 0.30, -0.30

COLORS = dict(kit.COLORS)
COLORS.update({
    "handle_black": (0.06, 0.06, 0.065),
    "slide_red": (0.85, 0.07, 0.05),
    "enclosure": (0.78, 0.79, 0.77),
    "board": (0.55, 0.52, 0.46),
    "plate_black": (0.06, 0.06, 0.06),
    "plate_yellow": (0.96, 0.78, 0.10),
    "plate_white": (0.93, 0.93, 0.91),
})


def handle_origin(slot: int) -> tuple[float, float, float]:
    return (FACE_X, SLOT_Y[slot], AXIS_Z)


def hole_world(slot: int) -> tuple[float, float, float]:
    """World centre of a slide's hole at OFF with the slide out."""
    return (FACE_X - HOLE_D, SLOT_Y[slot] + TAB_YC, AXIS_Z - HOLE_R - SLIDE_TRAVEL)


def _xbox(stage, path, center, size, material, angle_deg=0.0, collide=True):
    """A box rotated about x by angle_deg (size given in its own frame)."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddRotateXOp().Set(angle_deg)
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _span_box(stage, path, xs, ys, zs, material, collide=True):
    c = ((xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2, (zs[0] + zs[1]) / 2)
    s = (abs(xs[1] - xs[0]), abs(ys[1] - ys[0]), abs(zs[1] - zs[0]))
    return kit.box(stage, path, c, s, material, collide=collide)


def _mats(stage, root):
    out = {}
    for name, rgb in COLORS.items():
        path = f"{root}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45 if name != "handle_black" else 0.35)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        out[name] = material
    return out


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


def write_textures(out_dir: Path) -> None:
    """Engraved two-ply laminate nameplates: designation and load, black letters."""
    big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 112)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    w, h = 516, 180
    bg = {"yellow": (245, 200, 30), "white": (240, 240, 236)}
    for k, tag in enumerate(TAGS):
        img = Image.new("RGB", (w, h), bg[PLATE_COLORS[k]])
        d = ImageDraw.Draw(img)
        d.rectangle([5, 5, w - 6, h - 6], outline=(20, 20, 20), width=6)
        d.text((22, h / 2), tag, fill=(15, 15, 15), font=big, anchor="lm")
        d.text((w - 22, h / 2), LOADS[k].replace(" ", "\n"), fill=(15, 15, 15), font=small, anchor="rm",
               align="center", spacing=4)
        img.save(out_dir / f"iso_nameplate_{k}.png")


# Engraved letters as thin black strokes (geometry, not a texture: in some runs the renderer showed another
# process's texture on a textured plate). Glyph strokes in mm: u to the left as you face the plate, v up.
_GLYPHS = {
    "Q": [((0, 10), (0, -10)), ((-13, 10), (-13, -10)), ((0, 10), (-13, 10)), ((0, -10), (-13, -10)),
          ((-7, -4), (-14, -12))],
    "1": [((-5, 10), (-5, -10)), ((-5, 10), (-1, 6))],
    "2": [((0, 10), (-10, 10)), ((-10, 10), (-10, 0)), ((-10, 0), (0, 0)), ((0, 0), (0, -10)), ((0, -10), (-10, -10))],
}
_GLYPH_W = {"Q": 14, "1": 8, "2": 11}
STROKE = 0.0026


def _engrave(stage, path, sy, zc, text, mat):
    fx = FACE_X - PLATE_T - 0.0004
    total = sum(_GLYPH_W[c] for c in text) + 5 * (len(text) - 1)
    u0 = total / 2.0
    k = 0
    for c in text:
        for (ua, va), (ub, vb) in _GLYPHS[c]:
            ya, yb = sy + (u0 + ua) / 1000.0, sy + (u0 + ub) / 1000.0
            za, zb = zc + va / 1000.0, zc + vb / 1000.0
            length = math.hypot(yb - ya, zb - za) + STROKE
            ang = math.degrees(math.atan2(zb - za, yb - ya))
            _xbox(stage, f"{path}_{k}", (fx, (ya + yb) / 2, (za + zb) / 2), (0.0006, length, STROKE), mat, ang,
                  collide=False)
            k += 1
        u0 -= _GLYPH_W[c] + 5


def _handle_mesh(stage, path, material):
    data = np.load(Path(__file__).with_name("ohb45j6_mesh.npz"))
    v = data["vertices"].astype(float)
    # The file is in a handle frame whose grip points -y; turn it 180 deg about x so the grip points +y at OFF.
    v = np.c_[v[:, 0], -v[:, 1], -v[:, 2]]
    faces = data["faces"]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in v]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.reshape(-1).tolist()))
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _thin_contacts(stage, prefix):
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def _in_slot(y, z):
    return STOP_SLOT_Y[0] <= y <= STOP_SLOT_Y[1] and z <= STOP_SLOT_TOP


def _slot_ring(stage, b, slot, mat):
    """The fixed stop: polar cells of the annulus sector, minus those touching the slot."""
    ox, oy, oz = handle_origin(slot)
    xc = ox - (STOP_D[0] + STOP_D[1]) / 2
    k = 0
    n_r = round((STOP_R[1] - STOP_R[0]) / STOP_DR)
    n_a = round((STOP_ANGLES_DEG[1] - STOP_ANGLES_DEG[0]) / STOP_DA_DEG)
    for ir in range(n_r):
        r0 = STOP_R[0] + ir * STOP_DR
        r1, rc = r0 + STOP_DR, r0 + STOP_DR / 2
        for ia in range(n_a):
            a0 = STOP_ANGLES_DEG[0] + ia * STOP_DA_DEG
            a1, ac = a0 + STOP_DA_DEG, a0 + STOP_DA_DEG / 2
            pts = [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a))) for r in (r0, rc, r1)
                   for a in (a0, ac, a1)]
            if any(_in_slot(y, z) for y, z in pts):
                continue
            ar = math.radians(ac)
            chord = 2 * r1 * math.tan(math.radians(STOP_DA_DEG / 2)) + 0.0004
            # Local box axes: x depth, y radial (rotated by ac about x), z tangential.
            _xbox(stage, f"{b}/stop{slot}_{k}", (xc, oy + rc * math.cos(ar), oz + rc * math.sin(ar)),
                  (STOP_D[1] - STOP_D[0], STOP_DR, chord), mat, ac)
            UsdGeom.Imageable(stage.GetPrimAtPath(f"{b}/stop{slot}_{k}")).MakeInvisible()
            k += 1


def make_isolators(out_path: Path) -> None:
    root = "/isolators"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=True)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
        SOLVER_ITERATIONS)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    m = _mats(stage, root)
    base = kit.rigid_body(stage, f"{root}/base", (0, 0, 0), 5.0)
    kit.fix_to_world(stage, root, base.GetPath())
    b = f"{root}/base"
    # Backboard standing on the table, with two feet.
    _span_box(stage, f"{b}/board", (BOARD_X, BOARD_X + BOARD_T), BOARD_Y, (0.0, BOARD_H), m["board"])
    for k, y in enumerate((BOARD_Y[0] + 0.04, BOARD_Y[1] - 0.04)):
        _span_box(stage, f"{b}/board_foot{k}", (BOARD_X - 0.10, BOARD_X + BOARD_T + 0.04), (y - 0.02, y + 0.02),
                  (0.0, 0.015), m["board"])
    for slot, sy in enumerate(SLOT_Y):
        # Enclosure (one box) and its lid seam.
        _span_box(stage, f"{b}/enclosure{slot}", (FACE_X, BOARD_X), (sy - ENC_W / 2, sy + ENC_W / 2),
                  (AXIS_Z - ENC_H / 2, AXIS_Z + ENC_H / 2), m["enclosure"])
        _span_box(stage, f"{b}/lid_seam{slot}", (FACE_X + 0.012, FACE_X + 0.014), (sy - ENC_W / 2 - 0.0005,
                  sy + ENC_W / 2 + 0.0005), (AXIS_Z - ENC_H / 2 - 0.0005, AXIS_Z + ENC_H / 2 + 0.0005), m["grey"],
                  collide=False)
        # Nameplate, engraved laminate, on the front above the handle.
        zc = AXIS_Z + PLATE_DZ
        plate_mat = m["plate_yellow"] if PLATE_COLORS[slot] == "yellow" else m["plate_white"]
        _span_box(stage, f"{b}/plate{slot}", (FACE_X - PLATE_T, FACE_X), (sy - PLATE_W / 2, sy + PLATE_W / 2),
                  (zc - PLATE_H / 2, zc + PLATE_H / 2), plate_mat, collide=False)
        _engrave(stage, f"{b}/plate{slot}_text", sy, zc, TAGS[slot], m["plate_black"])
        _slot_ring(stage, b, slot, m["handle_black"])

        # Handle: body origin on the axis at the enclosure face.
        hv = f"{root}/handle_{slot}"
        kit.rigid_body(stage, hv, handle_origin(slot), HANDLE_MASS)
        _handle_mesh(stage, f"{hv}/visual", m["handle_black"])
        for k, ang in enumerate((0.0, 45.0)):
            _xbox(stage, f"{hv}/hub{k}", (-(HUB_D[0] + HUB_D[1]) / 2, 0, 0),
                  (HUB_D[1] - HUB_D[0], 2 * HUB_R, 2 * HUB_R), m["handle_black"], ang)
            UsdGeom.Imageable(stage.GetPrimAtPath(f"{hv}/hub{k}")).MakeInvisible()
        _span_box(stage, f"{hv}/grip", (-GRIP_D[1], -GRIP_D[0]), GRIP_Y, (-GRIP_HW, GRIP_HW), m["handle_black"])
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{hv}/grip")).MakeInvisible()
        # The red insert along the grip's lower edge (the slide's cover, as on the real handle).
        kit.revolute(stage, f"{root}/joints/handle_{slot}_joint", base.GetPath(), hv, "X", handle_origin(slot),
                     0.0, 90.0)
        # Padlock slide: body origin = handle origin (slide in).
        sl = f"{root}/slide_{slot}"
        kit.rigid_body(stage, sl, handle_origin(slot), SLIDE_MASS)
        red = m["slide_red"]
        ty = TAB_Y
        h_lo, h_hi = HOLE_R - HOLE / 2, HOLE_R + HOLE / 2
        hd_lo, hd_hi = HOLE_D - HOLE / 2, HOLE_D + HOLE / 2
        d0, d1 = TAB_D
        r0, r1 = TAB_R
        # The stripe along the grip's lower edge (flush), and the tail behind the tab back to the base's stop.
        _span_box(stage, f"{sl}/bar", (-BAR_D[1], -BAR_D[0]), BAR_Y, (-BAR_R[1], -BAR_R[0]), red)
        _span_box(stage, f"{sl}/tail", (-TAIL_D[1], -TAIL_D[0]), TAIL_Y, (-TAIL_R[1], -TAIL_R[0]), red)
        # The tab at the tip, with its padlock hole.
        _span_box(stage, f"{sl}/tab_above_hole", (-d1, -d0), ty, (-h_lo, -r0), red)
        _span_box(stage, f"{sl}/tab_below_hole", (-d1, -d0), ty, (-r1, -h_hi), red)
        _span_box(stage, f"{sl}/tab_front_strip", (-d1, -hd_hi), ty, (-h_hi, -h_lo), red)
        _span_box(stage, f"{sl}/tab_back_strip", (-hd_lo, -d0), ty, (-h_hi, -h_lo), red)
        # Finger ridges on the finger tab (visual).
        for k, z in enumerate((-r1 + 0.003, -r1 + 0.007, -r1 + 0.011)):
            _span_box(stage, f"{sl}/ridge{k}", (-d1 - 0.0006, -d0 + 0.0006), (ty[0] - 0.0006, ty[1] + 0.0006),
                      (z - 0.0006, z + 0.0006), m["dark_red"], collide=False)
        kit.prismatic(stage, f"{root}/joints/slide_{slot}_joint", hv, sl, "Z", (0, 0, 0), -SLIDE_TRAVEL, 0.0)
    _thin_contacts(stage, root)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_pedestal(out_path: Path) -> None:
    """The padlock lies flat on a small grey post (under the shackle half of its body), and the parking shelf under
    the desk."""
    root = "/pedestal"
    stage, _ = kit.new_stage(out_path, root)
    mats = kit.materials(stage, root, ("stand",))
    x, y = LOCK_START
    _span_box(stage, f"{root}/foot", (x - 0.03, x + 0.03), (y - 0.04, y + 0.02), (0.0, 0.01), mats["stand"])
    # One post under the shackle half of the body (the gripper takes the body's other half).
    _span_box(stage, f"{root}/post", (x - 0.008, x + 0.008), (y - 0.016, y - 0.002), (0.0, STAND_TOP),
              mats["stand"])
    kit.box(stage, f"{root}/park_shelf", (PARK_X, 0.0, PARK_Z - 0.005), (0.12, 0.60, 0.01), mats["stand"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    make_isolators(out_dir / "isolators.usda")
    make_pedestal(out_dir / "pedestal.usda")
    padlock_410.make_padlock_410(out_dir / "padlock_ml410.usda", padlock_props.PADLOCKS["ml410"])

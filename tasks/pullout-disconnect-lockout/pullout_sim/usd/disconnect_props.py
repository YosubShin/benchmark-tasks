"""Generate the props of the pull-out disconnect task as USD (dimensions in pullout_sim/spec.py).

- `disconnect_ac1.usda`, `disconnect_ac2.usda`, fixed-base articulations: a Milbank U3802 pull-out disconnect each.
  Root = the body (back, sides, top, bottom, the dead front with the pull-out's window, the recess the block slides
  in, four live clips and four insulated dummy pockets behind it, and the padlock tab out of the bottom). The cover
  (front plate with the AC-n label and a slot for the tab, hood, side skirts) turns on a revolute joint about the
  hinge pin (along y) near the top rear; positive = opening, stop at COVER_OPEN_LIMIT_DEG; shut = the joint's lower
  limit (where the real cover's edges rest on the body).
  Clips: each is two fixed fingers whose faces stand CLIP_GRIP_M inside the blade's faces, with a compliant-contact
  material (the contact force is a spring on the penetration: the clip's spring), and angled lead-ins. Pockets: the
  same shape with POCKET_CLEAR of play and no spring.
- `block.usda`, a rigid body: the pull-out (black body, grip bar, two copper jumpers = four blades on its back,
  ON / OFF marks on its face). Seated ON its blades are in the clips; turned half a turn about x (OFF) in the
  pockets.
- `fixtures.usda`, static: the plywood board, the padlock's pedestal, a parking shelf under the desk, conduits
  (visual).
- `padlock_ml410.usda`: the robot's red Master Lock 410 (articulated, sprung shackle), pullout_sim/usd/padlock_410.py.

Frames: disconnect_ac<n>.usda's origin is (0, BOX_Y[n], 0) in the desk frame (x and z are desk values); the block's
origin is its centre (x into the box, z up when ON); fixtures.usda is in the desk frame.

Usage: python -m pullout_sim.usd.disconnect_props <out dir>
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from pullout_sim import spec as S
from pullout_sim.usd import padlock_410, padlock_props
from pullout_sim.usd import usd_kit as kit

PARK_X, PARK_Z = 0.30, kit.PARK_Z
COLORS = {
    "box_grey": (0.60, 0.62, 0.62),
    "dead_front": (0.66, 0.68, 0.68),
    "insulator": (0.10, 0.10, 0.10),
    "block_black": (0.05, 0.05, 0.055),
    "copper": (0.80, 0.45, 0.22),
    "steel": (0.72, 0.74, 0.77),
    "black": (0.03, 0.03, 0.03),
    "white": (0.93, 0.93, 0.91),
    "board": (0.76, 0.66, 0.50),
    "stand": (0.40, 0.40, 0.42),
    "conduit": (0.50, 0.52, 0.55),
    "plate_yellow": (0.96, 0.78, 0.12),
    "plate_white": (0.94, 0.94, 0.92),
}
CONTACT_OFFSET = 0.002
SOLVER_ITERATIONS = 32
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LEAD_IN_DEG = 25.0
LEAD_IN_LEN = 0.004
CLIP_X = (S.BLOCK_SEAT_X + S.BLOCK_SIZE[0] / 2 + 0.0045, S.BLOCK_SEAT_X + S.BLOCK_SIZE[0] / 2 + S.BLADE_LEN + 0.003)
"""Clip fingers' x range: their angled lead-ins end just behind the seated block's back face."""
FINGER_T = 0.0015


def _mats(stage, root, names):
    out = {}
    for name in names:
        path = f"{root}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[name]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4 if name in ("copper", "steel") else 0.6)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        out[name] = material
    return out


def _physics_material(stage, root, name, mu, combine="min", compliant=None):
    mat = UsdShade.Material.Define(stage, f"{root}/PhysicsMaterials/{name}")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(mu)
    api.CreateDynamicFrictionAttr(mu)
    prim = mat.GetPrim()
    prim.AddAppliedSchema("PhysxMaterialAPI")
    prim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(combine)
    if compliant is not None:
        k, c = compliant
        prim.CreateAttribute("physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float).Set(k)
        prim.CreateAttribute("physxMaterial:compliantContactDamping", Sdf.ValueTypeNames.Float).Set(c)
    return mat


def _bind_physics(stage, path, mat):
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(mat, UsdShade.Tokens.weakerThanDescendants,
                                                                       "physics")


def span(stage, path, xr, yr, zr, material, collide=True):
    """A box given by its x, y and z ranges."""
    return kit.box(stage, path, ((xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2, (zr[0] + zr[1]) / 2),
                   (xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]), material, collide=collide)


def rbox(stage, path, center, size, yaw_deg, material, collide=True):
    """A box turned about z."""
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


def _contacts(stage, prefix="/"):
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def _text_runs(text: str, px: int = 14) -> tuple[list, int, int]:
    font = ImageFont.truetype(FONT, px)
    l, t, r, b = font.getbbox(text)
    w, h = r - l + 2, b - t + 2
    img = Image.new("1", (w, h), 0)
    ImageDraw.Draw(img).text((1 - l, 1 - t), text, fill=1, font=font)
    runs = []
    for row in range(h):
        col = 0
        while col < w:
            if img.getpixel((col, row)):
                c0 = col
                while col < w and img.getpixel((col, row)):
                    col += 1
                runs.append((row, c0, col))
            else:
                col += 1
    return runs, w, h


def text_boxes(stage, path, text, center, height, material, px=14, upside_down=False):
    """Letters as thin visual boxes (geometry, not a texture) on a plane facing -x, centred at ``center``, letters
    ``height`` tall; seen from -x the image's left is +y. ``upside_down`` turns the text half a turn in its plane."""
    runs, w, h = _text_runs(text, px)
    pix = height / h
    x, yc, zc = center
    sgn = -1.0 if upside_down else 1.0
    UsdGeom.Xform.Define(stage, path)
    for i, (row, c0, c1) in enumerate(runs):
        y0 = yc + sgn * (w / 2 - c0) * pix
        y1 = yc + sgn * (w / 2 - c1) * pix
        z = zc + sgn * (h / 2 - row - 0.5) * pix
        span(stage, f"{path}/r{i}", (x - 0.0003, x), (min(y0, y1), max(y0, y1)), (z - pix / 2, z + pix / 2), material,
             collide=False)


def _clip(stage, path, y, z, xr, mat, grip, phys):
    """Two fingers either side of a blade's path at (y, z), faces ``grip`` inside the blade's faces (negative =
    play), with lead-ins opening toward the block. Returns the finger prims' paths."""
    half_in = S.BLADE_T / 2 - grip
    h = S.BLADE_H + 0.003
    paths = []
    for k, s in enumerate((-1, 1)):
        yi = y + s * half_in
        p = span(stage, f"{path}_f{k}", xr, (min(yi, yi + s * FINGER_T), max(yi, yi + s * FINGER_T)),
                 (z - h / 2, z + h / 2), mat)
        paths.append(p.GetPath())
        # Lead-in: a short plate turned outward at the fingers' front (toward -x).
        a = math.radians(LEAD_IN_DEG)
        cx = xr[0] - LEAD_IN_LEN / 2 * math.cos(a)
        cy = yi + s * (FINGER_T / 2 + LEAD_IN_LEN / 2 * math.sin(a))
        lp = rbox(stage, f"{path}_lead{k}", (cx, cy, z), (LEAD_IN_LEN, FINGER_T, h), -s * LEAD_IN_DEG, mat)
        paths.append(lp.GetPath())
    for p in paths:
        _bind_physics(stage, p, phys)
    return paths


def make_disconnect(out_path: Path, k: int) -> None:
    root = "/disconnect"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
        SOLVER_ITERATIONS)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    m = _mats(stage, root, COLORS)
    clip_mat = _physics_material(stage, root, "clip", 0.30, "min", compliant=(S.CLIP_STIFFNESS, 20.0))
    plastic = _physics_material(stage, root, "plastic", 0.30, "min")
    steel = _physics_material(stage, root, "steel_sheet", 0.40, "min")
    body = kit.rigid_body(stage, f"{root}/body", (0, 0, 0), 2.0)
    kit.fix_to_world(stage, root, body.GetPath())
    b = f"{root}/body"
    W, t = S.BOX_W / 2, S.SHEET
    xf, xb = S.BODY_FRONT, S.BOARD_X
    z0, z1 = S.BOX_Z
    g = m["box_grey"]
    span(stage, f"{b}/back", (xb - t, xb), (-W, W), (z0, z1), g)
    span(stage, f"{b}/side_l", (xf, xb), (W - t, W), (z0, z1), g)
    span(stage, f"{b}/side_r", (xf, xb), (-W, -W + t), (z0, z1), g)
    span(stage, f"{b}/top", (xf, xb), (-W, W), (z1 - t, z1), g)
    span(stage, f"{b}/bottom", (xf, xb), (-W, W), (z0, z0 + t), g)
    # Dead front with the window for the pull-out.
    wy, (wz0, wz1) = S.WINDOW_Y, S.WINDOW_Z
    df = m["dead_front"]
    span(stage, f"{b}/df_below", (xf, xf + t), (-W, W), (z0, wz0), df)
    span(stage, f"{b}/df_above", (xf, xf + t), (-W, W), (wz1, z1), df)
    span(stage, f"{b}/df_left", (xf, xf + t), (wy, W), (wz0, wz1), df)
    span(stage, f"{b}/df_right", (xf, xf + t), (-W, -wy), (wz0, wz1), df)
    # The printed label on the dead front, below the window (visual).
    span(stage, f"{b}/df_label", (xf - 0.0004, xf), (-0.050, 0.050), (z0 + 0.012, wz0 - 0.010), m["white"],
         collide=False)
    # The recess the block slides in (insulating base), and the stops its back face rests on when seated.
    xr0, xr1 = xf + t, S.BLOCK_SEAT_X + S.BLOCK_SIZE[0] / 2
    ins = m["insulator"]
    span(stage, f"{b}/recess_floor", (xr0, xb - t), (-wy, wy), (wz0 - 0.004, wz0), ins)
    span(stage, f"{b}/recess_ceiling", (xr0, xb - t), (-wy, wy), (wz1, wz1 + 0.004), ins)
    span(stage, f"{b}/recess_l", (xr0, xb - t), (wy, wy + 0.004), (wz0, wz1), ins)
    span(stage, f"{b}/recess_r", (xr0, xb - t), (-wy - 0.004, -wy), (wz0, wz1), ins)
    for name in ("recess_floor", "recess_ceiling", "recess_l", "recess_r"):
        _bind_physics(stage, f"{b}/{name}", plastic)
    stop_x = (xr1, CLIP_X[0] - 0.0005)
    for j, (ya, yb) in enumerate(((-wy, -0.042), (-0.006, 0.006), (0.042, wy))):
        span(stage, f"{b}/stop{j}", (stop_x[0], xb - t), (ya, yb), (wz0, wz1), ins)
        _bind_physics(stage, f"{b}/stop{j}", plastic)
    # Line/load clips (live, z above the centre) and dummy pockets (z below), with the line and load straps behind.
    zc = S.BLOCK_Z
    for j, y in enumerate(S.BLADE_Y):
        _clip(stage, f"{b}/clip{j}", y, zc + S.BLADE_Z, CLIP_X, m["copper"], S.CLIP_GRIP_M, clip_mat)
        _clip(stage, f"{b}/pocket{j}", y, zc - S.BLADE_Z, CLIP_X, ins, -S.POCKET_CLEAR, plastic)
        span(stage, f"{b}/strap{j}", (CLIP_X[1], xb - t), (y - 0.003, y + 0.003),
             (zc + S.BLADE_Z - 0.003, zc + S.BLADE_Z + 0.003), m["copper"], collide=False)
    # Padlock tab out of the bottom, with its hole.
    tz = S.TAB_Z
    hx, hh, tw = S.TAB_HOLE_X, S.TAB_HOLE / 2, S.TAB_HALF_W
    span(stage, f"{b}/tab_base", (hx + hh, xf), (-tw, tw), tz, g)
    span(stage, f"{b}/tab_tip", (S.TAB_TIP_X, hx - hh), (-tw, tw), tz, g)
    span(stage, f"{b}/tab_side_l", (hx - hh, hx + hh), (hh, tw), tz, g)
    span(stage, f"{b}/tab_side_r", (hx - hh, hx + hh), (-tw, -hh), tz, g)
    for prim in stage.GetPrimAtPath(b).GetChildren():
        if prim.GetName().startswith(("back", "side", "top", "bottom", "df_", "tab_")) and prim.HasAPI(
                UsdPhysics.CollisionAPI):
            _bind_physics(stage, prim.GetPath(), steel)

    # Cover: body origin on the hinge axis.
    hx0, hz0 = S.HINGE_X, S.HINGE_Z
    cover = kit.rigid_body(stage, f"{root}/cover", (hx0, 0, hz0), S.COVER_MASS)
    cv = f"{root}/cover"

    def cspan(name, xr, yr, zr, mat, collide=True):
        return span(stage, f"{cv}/{name}", (xr[0] - hx0, xr[1] - hx0), yr, (zr[0] - hz0, zr[1] - hz0), mat, collide)

    cf, cw = S.COVER_FRONT, S.COVER_W / 2
    ctop = hz0 + 0.004
    plate_x = (cf, cf + t)
    sy, (sz0, sz1) = S.SLOT_HALF_W, S.SLOT_Z
    cspan("plate_upper", plate_x, (-cw, cw), (sz1, ctop), g)
    cspan("plate_left", plate_x, (sy, cw), (S.COVER_BOTTOM, sz1), g)
    cspan("plate_right", plate_x, (-cw, -sy), (S.COVER_BOTTOM, sz1), g)
    cspan("plate_bar", plate_x, (-sy, sy), (S.COVER_BOTTOM, sz0), g)
    lip_x = (cf + t, cf + t + S.LIP)
    cspan("lip_left", lip_x, (sy, cw), (S.COVER_BOTTOM, S.COVER_BOTTOM + t), g)
    cspan("lip_right", lip_x, (-cw, -sy), (S.COVER_BOTTOM, S.COVER_BOTTOM + t), g)
    cspan("hood", (cf, xb - 0.003), (-cw - t, cw + t), (ctop, ctop + t), g)
    for s, name in ((1, "skirt_l"), (-1, "skirt_r")):
        yr = (cw, cw + t) if s > 0 else (-cw - t, -cw)
        cspan(name, (cf, xf + 0.012), yr, (S.COVER_BOTTOM + 0.006, ctop), g)
    # Rain-hood drip edge and the hinge pin (visual).
    cspan("drip", (cf - 0.004, cf), (-cw - t, cw + t), (ctop - 0.010, ctop + t), g, collide=False)
    kit.cylinder(stage, f"{cv}/hinge_pin", (0, 0, 0), 0.003, S.COVER_W + 0.008, m["steel"], axis="Y")
    # The unit's label (engraved plate, AC-1 yellow / AC-2 white) on the cover's front.
    plate = m["plate_yellow" if k == 0 else "plate_white"]
    pz = (S.BOX_Z[0] + S.BOX_Z[1]) / 2 + 0.030
    cspan("label", (cf - 0.0012, cf), (-0.040, 0.040), (pz - 0.022, pz + 0.022), plate, collide=False)
    text_boxes(stage, f"{cv}/label_text", S.LABELS[k], (cf - 0.0013 - hx0, 0.0, pz - hz0), 0.022, m["black"])
    for prim in stage.GetPrimAtPath(cv).GetChildren():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            _bind_physics(stage, prim.GetPath(), steel)
    kit.revolute(stage, f"{root}/joints/cover_joint", body.GetPath(), cover.GetPath(), "Y", (hx0, 0, hz0), 0.0,
                 S.COVER_OPEN_LIMIT_DEG)
    _contacts(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_block(out_path: Path) -> None:
    root = "/block"
    stage, root_prim = kit.new_stage(out_path, root)
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(S.BLOCK_MASS)
    root_prim.AddAppliedSchema("PhysxRigidBodyAPI")
    root_prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    root_prim.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    root_prim.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(1.0)
    m = _mats(stage, root, ("block_black", "copper", "white"))
    dx, ly, hz = (v / 2 for v in S.BLOCK_SIZE)
    span(stage, f"{root}/body", (-dx, dx), (-ly, ly), (-hz, hz), m["block_black"])
    gx, gy, gz = S.GRIP_SIZE
    span(stage, f"{root}/grip", (-dx - gx, -dx), (-gy / 2, gy / 2), (-gz / 2, gz / 2), m["block_black"])
    # Finger recesses either side of the grip bar's ends (visual flare, as on the real pull-out).
    for s in (-1, 1):
        span(stage, f"{root}/grip_end{'lr'[s > 0]}", (-dx - gx * 0.6, -dx), (s * gy / 2, s * (gy / 2 + 0.006))
             if s > 0 else (s * (gy / 2 + 0.006), s * gy / 2), (-gz / 2 - 0.002, gz / 2 + 0.002), m["block_black"],
             collide=False)
    for j, y in enumerate(S.BLADE_Y):
        span(stage, f"{root}/blade{j}", (dx, dx + S.BLADE_LEN), (y - S.BLADE_T / 2, y + S.BLADE_T / 2),
             (S.BLADE_Z - S.BLADE_H / 2, S.BLADE_Z + S.BLADE_H / 2), m["copper"])
    # Jumper straps joining each pole's two blades across the back (visual).
    for j, (ya, yb) in enumerate(((S.BLADE_Y[0], S.BLADE_Y[1]), (S.BLADE_Y[2], S.BLADE_Y[3]))):
        span(stage, f"{root}/jumper{j}", (dx, dx + 0.002), (ya, yb), (S.BLADE_Z - S.BLADE_H / 2,
                                                                       S.BLADE_Z + S.BLADE_H / 2), m["copper"],
             collide=False)
    # ON (reads upright when the block is in ON) and OFF (upside down; reads upright when turned OFF), on the face
    # at the +y end.
    ye = ly - 0.012
    text_boxes(stage, f"{root}/mark_on", "ON", (-dx - 0.0002, ye, -0.010), 0.008, m["white"])
    text_boxes(stage, f"{root}/mark_off", "OFF", (-dx - 0.0002, -ye, 0.010), 0.008, m["white"], upside_down=True)
    kit.friction(stage, root, [root], static=0.8, dynamic=0.7, name="grip_plastic")
    for j in range(len(S.BLADE_Y)):
        mat = _physics_material(stage, root, "blade_copper", 0.30, "min")
        _bind_physics(stage, f"{root}/blade{j}", mat)
    _contacts(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_fixtures(out_path: Path) -> None:
    root = "/fixtures"
    stage, _ = kit.new_stage(out_path, root)
    m = _mats(stage, root, COLORS)
    bx, (y0, y1) = S.BOARD_X, S.BOARD_Y
    span(stage, f"{root}/board", (bx, bx + S.BOARD_T), (y0, y1), (0.0, S.BOARD_H), m["board"])
    for k, y in enumerate((y0 + 0.03, y1 - 0.03)):
        span(stage, f"{root}/board_foot{k}", (bx + S.BOARD_T, bx + 0.14), (y - 0.02, y + 0.02), (0.0, 0.03),
             m["board"])
    # Conduits from the boxes' bottoms down to the table (visual).
    for k, yb in enumerate(S.BOX_Y):
        for s in (-1, 1):
            kit.cylinder(stage, f"{root}/conduit{k}{'lr'[s > 0]}", (bx - 0.020, yb + s * 0.045, S.BOX_Z[0] / 2),
                         0.009, S.BOX_Z[0], m["conduit"])
    x, y = S.PEDESTAL_XY
    kit.box(stage, f"{root}/pedestal_foot", (x, y, 0.005), (0.06, 0.05, 0.01), m["stand"])
    kit.box(stage, f"{root}/pedestal_column", (x, y, S.PEDESTAL_TOP / 2), (0.030, 0.009, S.PEDESTAL_TOP), m["stand"])
    kit.box(stage, f"{root}/park_shelf", (PARK_X, 0.0, PARK_Z - 0.005), (0.12, 0.60, 0.01), m["stand"])
    _contacts(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    make_disconnect(out_dir / "disconnect_ac1.usda", 0)
    make_disconnect(out_dir / "disconnect_ac2.usda", 1)
    make_block(out_dir / "block.usda")
    make_fixtures(out_dir / "fixtures.usda")
    red = padlock_props.PadlockVariant("ml410", "Master Lock 410 Zenex, red (410RED)", (0.038, 0.019, 0.044), 0.00635,
                                       0.020, 0.038, 0.10, "lock_red")
    padlock_410.make_padlock_410(out_dir / "padlock_ml410.usda", red)

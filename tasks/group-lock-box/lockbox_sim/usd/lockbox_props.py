"""Generate the props of the group-lock-box task as USD (dimensions in lockbox_sim/spec.py).

- `lockbox.usda`, a fixed-base articulation: the Master Lock S601 group lock box on its riser. Root = the body
  (riser, floor, walls, and along both long sides a horizontal flange 17.5 mm wide with nine 11 x 14 mm holes); the
  lid (top with the key slot near the latch end, short skirts, and its own flanges with the same holes, lying on the
  body's flanges) turns on a revolute joint about the hinge pin at the left (+y) end. The task holds the lid shut as
  the latch does, until the after-release check. Nothing else holds it: a shackle through a lid-flange hole and the
  body-flange hole under it does, by collision (the shackle's loop around both flanges).
- `fixtures.usda`, static: the board with the two pump starter panels (nameplate, rotary handle and red job
  padlock, visual), a key peg under each panel, the robot padlock's pedestal and a parking shelf under the desk.
- `key.usda`, a rigid body: the job lock's key (bow with a hole, blade), hung on a peg by its bow hole.
- `padlock_ml410.usda` (the robot's purple Master Lock 410, articulated) and `coworker.usda` (the coworker's yellow
  410, one rigid body, shut, with its danger tag): lockbox_sim/usd/padlock_410.py.

Frames: lockbox.usda's origin is (BOX_X, BOX_Y, 0) in the desk frame; fixtures.usda and key.usda are in the desk frame
(key: origin at the bow hole's centre, the key's plane y-z, blade down along -z).

Usage: python -m lockbox_sim.usd.lockbox_props <out dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from lockbox_sim import spec as S
from lockbox_sim.usd import padlock_410, padlock_props
from lockbox_sim.usd import usd_kit as kit

PARK_X, PARK_Z = 0.30, kit.PARK_Z
COLORS = {
    "box_red": (0.78, 0.05, 0.04),
    "riser": (0.40, 0.41, 0.43),
    "window": (0.10, 0.11, 0.12),
    "steel": (0.72, 0.74, 0.77),
    "black": (0.04, 0.04, 0.04),
    "board": (0.76, 0.66, 0.50),
    "panel": (0.80, 0.80, 0.77),
    "brass": (0.80, 0.66, 0.30),
    "lock_red": (0.80, 0.10, 0.08),
    "stand": (0.40, 0.40, 0.42),
    "dark": (0.12, 0.12, 0.13),
    "plate_yellow": (0.96, 0.78, 0.12),
    "plate_white": (0.94, 0.94, 0.92),
}
CONTACT_OFFSET = 0.002
SOLVER_ITERATIONS = 32
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _mats(stage, root, names):
    out = {}
    for name in names:
        path = f"{root}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[name]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35 if name in ("box_red", "steel") else 0.6)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        out[name] = material
    return out


def span(stage, path, xr, yr, zr, material, collide=True):
    """A box given by its x, y and z ranges."""
    return kit.box(stage, path, ((xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2, (zr[0] + zr[1]) / 2),
                   (xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]), material, collide=collide)


def _thin_contacts(stage, prefix):
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def _text_runs(text: str, px: int = 14) -> tuple[list, int, int]:
    """The text as runs of dark pixels: [(row, col0, col1)], image width, height (rendered with PIL, 1 bit)."""
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


def text_boxes(stage, path, text, center, height, material, px=14):
    """Letters as thin visual boxes (geometry, not a texture) on a plane facing -x, centred at ``center`` (x, y, z),
    letters ``height`` tall. Seen from -x the image's left is +y. Textures were seen to be swapped for another
    scene's image by the renderer when several simulations shared the machine; geometry cannot be."""
    runs, w, h = _text_runs(text, px)
    pix = height / h
    x, yc, zc = center
    UsdGeom.Xform.Define(stage, path)
    for i, (row, c0, c1) in enumerate(runs):
        y0 = yc + (w / 2 - c0) * pix
        y1 = yc + (w / 2 - c1) * pix
        z = zc + (h / 2 - row - 0.5) * pix
        span(stage, f"{path}/r{i}", (x - 0.0003, x), (y1, y0), (z - pix / 2, z + pix / 2), material, collide=False)


def _flange(stage, path, side, zr, material):
    """A flange along a long side (side = -1 front, +1 back) with its nine holes, in the box frame."""
    w0, w1 = S.BODY_W / 2, S.BODY_W / 2 + S.FLANGE_W
    hc = S.BODY_W / 2 + S.HOLE_OUT
    hi, ho = hc - S.HOLE_X / 2, hc + S.HOLE_X / 2
    xr = lambda a, b: (side * b, side * a) if side < 0 else (a, b)  # noqa: E731
    L = S.BODY_L / 2
    span(stage, f"{path}_inner", xr(w0, hi), (-L, L), zr, material)
    span(stage, f"{path}_outer", xr(ho, w1), (-L, L), zr, material)
    ys = sorted((5 - k) * S.HOLE_PITCH for k in range(1, S.N_HOLES + 1))
    edges = [-L] + [e for y in ys for e in (y - S.HOLE_Y / 2, y + S.HOLE_Y / 2)] + [L]
    for i in range(0, len(edges), 2):
        span(stage, f"{path}_bridge{i // 2}", xr(hi, ho), (edges[i], edges[i + 1]), zr, material)


def make_lockbox(out_path: Path) -> None:
    root = "/lockbox"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
        SOLVER_ITERATIONS)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    m = _mats(stage, root, COLORS)
    body = kit.rigid_body(stage, f"{root}/body", (0, 0, 0), 2.0)
    kit.fix_to_world(stage, root, body.GetPath())
    b = f"{root}/body"
    W, L, t = S.BODY_W / 2, S.BODY_L / 2, S.WALL_T
    z0, zf = S.Z_BASE, S.Z_BODY_FLANGE[1]
    if z0 > 0:
        span(stage, f"{b}/riser", (-W - 0.012, W + 0.012), (-L - 0.01, L + 0.01), (0.0, z0), m["riser"])
    span(stage, f"{b}/floor", (-W, W), (-L, L), (z0, z0 + 0.002), m["box_red"])
    for s, name in ((-1, "front"), (1, "back")):
        span(stage, f"{b}/wall_{name}", (W - t, W) if s > 0 else (-W, -W + t), (-L, L), (z0, zf), m["box_red"])
        span(stage, f"{b}/end_{name}", (-W, W), (L - t, L) if s > 0 else (-L, -L + t), (z0, zf), m["box_red"])
        _flange(stage, f"{b}/flange_{name}", s, S.Z_BODY_FLANGE, m["box_red"])
    # Key window on the front near the latch end, and the latch's staple on the right end (visual).
    span(stage, f"{b}/window", (-W - 0.0008, -W), (-L + 0.012, -L + 0.095), (z0 + 0.025, z0 + 0.090), m["window"],
         collide=False)
    span(stage, f"{b}/latch_staple", (-0.012, 0.012), (-L - 0.008, -L), (zf - 0.035, zf - 0.010), m["box_red"],
         collide=False)
    # Lid: body origin on the hinge axis.
    hy, hz = S.HINGE_Y - S.BOX_Y, S.HINGE_Z
    lid = kit.rigid_body(stage, f"{root}/lid", (0, hy, hz), S.LID_MASS)
    lv = f"{root}/lid"

    def lspan(name, xr, yr, zr, mat, collide=True):
        return span(stage, f"{lv}/{name}", xr, (yr[0] - hy, yr[1] - hy), (zr[0] - hz, zr[1] - hz), mat, collide)

    zt = S.Z_TOP
    sx, sy = S.KEY_SLOT[0] - S.BOX_X, S.KEY_SLOT[1] - S.BOX_Y
    sl, sw = S.KEY_SLOT_SIZE
    top = (zt - t, zt)
    lspan("top_a", (-W, W), (sy + sw / 2, L + 0.001), top, m["box_red"])
    lspan("top_b", (-W, W), (-L - 0.001, sy - sw / 2), top, m["box_red"])
    lspan("top_c", (-W, sx - sl / 2), (sy - sw / 2, sy + sw / 2), top, m["box_red"])
    lspan("top_d", (sx + sl / 2, W), (sy - sw / 2, sy + sw / 2), top, m["box_red"])
    # The slot's plate with its two rivets (visual).
    lspan("slot_plate_l", (sx - sl / 2 - 0.010, sx - sl / 2), (sy - 0.008, sy + 0.008), (zt, zt + 0.0008),
          m["box_red"], collide=False)
    lspan("slot_plate_r", (sx + sl / 2, sx + sl / 2 + 0.010), (sy - 0.008, sy + 0.008), (zt, zt + 0.0008),
          m["box_red"], collide=False)
    zl = S.Z_LID_FLANGE[1]
    for s, name in ((-1, "front"), (1, "back")):
        lspan(f"skirt_{name}", (W - t, W) if s > 0 else (-W, -W + t), (-L, L), (zl, zt - t), m["box_red"])
        lspan(f"skirt_end_{name}", (-W, W), (L - t, L) if s > 0 else (-L, -L + t), (zl - t, zt - t), m["box_red"])
        _flange(stage, f"{lv}/flange_{name}", s, (S.Z_LID_FLANGE[0] - hz, S.Z_LID_FLANGE[1] - hz), m["box_red"])
    # The flanges were built in box coordinates: shift the lid's to its frame.
    for prim in list(stage.GetPrimAtPath(lv).GetChildren()):
        if prim.GetName().startswith("flange_"):
            op = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]
            p = op.Get()
            op.Set(Gf.Vec3d(p[0], p[1] - hy, p[2]))
    # Folding handle on the top (visual) and the hinge pin.
    lspan("handle_bar", (-0.006, 0.006), (-0.075, 0.075), (zt, zt + 0.007), m["box_red"], collide=False)
    lspan("latch", (-0.013, 0.013), (-L - 0.006, -L + 0.004), (zt - 0.030, zt + 0.001), m["box_red"], collide=False)
    kit.cylinder(stage, f"{lv}/hinge_pin", (0, 0, 0), 0.004, S.BODY_W + 0.02, m["steel"], axis="X")
    for k in range(1, S.N_HOLES + 1):
        yk = (5 - k) * S.HOLE_PITCH - hy
        text_boxes(stage, f"{lv}/num{k}", str(k), (-W - 0.0004, yk, (zl + zt) / 2 - hz), 0.011, m["black"])
    joint = kit.revolute(stage, f"{root}/joints/lid_joint", body.GetPath(), lid.GetPath(), "X", (0, hy, hz),
                         -100.0, 0.0)
    _thin_contacts(stage, f"{b}/flange")
    _thin_contacts(stage, f"{lv}/flange")
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")
    return joint


def _job_lock(stage, path, center, m):
    """A shut red Master Lock 410 on a panel's handle (visual only)."""
    x, y, z = center
    span(stage, f"{path}_body", (x - 0.0095, x + 0.0095), (y - 0.019, y + 0.019), (z - 0.022, z + 0.022),
         m["lock_red"], collide=False)
    for k, dy in enumerate((-0.0127, 0.0127)):
        span(stage, f"{path}_leg{k}", (x - 0.003, x + 0.003), (y + dy - 0.003, y + dy + 0.003), (z + 0.022, z + 0.062),
             m["steel"], collide=False)
    span(stage, f"{path}_bend", (x - 0.003, x + 0.003), (y - 0.016, y + 0.016), (z + 0.059, z + 0.065), m["steel"],
         collide=False)


def make_fixtures(out_path: Path) -> None:
    root = "/fixtures"
    stage, _ = kit.new_stage(out_path, root)
    m = _mats(stage, root, COLORS)
    bx, (y0, y1) = S.BOARD_X, S.BOARD_Y
    span(stage, f"{root}/board", (bx, bx + 0.018), (y0, y1), (0.0, S.BOARD_H), m["board"])
    for k, y in enumerate((y0 + 0.03, y1 - 0.03)):
        span(stage, f"{root}/board_foot{k}", (bx - 0.08, bx + 0.06), (y - 0.02, y + 0.02), (0.0, 0.015), m["board"])
    z0, z1 = S.PANEL_Z
    for k, y in enumerate(S.PANEL_Y):
        p = f"{root}/panel{k}"
        fx = bx - S.PANEL_D
        span(stage, p, (fx, bx), (y - S.PANEL_W / 2, y + S.PANEL_W / 2), (z0, z1), m["panel"])
        # Nameplate on the front, above the handle.
        pw, ph, pz = 0.064, 0.024, z1 - 0.020
        # The plate itself is solid colour (laminate); the engraved letters are a texture on a slightly smaller quad.
        span(stage, f"{p}_plate", (fx - 0.002, fx), (y - pw / 2, y + pw / 2), (pz - ph / 2, pz + ph / 2),
             m["plate_yellow" if k == 0 else "plate_white"], collide=False)
        text_boxes(stage, f"{p}_plate_text", S.PANEL_NAMES[k], (fx - 0.0021, y, pz), 0.012, m["black"])
        # Rotary handle at OFF (pointing left) and its job lock hanging from the handle's lock-off ear (visual).
        hz = (z0 + z1) / 2 - 0.012
        kit.cylinder(stage, f"{p}_hub", (fx - 0.006, y, hz), 0.016, 0.012, m["black"], axis="X")
        span(stage, f"{p}_grip", (fx - 0.020, fx - 0.004), (y, y + 0.028), (hz - 0.007, hz + 0.007), m["black"],
             collide=False)
        _job_lock(stage, f"{p}_joblock", (fx - 0.014, y + 0.012, hz - 0.048), m)
        # Key peg on the panel's right side (4 mm square steel rod).
        ys = y - S.PANEL_W / 2
        pz0, px = S.PEG_Z, S.PEG_X
        span(stage, f"{p}_peg", (px - S.PEG_SIDE / 2, px + S.PEG_SIDE / 2), (ys - S.PEG_LEN, ys),
             (pz0 - S.PEG_SIDE / 2, pz0 + S.PEG_SIDE / 2), m["steel"])
    # The robot padlock's pedestal: a column thinner than the lock body (fingers pass beside it).
    x, y = S.PEDESTAL_XY
    kit.box(stage, f"{root}/pedestal_foot", (x, y, 0.005), (0.06, 0.05, 0.01), m["stand"])
    kit.box(stage, f"{root}/pedestal_column", (x, y, S.PEDESTAL_TOP / 2), (0.030, 0.009, S.PEDESTAL_TOP), m["stand"])
    kit.box(stage, f"{root}/park_shelf", (PARK_X, 0.0, PARK_Z - 0.005), (0.12, 0.60, 0.01), m["stand"])
    _thin_contacts(stage, f"{root}/panel")
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_key(out_path: Path) -> None:
    root = "/key"
    stage, root_prim = kit.new_stage(out_path, root)
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(S.KEY_MASS)
    root_prim.AddAppliedSchema("PhysxRigidBodyAPI")
    root_prim.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(2.0)
    root_prim.CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float).Set(0.5)
    m = _mats(stage, root, ("brass",))
    t, h, bw = S.KEY_T / 2, S.KEY_HOLE / 2, S.KEY_BOW_W / 2
    ztop, zbot = S.KEY_BOW
    span(stage, f"{root}/bow_top", (-t, t), (-bw, bw), (h, ztop), m["brass"])
    span(stage, f"{root}/bow_bottom", (-t, t), (-bw, bw), (zbot, -h), m["brass"])
    for k, s in enumerate((-1, 1)):
        span(stage, f"{root}/bow_side{k}", (-t, t), (h, bw) if s > 0 else (-bw, -h), (-h, h), m["brass"])
    bl = S.KEY_BLADE_W / 2
    span(stage, f"{root}/blade", (-t, t), (-bl, bl), (S.KEY_TIP_Z, zbot), m["brass"])
    # Cuts on the blade's top edge (visual).
    for k in range(4):
        zc = zbot - 0.006 - 0.0065 * k
        span(stage, f"{root}/bit{k}", (-t, t), (bl, bl + 0.0025), (zc - 0.002, zc + 0.002), m["brass"], collide=False)
    kit.friction(stage, root, [root], static=0.8, dynamic=0.6, name="brass")
    _thin_contacts(stage, root)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    make_lockbox(out_dir / "lockbox.usda")
    make_fixtures(out_dir / "fixtures.usda")
    make_key(out_dir / "key.usda")
    padlock_410.make_padlock_410(out_dir / "padlock_ml410.usda", padlock_props.PADLOCKS["ml410"])
    padlock_410.make_prelocked_410(out_dir / "coworker.usda")

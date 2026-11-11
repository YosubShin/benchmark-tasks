"""Generate the props of the wall switch lockout task as USD.

Real parts: two single-pole wall toggle switches (15 A, 120 V residential, e.g. Leviton 1451-2W) in flush boxes with
plain one-gang plates (70 x 115 mm), each carrying a Master Lock 496B universal wall switch lockout (red, about 120 x 45
x 25 mm, polycarbonate; padlock shackle up to 5/16 in). How the 496B works (Master Lock product page and installation
videos): its base frame goes over the switch plate and is held by the plate's own two screws (installed once; the
hinged cover then opens for normal use); the cover is hinged at the top of the base and, swung down, encloses the
toggle; a padlock ring on the base sticks out through a slot near the cover's bottom, and a padlock through the ring
keeps the cover from swinging open. With the cover shut nobody can reach the toggle.

Modelled:
- `switches.usda`, a fixed-base articulation. Root (fixed): the plywood wall board on its feet, both switch plates,
  both 496B bases (frame around the toggle, hinge leaf, padlock ring on a neck), the engraved labels. Per switch:
  `toggle_<k>` on a revolute joint about y at its pivot (6 mm behind the plate face; +30 deg = up = ON, -30 deg = down =
  OFF; the end stops are joint limits; the snap is a capped drive set by the task); `cover_<k>` on a revolute joint
  about y at the base's top front edge (0 = shut, the joint's upper limit = open, leaning back on the wall).
- The lock-out is geometry: shut, a rib inside the cover sits at the middle of the toggle's throw; it clears the
  toggle at either end but stops it about 15 deg short of the middle, so it cannot be pushed from OFF to ON (and the
  toggle cannot be reached through the shut cover). A shackle through the ring sits right in front of the cover's
  face: the cover can swing out about 7 mm at the ring (under 4 deg), then meets the shackle.
- `pedestal.usda`: the post the open padlock stands on, and the parking shelf.
- `padlock_ml410.usda`: the Master Lock 410 (same model as the other lockout tasks).

World (desk) frame: x forward from the arm bases, y left, z up from the table top.

Usage: python -m switch_lockout_sim.usd.switch_props switch_lockout_sim/assets
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from switch_lockout_sim.spec import LABEL_COLORS, TAGS
from switch_lockout_sim.usd import padlock_410, padlock_props
from switch_lockout_sim.usd import usd_kit as kit

# Layout -------------------------------------------------------------------------------------------------------
WALL_X = 0.320
"""World x of the wall board's front face (the switch boxes are flush in it)."""
BOARD_T = 0.018
BOARD_Y = (-0.20, 0.28)
BOARD_H = 0.46
SLOT_Y = (-0.03, 0.17)
"""World y of the two switches: slot 0 (S1, the named one) and slot 1 (S2)."""
ZC = 0.23
"""World z of both toggles' pivots (the plates' centres)."""

# Switch plate (one-gang, 2.75 x 4.5 in) and toggle --------------------------------------------------------------
PLATE_W, PLATE_H, PLATE_T = 0.070, 0.115, 0.006
PLATE_X = WALL_X - PLATE_T
"""Plate face."""
OPEN_W, OPEN_H = 0.010, 0.024
"""Toggle opening in the plate (0.40 x 0.94 in)."""
PIVOT_X = WALL_X
TOGGLE_L = 0.026
"""Pivot to the toggle's tip: it stands 16.5 mm out of the plate at either end of its throw."""
TOGGLE_R0 = 0.008
TOGGLE_W, TOGGLE_T = 0.008, 0.005
THROW_DEG = 30.0
TOGGLE_MASS = 0.005

# Master Lock 496B -------------------------------------------------------------------------------------------------
BASE_W, BASE_H, BASE_T = 0.045, 0.120, 0.003
BASE_Z = (ZC - 0.060, ZC + 0.060)
BASE_X = (PLATE_X - BASE_T, PLATE_X)
WIN_W, WIN_H = 0.016, 0.032
"""The base's window around the toggle."""
COVER_X = (PLATE_X - 0.0215, BASE_X[0] - 0.0005)
"""Shut cover: front face 21.5 mm in front of the plate, back edge 0.5 mm off the base."""
COVER_WALL = 0.0025
COVER_INNER_X = COVER_X[0] + COVER_WALL
"""Inner face of the shut cover's front: 25 mm in front of the toggle's pivot."""
HINGE = (COVER_X[0] + 0.008, BASE_Z[1] + 0.0005)
"""Hinge axis (x, z) at the top of the base, 8 mm behind the cover's face: shut, the cover's centre of mass hangs
2.4 mm in front of it, so the cover rests shut."""
COVER_OPEN_DEG = 185.0
"""Open, the cover leans back on the wall above the plate (it passes its balance point at about 178 deg)."""
RIB_Z = (ZC - 0.0025, ZC + 0.0025)
RIB_X = (None, BASE_X[0] - 0.0015)
RIB_W = 0.016
"""A rib inside the cover at the middle of the toggle's throw (inferred: Master Lock says the 496B holds the switch ON
or OFF): it clears the toggle at either end and stops it before the middle, even with the cover a few degrees open."""
COVER_MASS = 0.030
RING_Z = (0.177, 0.181)
"""The padlock ring, 4 mm thick, its hole's axis vertical."""
HOLE = 0.0085
"""Square hole (the 496B takes shackles up to 5/16 in = 7.9 mm)."""
RING_WALL = 0.004
HOLE_X = COVER_X[0] - 0.0105
"""Hole centre: the ring stands proud of the shut cover's face on a neck through the cover's slot."""
NECK_W = 0.008
SLOT_W, SLOT_Z = 0.019, (RING_Z[0] - 0.0015, RING_Z[1] + 0.0015)

# Labels (engraved two-ply laminate on the wall beside each plate, on its left) ------------------------------------------
LABEL_W, LABEL_H, LABEL_T = 0.050, 0.032, 0.0016
LABEL_ZC = ZC
LABEL_DY = PLATE_W / 2 + 0.008 + LABEL_W / 2

# Padlock post ----------------------------------------------------------------------------------------------------
LOCK_START = (0.215, 0.12)
"""World (x, y) of the open padlock's body centre: it stands upright on its post, shackle up."""
LOCK_START_Z = 0.140
POST_TOP = LOCK_START_Z - 0.022
PARK_X, PARK_Z = 0.30, -0.30

CONTACT_OFFSET = 0.001
SOLVER_ITERATIONS = 32

COLORS = dict(kit.COLORS)
COLORS.update({
    "board": (0.62, 0.55, 0.44),
    "plate_ivory": (0.93, 0.92, 0.88),
    "toggle_ivory": (0.96, 0.95, 0.91),
    "base_red": (0.62, 0.05, 0.04),
    "cover_red": (0.86, 0.12, 0.08),
    "label_black": (0.06, 0.06, 0.06),
    "label_yellow": (0.96, 0.78, 0.10),
    "label_white": (0.95, 0.95, 0.93),
    "sticker_white": (0.97, 0.97, 0.97),
    "sticker_red": (0.80, 0.06, 0.05),
    "screw": (0.75, 0.76, 0.78),
})


def pivot(slot: int) -> tuple[float, float, float]:
    return (PIVOT_X, SLOT_Y[slot], ZC)


def hinge(slot: int) -> tuple[float, float, float]:
    return (HINGE[0], SLOT_Y[slot], HINGE[1])


def hole_world(slot: int) -> tuple[float, float, float]:
    """World centre of a ring's hole (mid-thickness)."""
    return (HOLE_X, SLOT_Y[slot], (RING_Z[0] + RING_Z[1]) / 2)


def _span(stage, path, xs, ys, zs, material, collide=True):
    c = ((xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2, (zs[0] + zs[1]) / 2)
    s = (abs(xs[1] - xs[0]), abs(ys[1] - ys[0]), abs(zs[1] - zs[0]))
    return kit.box(stage, path, c, s, material, collide=collide)


def _ybox(stage, path, center, size, material, angle_deg, collide=True):
    """A box rotated about y by angle_deg."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddRotateYOp().Set(angle_deg)
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _xbox(stage, path, center, size, material, angle_deg):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddRotateXOp().Set(angle_deg)
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    return cube


def _mats(stage, root):
    out = {}
    for name, rgb in COLORS.items():
        path = f"{root}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3 if name == "cover_red" else 0.5)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        out[name] = material
    return out


# Engraved letters as thin black strokes (geometry, not a texture: textured plates sometimes showed another
# process's image). Glyph strokes in mm: u to the left as you face the label, v up.
_GLYPHS = {
    "S": [((-2.5, 10), (-10, 10)), ((0, 7.5), (-2.5, 10)), ((0, 7.5), (0, 2.5)), ((0, 2.5), (-2.5, 0)),
          ((-2.5, 0), (-7.5, 0)), ((-7.5, 0), (-10, -2.5)), ((-10, -2.5), (-10, -7.5)), ((-10, -7.5), (-7.5, -10)),
          ((-7.5, -10), (0, -10))],
    "1": [((-5, 10), (-5, -10)), ((-5, 10), (-1, 6))],
    "2": [((0, 10), (-10, 10)), ((-10, 10), (-10, 0)), ((-10, 0), (0, 0)), ((0, 0), (0, -10)), ((0, -10), (-10, -10))],
}
_GLYPH_W = {"S": 11, "1": 8, "2": 11}
STROKE = 0.0026


def _engrave(stage, path, sy, zc, text, mat):
    fx = WALL_X - LABEL_T - 0.0004
    total = sum(_GLYPH_W[c] for c in text) + 5 * (len(text) - 1)
    u0 = total / 2.0
    k = 0
    for c in text:
        for (ua, va), (ub, vb) in _GLYPHS[c]:
            ya, yb = sy + (u0 + ua) / 1000.0, sy + (u0 + ub) / 1000.0
            za, zb = zc + va / 1000.0, zc + vb / 1000.0
            length = math.hypot(yb - ya, zb - za) + STROKE
            ang = math.degrees(math.atan2(zb - za, yb - ya))
            _xbox(stage, f"{path}_{k}", (fx, (ya + yb) / 2, (za + zb) / 2), (0.0006, length, STROKE), mat, ang)
            k += 1
        u0 -= _GLYPH_W[c] + 5


def _thin_contacts(stage, prefix):
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def _frame(stage, path, xs, yc, zc, outer, inner, mat, collide=True):
    """A rectangular frame in a y-z plane: outer (w, h), inner opening (w, h), centred at (yc, zc)."""
    (ow, oh), (iw, ih) = outer, inner
    _span(stage, f"{path}_top", xs, (yc - ow / 2, yc + ow / 2), (zc + ih / 2, zc + oh / 2), mat, collide)
    _span(stage, f"{path}_bot", xs, (yc - ow / 2, yc + ow / 2), (zc - oh / 2, zc - ih / 2), mat, collide)
    _span(stage, f"{path}_l", xs, (yc + iw / 2, yc + ow / 2), (zc - ih / 2, zc + ih / 2), mat, collide)
    _span(stage, f"{path}_r", xs, (yc - ow / 2, yc - iw / 2), (zc - ih / 2, zc + ih / 2), mat, collide)


def make_switches(out_path: Path) -> None:
    root = "/switches"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=True)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
        SOLVER_ITERATIONS)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    m = _mats(stage, root)
    base = kit.rigid_body(stage, f"{root}/base", (0, 0, 0), 5.0)
    kit.fix_to_world(stage, root, base.GetPath())
    b = f"{root}/base"
    # Wall board standing on the table on two feet; the switch boxes are flush in it (a hole behind each plate).
    bx = (WALL_X, WALL_X + BOARD_T)
    _span(stage, f"{b}/board_low", bx, BOARD_Y, (0.0, ZC - PLATE_H / 2 + 0.01), m["board"])
    _span(stage, f"{b}/board_high", bx, BOARD_Y, (ZC + PLATE_H / 2 - 0.01, BOARD_H), m["board"])
    edges = [BOARD_Y[0]] + [v for sy in SLOT_Y for v in (sy - PLATE_W / 2 + 0.01, sy + PLATE_W / 2 - 0.01)] + [BOARD_Y[1]]
    edges = sorted(edges)
    for k in range(0, len(edges), 2):
        _span(stage, f"{b}/board_mid{k}", bx, (edges[k], edges[k + 1]), (ZC - PLATE_H / 2 + 0.01,
              ZC + PLATE_H / 2 - 0.01), m["board"])
    for k, y in enumerate((BOARD_Y[0] + 0.04, BOARD_Y[1] - 0.04)):
        _span(stage, f"{b}/board_foot{k}", (WALL_X - 0.08, WALL_X + BOARD_T + 0.06), (y - 0.02, y + 0.02),
              (0.0, 0.015), m["board"])
    for slot, sy in enumerate(SLOT_Y):
        # Plate with its toggle opening; the device's yoke behind it (dark, visual).
        _frame(stage, f"{b}/plate{slot}", (PLATE_X, WALL_X), sy, ZC, (PLATE_W, PLATE_H), (OPEN_W, OPEN_H),
               m["plate_ivory"])
        _span(stage, f"{b}/device{slot}", (WALL_X - 0.0005, WALL_X + 0.012), (sy - 0.016, sy + 0.016),
              (ZC - 0.030, ZC + 0.030), m["dark"], collide=False)
        # 496B base: frame around the toggle on the plate, held by the plate's two screws.
        _frame(stage, f"{b}/base{slot}", BASE_X, sy, ZC, (BASE_W, BASE_H), (WIN_W, WIN_H), m["base_red"])
        for k, dz in enumerate((-0.0415, 0.0415)):
            kit.cylinder(stage, f"{b}/screw{slot}_{k}", (BASE_X[0] - 0.0004, sy, ZC + dz), 0.0035, 0.001,
                         m["screw"], axis="X")
        # Hinge leaf and knuckle over the base's top (visual).
        _span(stage, f"{b}/hinge_leaf{slot}", (HINGE[0] - 0.002, BASE_X[1]), (sy - 0.018, sy + 0.018),
              (BASE_Z[1] + 0.0006, BASE_Z[1] + 0.0036), m["base_red"], collide=False)
        for k, dy in enumerate((-0.014, 0.014)):
            kit.cylinder(stage, f"{b}/knuckle{slot}_{k}", (HINGE[0], sy + dy, HINGE[1] + 0.001), 0.0028, 0.008,
                         m["base_red"], axis="Y")
        # Padlock ring on its neck (through the shut cover's slot).
        ring_out = HOLE + 2 * RING_WALL
        _span(stage, f"{b}/neck{slot}", (HOLE_X + ring_out / 2, BASE_X[1]), (sy - NECK_W / 2, sy + NECK_W / 2),
              RING_Z, m["base_red"])
        _frame_xy = [((HOLE_X - ring_out / 2, HOLE_X + ring_out / 2), (sy + HOLE / 2, sy + ring_out / 2)),
                     ((HOLE_X - ring_out / 2, HOLE_X + ring_out / 2), (sy - ring_out / 2, sy - HOLE / 2)),
                     ((HOLE_X - ring_out / 2, HOLE_X - HOLE / 2), (sy - HOLE / 2, sy + HOLE / 2)),
                     ((HOLE_X + HOLE / 2, HOLE_X + ring_out / 2), (sy - HOLE / 2, sy + HOLE / 2))]
        for k, (xs, ys) in enumerate(_frame_xy):
            _span(stage, f"{b}/ring{slot}_{k}", xs, ys, RING_Z, m["base_red"])
        # Engraved label under the plate.
        lab = m["label_yellow"] if LABEL_COLORS[slot] == "yellow" else m["label_white"]
        ly = sy + LABEL_DY
        _span(stage, f"{b}/label{slot}", (WALL_X - LABEL_T, WALL_X), (ly - LABEL_W / 2, ly + LABEL_W / 2),
              (LABEL_ZC - LABEL_H / 2, LABEL_ZC + LABEL_H / 2), lab, collide=False)
        _engrave(stage, f"{b}/label{slot}_text", ly, LABEL_ZC, TAGS[slot], m["label_black"])

        # Toggle: body origin at the pivot; at joint angle 0 the lever points straight out (-x).
        tg = f"{root}/toggle_{slot}"
        kit.rigid_body(stage, tg, pivot(slot), TOGGLE_MASS)
        _span(stage, f"{tg}/lever", (-TOGGLE_L, -TOGGLE_R0), (-TOGGLE_W / 2, TOGGLE_W / 2),
              (-TOGGLE_T / 2, TOGGLE_T / 2), m["toggle_ivory"])
        _span(stage, f"{tg}/stem", (-TOGGLE_R0, -0.002), (-0.004, 0.004), (-0.0025, 0.0025), m["toggle_ivory"],
              collide=False)
        kit.revolute(stage, f"{root}/joints/toggle_{slot}_joint", base.GetPath(), tg, "Y", pivot(slot),
                     -THROW_DEG, THROW_DEG)

        # Cover: body origin on the hinge axis; built shut.
        cv = f"{root}/cover_{slot}"
        hx, _, hz = hinge(slot)
        kit.rigid_body(stage, cv, hinge(slot), COVER_MASS)
        red = m["cover_red"]
        x0, x1 = COVER_X[0] - hx, COVER_X[1] - hx
        xi = COVER_INNER_X - hx
        z0, z1 = BASE_Z[0] - hz, BASE_Z[1] - hz
        hw = BASE_W / 2
        s0, s1 = SLOT_Z[0] - hz, SLOT_Z[1] - hz
        # Front face, with the slot the padlock ring comes through.
        _span(stage, f"{cv}/front_upper", (x0, xi), (-hw, hw), (s1, z1), red)
        _span(stage, f"{cv}/front_lower", (x0, xi), (-hw, hw), (z0, s0), red)
        _span(stage, f"{cv}/front_l", (x0, xi), (SLOT_W / 2, hw), (s0, s1), red)
        _span(stage, f"{cv}/front_r", (x0, xi), (-hw, -SLOT_W / 2), (s0, s1), red)
        for k, s in enumerate((-1, 1)):
            _span(stage, f"{cv}/side{k}", (xi, x1), (s * (hw - COVER_WALL), s * hw), (z0, z1), red)
        _span(stage, f"{cv}/rib", (xi, RIB_X[1] - hx), (-RIB_W / 2, RIB_W / 2), (RIB_Z[0] - hz, RIB_Z[1] - hz), red)
        _span(stage, f"{cv}/top", (xi, x1), (-hw + COVER_WALL, hw - COVER_WALL), (z1 - COVER_WALL, z1), red)
        _span(stage, f"{cv}/bottom", (xi, x1), (-hw + COVER_WALL, hw - COVER_WALL), (z0, z0 + COVER_WALL), red)
        # DANGER LOCKED OUT label on the cover's face (visual).
        lz = (z0 + z1) / 2 + 0.012
        _span(stage, f"{cv}/sticker", (x0 - 0.0004, x0), (-0.017, 0.017), (lz - 0.032, lz + 0.032),
              m["sticker_white"], collide=False)
        _span(stage, f"{cv}/sticker_band", (x0 - 0.0007, x0 - 0.0003), (-0.014, 0.014), (lz + 0.018, lz + 0.027),
              m["sticker_red"], collide=False)
        for k, dz in enumerate((0.006, -0.004, -0.014)):
            _span(stage, f"{cv}/sticker_line{k}", (x0 - 0.0007, x0 - 0.0003), (-0.011, 0.011),
                  (lz + dz - 0.0012, lz + dz + 0.0012), m["label_black"], collide=False)
        kit.revolute(stage, f"{root}/joints/cover_{slot}_joint", base.GetPath(), cv, "Y", hinge(slot), 0.0,
                     COVER_OPEN_DEG)
    _thin_contacts(stage, root)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_pedestal(out_path: Path) -> None:
    """The padlock stands upright (shackle up) on a small grey post with a shallow cradle, and the parking shelf."""
    root = "/pedestal"
    stage, _ = kit.new_stage(out_path, root)
    mats = kit.materials(stage, root, ("stand",))
    x, y = LOCK_START
    _span(stage, f"{root}/foot", (x - 0.03, x + 0.03), (y - 0.03, y + 0.03), (0.0, 0.01), mats["stand"])
    _span(stage, f"{root}/post", (x - 0.007, x + 0.007), (y - 0.007, y + 0.007), (0.0, POST_TOP), mats["stand"])
    # Cradle lips 4 mm high around the body's foot (19 x 38 mm, the lock's thickness along x).
    lip = 0.004
    _span(stage, f"{root}/lip_front", (x - 0.0115, x - 0.0100), (y - 0.012, y + 0.012), (POST_TOP - 0.002,
          POST_TOP + lip), mats["stand"])
    _span(stage, f"{root}/lip_back", (x + 0.0100, x + 0.0115), (y - 0.012, y + 0.012), (POST_TOP - 0.002,
          POST_TOP + lip), mats["stand"])
    _span(stage, f"{root}/cradle", (x - 0.0115, x + 0.0115), (y - 0.012, y + 0.012), (POST_TOP - 0.004,
          POST_TOP), mats["stand"])
    kit.box(stage, f"{root}/park_shelf", (PARK_X, 0.0, PARK_Z - 0.005), (0.12, 0.60, 0.01), mats["stand"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    make_switches(out_dir / "switches.usda")
    make_pedestal(out_dir / "pedestal.usda")
    padlock_410.make_padlock_410(out_dir / "padlock_ml410.usda", padlock_props.PADLOCKS["ml410"])

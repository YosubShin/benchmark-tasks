"""Generate the props of the lock-and-tag task as USD.

Parts (each follows a real product; values marked "inferred" are not published and are to measure):

- Disconnect switch (`switch.usda`), a fixed-base articulation: the enclosure, a fixed lock-off ear, and the
  operating handle on a revolute joint about the axis normal to the side wall. The handle of the group-hasp tasks
  (Square D heavy-duty safety switch, HU361 class: the handle padlocks OFF through a hole in the handle that lines up
  with a hole in a fixed part of the enclosure only at OFF). All handle dimensions are inferred: lock-off holes
  12 mm square (Square D quotes a 3/8 in shackle maximum), 3 mm steel tab and ear 1 mm apart, hole 70 mm below the
  pivot, 90 degrees OFF to ON, the plate stack 65 mm out from the side wall (the handle mechanism's depth).
  Simplification: the lever points up at OFF and turns to horizontal (toward -y) at ON.
- Danger tag (`tag.usda`), a floating articulation of three thin plates hinged across the tag (a stiff-ish laminate
  that bends): Brady 65502 (5-3/4 x 3 in = 146 x 76 mm polyester laminate, 3/8 in = 9.5 mm metal grommet; Master
  Lock 497A is the same size with a 7/16 in grommet). Inferred: 20 mm grommet flange, 2.5 mm grommet thickness, the
  grommet's centre 13 mm below the top edge, 8 g. The card collider is 1.2 mm thick (the real laminate is about
  0.4 mm; thinner colliders let the 6.35 mm shackle tunnel). The grommet's hole is an octagon of 8 bars (9.5 mm
  across flats); the card has a 20 mm square opening behind the grommet. Hinges: 0.02 N m/rad, +-60 degrees.
- The robot's padlock, a red Master Lock 410 (`padlock_ml410.usda`): the box model of locktag_sim/usd/padlock_props.py
  in the 410's real shape by locktag_sim/usd/padlock_410.py (prismatic shackle that pops up 16 mm, no swivel).
- Stands (`stands.usda`): a column the open padlock lies on (on its narrow side, shackle toward +x, long leg up),
  and the tag pocket (a block with a 4 mm slot, 80 mm deep, the tag standing in it grommet up, its face toward the
  robot).

Frames: the switch's origin is on the side wall at (WALL_X, HOLE_Y, 0) in the world; the tag's origin is its
grommet's centre, its face normal along local x, its width along local y, its top edge at local z = +13 mm.

Usage: python -m locktag_sim.usd.locktag_props <out dir>
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from locktag_sim.spec import HOLE_X, HOLE_Z, LOCK_PEDESTAL_TOP, LOCK_START_XY, TAG_GROMMET, WALL_X
from locktag_sim.usd import padlock_410, padlock_props, tag_texture
from locktag_sim.usd import usd_kit as kit

# Switch handle (Square D heavy-duty style; all inferred) --------------------------------------------
LOCKOFF_HOLE = 0.012
PLATE_T = 0.003
PLATE_GAP = 0.001
TAB_X = (HOLE_X - PLATE_GAP / 2 - PLATE_T, HOLE_X - PLATE_GAP / 2)
EAR_X = (HOLE_X + PLATE_GAP / 2, HOLE_X + PLATE_GAP / 2 + PLATE_T)
PLATE_HALF_W = 0.012
PLATE_BELOW = 0.004
"""Steel below the holes' lower edges."""
PIVOT_R = 0.07
PIVOT_Z = HOLE_Z + PIVOT_R
PIVOT_X = HOLE_X - 0.009
HANDLE_ON_DEG = 90.0
LEVER_LEN = 0.10
EAR_ARM_DZ = 0.045
"""The ear's standoff arm runs back to the wall this far above the holes' centres (clear of a hanging tag)."""
ENCLOSURE = (0.14, (-0.12, 0.08), 0.44)
"""Depth behind the wall, y range, height."""

# Danger tag (Brady 65502) -----------------------------------------------------------------------------
TAG_W, TAG_H = 0.076, 0.146
TAG_T = 0.0012
GROMMET_FROM_TOP = 0.013
GROMMET_HOLE = 0.0095
GROMMET_OD = 0.020
GROMMET_T = 0.0025
TAG_OPENING = 0.020
TAG_SEG_LEN = (0.040, 0.053, 0.053)
TAG_SEG_MASS = (0.004, 0.002, 0.002)
TAG_HINGE_K = 0.02
TAG_HINGE_D = 0.0005
TAG_TOP = GROMMET_FROM_TOP
"""Top edge in the tag frame."""
# Hinge z (tag frame) between segments.
TAG_HINGES = (TAG_TOP - TAG_SEG_LEN[0], TAG_TOP - TAG_SEG_LEN[0] - TAG_SEG_LEN[1])
TAG_BOTTOM = TAG_TOP - TAG_H

# Stands ----------------------------------------------------------------------------------------------
POCKET_SLOT = 0.004
POCKET_DEPTH = 0.080
POCKET_LEN = 0.090
POCKET_WALL = 0.006
POCKET_FLOOR = TAG_GROMMET[2] + TAG_BOTTOM
"""World z of the pocket's floor (the tag's bottom edge rests on it)."""
PARK_X, PARK_Z = 0.30, -0.30

PADLOCK = padlock_props.PadlockVariant("ml410", "Master Lock 410 Zenex, red (410RED)", (0.038, 0.019, 0.044),
                                       0.00635, 0.020, 0.038, 0.10, "lock_red")

CONTACT_OFFSET = 0.002
SOLVER_ITERATIONS = 32


def _yz_plate_with_hole(stage, path, x_range, y_half, z_range, hole_z, hole, material):
    """A plate in the y-z plane (thickness along x) with one square hole centred at y = 0, z = hole_z."""
    xc, t = (x_range[0] + x_range[1]) / 2, x_range[1] - x_range[0]
    z0, z1 = z_range
    lo, hi = hole_z - hole / 2, hole_z + hole / 2
    kit.box(stage, f"{path}_below", (xc, 0, (z0 + lo) / 2), (t, 2 * y_half, lo - z0), material)
    kit.box(stage, f"{path}_above", (xc, 0, (hi + z1) / 2), (t, 2 * y_half, z1 - hi), material)
    side = y_half - hole / 2
    for k, s in enumerate((-1, 1)):
        kit.box(stage, f"{path}_side{k}", (xc, s * (hole / 2 + side / 2), hole_z), (t, side, hole), material)


def _thin_contacts(stage, root_prim, paths_prefix: str, offset=CONTACT_OFFSET) -> None:
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
        SOLVER_ITERATIONS)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(paths_prefix) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(offset)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def make_switch(out_path: Path) -> None:
    """Enclosure, fixed lock-off ear, and the operating handle. Frame: world shifted by (WALL_X, HOLE_Y, 0)."""
    root = "/switch"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim)
    mats = kit.materials(stage, root, ("grey", "steel", "red", "black", "white", "dark"))
    base = kit.rigid_body(stage, f"{root}/enclosure", (0, 0, 0), 10.0)
    depth, (y0, y1), height = ENCLOSURE
    kit.box(stage, f"{root}/enclosure/box", (depth / 2, (y0 + y1) / 2, height / 2), (depth, y1 - y0, height),
            mats["grey"])
    kit.fix_to_world(stage, root, base.GetPath())
    wx = lambda x: x - WALL_X  # noqa: E731  world x -> switch frame
    lo, hi = HOLE_Z - LOCKOFF_HOLE / 2, HOLE_Z + LOCKOFF_HOLE / 2
    # Fixed lock-off ear: a plate parallel to the wall with the lock-off hole, on a standoff arm back to the wall.
    ear_top = HOLE_Z + EAR_ARM_DZ + 0.003
    _yz_plate_with_hole(stage, f"{root}/enclosure/ear", (wx(EAR_X[0]), wx(EAR_X[1])), PLATE_HALF_W,
                        (lo - PLATE_BELOW, ear_top), HOLE_Z, LOCKOFF_HOLE, mats["steel"])
    kit.box(stage, f"{root}/enclosure/ear_arm", (wx(EAR_X[0]) / 2, 0, HOLE_Z + EAR_ARM_DZ),
            (-wx(EAR_X[0]), 2 * PLATE_HALF_W, 0.006), mats["steel"])
    # The operating mechanism's boss: the handle's shaft comes out of the wall here.
    boss_x0 = wx(PIVOT_X + 0.006)
    kit.box(stage, f"{root}/enclosure/boss", (boss_x0 / 2, 0, PIVOT_Z), (-boss_x0, 0.03, 0.03), mats["dark"])
    # ON / OFF marks on the wall (visual): OFF above the pivot (lever up), ON toward -y.
    kit.box(stage, f"{root}/enclosure/mark_off", (-0.0005, 0, PIVOT_Z + LEVER_LEN + 0.012), (0.001, 0.02, 0.008),
            mats["white"], collide=False)
    kit.box(stage, f"{root}/enclosure/mark_on", (-0.0005, -LEVER_LEN - 0.012, PIVOT_Z), (0.001, 0.008, 0.02),
            mats["red"], collide=False)
    # Operating handle: body origin at the pivot.
    px = wx(PIVOT_X)
    handle = kit.rigid_body(stage, f"{root}/handle", (px, 0, PIVOT_Z), 0.15)
    hx = lambda x: wx(x) - px  # noqa: E731  world x -> handle frame
    _yz_plate_with_hole(stage, f"{root}/handle/tab", (hx(TAB_X[0]), hx(TAB_X[1])), PLATE_HALF_W,
                        (lo - PLATE_BELOW - PIVOT_Z, 0.0), HOLE_Z - PIVOT_Z, LOCKOFF_HOLE, mats["steel"])
    hub_x = (hx(HOLE_X - 0.016), hx(TAB_X[1]))
    kit.box(stage, f"{root}/handle/hub", ((hub_x[0] + hub_x[1]) / 2, 0, 0), (hub_x[1] - hub_x[0], 0.024, 0.024),
            mats["dark"])
    lever_x = (hx(HOLE_X - 0.016), hx(HOLE_X - 0.005))
    kit.box(stage, f"{root}/handle/lever", ((lever_x[0] + lever_x[1]) / 2, 0, LEVER_LEN / 2),
            (lever_x[1] - lever_x[0], 0.016, LEVER_LEN), mats["black"])
    kit.box(stage, f"{root}/handle/grip", ((lever_x[0] + lever_x[1]) / 2, 0, LEVER_LEN - 0.015),
            (lever_x[1] - lever_x[0] + 0.004, 0.022, 0.03), mats["red"])
    kit.revolute(stage, f"{root}/joints/handle_joint", base.GetPath(), handle.GetPath(), "X", (px, 0, PIVOT_Z),
                 0.0, HANDLE_ON_DEG)
    _thin_contacts(stage, root_prim, f"{root}/handle")
    _thin_contacts(stage, root_prim, f"{root}/enclosure/ear")
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


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


def _quad(stage, path, corners, uvs, material):
    """A textured quad (visual)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(False)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(*uv) for uv in uvs]))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _rbox_x(stage, path, center, size, material, rot_x_deg):
    """A box turned about its x axis (collider)."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddRotateXOp().Set(rot_x_deg)
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def make_tag(out_path: Path) -> None:
    """The danger tag: three hinged plates; the top one carries the grommet. Frame at the grommet's centre."""
    root = "/tag"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim)
    mats = kit.materials(stage, root, ("white",))
    brass = UsdShade.Material.Define(stage, f"{root}/Looks/brass")
    sh = UsdShade.Shader.Define(stage, f"{root}/Looks/brass/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.78, 0.60, 0.22))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.6)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    brass.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    front = _tex_material(stage, root, "locktag_front", f"./{tag_texture.FILE_FRONT}")
    back = _tex_material(stage, root, "locktag_back", f"./{tag_texture.FILE_BACK}")
    hw, t = TAG_W / 2, TAG_T
    tops = (TAG_TOP, TAG_HINGES[0], TAG_HINGES[1])
    bodies = []
    for k in range(3):
        z1, z0 = tops[k], tops[k] - TAG_SEG_LEN[k]
        origin = (0.0, 0.0, 0.0 if k == 0 else z1)
        body = kit.rigid_body(stage, f"{root}/seg{k}", origin, TAG_SEG_MASS[k])
        bodies.append(body)
        p = f"{root}/seg{k}"
        lz = lambda z: z - origin[2]  # noqa: E731  tag frame z -> segment frame
        if k == 0:
            o = TAG_OPENING / 2
            kit.box(stage, f"{p}/card_below", (0, 0, lz((z0 - o) / 2)), (t, TAG_W, -o - z0), mats["white"])
            kit.box(stage, f"{p}/card_above", (0, 0, lz((o + z1) / 2)), (t, TAG_W, z1 - o), mats["white"])
            for j, s in enumerate((-1, 1)):
                kit.box(stage, f"{p}/card_side{j}", (0, s * (o + hw) / 2, 0), (t, hw - o, 2 * o), mats["white"])
            # Grommet: an octagon of 8 bars (hole 9.5 mm across flats, 20 mm across the flange's flats).
            rc = (GROMMET_HOLE + GROMMET_OD) / 4
            bar_w = (GROMMET_OD - GROMMET_HOLE) / 2
            seg = GROMMET_OD * math.tan(math.radians(22.5))
            for i in range(8):
                a = math.radians(45 * i)
                _rbox_x(stage, f"{p}/grommet_{i}", (0, rc * math.cos(a), rc * math.sin(a)), (GROMMET_T, bar_w, seg),
                        brass, math.degrees(a))
        else:
            kit.box(stage, f"{p}/card", (0, 0, lz((z0 + z1) / 2)), (t, TAG_W, z1 - z0), mats["white"])
        for name in [c.GetName() for c in stage.GetPrimAtPath(p).GetChildren()]:
            if name.startswith("card"):
                UsdGeom.Imageable(stage.GetPrimAtPath(f"{p}/{name}")).MakeInvisible()
        # Printed faces (visual quads, slightly off the collider), image v from bottom (0) to top (1).
        v0, v1 = (z0 - TAG_BOTTOM) / TAG_H, (z1 - TAG_BOTTOM) / TAG_H
        xf = -t / 2 - 0.0002
        _quad(stage, f"{p}/face_front", [(xf, hw, lz(z0)), (xf, -hw, lz(z0)), (xf, -hw, lz(z1)), (xf, hw, lz(z1))],
              [(0, v0), (1, v0), (1, v1), (0, v1)], front)
        xb = t / 2 + 0.0002
        _quad(stage, f"{p}/face_back", [(xb, -hw, lz(z0)), (xb, hw, lz(z0)), (xb, hw, lz(z1)), (xb, -hw, lz(z1))],
              [(0, v0), (1, v0), (1, v1), (0, v1)], back)
    # Grommet visual: two brass rings (front and back flanges).
    for j, x in enumerate((-GROMMET_T / 2, GROMMET_T / 2)):
        ring = UsdGeom.Mesh.Define(stage, f"{root}/seg0/grommet_vis{j}")
        n = 24
        pts, faces = [], []
        for i in range(n):
            a = 2 * math.pi * i / n
            pts.append((x, GROMMET_HOLE / 2 * math.cos(a), GROMMET_HOLE / 2 * math.sin(a)))
            pts.append((x, GROMMET_OD / 2 * math.cos(a), GROMMET_OD / 2 * math.sin(a)))
        for i in range(n):
            a, b = 2 * i, 2 * ((i + 1) % n)
            faces.append([a, a + 1, b + 1, b])
        ring.CreatePointsAttr([Gf.Vec3f(*q) for q in pts])
        ring.CreateFaceVertexCountsAttr([4] * n)
        ring.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
        ring.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(ring.GetPrim()).Bind(brass)
    for k in (1, 2):
        j = kit.revolute(stage, f"{root}/joints/hinge{k}", bodies[k - 1].GetPath(), bodies[k].GetPath(), "Y",
                         (0, 0, tops[k] - (0.0 if k == 1 else tops[k - 1])), -60.0, 60.0)
        del j  # stiffness and damping are set by the task's actuator config (TAG_HINGE_K, TAG_HINGE_D)
    _thin_contacts(stage, root_prim, root, offset=0.002)
    # Brass on steel and laminate on steel: friction 0.6 that wins over the padlock's slick short leg (combine "max"),
    # so the grommet does not run along a tilted leg by itself.
    kit.friction(stage, root, [f"{root}/seg{k}" for k in range(3)], static=0.6, dynamic=0.5, name="grommet",
                 combine="max")
    for body in bodies:
        # Speculative CCD: the 2.5 mm grommet must not let a moving 6.35 mm shackle pass through it.
        body.GetPrim().AddAppliedSchema("PhysxRigidBodyAPI")
        body.GetPrim().CreateAttribute("physxRigidBody:enableSpeculativeCCD", Sdf.ValueTypeNames.Bool).Set(True)
    stage.GetRootLayer().Save()
    tag_texture.write(out_path.parent)
    print(f"wrote {out_path}")


def make_stands(out_path: Path) -> None:
    root = "/stands"
    stage, _ = kit.new_stage(out_path, root)
    mats = kit.materials(stage, root, ("stand",))
    m = mats["stand"]
    # Padlock column: under the body only (the shackle overhangs its +x edge; the fingers take the body's -x end).
    x, y = LOCK_START_XY
    kit.box(stage, f"{root}/lock_foot", (x, y, 0.005), (0.05, 0.05, 0.01), m)
    kit.box(stage, f"{root}/lock_column", (x + 0.016, y, LOCK_PEDESTAL_TOP / 2), (0.012, 0.016, LOCK_PEDESTAL_TOP), m)
    # Tag pocket: two walls either side of a 4 mm slot along y, floor at POCKET_FLOOR.
    tx, ty, _ = TAG_GROMMET
    top = POCKET_FLOOR + POCKET_DEPTH
    kit.box(stage, f"{root}/pocket_foot", (tx, ty, 0.005), (0.06, POCKET_LEN + 0.02, 0.01), m)
    kit.box(stage, f"{root}/pocket_base", (tx, ty, POCKET_FLOOR / 2), (POCKET_SLOT + 2 * POCKET_WALL, POCKET_LEN,
                                                                        POCKET_FLOOR), m)
    for k, s in enumerate((-1, 1)):
        kit.box(stage, f"{root}/pocket_wall{k}", (tx + s * (POCKET_SLOT + POCKET_WALL) / 2, ty, (POCKET_FLOOR + top) / 2),
                (POCKET_WALL, POCKET_LEN, top - POCKET_FLOOR), m)
    kit.box(stage, f"{root}/park_shelf", (PARK_X, 0.0, PARK_Z - 0.005), (0.12, 0.60, 0.01), m)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    make_switch(out_dir / "switch.usda")
    make_tag(out_dir / "tag.usda")
    make_stands(out_dir / "stands.usda")
    padlock_410.make_padlock_410(out_dir / "padlock_ml410.usda", PADLOCK)
    lock_stage = Usd.Stage.Open(str(out_dir / "padlock_ml410.usda"))
    for path in ("/padlock/body", "/padlock/shackle"):
        prim = lock_stage.GetPrimAtPath(path)
        prim.AddAppliedSchema("PhysxRigidBodyAPI")
        prim.CreateAttribute("physxRigidBody:enableSpeculativeCCD", Sdf.ValueTypeNames.Bool).Set(True)
    lock_stage.GetRootLayer().Save()

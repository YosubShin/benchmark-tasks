"""Generate the props of the hasp-apply task as USD.

Parts (each follows a real product; values marked "inferred" are not published and are to measure):

- Disconnect switch (`switch.usda`), a fixed-base articulation: the enclosure's side wall, a fixed
  lock-off ear, and the operating handle on a revolute joint about the axis normal to the wall.
  Based on the side-operated handle of a Square D heavy-duty safety switch (HU361 class, the switch in
  Master Lock's 420 application photo and in the Brady video), whose handle padlocks OFF through a
  hole in the handle that lines up with a hole in a fixed part of the enclosure only at OFF. All handle
  dimensions are inferred: lock-off hole 12 mm square (Square D quotes a 3/8 in shackle maximum for its
  handles), hole 50 mm below the pivot, 90 degrees OFF to ON, 3 mm steel plates, ear 3 mm behind the
  handle's tab. Simplification: the lever points up at OFF and turns to horizontal (toward -y) at ON;
  the real Square D lever points down at OFF and up at ON. Flipped so the lever clears the hanging hasp
  and the robot's wrists.
- Master Lock 420 scissor hasp (`hasp.usda`), a floating articulation: jaw A (the root) and jaw B,
  hinged at the rivet. Published (masterlock.com 420 page and technical literature): 1 in (25 mm)
  inside jaw diameter, 4-1/2 in (114 mm) overall length, 1-3/4 in (44 mm) overall width, six lock
  holes, shackles up to 9/32 in (7 mm). Inferred from the product photo: 8 mm ring width, 2 mm jaw
  plates, 40 mm wide tail, holes 8 mm square in 2 columns 22 mm apart and 3 rows 16 mm apart, rivet
  26 mm below the ring centre, 30 g per jaw. The ring is an octagon of flat bars; jaw A owns the top
  bar (the one through the handle), the left side and the lower left; jaw B the right side. Closed, the
  two tails lie flat on each other with their holes aligned; opening turns B about the rivet, which
  spreads the tails and splits the ring at its upper right. Nothing in the asset holds it shut: a
  shackle through an aligned hole pair does, by collision.
- The robot's padlock, Master Lock 410 (`padlock_ml410.usda`) and the coworker's yellow 410 with a danger
  tag (`coworker.usda`): the box model of haspapply_sim/usd/padlock_props.py in the 410's real shape (round
  shackle and bend, rounded body, label, keyway) by haspapply_sim/usd/padlock_410.py.
- Stands (`pedestal.usda`): a thin column the open padlock lies on (on its side, shackle toward -y), the hasp
  holder (a block with a 7 mm slot the closed hasp's tails stand in), and a shelf under the desk (unused).

Frames: the hasp's origin is its ring centre, jaws in the x-z plane (normal +y), jaw A on +y of the
mid-plane and jaw B on -y. The switch's origin is on the wall at (WALL_X, HASP_Y, 0) in the world.

Usage: python -m haspapply_sim.usd.hasp_props <out dir>
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from haspapply_sim.spec import BOTTOM_HOLE_Z, HASP_START_DXY, HASP_START_DZ, HASP_Y, PEDESTAL_XY, RING_X, WALL_X
from haspapply_sim.usd import padlock_410, padlock_props
from haspapply_sim.usd import usd_kit as kit

# Master Lock 420 hasp ----------------------------------------------------------------------------
JAW_R = 0.0125
"""Inside radius of the ring (1 in jaw, published)."""
JAW_W = 0.008
"""Radial width of the ring bars (inferred)."""
JAW_T = 0.002
"""Thickness of each jaw plate (inferred)."""
HASP_LEN = 0.114
TAIL_W = 0.040
TAIL_TOP = -0.032
TAIL_BOTTOM = JAW_R + JAW_W - HASP_LEN
"""The tail's lower edge (hasp frame): 114 mm below the ring's top."""
RIVET_Z = -0.026
HOLE = 0.008
"""Tail holes, square (a round 7 mm shackle limit; inferred 8 mm)."""
HOLE_COLS = (-0.011, 0.011)
HOLE_ROWS = tuple(TAIL_BOTTOM + 0.009 + 0.016 * k for k in range(3))
JAW_MASS = 0.03
HASP_DAMPING = 1.0
"""Body damping standing in for the friction of a real hasp in the handle's hole."""

# Switch handle (Square D heavy-duty style; all inferred) --------------------------------------------
LOCKOFF_HOLE = 0.012
PLATE_T = 0.003
TAB_X = (RING_X - 0.0045, RING_X - 0.0015)
"""World x range of the handle's lock-off tab."""
EAR_X = (RING_X + 0.0015, RING_X + 0.0045)
"""World x range of the fixed lock-off ear."""
RING_Z = BOTTOM_HOLE_Z - HOLE_ROWS[0]
"""World z of the ring centre when the ring's top bar rests on the lock-off holes' lower edges."""
LOCKOFF_Z = RING_Z + JAW_R + LOCKOFF_HOLE / 2
"""World z of the lock-off holes' centres (handle at OFF)."""
PIVOT_R = 0.05
PIVOT_Z = LOCKOFF_Z + PIVOT_R
PIVOT_X = RING_X - 0.009
HANDLE_ON_DEG = 90.0
LEVER_LEN = 0.10
ENCLOSURE = (0.12, (-0.22, 0.03), 0.44)
"""Depth behind the wall, y range, height."""

PEDESTAL_TOP = 0.138
HOLDER_SLOT = 0.007
"""Width of the hasp holder's slot (the closed tails are 4 mm thick)."""
HOLDER_DEPTH = 0.015
HOLDER_LEN = 0.05
PARK_X, PARK_Z = 0.30, -0.30

COLORS = dict(kit.COLORS)


def hole_local(col: int, row: int) -> tuple[float, float, float]:
    """A tail hole's centre in the hasp (jaw A) frame."""
    return (HOLE_COLS[col], 0.0, HOLE_ROWS[row])


def nominal_hasp_pos() -> tuple[float, float, float]:
    return (RING_X, HASP_Y, RING_Z)


def _rbox(stage, path, center, size, material, rot_y_deg=0.0, collide=True):
    """A box turned about its y axis."""
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


def _plate_with_holes(stage, path, x_range, z_range, holes_x, holes_z, hole, y_center, t, material):
    """A plate in the x-z plane (thickness t along y) with square holes on a grid, built from boxes."""
    x0, x1 = x_range
    z0, z1 = z_range
    edges = [z0]
    for zc in sorted(holes_z):
        edges += [zc - hole / 2, zc + hole / 2]
    edges.append(z1)
    k = 0
    for i in range(0, len(edges), 2):  # full-width bands between hole rows
        za, zb = edges[i], edges[i + 1]
        if zb - za > 1e-5:
            kit.box(stage, f"{path}_band{k}", ((x0 + x1) / 2, y_center, (za + zb) / 2), (x1 - x0, t, zb - za), material)
            k += 1
    xs = [x0]
    for xc in sorted(holes_x):
        xs += [xc - hole / 2, xc + hole / 2]
    xs.append(x1)
    for zc in holes_z:  # bars beside the holes
        for i in range(0, len(xs), 2):
            xa, xb = xs[i], xs[i + 1]
            kit.box(stage, f"{path}_bar{k}", ((xa + xb) / 2, y_center, zc), (xb - xa, t, hole), material)
            k += 1


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


def _damped(prim, damping):
    prim.AddAppliedSchema("PhysxRigidBodyAPI")
    prim.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(damping)
    prim.CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float).Set(damping)


CONTACT_OFFSET = 0.002
SOLVER_ITERATIONS = 32


def _thin_contacts(stage, root_prim, paths_prefix: str) -> None:
    """The hasp's plates are 2-3 mm thick and carry up to ~30 N in the function test: small contact offsets
    and more solver iterations keep PhysX from pushing them through each other."""
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
        SOLVER_ITERATIONS)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(paths_prefix) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def make_hasp(out_path: Path) -> None:
    root = "/hasp"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim)
    mats = kit.materials(stage, root, ("steel", "red", "dark"))
    jaw_a = kit.rigid_body(stage, f"{root}/jaw_a", (0, 0, 0), JAW_MASS)
    jaw_b = kit.rigid_body(stage, f"{root}/jaw_b", (0, 0, RIVET_Z), JAW_MASS)
    for body in (jaw_a, jaw_b):
        _damped(body.GetPrim(), HASP_DAMPING)
    rc = JAW_R + JAW_W / 2
    seg_len = 2 * (JAW_R + JAW_W) * math.tan(math.radians(22.5))
    owners = {"a": (90, 135, 180, 225), "b": (45, 0, -45)}
    for name, angles in owners.items():
        y = JAW_T / 2 if name == "a" else -JAW_T / 2
        dz = 0.0 if name == "a" else -RIVET_Z
        path = f"{root}/jaw_{name}"
        for a in angles:
            ar = math.radians(a)
            _rbox(stage, f"{path}/ring_{a + 90}", (rc * math.cos(ar), y, rc * math.sin(ar) + dz),
                  (seg_len, JAW_T, JAW_W), mats["steel"], rot_y_deg=-(a + 90))
        # Neck from the ring's bottom to the tail (closes the ring at the bottom), steel.
        kit.box(stage, f"{path}/neck", (0, y, (-JAW_R + TAIL_TOP) / 2 + dz), (0.018, JAW_T, -JAW_R - TAIL_TOP),
                mats["steel"])
        _plate_with_holes(stage, f"{path}/tail", (-TAIL_W / 2, TAIL_W / 2), (TAIL_BOTTOM + dz, TAIL_TOP + dz),
                          HOLE_COLS, [z + dz for z in HOLE_ROWS], HOLE, y, JAW_T, mats["red"])
    kit.cylinder(stage, f"{root}/jaw_a/rivet", (0, 0, RIVET_Z), 0.004, 2 * JAW_T + 0.002, mats["dark"], axis="Y")
    joint = kit.revolute(stage, f"{root}/joints/jaw_joint", jaw_a.GetPath(), jaw_b.GetPath(), "Y", (0, 0, RIVET_Z),
                         0.0, 40.0)
    _thin_contacts(stage, root_prim, root)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")
    return joint


def make_switch(out_path: Path) -> None:
    """Enclosure side wall, fixed lock-off ear, and the operating handle. Frame: world shifted by
    (WALL_X, HASP_Y, 0)."""
    root = "/switch"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim)
    mats = kit.materials(stage, root, ("grey", "light_grey", "steel", "red", "black", "white", "dark", "enclosure"))
    base = kit.rigid_body(stage, f"{root}/enclosure", (0, 0, 0), 10.0)
    depth, (y0, y1), height = ENCLOSURE
    kit.box(stage, f"{root}/enclosure/box", (depth / 2, (y0 + y1) / 2, height / 2), (depth, y1 - y0, height), mats["grey"])
    kit.fix_to_world(stage, root, base.GetPath())
    wx = lambda x: x - WALL_X  # noqa: E731  world x -> switch frame
    lo, hi = LOCKOFF_Z - LOCKOFF_HOLE / 2, LOCKOFF_Z + LOCKOFF_HOLE / 2
    # Fixed lock-off ear: a plate parallel to the wall with the lock-off hole, on a standoff arm above
    # the hole (the hasp ring passes between the ear and the wall).
    _yz_plate_with_hole(stage, f"{root}/enclosure/ear", (wx(EAR_X[0]), wx(EAR_X[1])), 0.012,
                        (lo - 0.004, hi + 0.014), LOCKOFF_Z, LOCKOFF_HOLE, mats["steel"])
    kit.box(stage, f"{root}/enclosure/ear_arm", (wx(EAR_X[0]) / 2, 0, hi + 0.011), (-wx(EAR_X[0]), 0.024, 0.006),
            mats["steel"])
    # The operating mechanism's boss: the handle's shaft comes out of the wall here (fixed; the handle turns on it).
    kit.box(stage, f"{root}/enclosure/boss", ((wx(RING_X + 0.006)) / 2, 0, PIVOT_Z), (-wx(RING_X + 0.006), 0.03, 0.03),
            mats["dark"])
    # ON / OFF marks on the wall (visual): OFF above the pivot (lever up), ON toward -y.
    kit.box(stage, f"{root}/enclosure/mark_off", (-0.0005, 0, PIVOT_Z + LEVER_LEN + 0.012), (0.001, 0.02, 0.008),
            mats["white"], collide=False)
    kit.box(stage, f"{root}/enclosure/mark_on", (-0.0005, -LEVER_LEN - 0.012, PIVOT_Z), (0.001, 0.008, 0.02),
            mats["red"], collide=False)
    # Operating handle: body origin at the pivot.
    px = wx(PIVOT_X)
    handle = kit.rigid_body(stage, f"{root}/handle", (px, 0, PIVOT_Z), 0.15)
    hx = lambda x: wx(x) - px  # noqa: E731  world x -> handle frame
    tab_lo = lo - 0.004 - PIVOT_Z
    _yz_plate_with_hole(stage, f"{root}/handle/tab", (hx(TAB_X[0]), hx(TAB_X[1])), 0.012, (tab_lo, 0.0),
                        LOCKOFF_Z - PIVOT_Z, LOCKOFF_HOLE, mats["steel"])
    hub_x = (hx(RING_X - 0.016), hx(TAB_X[1]))
    kit.box(stage, f"{root}/handle/hub", ((hub_x[0] + hub_x[1]) / 2, 0, 0), (hub_x[1] - hub_x[0], 0.024, 0.024),
            mats["dark"])
    lever_x = (hx(RING_X - 0.016), hx(RING_X - 0.005))
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


def make_pedestal(out_path: Path) -> None:
    root = "/pedestal"
    stage, _ = kit.new_stage(out_path, root)
    mats = kit.materials(stage, root, ("stand",))
    x, y = PEDESTAL_XY
    kit.box(stage, f"{root}/foot", (x, y, 0.005), (0.05, 0.06, 0.01), mats["stand"])
    kit.box(stage, f"{root}/column", (x, y, PEDESTAL_TOP / 2), (0.012, 0.038, PEDESTAL_TOP), mats["stand"])
    # Hasp holder: a grey block with a slot along x that the hasp's closed tails stand in, ring up, in the same
    # orientation as on the handle.
    hx, hy = RING_X + HASP_START_DXY[0], HASP_Y + HASP_START_DXY[1]
    floor = RING_Z + HASP_START_DZ + TAIL_BOTTOM
    wall_t = 0.007
    kit.box(stage, f"{root}/holder_foot", (hx, hy, 0.005), (0.07, 0.05, 0.01), mats["stand"])
    kit.box(stage, f"{root}/holder_column", (hx, hy, floor / 2), (HOLDER_LEN, HOLDER_SLOT + 2 * wall_t, floor),
            mats["stand"])
    for k, sgn in enumerate((-1, 1)):
        kit.box(stage, f"{root}/holder_wall{k}", (hx, hy + sgn * (HOLDER_SLOT / 2 + wall_t / 2), floor + HOLDER_DEPTH / 2),
                (HOLDER_LEN, wall_t, HOLDER_DEPTH), mats["stand"])
    kit.box(stage, f"{root}/park_shelf", (PARK_X, 0.0, PARK_Z - 0.005), (0.12, 0.60, 0.01), mats["stand"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    make_switch(out_dir / "switch.usda")
    make_hasp(out_dir / "hasp.usda")
    make_pedestal(out_dir / "pedestal.usda")
    padlock_410.make_padlock_410(out_dir / "padlock_ml410.usda", padlock_props.PADLOCKS["ml410"])

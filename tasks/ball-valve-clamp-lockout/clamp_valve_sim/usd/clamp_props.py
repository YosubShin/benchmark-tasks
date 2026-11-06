"""The clamp-on ball valve lockout (Master Lock S3080 adjustable ball valve lockout) as a USD articulation.

How the real S3080 works (Master Lock sell sheet 7000-0206, product page and how-to videos; see README.md): two red
thermoplastic halves that slide together. Half A is a channel that goes over the closed valve's lever, with a pair
of legs at its stem end; half B is a second pair of legs with a housing that slides along a tongue of half A. Fitted
on a valve, the lever lies in A's channel, A's legs stand on the lever's side of the pipe and B's legs on the far side,
the legs of each pair beside the two ends of the valve body. Pushed together, the two leg pairs grip the pipe between
them (adjustable for 1/2 in to 2 in valves); the padlock's shackle then goes through the aligned holes of B's housing
(four holes, one per setting) and A's tongue, so the halves cannot slide apart. With the legs astride the pipe the
device cannot turn about the stem, and the lever, held in the channel, cannot turn either. Published: 134.6 mm long,
76.9 mm wide, 104.1 mm tall, 156 g, polycarbonate and PBT, four locking holes. Everything else (wall thicknesses,
hole size and pitch, the tongue) is inferred from the photos and videos.

Frame (the device as fitted on the valve, half B pushed to its first setting, q = 0): origin on the valve's stem axis
at the lever's mid-plane, axes as the world's (+x away from the robot, the closed lever along -x, the pipe along y).
Half A is the articulation's root body; half B slides on a prismatic joint along +x (q = how far B's legs are out from
the first setting, 0 to B_TRAVEL). Self collisions are off: the joint's limits are the slide's ends, and the padlock
is what collides with both halves.

Usage: imported by valve_props (``make_device``).
"""

from __future__ import annotations

from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from clamp_valve_sim.usd import usd_kit as kit

# Published outline ---------------------------------------------------------------------------------------------
LENGTH, WIDTH, HEIGHT = 0.1346, 0.0769, 0.1041
MASS_A, MASS_B = 0.095, 0.061
"""156 g in all (published); the split is inferred."""

# Inferred geometry (device frame) ---------------------------------------------------------------------------
TOP_Z = 0.0035
"""Underside of the channel's top plate and of the crossbars: 0.5 mm above the lever's grip (6 mm thick), which it
rests on."""
PLATE_T = 0.005
CH_X = (-0.100, -0.0115)
CH_IN, CH_WALL = 0.013, 0.003
"""Channel: 26 mm inside (it takes levers of 1/2 to 2 in valves; the removable insert, not modelled, narrows it), 3 mm
walls."""
CH_WALL_BOTTOM = -0.010
LEG_X_A = (-0.0215, -0.0115)
LEG_X_B = (0.0115, 0.0215)
"""Leg pairs 10 mm thick along x; at the first setting their inner faces are 23 mm apart (the 21.3 mm pipe fits
with 0.85 mm each side)."""
LEG_Y = (0.0315, 0.0385)
"""Legs beside the valve body's ends (the body is 56 mm long, so 3.5 mm clear of it each side); 77 mm over both legs
(published width)."""
LEG_BOTTOM = TOP_Z + PLATE_T + 0.0150 - HEIGHT
"""Device height: B's housing top to the leg tips = 104.1 mm (published)."""
# Locking: B's housing wall (outer) and A's tongue (inner) on the device's right side (-y), holes along y.
WALL_Y = (-0.0385, -0.0355)
TONGUE_Y = (-0.0350, -0.0315)
HOLE = 0.009
HOLE_Z = 0.012
"""Square 9 mm holes (Master Lock lockouts take shackles up to about 7 mm; the 410's is 6.35 mm), centre 12 mm
above the lever's mid-plane, over the stem nut."""
HOLE_PITCH = 0.012
N_HOLES = 4
TONGUE_X = (LEG_X_A[0], 0.016)
TONGUE_Z = (TOP_Z, HOLE_Z + HOLE / 2 + 0.004)
WALL_Z = (-0.006, TONGUE_Z[1])
HOUSING_TOP_T = 0.003
HOUSING_X = (-(N_HOLES - 1) * HOLE_PITCH - HOLE / 2 - 0.004, LEG_X_B[1])
B_TRAVEL = 0.039
"""B slides out to 62 mm between the leg pairs (a 2 in valve)."""
B_START = 0.017
"""The device starts opened to 40 mm between the leg pairs."""
REST_TOP_BELOW_A = CH_WALL_BOTTOM
"""The rest block's top is level with the channel walls' bottom edge."""
DEVICE_START_Z = -LEG_BOTTOM + 0.0005
"""A's origin height when the device stands on its legs on the table."""
REST_TOP = DEVICE_START_Z + REST_TOP_BELOW_A - 0.0005

RED = (0.80, 0.07, 0.06)


def hole_x_b(k: int) -> float:
    """Centre x of B's hole k in B's frame (hole k lines up with A's tongue hole at q = k * HOLE_PITCH)."""
    return -k * HOLE_PITCH


def _box(stage, path, lo, hi, mat):
    c = tuple((a + b) / 2 for a, b in zip(lo, hi))
    s = tuple(b - a for a, b in zip(lo, hi))
    return kit.box(stage, path, c, s, mat)


def _plate_with_holes(stage, path, x_range, y_range, z_range, holes_x, mat):
    """A plate in the x-z plane (thickness along y) with square holes of side HOLE at (x, HOLE_Z)."""
    x0, x1 = x_range
    z0, z1 = z_range
    h0, h1 = HOLE_Z - HOLE / 2, HOLE_Z + HOLE / 2
    _box(stage, f"{path}_below", (x0, y_range[0], z0), (x1, y_range[1], h0), mat)
    _box(stage, f"{path}_above", (x0, y_range[0], h1), (x1, y_range[1], z1), mat)
    edges = sorted(holes_x)
    cuts = [x0] + [e for hx in edges for e in (hx - HOLE / 2, hx + HOLE / 2)] + [x1]
    for k in range(0, len(cuts), 2):
        a, b = cuts[k], cuts[k + 1]
        if b - a > 1e-5:
            _box(stage, f"{path}_web{k // 2}", (a, y_range[0], h0), (b, y_range[1], h1), mat)


def make_device(out_path: Path) -> None:
    root = "/s3080"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    mat = UsdShade.Material.Define(stage, f"{root}/Looks/red")
    shader = UsdShade.Shader.Define(stage, f"{root}/Looks/red/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*RED))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    top = TOP_Z + PLATE_T

    # Half A: channel, crossbar, legs, tongue.
    a = f"{root}/half_a"
    kit.rigid_body(stage, a, (0, 0, 0), MASS_A)
    _box(stage, f"{a}/channel_top", (CH_X[0], -CH_IN - CH_WALL, TOP_Z), (CH_X[1], CH_IN + CH_WALL, top), mat)
    for k, sgn in enumerate((-1, 1)):
        y0, y1 = sorted((sgn * CH_IN, sgn * (CH_IN + CH_WALL)))
        _box(stage, f"{a}/channel_wall{k}", (CH_X[0], y0, CH_WALL_BOTTOM), (CH_X[1], y1, TOP_Z), mat)
    _box(stage, f"{a}/crossbar", (LEG_X_A[0], TONGUE_Y[1], TOP_Z), (LEG_X_A[1], LEG_Y[1], top), mat)
    _box(stage, f"{a}/leg_left", (LEG_X_A[0], LEG_Y[0], LEG_BOTTOM), (LEG_X_A[1], LEG_Y[1], TOP_Z), mat)
    _box(stage, f"{a}/leg_right", (LEG_X_A[0], TONGUE_Y[0], LEG_BOTTOM), (LEG_X_A[1], TONGUE_Y[1], TOP_Z), mat)
    _plate_with_holes(stage, f"{a}/tongue", TONGUE_X, TONGUE_Y, TONGUE_Z, [0.0], mat)

    # Half B: crossbar, legs, housing (outer wall with four holes, top plate).
    b = f"{root}/half_b"
    kit.rigid_body(stage, b, (0, 0, 0), MASS_B)
    _box(stage, f"{b}/crossbar", (LEG_X_B[0], TONGUE_Y[1], TOP_Z), (LEG_X_B[1], LEG_Y[1], top), mat)
    _box(stage, f"{b}/leg_left", (LEG_X_B[0], LEG_Y[0], LEG_BOTTOM), (LEG_X_B[1], LEG_Y[1], TOP_Z), mat)
    _box(stage, f"{b}/leg_right_out", (LEG_X_B[0], WALL_Y[0], LEG_BOTTOM), (LEG_X_B[1], WALL_Y[1], WALL_Z[0]), mat)
    _plate_with_holes(stage, f"{b}/wall", HOUSING_X, WALL_Y, WALL_Z, [hole_x_b(k) for k in range(N_HOLES)], mat)
    _box(stage, f"{b}/housing_top", (HOUSING_X[0], WALL_Y[0], WALL_Z[1]), (HOUSING_X[1], TONGUE_Y[1],
                                                                             WALL_Z[1] + HOUSING_TOP_T), mat)
    kit.prismatic(stage, f"{root}/joints/slide_joint", a, b, "X", (0, 0, 0), 0.0, B_TRAVEL)
    kit.friction(stage, root, [a, b], static=0.5, dynamic=0.4, name="plastic", combine="average")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.001)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")

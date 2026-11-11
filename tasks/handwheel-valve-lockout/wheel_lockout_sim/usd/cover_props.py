"""The rotating gate valve lockout (Master Lock 480, for 1 to 3 in handwheels) as a USD articulation.

How the real 480 works (Master Lock product page and technical data, product photos, the 480-484 how-to video; see
README.md): a red thermoplastic cover in two pieces that together enclose a valve handwheel. Each piece is a hollow
sector of a flat drum: a front face, a back face and a rim wall between them. The fixed piece (A, with the "DANGER"
label) leaves an open sector; its back face has a stem opening at the centre (19 mm, published) reached through that
sector. The rotating piece (B) turns about the drum's axis on a pivot at the centre of the front face; opened, it nests
inside A, closed, it fills A's open sector. Each piece has a flat padlock ear with two holes in the plane of its back
face (product photos). Fitting: with B nested, A is slid sideways over the handwheel, open sector first, so the wheel
passes between the faces and the stem slides into the stem opening; then B is swung round until its ear lies against
A's. A padlock through the lined-up ear holes stops B swinging back. Closed, the drum encloses the wheel front, back
and round its rim: the wheel cannot be gripped, the drum turns freely on the stem, and the back face behind the rim
stops it being pulled off. Published: fits 1 to 3 in (25 to 76 mm) handwheels, depth 1-1/16 in (30 mm), stem opening
3/4 in (19 mm), Zenex thermoplastic, takes several padlocks. Everything else (outer diameter, wall thickness, sector
angles, ear size, hole size and position, mass) is inferred from the product photos and the video.

Frame (fitted on the wheel, B closed, q = 0): origin on the drum axis at mid-depth, z along the axis (up on the
valve); the ears point at the robot (-x); angles phi are measured from +x toward +y. A covers phi in [180, 410] deg
(the right side and the front), open sector [50, 180] (left). B covers [48, 182] at q = 0 and turns about z; q in
[Q_OPEN, 0] (negative = opened, clockwise seen from above). B's ear runs under A's back face and, closed, lies just
under A's ear; their holes run along z. A's pivot boss, through a hole in B's front face, rests on the wheel nut.

Collision shapes: convex hulls of 10 degree annular sectors, and boxes for the ears and the boss.
"""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from wheel_lockout_sim.usd import usd_kit as kit

# Inferred geometry (cover frame) --------------------------------------------------------------------------------
A_R = (0.0410, 0.0430)
"""A's rim wall, inner and outer radius: 86 mm across (fits up to a 76 mm wheel inside B)."""
B_R = (0.0380, 0.0400)
A_FACE_Z = 0.0180
"""A's faces: inner surfaces at +-18 mm, 2.5 mm thick (outer depth 41 mm)."""
B_FACE_Z = 0.0150
"""B's faces: inner surfaces at +-15 mm (30 mm inside depth, the published depth), 2.5 mm thick, nested 0.5 mm inside
A's."""
FACE_T = 0.0025
BOSS_Z = 0.0145
"""A's pivot boss: hangs from A's front face through a hole in B's front face; its underside (14.5 mm above the
drum's mid-plane, below B's face) is what the cover rests on, on the wheel nut, so B swings freely under A."""
BOSS_HALF = 0.007
B_FACE_HOLE = 0.0105
STEM_R = 0.0095
"""Stem opening, 19 mm (published), in both back faces."""
A_PHI = (180.0, 410.0)
B_PHI = (48.0, 182.0)
Q_OPEN = -124.0
"""B opened: nested inside A, its tab just clear of A's rim end (phi = 52 to 56 deg)."""
EAR_X = (-0.076, -0.039)
"""Padlock ears: flat plates in the back faces' planes at phi = 180 deg (pointing at the robot), from inside the rims
to 76 mm from the axis. Closed, B's ear lies on A's ear and their holes line up along the drum's axis (product
photos: two-hole ears, stacked)."""
EAR_HALF_W = 0.008
A_EAR_Z = (-A_FACE_Z - FACE_T, -A_FACE_Z)
B_EAR_Z = (A_EAR_Z[0] - 0.0005 - FACE_T, A_EAR_Z[0] - 0.0005)
"""B's ear runs under A's back face (B nests inside A, so its ear reaches out beneath A's), and closed it lies just
under A's ear."""
HOLE = 0.009
"""Square 9 mm holes (Master Lock rates its valve lockouts for shackles up to 5/16 in)."""
HOLE_XS = (-0.056, -0.067)
"""Two holes in a row along the ear; the padlock goes through the inner one."""
STACK_Z = (B_EAR_Z[0], A_EAR_Z[1])
MASS_A, MASS_B = 0.065, 0.040
"""Not published; inferred from a 2.5 mm thermoplastic shell this size."""

RED = (0.78, 0.06, 0.05)
SEG_DEG = 10.0


def _mat(stage, root, name, rgb, rough=0.45):
    path = f"{root}/Looks/{name}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def sector_points(r0, r1, z0, z1, phi0_deg, phi1_deg, n=5):
    """Vertices of an annular sector prism: n points along each arc, inner/outer, bottom/top."""
    pts = []
    for z in (z0, z1):
        for r in (r0, r1):
            for k in range(n):
                a = math.radians(phi0_deg + (phi1_deg - phi0_deg) * k / (n - 1))
                pts.append((r * math.cos(a), r * math.sin(a), z))
    return pts


def _sector_mesh(stage, path, r0, r1, z0, z1, phi0, phi1, mat, collide=True, n=5):
    pts = sector_points(r0, r1, z0, z1, phi0, phi1, n)

    def idx(zi, ri, k):
        return zi * 2 * n + ri * n + k

    counts, faces = [], []
    for k in range(n - 1):
        # bottom, top, inner, outer quads
        for quad in ((idx(0, 0, k), idx(0, 0, k + 1), idx(0, 1, k + 1), idx(0, 1, k)),
                     (idx(1, 0, k), idx(1, 1, k), idx(1, 1, k + 1), idx(1, 0, k + 1)),
                     (idx(0, 0, k), idx(1, 0, k), idx(1, 0, k + 1), idx(0, 0, k + 1)),
                     (idx(0, 1, k), idx(0, 1, k + 1), idx(1, 1, k + 1), idx(1, 1, k))):
            counts.append(4)
            faces.extend(quad)
    for k, flip in ((0, False), (n - 1, True)):
        quad = [idx(0, 0, k), idx(0, 1, k), idx(1, 1, k), idx(1, 0, k)]
        counts.append(4)
        faces.extend(reversed(quad) if flip else quad)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    return mesh


def _segments(phi0, phi1, step=SEG_DEG):
    n = max(1, math.ceil((phi1 - phi0) / step - 1e-6))
    return [(phi0 + (phi1 - phi0) * k / n, phi0 + (phi1 - phi0) * (k + 1) / n) for k in range(n)]


def _shell(stage, path, phis, r_rim, face_z, mat, front_hole=0.0):
    """Front face (with a centre hole of radius front_hole), back face (with the stem opening) and rim wall over the
    angle range phis."""
    r0, r1 = r_rim
    for k, (a, b) in enumerate(_segments(*phis)):
        _sector_mesh(stage, f"{path}/front{k}", front_hole, r1, face_z, face_z + FACE_T, a, b, mat)
        _sector_mesh(stage, f"{path}/back{k}", STEM_R, r1, -face_z - FACE_T, -face_z, a, b, mat)
        _sector_mesh(stage, f"{path}/rim{k}", r0, r1, -face_z, face_z, a, b, mat)


def _box(stage, path, lo, hi, mat):
    c = tuple((p + q) / 2 for p, q in zip(lo, hi))
    s = tuple(q - p for p, q in zip(lo, hi))
    return kit.box(stage, path, c, s, mat)


def _ear(stage, path, z_range, x0, mat):
    """A flat plate (thickness along z) with two square holes along x, at HOLE_XS."""
    x1 = EAR_X[0]
    w = EAR_HALF_W
    h = HOLE / 2
    _box(stage, f"{path}_side0", (x1, -w, z_range[0]), (x0, -h, z_range[1]), mat)
    _box(stage, f"{path}_side1", (x1, h, z_range[0]), (x0, w, z_range[1]), mat)
    xs = sorted(HOLE_XS)
    cuts = [x1, xs[0] - h, xs[0] + h, xs[1] - h, xs[1] + h, x0]
    for k in range(0, len(cuts), 2):
        _box(stage, f"{path}_web{k // 2}", (cuts[k], -h, z_range[0]), (cuts[k + 1], h, z_range[1]), mat)


def _label(stage, path, mat):
    """The DANGER / DO NOT OPERATE label on A's front face (visual)."""
    z = A_FACE_Z + FACE_T + 0.0003
    pts, n = [], 9
    phi0, phi1 = 200.0, 390.0
    for k in range(n):
        a = math.radians(phi0 + (phi1 - phi0) * k / (n - 1))
        for r in (0.016, 0.036):
            pts.append((r * math.cos(a), r * math.sin(a), z))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    counts, faces, st = [], [], []
    for k in range(n - 1):
        counts.append(4)
        faces.extend([2 * k, 2 * k + 1, 2 * k + 3, 2 * k + 2])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return mesh


def make_cover(out_path: Path) -> None:
    root = "/ml480"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    red = _mat(stage, root, "red", RED, 0.4)
    red_b = _mat(stage, root, "red_b", (0.70, 0.05, 0.04), 0.4)
    white = _mat(stage, root, "label", (0.95, 0.95, 0.93), 0.6)

    a = f"{root}/half_a"
    kit.rigid_body(stage, a, (0, 0, 0), MASS_A)
    _shell(stage, a, A_PHI, A_R, A_FACE_Z, red)
    _ear(stage, f"{a}/ear", A_EAR_Z, -A_R[0] - 0.0005, red)
    _label(stage, f"{a}/label", white)
    # The pivot boss at the centre of A's front face (visual on top; below, the collider the cover rests on).
    kit.cylinder(stage, f"{a}/boss", (0, 0, A_FACE_Z + FACE_T + 0.0015), 0.011, 0.003, red)
    _box(stage, f"{a}/boss_pin", (-BOSS_HALF, -BOSS_HALF, BOSS_Z), (BOSS_HALF, BOSS_HALF, A_FACE_Z), red)

    b = f"{root}/half_b"
    kit.rigid_body(stage, b, (0, 0, 0), MASS_B)
    _shell(stage, b, B_PHI, B_R, B_FACE_Z, red_b, front_hole=B_FACE_HOLE)
    _ear(stage, f"{b}/ear", B_EAR_Z, -B_R[0] - 0.0005, red_b)
    kit.revolute(stage, f"{root}/joints/swing_joint", a, b, "Z", (0, 0, 0), Q_OPEN, 0.0)
    kit.friction(stage, root, [a, b], static=0.5, dynamic=0.4, name="plastic", combine="average")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.001)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def hole_in_cover(k: int = 0) -> tuple[float, float, float]:
    """Centre of ear hole k (0 = inner), cover frame, mid-way through the two stacked ears."""
    return (HOLE_XS[k], 0.0, (STACK_Z[0] + STACK_Z[1]) / 2)

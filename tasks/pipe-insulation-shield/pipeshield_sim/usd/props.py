"""Generate the props of the pipe insulation shield task as USD (dimensions and sources: pipeshield_sim/spec.py).

Files (desk frame, see spec.py):
- ``rig.usda`` (fixed): two B-Line B3100-2-1/2 clevis hangers on 1/2 in rods from an overhead strut (A at y = 0 with
  no shield; B at y = -0.58 with its B3151 shield already in), and a wooden block the loose shield lies on. The
  lower straps (1 in wide) are 10 degree strips round the bottom and straight legs up to the cross bolt; colliders.
- ``pipe.usda``: a free rigid body: 1.24 m of 1/2 in copper tube with 1 in fiberglass insulation (2.83 in OD), the
  insulation one convex 32-sided prism; mass of the tube full of water plus insulation. Origin at the section's
  middle on its axis.
- ``shield.usda``: a free rigid body: the B3151-3X shield, a half-round of 18 ga sheet 12 in long (18 flat strips of
  10 degrees), a 12 mm red tape mark round its middle, inside and out. Origin on its axis at its middle (the centre of
  its curvature), open side up.

Usage: python -m pipeshield_sim.usd.props pipeshield_sim/assets
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from pipeshield_sim import spec as S
from pipeshield_sim.usd import usd_kit as kit

COLORS = {
    "galv": (0.70, 0.72, 0.74),
    "zinc": (0.78, 0.80, 0.82),
    "strut": (0.55, 0.57, 0.60),
    "jacket": (0.93, 0.93, 0.90),
    "copper": (0.80, 0.45, 0.20),
    "red": (0.85, 0.04, 0.03),
    "wood": (0.72, 0.56, 0.36),
    "dark": (0.15, 0.15, 0.16),
}
METAL = ("galv", "zinc", "strut", "copper")


def mats(stage, root):
    out = {}
    for name, rgb in COLORS.items():
        path = f"{root}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9 if name in ("red", "wood", "jacket") else 0.45)
        if name in METAL:
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.5)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        out[name] = material
    return out


def span(stage, path, xr, yr, zr, material, collide=True):
    return kit.box(stage, path, ((xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2, (zr[0] + zr[1]) / 2),
                   (xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]), material, collide=collide)


def tilted_box(stage, path, center, size, rot_y_deg, material, collide=True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddRotateYOp().Set(float(rot_y_deg))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def arc_strips(stage, path, cx, cz, r_in, t, y0, y1, material, n=18, a0=180.0, a1=360.0, collide=True):
    """A circular arc of sheet (inside radius r_in, thickness t) between angles a0..a1 (x-z plane, degrees from +x,
    270 = straight down), as n flat strips, extruded along y from y0 to y1."""
    r_mid = r_in + t / 2
    step = (a1 - a0) / n
    width = 2 * r_mid * math.tan(math.radians(step / 2)) + 0.0004
    for k in range(n):
        th = a0 + step * (k + 0.5)
        c = (cx + r_mid * math.cos(math.radians(th)), (y0 + y1) / 2, cz + r_mid * math.sin(math.radians(th)))
        tilted_box(stage, f"{path}_{k:02d}", c, (width, y1 - y0, t), -(th + 90.0), material, collide)


def xz_prism(stage, path, poly_xz, y0, y1, material, collide=True):
    pts, faces, counts = [], [], []
    n = len(poly_xz)
    for y in (y0, y1):
        pts += [Gf.Vec3f(float(x), float(y), float(z)) for x, z in poly_xz]
    faces += list(range(n))
    counts.append(n)
    for k in range(n):
        k1 = (k + 1) % n
        faces += [k1, k, n + k, n + k1]
        counts.append(4)
    faces += list(range(2 * n - 1, n - 1, -1))
    counts.append(n)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    return mesh


def ngon(radius_flat, n, phase_deg=0.0):
    r = radius_flat / math.cos(math.pi / n)
    return [(r * math.cos(math.radians(phase_deg) + 2 * math.pi * (k + 0.5) / n),
             r * math.sin(math.radians(phase_deg) + 2 * math.pi * (k + 0.5) / n)) for k in range(n)]


def physics_material(stage, root, name, mu, combine="average"):
    mat = UsdShade.Material.Define(stage, f"{root}/PhysicsMaterials/{name}")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(mu)
    api.CreateDynamicFrictionAttr(mu)
    mat.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    mat.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(combine)
    return mat


def bind(stage, path, mat):
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(mat, UsdShade.Tokens.weakerThanDescendants,
                                                                       "physics")


def thin_contacts(stage, offset=0.001):
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(offset)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


# -- rig ----------------------------------------------------------------------------------------------------------------
def clevis(stage, path, y, z_cb, m):
    """A B3100 clevis: lower strap (bottom arc and legs, colliders), cross bolt (collider), upper yoke, rod, nuts."""
    x = S.PIPE_X
    cz = z_cb + S.CLEVIS_R
    t, w = S.STRAP_T, S.STRAP_W
    y0, y1 = y - w / 2, y + w / 2
    arc_strips(stage, f"{path}/strap", x, cz, S.CLEVIS_R, t, y0, y1, m["zinc"], n=18)
    z_bolt = z_cb + S.INS_R + S.BOLT_ABOVE_PIPE
    for k, sx in enumerate((-1, 1)):
        xs = sorted((x + sx * S.CLEVIS_R, x + sx * (S.CLEVIS_R + t)))
        span(stage, f"{path}/leg{k}", xs, (y0, y1), (cz, z_bolt + 0.010), m["zinc"])
    span(stage, f"{path}/bolt", (x - S.CLEVIS_R - t - 0.008, x + S.CLEVIS_R + t + 0.008),
         (y - S.BOLT_D / 2, y + S.BOLT_D / 2), (z_bolt - S.BOLT_D / 2, z_bolt + S.BOLT_D / 2), m["zinc"])
    kit.cylinder(stage, f"{path}/bolt_vis", (x, y, z_bolt), S.BOLT_D / 2 + 0.0003, 2 * (S.CLEVIS_R + t + 0.008),
                 m["zinc"], axis="X")
    for k, sx in enumerate((-1, 1)):
        kit.cylinder(stage, f"{path}/bolt_nut{k}", (x + sx * (S.CLEVIS_R + t + 0.011), y, z_bolt), 0.011, 0.008,
                     m["zinc"], axis="X")
    # Upper yoke (formed over the lower strap's legs, to the rod) and the rod with its nuts.
    top = z_cb + S.INS_R + S.SH_T + S.TOP_ABOVE_PIPE
    xo = S.CLEVIS_R + t
    for k, sx in enumerate((-1, 1)):
        span(stage, f"{path}/yoke{k}", sorted((x + sx * xo, x + sx * (xo + t))), (y0 - 0.001, y1 + 0.001),
             (z_bolt - 0.012, top - 0.020), m["zinc"], collide=False)
        span(stage, f"{path}/yoke_slope{k}", sorted((x + sx * 0.008, x + sx * (xo + t))), (y0 - 0.001, y1 + 0.001),
             (top - 0.020, top - 0.020 + t), m["zinc"], collide=False)
    span(stage, f"{path}/yoke_top", (x - 0.012, x + 0.012), (y0 - 0.001, y1 + 0.001), (top - 0.020, top), m["zinc"],
         collide=False)
    kit.cylinder(stage, f"{path}/rod", (x, y, (top - 0.010 + S.BEAM_Z) / 2), S.ROD_D / 2, S.BEAM_Z - top + 0.010,
                 m["zinc"])
    for k, zz in enumerate((top + 0.006, S.BEAM_Z - 0.006)):
        prism_z(stage, f"{path}/rod_nut{k}", [(x + a, y + b) for a, b in ngon(0.0095, 6)], zz - 0.005, zz + 0.005,
                m["zinc"])


def prism_z(stage, path, poly, z0, z1, material):
    pts = [Gf.Vec3f(float(px), float(py), float(z)) for z in (z0, z1) for px, py in poly]
    n = len(poly)
    faces, counts = list(range(n - 1, -1, -1)), [n]
    for k in range(n):
        k1 = (k + 1) % n
        faces += [k, k1, n + k1, n + k]
        counts.append(4)
    faces += list(range(n, 2 * n))
    counts.append(n)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def make_rig(out_path: Path) -> None:
    root = "/rig"
    stage, root_prim = kit.new_stage(out_path, root)
    m = mats(stage, root)
    clevis(stage, f"{root}/hanger_a", S.HANGER_A_Y, S.Z_CB, m)
    clevis(stage, f"{root}/hanger_b", S.HANGER_B_Y, S.Z_CB_B, m)
    # Hanger B's shield, already in (the same B3151-3X).
    arc_strips(stage, f"{root}/shield_b/strip", S.PIPE_X, S.Z_CB_B + S.SH_RO, S.SH_RI, S.SH_T,
               S.HANGER_B_Y - S.SH_L / 2, S.HANGER_B_Y + S.SH_L / 2, m["galv"], n=S.SH_SEGS)
    # Overhead strut (1-5/8 in) the rods hang from, on two posts behind the pipe (visual).
    sw = 1.625 * IN_
    span(stage, f"{root}/beam", (S.PIPE_X - sw / 2, S.PIPE_X + sw / 2), (-0.70, 0.66), (S.BEAM_Z, S.BEAM_Z + sw),
         m["strut"], collide=False)
    for k, py in enumerate((-0.68, 0.64)):
        span(stage, f"{root}/post{k}", (0.58, 0.58 + sw), (py - sw / 2, py + sw / 2), (0.0, S.BEAM_Z + sw), m["strut"],
             collide=False)
        span(stage, f"{root}/arm{k}", (S.PIPE_X + sw / 2, 0.58), (py - sw / 2, py + sw / 2),
             (S.BEAM_Z, S.BEAM_Z + sw), m["strut"], collide=False)
    # A wooden post (a 2x2 offcut on end) the loose shield lies balanced on.
    bx, by, bz = S.BLOCK_SIZE
    span(stage, f"{root}/block0", (S.SH0_X - bx / 2, S.SH0_X + bx / 2), (S.SH0_Y - by / 2, S.SH0_Y + by / 2),
         (0.0, bz), m["wood"])
    bind(stage, root, physics_material(stage, root, "galvanized", 0.30))
    bind(stage, f"{root}/block0", physics_material(stage, root, "wood", 0.50))
    thin_contacts(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


IN_ = S.IN


def _rigid_root(stage, root_prim, mass, inertia=None, com=(0.0, 0.0, 0.0)):
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    api = UsdPhysics.MassAPI.Apply(root_prim)
    api.CreateMassAttr(mass)
    if inertia is not None:
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        api.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
        api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    root_prim.AddAppliedSchema("PhysxRigidBodyAPI")
    root_prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    root_prim.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    root_prim.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float).Set(0.5)


def make_pipe(out_path: Path) -> None:
    root = "/pipe"
    stage, root_prim = kit.new_stage(out_path, root)
    L = S.PIPE_Y[1] - S.PIPE_Y[0]
    M = S.PIPE_MASS
    r2 = S.INS_R ** 2 * 0.6
    _rigid_root(stage, root_prim, M, inertia=(M * L * L / 12, M * r2, M * L * L / 12))
    m = mats(stage, root)
    xz_prism(stage, f"{root}/insulation", ngon(S.INS_R, 32), -L / 2, L / 2, m["jacket"])
    for k, sy in enumerate((-1, 1)):
        kit.cylinder(stage, f"{root}/tube_end{k}", (0.0, sy * (L / 2 + 0.010), 0.0), S.CU_R, 0.020, m["copper"],
                     axis="Y")
        kit.cylinder(stage, f"{root}/bore{k}", (0.0, sy * (L / 2 + 0.0201), 0.0), S.CU_RI, 0.0004, m["dark"], axis="Y")
        kit.cylinder(stage, f"{root}/cut{k}", (0.0, sy * (L / 2 + 0.0002), 0.0), S.INS_R * 0.995, 0.0004, m["wood"],
                     axis="Y")
    bind(stage, root, physics_material(stage, root, "asj_jacket", 0.40))
    thin_contacts(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_shield(out_path: Path) -> None:
    root = "/shield"
    stage, root_prim = kit.new_stage(out_path, root)
    M, R, L = S.SH_MASS, S.SH_RI + S.SH_T / 2, S.SH_L
    # A half-round's centre of mass lies 2R/pi below its axis (23 mm), not on it.
    _rigid_root(stage, root_prim, M, inertia=(M * (L * L / 12 + R * R / 2), M * R * R, M * (L * L / 12 + R * R / 2)),
                com=(0.0, 0.0, -2 * R / math.pi))
    root_prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(False)
    m = mats(stage, root)
    arc_strips(stage, f"{root}/strip", 0.0, 0.0, S.SH_RI, S.SH_T, -L / 2, L / 2, m["galv"], n=S.SH_SEGS)
    # Red tape (12 mm) round the middle, inside and out: the centre mark an installer lines up with the hanger.
    w = S.MARK_W / 2
    arc_strips(stage, f"{root}/mark_out", 0.0, 0.0, S.SH_RO, 0.0003, -w, w, m["red"], n=S.SH_SEGS, collide=False)
    arc_strips(stage, f"{root}/mark_in", 0.0, 0.0, S.SH_RI - 0.0003, 0.0003, -w, w, m["red"], n=S.SH_SEGS,
               collide=False)
    bind(stage, root, physics_material(stage, root, "galv_sheet", 0.30))
    thin_contacts(stage, offset=0.0008)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    make_rig(out_dir / "rig.usda")
    make_pipe(out_dir / "pipe.usda")
    make_shield(out_dir / "shield.usda")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets")

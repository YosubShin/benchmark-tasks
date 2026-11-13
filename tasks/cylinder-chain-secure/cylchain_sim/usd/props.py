"""Generate the props of the cylinder chain task as USD (dimensions and sources: cylchain_sim/spec.py).

Files (desk frame, see spec.py): ``bracket.usda`` and ``chain.usda`` (cylchain_sim/usd/bm_props.py: the
Bristol Maid CYB/DE2 bracket with its J-hooks, and the long-link chain); ``cylinder.usda`` (free rigid body, origin
at the middle of its base): MD aluminium cylinder (PhysX cylinder for the straight wall, convex hull for shoulder
and neck), CGA 870 post valve and toggle; green shoulder.

Usage: python -m cylchain_sim.usd.props cylchain_sim/assets
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from cylchain_sim import spec as S
from cylchain_sim.usd import usd_kit as kit

COLORS = {
    "plywood": (0.86, 0.80, 0.68),
    "bracket": (0.42, 0.47, 0.50),
    "bumper": (0.04, 0.04, 0.04),
    "alu": (0.80, 0.81, 0.83),
    "green": (0.05, 0.55, 0.20),
    "brass": (0.78, 0.62, 0.25),
    "black": (0.05, 0.05, 0.05),
    "zinc": (0.70, 0.72, 0.75),
    "red": (0.75, 0.02, 0.02),
    "white": (0.95, 0.95, 0.95),
}
METAL = ("alu", "brass", "zinc", "bracket")


def mats(stage, root):
    out = {}
    for name, rgb in COLORS.items():
        path = f"{root}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9 if name == "red" else 0.4)
        if name in METAL:
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.5)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        out[name] = material
    return out


def span(stage, path, xr, yr, zr, material, collide=True):
    return kit.box(stage, path, ((xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2, (zr[0] + zr[1]) / 2),
                   (xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]), material, collide=collide)


def _mat_q(m) -> tuple:
    from scipy.spatial.transform import Rotation

    return tuple(float(v) for v in Rotation.from_matrix(np.asarray(m)).as_quat())


def _set_orient(xf, rot):
    x, y, z, w = _mat_q(rot)
    xf.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))


def capsule(stage, path, a, b, radius, material, collide=True):
    """A capsule from point a to point b (parent frame)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    length = float(np.linalg.norm(d))
    x = d / length
    ref = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(ref, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    cap = UsdGeom.Capsule.Define(stage, path)
    cap.CreateAxisAttr("X")
    cap.CreateRadiusAttr(radius)
    cap.CreateHeightAttr(length)
    xf = UsdGeom.Xformable(cap)
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in (a + b) / 2]))
    _set_orient(xf, np.stack([x, y, z], 1))
    UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    return cap


def body_at(stage, path, pos, rot, mass, inertia):
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    _set_orient(xf, rot)
    UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
    m = UsdPhysics.MassAPI.Apply(xf.GetPrim())
    m.CreateMassAttr(mass)
    com, diag = inertia
    m.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*diag))
    m.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
    return xf


def joint_at(stage, path, body0, f0, body1, f1, point, rot, axis, lo, hi):
    """Revolute joint at a world point whose frame is ``rot`` in the authored pose (angle 0 there)."""
    j = UsdPhysics.RevoluteJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(str(body0))])
    j.CreateBody1Rel().SetTargets([Sdf.Path(str(body1))])
    j.CreateAxisAttr(axis)
    for (p, r), pos_attr, rot_attr in ((f0, j.CreateLocalPos0Attr, j.CreateLocalRot0Attr),
                                       (f1, j.CreateLocalPos1Attr, j.CreateLocalRot1Attr)):
        pos_attr(Gf.Vec3f(*[float(v) for v in r.T @ (np.asarray(point) - p)]))
        x, y, z, w = _mat_q(r.T @ rot)
        rot_attr(Gf.Quatf(float(w), float(x), float(y), float(z)))
    j.CreateLowerLimitAttr(lo)
    j.CreateUpperLimitAttr(hi)
    UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    return j


def thin_contacts(stage, prefix, offset=0.001):
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(offset)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


# -- cylinder ------------------------------------------------------------------------------------------------------
def _hull_mesh(stage, path, rings, material, collide=True, n=32):
    """A closed surface of revolution about z from (radius, z) rings, as a mesh (convex: a convex-hull collider)."""
    pts, faces, counts = [], [], []
    for r, z in rings:
        for i in range(n):
            a = 2 * math.pi * i / n
            pts.append((r * math.cos(a), r * math.sin(a), z))
    for k in range(len(rings) - 1):
        for i in range(n):
            a, b = k * n + i, k * n + (i + 1) % n
            faces += [a, b, b + n, a + n]
            counts.append(4)
    faces += list(range(n - 1, -1, -1))
    counts.append(n)
    top = (len(rings) - 1) * n
    faces += list(range(top, top + n))
    counts.append(n)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    return mesh


def make_cylinder(out_path: Path) -> None:
    root = "/cylinder"
    stage, root_prim = kit.new_stage(out_path, root)
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    mass = UsdPhysics.MassAPI.Apply(root_prim)
    mass.CreateMassAttr(S.CYL_MASS)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, S.CYL_COM_Z))
    h = S.CYL_TOP
    ixx = S.CYL_MASS * (3 * S.CYL_R ** 2 + h ** 2) / 12
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(ixx, ixx, S.CYL_MASS * S.CYL_R ** 2 / 2))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
    m = mats(stage, root)
    r, nr, top = S.CYL_R, S.CYL_NECK_R, S.CYL_TOP
    # Collision: one convex hull (base edge rounded by 3 mm, straight wall, shoulder, neck).
    rings = [(r - 0.003, 0.0), (r, 0.003), (r, S.CYL_BODY_H), (0.8 * r, S.CYL_BODY_H + 0.025), (nr, S.CYL_NECK_Z),
             (nr, top)]
    # Straight wall: PhysX's own cylinder shape (smooth, so a chain slides round it); shoulder and neck: a hull.
    body = UsdGeom.Cylinder.Define(stage, f"{root}/hull")
    body.CreateRadiusAttr(r)
    body.CreateHeightAttr(S.CYL_BODY_H)
    body.CreateAxisAttr("Z")
    UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3d(0, 0, S.CYL_BODY_H / 2))
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(m["alu"])
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    _hull_mesh(stage, f"{root}/shoulder", rings[2:], m["alu"], n=24)
    # Visual: green shoulder over the hull's top (slightly larger), white label band.
    _hull_mesh(stage, f"{root}/shoulder_paint", [(r + 0.0006, S.CYL_BODY_H - 0.035), (r + 0.0006, S.CYL_BODY_H),
                                                 (0.8 * r + 0.0006, S.CYL_BODY_H + 0.025),
                                                 (nr + 0.0006, S.CYL_NECK_Z), (nr + 0.0006, top)],
               m["green"], collide=False)
    _hull_mesh(stage, f"{root}/label", [(r + 0.0004, 0.12), (r + 0.0004, 0.20)], m["white"], collide=False)
    vx, vy, vz = S.VALVE
    span(stage, f"{root}/valve", vx, vy, vz, m["brass"])
    tx, ty, tz = S.TOGGLE
    span(stage, f"{root}/toggle", tx, ty, tz, m["black"])
    # Outlet post (CGA 870 yoke post) toward the wall, gas outlet hole visual.
    span(stage, f"{root}/post", (vx[1], vx[1] + 0.012), (-0.009, 0.009), (vz[0] + 0.012, vz[0] + 0.032), m["brass"])
    kit.friction(stage, root, [f"{root}/hull", f"{root}/shoulder"], static=S.CYL_FRICTION, dynamic=0.3, name="alu", combine="min")
    thin_contacts(stage, root)
    root_prim.AddAppliedSchema("PhysxRigidBodyAPI")
    root_prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def main(out_dir: Path) -> None:
    from cylchain_sim.usd.bm_props import make_bracket, make_chain

    out_dir.mkdir(parents=True, exist_ok=True)
    make_bracket(out_dir / "bracket.usda")
    make_cylinder(out_dir / "cylinder.usda")
    make_chain(out_dir / "chain.usda")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets")

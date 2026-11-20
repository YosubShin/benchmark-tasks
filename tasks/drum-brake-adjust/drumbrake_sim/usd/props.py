"""Generate the props of the drum brake adjust task as USD (dimensions and sources: drumbrake_sim/spec.py).

Files (desk frame, see spec.py):
- ``bench.usda`` (fixed colliders): the spoon's two wooden rest blocks and the adjusting hole's rubber cover lying
  on the bench (visual).
- ``brake.usda``: one fixed-base articulation. Root body ``plate`` (fixed to the world): the backing plate with its
  adjusting hole (the plate is built from convex pieces round the hole, so the hole is a real opening), the knuckle
  and the hub stand's post and base. ``drum`` turns on the axle (revolute ``drum``, about y): the drum's closed face
  and its five wheel studs collide, its rim is visual. ``star`` turns on the adjuster's axis (revolute ``star``,
  about x): a hub and 24 teeth, each a convex tapered tooth, sitting 6 mm inside the plate behind the hole. The drag
  and the pawl detent are rules in drumbrake_sim.task. Self-collisions are off.
- ``spoon.usda``: the brake adjusting spoon, a free rigid body (origin at its tip, the spoon along +y): a flat-bar
  handle and a thin blade whose last 4 mm taper to a 1 mm edge.

Usage: python -m drumbrake_sim.usd.props drumbrake_sim/assets
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from drumbrake_sim import spec as S
from drumbrake_sim.usd import usd_kit as kit

COLORS = {
    "plate": (0.10, 0.10, 0.11),
    "drum": (0.36, 0.34, 0.33),
    "drum_face": (0.28, 0.27, 0.27),
    "zinc": (0.74, 0.76, 0.78),
    "steel": (0.62, 0.64, 0.66),
    "black_oxide": (0.08, 0.08, 0.09),
    "grip": (0.78, 0.05, 0.03),
    "stand": (0.20, 0.32, 0.55),
    "rubber": (0.03, 0.03, 0.03),
    "wood": (0.72, 0.56, 0.36),
    "dark": (0.04, 0.04, 0.04),
    "shoe": (0.45, 0.40, 0.33),
}
METAL = ("zinc", "steel", "drum", "drum_face")


def mats(stage, root):
    out = {}
    for name, rgb in COLORS.items():
        path = f"{root}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9 if name in ("wood", "rubber") else 0.5)
        if name in METAL:
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.5)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        out[name] = material
    return out


def span(stage, path, xr, yr, zr, material, collide=True):
    return kit.box(stage, path, ((xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2, (zr[0] + zr[1]) / 2),
                   (xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]), material, collide=collide)


def _mesh(stage, path, pts, faces, counts, material, collide):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    return mesh


def extrude(stage, path, poly, axis, a0, a1, material, collide=True):
    """A convex prism: a CCW polygon in the plane normal to ``axis`` ('x': (y, z); 'y': (x, z); 'z': (x, y)),
    extruded from a0 to a1 along the axis."""
    n = len(poly)

    def pt(u, v, a):
        return {"x": (a, u, v), "y": (u, a, v), "z": (u, v, a)}[axis]

    pts = [pt(u, v, a0) for u, v in poly] + [pt(u, v, a1) for u, v in poly]
    faces, counts = list(range(n - 1, -1, -1)), [n]
    for k in range(n):
        k1 = (k + 1) % n
        faces += [k, k1, n + k1, n + k]
        counts.append(4)
    faces += list(range(n, 2 * n))
    counts.append(n)
    return _mesh(stage, path, pts, faces, counts, material, collide)


def circle(cx, cz, r, n=48):
    return [(cx + r * math.cos(2 * math.pi * k / n), cz + r * math.sin(2 * math.pi * k / n)) for k in range(n)]


def clip(poly, a, b, c):
    """Keep the part of a convex polygon where a*u + b*v <= c."""
    out = []
    n = len(poly)
    for k in range(n):
        p, q = poly[k], poly[(k + 1) % n]
        fp, fq = a * p[0] + b * p[1] - c, a * q[0] + b * q[1] - c
        if fp <= 0:
            out.append(p)
        if (fp <= 0) != (fq <= 0):
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def physics_material(stage, root, name, mu, combine="average"):
    mat = UsdShade.Material.Define(stage, f"{root}/PhysicsMaterials/{name}")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(mu)
    api.CreateDynamicFrictionAttr(mu)
    mat.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    mat.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(combine)
    return mat


def thin_contacts(stage, offset):
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(offset)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def _body(stage, path, translate, mass, inertia=None):
    xf = kit.rigid_body(stage, path, translate, mass)
    if inertia is not None:
        api = UsdPhysics.MassAPI(xf.GetPrim())
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        api.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
    prim = xf.GetPrim()
    prim.AddAppliedSchema("PhysxRigidBodyAPI")
    prim.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float).Set(0.5)
    return xf


def _revolute(stage, path, b0, b1, axis, pos0):
    j = UsdPhysics.RevoluteJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(str(b0))])
    j.CreateBody1Rel().SetTargets([Sdf.Path(str(b1))])
    j.CreateAxisAttr(axis)
    j.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    return j


# -- bench --------------------------------------------------------------------------------------------------------
def make_bench(out_path: Path) -> None:
    root = "/bench"
    stage, _ = kit.new_stage(out_path, root)
    m = mats(stage, root)
    for name, (xr, yr, top) in (("rest_a", S.BLOCK_A), ("rest_b", S.BLOCK_B)):
        span(stage, f"{root}/{name}", xr, yr, (0.0, top), m["wood"])
    # The adjusting hole's rubber cover, pulled out and lying on the bench (visual).
    px, py, _ = S.PLUG_POS
    kit.cylinder(stage, f"{root}/plug_body", (px, py, 0.003), 0.012, 0.006, m["rubber"])
    kit.cylinder(stage, f"{root}/plug_lip", (px, py, 0.0075), 0.0085, 0.003, m["rubber"])
    wood = physics_material(stage, root, "wood", 0.5)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(root)).Bind(wood, UsdShade.Tokens.weakerThanDescendants,
                                                                      "physics")
    thin_contacts(stage, 0.001)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


# -- brake ----------------------------------------------------------------------------------------------------------
def make_brake(out_path: Path) -> None:
    root = "/brake"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    m = mats(stage, root)
    ax, ay, az = S.AXLE

    # Fixed part: backing plate with the adjusting hole, knuckle, stand.
    plate = _body(stage, f"{root}/plate", (0.0, 0.0, 0.0), 20.0)
    kit.fix_to_world(stage, root, plate.GetPath())
    disc = circle(ax, az, S.BP_R, 64)
    x0, x1 = S.SW[0] - S.SLOT_W / 2, S.SW[0] + S.SLOT_W / 2
    pieces = {
        "top": clip(disc, 0.0, -1.0, -S.SLOT_Z1),
        "bottom": clip(disc, 0.0, 1.0, S.SLOT_Z0),
        "left": clip(clip(clip(disc, 0.0, -1.0, -S.SLOT_Z0), 0.0, 1.0, S.SLOT_Z1), 1.0, 0.0, x0),
        "right": clip(clip(clip(disc, 0.0, -1.0, -S.SLOT_Z0), 0.0, 1.0, S.SLOT_Z1), -1.0, 0.0, -x1),
    }
    for name, poly in pieces.items():
        extrude(stage, f"{root}/plate/bp_{name}", poly, "y", S.BP_Y, S.BP_Y + S.BP_T, m["plate"])
    # Raised centre boss and the plate's outer lip (visual).
    kit.cylinder(stage, f"{root}/plate/boss", (ax, S.BP_Y - 0.004, az), 0.055, 0.008, m["plate"], axis="Y")
    span(stage, f"{root}/plate/knuckle", *S.KNUCKLE, m["stand"])
    span(stage, f"{root}/plate/post", *S.POST, m["stand"])
    span(stage, f"{root}/plate/base", *S.BASE, m["stand"])
    # Inside the drum (visual, hidden once the drum is on): shoes and the adjuster screw.
    for k, sx in enumerate((-1, 1)):
        span(stage, f"{root}/plate/shoe{k}", (ax + sx * 0.112 - 0.004, ax + sx * 0.112 + 0.004),
             (S.BP_Y + 0.008, S.BP_Y + 0.008 + 2.0 * S.IN), (az - 0.07, az + 0.07), m["shoe"], collide=False)
    kit.cylinder(stage, f"{root}/plate/adj_screw", (S.SW[0], S.SW[1], S.SW[2]), 0.0045, 0.11, m["steel"], axis="X")

    # Drum: rim visual, closed face and studs collide. Body origin on the axle.
    drum = _body(stage, f"{root}/drum", (ax, ay, az), S.DRUM_MASS, inertia=(0.05, 0.096, 0.05))
    y_mid = S.DRUM_Y0 + S.DRUM_DEPTH / 2 - ay
    # Rim: a hollow ring (48 visual segments) so the inside stays open toward the plate.
    n_seg, r_mid, thick = 48, (S.DRUM_RO + S.DRUM_RI) / 2, S.DRUM_RO - S.DRUM_RI
    for k in range(n_seg):
        a0, a1 = 2 * math.pi * k / n_seg, 2 * math.pi * (k + 1) / n_seg
        seg = [(S.DRUM_RI * math.cos(a0), S.DRUM_RI * math.sin(a0)), (S.DRUM_RO * math.cos(a0), S.DRUM_RO * math.sin(a0)),
               (S.DRUM_RO * math.cos(a1), S.DRUM_RO * math.sin(a1)), (S.DRUM_RI * math.cos(a1), S.DRUM_RI * math.sin(a1))]
        extrude(stage, f"{root}/drum/rim{k:02d}", seg, "y", S.DRUM_Y0 - ay, S.DRUM_FACE_Y - ay, m["drum"], collide=False)
    del y_mid, r_mid, thick
    kit.cylinder(stage, f"{root}/drum/rim_lip", (0.0, S.DRUM_Y0 - ay + 0.004, 0.0), S.DRUM_RO + 0.004, 0.008,
                 m["drum"], axis="Y")
    extrude(stage, f"{root}/drum/face", circle(0.0, 0.0, S.DRUM_RO - 0.004, 32), "y",
            S.DRUM_FACE_Y - S.DRUM_FACE_T - ay, S.DRUM_FACE_Y - ay + 0.0005, m["drum_face"])
    kit.cylinder(stage, f"{root}/drum/hub_cap", (0.0, S.DRUM_FACE_Y - ay + 0.008, 0.0), 0.032, 0.016, m["dark"],
                 axis="Y")
    for k in range(5):
        a = math.radians(S.STUD_ALPHA0 + 72.0 * k)
        sx, sz = -S.STUD_R * math.cos(a), S.STUD_R * math.sin(a)
        ring = [(sx + S.STUD_D / 2 / math.cos(math.pi / 8) * math.cos(math.pi / 8 + 2 * math.pi * i / 8),
                 sz + S.STUD_D / 2 / math.cos(math.pi / 8) * math.sin(math.pi / 8 + 2 * math.pi * i / 8))
                for i in range(8)]
        extrude(stage, f"{root}/drum/stud{k}", ring, "y", S.DRUM_FACE_Y - ay, S.DRUM_FACE_Y - ay + S.STUD_L,
                m["zinc"])
    _revolute(stage, f"{root}/joints/drum", plate.GetPath(), drum.GetPath(), "Y", (ax, ay, az))

    # Star wheel: hub + 24 tapered teeth in the y-z plane, extruded along x. Body origin on its axis. At joint 0 a
    # gap between two teeth faces the hole (the -y side).
    sw = _body(stage, f"{root}/star", S.SW, S.SW_MASS, inertia=(1e-6, 8e-7, 8e-7))
    hw = S.SW_W / 2
    hub = [(S.SW_RR * math.cos(2 * math.pi * k / 24), S.SW_RR * math.sin(2 * math.pi * k / 24)) for k in range(24)]
    extrude(stage, f"{root}/star/hub", hub, "x", -hw, hw, m["steel"])
    for k in range(S.SW_TEETH):
        # Tooth k centred at angle (k + 0.5) pitch from the -y direction (gap at 0).
        th = math.pi + (k + 0.5) * S.PITCH_ANG  # angle in the (y, z) plane from +y
        c, s = math.cos(th), math.sin(th)
        r0, r1 = S.SW_RR - 0.0004, S.SW_RT
        w0, w1 = S.SW_TOOTH_ROOT_W / 2, S.SW_TOOTH_TIP_W / 2
        local = [(r0, -w0), (r1, -w1), (r1, w1), (r0, w0)]  # (radial, tangential), CCW
        poly = [(c * r - s * t, s * r + c * t) for r, t in local]
        extrude(stage, f"{root}/star/tooth{k:02d}", poly, "x", -hw, hw, m["steel"], collide=False)
    kit.cylinder(stage, f"{root}/star/thread", (0.0, 0.0, 0.0), 0.0046, 0.004 + S.SW_W, m["steel"], axis="X")
    _revolute(stage, f"{root}/joints/star", plate.GetPath(), sw.GetPath(), "X", S.SW)

    steel = physics_material(stage, root, "steel", 0.30)
    UsdShade.MaterialBindingAPI(root_prim).Bind(steel, UsdShade.Tokens.weakerThanDescendants, "physics")
    dry = physics_material(stage, root, "dry_steel", 0.5)
    UsdShade.MaterialBindingAPI.Apply(sw.GetPrim()).Bind(dry, UsdShade.Tokens.weakerThanDescendants, "physics")
    thin_contacts(stage, 0.0004)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


# -- spoon ----------------------------------------------------------------------------------------------------------
def make_spoon(out_path: Path) -> None:
    root = "/spoon"
    stage, root_prim = kit.new_stage(out_path, root)
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    mass = UsdPhysics.MassAPI.Apply(root_prim)
    mass.CreateMassAttr(S.SPOON_MASS)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, -0.150, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(5e-4, 4e-6, 5e-4))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
    root_prim.AddAppliedSchema("PhysxRigidBodyAPI")
    root_prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    root_prim.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float).Set(0.5)
    m = mats(stage, root)
    L, hl = S.SPOON_L, S.HANDLE_L
    span(stage, f"{root}/handle", (-S.HANDLE_W / 2, S.HANDLE_W / 2), (-L, -L + hl), (-S.HANDLE_T / 2, S.HANDLE_T / 2),
         m["grip"])
    t0 = S.TIP_TAPER + S.TIP_FLAT
    span(stage, f"{root}/blade", (-S.BLADE_W / 2, S.BLADE_W / 2), (-L + hl, -t0),
         (-S.BLADE_T / 2, S.BLADE_T / 2), m["black_oxide"])
    b, t = S.BLADE_T / 2, S.TIP_T / 2
    extrude(stage, f"{root}/taper", [(-t0, -b), (-S.TIP_FLAT, -t), (-S.TIP_FLAT, t), (-t0, b)], "x",
            -S.BLADE_W / 2, S.BLADE_W / 2, m["black_oxide"])
    span(stage, f"{root}/tip", (-S.BLADE_W / 2, S.BLADE_W / 2), (-S.TIP_FLAT, 0.0), (-t, t), m["black_oxide"])
    steel = physics_material(stage, root, "spoon_steel", 0.5)
    UsdShade.MaterialBindingAPI(root_prim).Bind(steel, UsdShade.Tokens.weakerThanDescendants, "physics")
    grip = physics_material(stage, root, "dipped_grip", 0.9)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"{root}/handle")).Bind(
        grip, UsdShade.Tokens.weakerThanDescendants, "physics")
    thin_contacts(stage, 0.0003)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    make_bench(out_dir / "bench.usda")
    make_brake(out_dir / "brake.usda")
    make_spoon(out_dir / "spoon.usda")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets")

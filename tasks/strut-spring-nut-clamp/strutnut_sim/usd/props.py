"""Generate the props of the strut spring-nut task as USD (dimensions and sources: strutnut_sim/spec.py).

Files (desk frame, see spec.py):
- ``stand.usda`` (fixed colliders): a 2 ft length of Unistrut P1000 channel, open side up, bolted across two P1000
  posts on square post bases; its lips modelled as the catalog section shows them (top flange, return turned down
  7.1 mm, the bend at the slot edge as a chamfer); a 12 mm red tape mark across the near wall and lip; the wooden
  parts block whose pocket the kit stands in.
- ``kit.usda``: one floating articulation, rooted at the spring nut (origin: the middle of the nut's top face). The
  nut is a convex outline with its two leading corners rounded (a quarter turn clockwise inside the channel, square
  corners stop it) and two grooves across its top under the lip returns; its spring is a sprung prismatic foot.
  The bolt is screwed into the nut: a prismatic ``thread`` joint (head underside height above the nut's top) and a
  revolute ``turn`` joint, coupled per control step by strutnut_sim.task (the screw rule). Self-collisions are off.
- ``fitting.usda``: the P1064 flat plate fitting (square washer), a free rigid body (origin: the middle of its
  underside) with a 9/16 in hole round the bolt's shank; it is held on the bolt only by geometry (the head above,
  the nut below).

Usage: python -m strutnut_sim.usd.props strutnut_sim/assets
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from strutnut_sim import spec as S
from strutnut_sim.usd import usd_kit as kit

COLORS = {
    "perma_green": (0.18, 0.42, 0.24),
    "zinc": (0.74, 0.76, 0.78),
    "bolt": (0.62, 0.64, 0.66),
    "nut": (0.66, 0.66, 0.60),
    "spring": (0.30, 0.30, 0.32),
    "red": (0.85, 0.04, 0.03),
    "wood": (0.72, 0.56, 0.36),
    "galv": (0.60, 0.62, 0.64),
    "dark": (0.15, 0.15, 0.16),
}
METAL = ("zinc", "bolt", "nut", "galv", "spring")


def mats(stage, root):
    out = {}
    for name, rgb in COLORS.items():
        path = f"{root}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9 if name in ("red", "wood") else 0.45)
        if name in METAL:
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.5)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        out[name] = material
    return out


def span(stage, path, xr, yr, zr, material, collide=True):
    return kit.box(stage, path, ((xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2, (zr[0] + zr[1]) / 2),
                   (xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]), material, collide=collide)


def hull_mesh(stage, path, rings, material, collide=True):
    """A closed convex solid from a list of polygon rings [(z, [(x, y), ...]), ...] (same vertex count, CCW)."""
    pts, faces, counts = [], [], []
    n = len(rings[0][1])
    for z, poly in rings:
        pts += [Gf.Vec3f(float(x), float(y), float(z)) for x, y in poly]
    faces += list(range(n - 1, -1, -1))
    counts.append(n)
    for r in range(len(rings) - 1):
        a0, b0 = r * n, (r + 1) * n
        for k in range(n):
            k1 = (k + 1) % n
            faces += [a0 + k, a0 + k1, b0 + k1, b0 + k]
            counts.append(4)
    top = (len(rings) - 1) * n
    faces += list(range(top, top + n))
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


def prism(stage, path, poly, z0, z1, material, collide=True):
    return hull_mesh(stage, path, [(z0, poly), (z1, poly)], material, collide)


def xz_prism(stage, path, poly_xz, y0, y1, material, collide=True):
    """A convex prism from a CCW polygon in the x-z plane, extruded along y (as rings in y, mapped to z)."""
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
    """An n-gon with a flat facing phase_deg, at distance radius_flat from the centre (CCW)."""
    r = radius_flat / math.cos(math.pi / n)
    out = []
    for k in range(n):
        a = math.radians(phase_deg) + 2 * math.pi * (k + 0.5) / n
        out.append((r * math.cos(a), r * math.sin(a)))
    return out


def physics_material(stage, root, name, mu, combine="average"):
    mat = UsdShade.Material.Define(stage, f"{root}/PhysicsMaterials/{name}")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(mu)
    api.CreateDynamicFrictionAttr(mu)
    mat.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    mat.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(combine)
    return mat


def bind_physics(stage, path, mat):
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(mat, UsdShade.Tokens.weakerThanDescendants,
                                                                       "physics")


def thin_contacts(stage, offset=0.001):
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(offset)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


# -- stand ------------------------------------------------------------------------------------------------------------
def make_stand(out_path: Path) -> None:
    root = "/stand"
    stage, _ = kit.new_stage(out_path, root)
    m = mats(stage, root)
    cx, w, t = S.CH_X, S.CH_W / 2, S.CH_T
    y0, y1 = S.CH_Y
    z0, top = S.CH_Z0, S.CH_TOP
    g = m["perma_green"]
    span(stage, f"{root}/ch_back", (cx - w, cx + w), (y0, y1), (z0, z0 + t), g)
    for k, sx in enumerate((-1, 1)):
        xs = sorted((cx + sx * (w - t), cx + sx * w))
        span(stage, f"{root}/ch_wall{k}", xs, (y0, y1), (z0, top), g)
        # Lip: the top flange from the wall to the slot edge, and the return turned down at the slot edge (one convex
        # piece with the bend at the slot edge as a chamfer).
        a, b = S.SLOT / 2, S.SLOT / 2 + t
        xs = sorted((cx + sx * (a + t), cx + sx * (w - t)))
        span(stage, f"{root}/ch_lip{k}", xs, (y0, y1), (top - t, top), g)
        xs = sorted((cx + sx * a, cx + sx * b))
        span(stage, f"{root}/ch_return{k}", xs, (y0, y1), (top - S.LIP_RETURN, top - t), g)
        # The lip's rounded bend at the slot edge: a 45 degree bevel (visual and collision), in 10 cm pieces.
        poly = [(a, top - t), (b, top - t), (b, top), (a + S.LIP_ROUND, top), (a, top - S.LIP_ROUND)]
        if sx < 0:
            poly = [(-x, z) for x, z in reversed(poly)]
        ys = [y0 + (y1 - y0) * i / 6 for i in range(7)]
        for i in range(6):
            xz_prism(stage, f"{root}/ch_bend{k}_{i}", [(cx + x, z) for x, z in poly], ys[i], ys[i + 1], g)
    # Anchors: 3/8 in bolts with square washers through the channel's back into the bench (visual).
    for k, ay in enumerate(S.ANCHOR_Y):
        span(stage, f"{root}/anchor_washer{k}", (cx - 0.0105, cx + 0.0105), (ay - 0.0105, ay + 0.0105),
             (z0 + t, z0 + t + 0.003), m["galv"], collide=False)
        prism(stage, f"{root}/anchor_head{k}", [(cx + x, ay + y) for x, y in ngon(0.0071, 6)], z0 + t + 0.003,
              z0 + t + 0.009, m["galv"], collide=False)
    # Red tape mark across the near wall's face and the near lip's top.
    my0, my1 = S.MARK_Y - S.MARK_W / 2, S.MARK_Y + S.MARK_W / 2
    span(stage, f"{root}/mark_face", (cx - w - 0.0004, cx - w), (my0, my1), (top - 0.030, top), m["red"], collide=False)
    span(stage, f"{root}/mark_lip", (cx - w, cx - S.SLOT / 2 - 0.0015), (my0, my1), (top, top + 0.0004), m["red"],
         collide=False)
    # Parts block (wood) with the pocket the kit's nut and spring hang in.
    hx, hy = S.HOLDER_XY
    bx, by, bz = S.HOLDER_SIZE
    px, py = S.POCKET
    pz = bz - S.POCKET_DEPTH
    wood = m["wood"]
    span(stage, f"{root}/holder_floor", (hx - bx / 2, hx + bx / 2), (hy - by / 2, hy + by / 2), (0.0, pz), wood)
    span(stage, f"{root}/holder_n", (hx - bx / 2, hx - px / 2), (hy - by / 2, hy + by / 2), (pz, bz), wood)
    span(stage, f"{root}/holder_f", (hx + px / 2, hx + bx / 2), (hy - by / 2, hy + by / 2), (pz, bz), wood)
    span(stage, f"{root}/holder_r", (hx - px / 2, hx + px / 2), (hy - by / 2, hy - py / 2), (pz, bz), wood)
    span(stage, f"{root}/holder_l", (hx - px / 2, hx + px / 2), (hy + py / 2, hy + by / 2), (pz, bz), wood)
    steel = physics_material(stage, root, "painted_steel", 0.30)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(root)).Bind(steel, UsdShade.Tokens.weakerThanDescendants,
                                                                       "physics")
    thin_contacts(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


# -- kit: spring nut + spring + bolt ---------------------------------------------------------------------------------
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


def _prismatic(stage, path, b0, b1, pos0, lo, hi):
    j = UsdPhysics.PrismaticJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(str(b0))])
    j.CreateBody1Rel().SetTargets([Sdf.Path(str(b1))])
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLowerLimitAttr(lo)
    j.CreateUpperLimitAttr(hi)
    UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    return j


def make_kit(out_path: Path) -> None:
    root = "/kit"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    m = mats(stage, root)
    e0 = S.START_TURNS * S.PITCH
    q_thread0 = S.BOLT_L - e0

    # Nut: origin at the middle of its top face; x across its width, y along its length.
    nut = _body(stage, f"{root}/nut", (0.0, 0.0, 0.0), S.NUT_MASS)
    outline = S.nut_outline()
    lower_top = -S.GROOVE_D
    # Lower body: the outline at the groove floor, bevelled along its long lower edges.
    inset = [(x - math.copysign(S.NUT_CHAMFER, x) if abs(abs(x) - S.NUT_W / 2) < 1e-6 else x, y) for x, y in outline]
    hull_mesh(stage, f"{root}/nut/body", [(-S.NUT_T, inset), (-S.NUT_T + S.NUT_CHAMFER, outline), (lower_top, outline)],
              m["nut"])
    # Top: the middle land between the grooves and the two end lands outside them.
    g0, g1 = S.GROOVE_C - S.GROOVE_W / 2, S.GROOVE_C + S.GROOVE_W / 2
    hw = S.NUT_W / 2
    prism(stage, f"{root}/nut/land_mid", [(hw, -g0), (hw, g0), (-hw, g0), (-hw, -g0)], lower_top, 0.0, m["nut"])

    def clip(poly, ymin):
        out = []
        n = len(poly)
        for k in range(n):
            a, b = poly[k], poly[(k + 1) % n]
            ina, inb = a[1] >= ymin, b[1] >= ymin
            if ina:
                out.append(a)
            if ina != inb:
                t = (ymin - a[1]) / (b[1] - a[1])
                out.append((a[0] + t * (b[0] - a[0]), ymin))
        return out

    end_pos = clip(outline, g1)
    end_neg = [(-x, -y) for x, y in end_pos]
    for name, poly in (("land_end_p", end_pos), ("land_end_n", end_neg)):
        hull_mesh(stage, f"{root}/nut/{name}", [(lower_top, poly), (0.0, poly)], m["nut"])
    kit.cylinder(stage, f"{root}/nut/hole_vis", (0.0, 0.0, 0.0002), S.BOLT_D / 2 * 0.8, 0.0006, m["dark"])

    # Spring foot: origin at its underside, spring free (joint 0) below the nut.
    foot_z = -S.NUT_T - S.SPRING_FREE
    foot = _body(stage, f"{root}/foot", (0.0, 0.0, foot_z), S.FOOT_MASS)
    prism(stage, f"{root}/foot/disc", ngon(S.FOOT_R, 12), 0.0, S.FOOT_T, m["spring"])
    for i in range(7):
        kit.cylinder(stage, f"{root}/foot/coil{i}", (0.0, 0.0, S.FOOT_T + 0.002 + i * 0.0042),
                     0.0075 - 0.0003 * i, 0.0011, m["spring"])
    _prismatic(stage, f"{root}/joints/spring", nut.GetPath(), foot.GetPath(), (0.0, 0.0, foot_z), -0.001,
               S.SPRING_MAX)

    # Bolt: thread slide (prismatic) then turn (revolute); origin at the head's underside on the axis.
    slide = _body(stage, f"{root}/slide", (0.0, 0.0, q_thread0), S.SLIDE_MASS, inertia=(4e-7, 4e-7, 4e-7))
    bolt = _body(stage, f"{root}/bolt", (0.0, 0.0, q_thread0), S.BOLT_MASS)
    prism(stage, f"{root}/bolt/head", ngon(S.HEAD_AF / 2, 6, phase_deg=90.0 - 30.0), 0.0, S.HEAD_H, m["bolt"])
    prism(stage, f"{root}/bolt/shank", ngon(S.BOLT_D / 2 * 0.985, 8), -S.BOLT_L, 0.0, m["bolt"])
    _prismatic(stage, f"{root}/joints/thread", nut.GetPath(), slide.GetPath(), (0.0, 0.0, 0.0),
               S.BOLT_L - S.NUT_T - 0.004, S.BOLT_L)
    turn = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/joints/turn")
    turn.CreateBody0Rel().SetTargets([slide.GetPath()])
    turn.CreateBody1Rel().SetTargets([bolt.GetPath()])
    turn.CreateAxisAttr("Z")
    turn.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    turn.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    UsdPhysics.DriveAPI.Apply(turn.GetPrim(), "angular")

    zinc = physics_material(stage, root, "zinc", 0.30)
    UsdShade.MaterialBindingAPI(root_prim).Bind(zinc, UsdShade.Tokens.weakerThanDescendants, "physics")
    thin_contacts(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_fitting(out_path: Path) -> None:
    root = "/fitting"
    stage, root_prim = kit.new_stage(out_path, root)
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(S.FIT_MASS)
    root_prim.AddAppliedSchema("PhysxRigidBodyAPI")
    root_prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    root_prim.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float).Set(0.5)
    m = mats(stage, root)
    w, h, t = S.FIT_W / 2, S.FIT_HOLE / 2, S.FIT_T
    z = (0.0, t)
    span(stage, f"{root}/plate_n", (-w, -h), (-w, w), z, m["zinc"])
    span(stage, f"{root}/plate_f", (h, w), (-w, w), z, m["zinc"])
    span(stage, f"{root}/plate_r", (-h, h), (-w, -h), z, m["zinc"])
    span(stage, f"{root}/plate_l", (-h, h), (h, w), z, m["zinc"])
    zinc = physics_material(stage, root, "zinc", 0.30)
    UsdShade.MaterialBindingAPI(root_prim).Bind(zinc, UsdShade.Tokens.weakerThanDescendants, "physics")
    thin_contacts(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    make_stand(out_dir / "stand.usda")
    make_kit(out_dir / "kit.usda")
    make_fitting(out_dir / "fitting.usda")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets")

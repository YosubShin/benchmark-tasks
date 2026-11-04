"""USD helpers for the prop generator (clamp_props.py). Collision shapes are boxes; cylinders are visual only."""

from __future__ import annotations

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


COLORS = {
    "red": (0.80, 0.06, 0.05),
    "dark_red": (0.55, 0.04, 0.03),
    "blue": (0.10, 0.30, 0.85),
    "green": (0.10, 0.62, 0.20),
    "yellow": (0.95, 0.78, 0.10),
    "white": (0.95, 0.95, 0.95),
    "orange": (0.92, 0.45, 0.08),
    "purple": (0.45, 0.20, 0.60),
    "black": (0.05, 0.05, 0.05),
    "dark": (0.12, 0.12, 0.13),
    "grey": (0.55, 0.57, 0.60),
    "light_grey": (0.78, 0.79, 0.80),
    "enclosure": (0.80, 0.80, 0.76),
    "steel": (0.72, 0.74, 0.77),
    "copper": (0.80, 0.45, 0.20),
    "stand": (0.40, 0.40, 0.42),
}


def new_stage(out_path, root: str):
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prim = UsdGeom.Xform.Define(stage, root).GetPrim()
    stage.SetDefaultPrim(prim)
    return stage, prim


def materials(stage, root: str, names=None) -> dict:
    out = {}
    for name in names or COLORS:
        path = f"{root}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[name]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        out[name] = material
    return out


def box(stage, path, center, size, material, collide=True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def cylinder(stage, path, center, radius, height, material, axis="Z"):
    """A visual-only cylinder (collision stays on boxes)."""
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr(axis)
    UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(material)
    return cyl


def rigid_body(stage, path, translate, mass, kinematic=False):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    rb = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(mass)
    return xform


def articulation_root(stage, root_prim, self_collisions=False):
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    root_prim.AddAppliedSchema("PhysxArticulationAPI")
    root_prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(self_collisions)


def fix_to_world(stage, root: str, body_path) -> None:
    fixed = UsdPhysics.FixedJoint.Define(stage, f"{root}/joints/world_fixed")
    fixed.CreateBody1Rel().SetTargets([Sdf.Path(str(body_path))])


def friction(stage, root, paths, static=1.0, dynamic=0.8, name="rubber", combine="max"):
    mat = UsdShade.Material.Define(stage, f"{root}/PhysicsMaterials/{name}")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    mat.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    mat.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(combine)
    for path in paths:
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).Bind(mat, UsdShade.Tokens.weakerThanDescendants,
                                                                    "physics")


def revolute(stage, path, body0, body1, axis, pos0, lower_deg, upper_deg, rot0=None):
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(str(body0))])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(str(body1))])
    joint.CreateAxisAttr(axis)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    if rot0 is not None:
        joint.CreateLocalRot0Attr(Gf.Quatf(*rot0))
        joint.CreateLocalRot1Attr(Gf.Quatf(*rot0))
    joint.CreateLowerLimitAttr(lower_deg)
    joint.CreateUpperLimitAttr(upper_deg)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    return joint


def prismatic(stage, path, body0, body1, axis, pos0, lower, upper):
    joint = UsdPhysics.PrismaticJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(str(body0))])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(str(body1))])
    joint.CreateAxisAttr(axis)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    return joint


def hide(stage, path) -> None:
    UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()

"""USD helpers and the breaker's published dimensions for the breaker-lockout props (boxes for collision).

Values marked (P) are published, (I) inferred, (TM) to measure on a bought part.
"""

from __future__ import annotations

from dataclasses import dataclass

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


@dataclass(frozen=True)
class BreakerVariant:
    """A 1-pole DIN-rail miniature circuit breaker. Depth is from the mounting plate to the toggle's tip."""

    name: str
    source: str
    pitch: float
    height: float
    depth: float
    color: str
    front_h: float = 0.045
    """DIN 43880 front window (I)."""
    shoulder: float = 0.044
    """Mounting plate to the step where the front window starts (I/TM)."""
    toggle: tuple[float, float, float] = (0.009, 0.007, 0.007)
    """Toggle protrusion (x), width (y), thickness (z) (TM; 7 x 7 mm, 9 mm proud, inferred from photos). The
    width matters: a 24 mm lockout on a 17.5 mm row clears the neighbours' toggles by (17.5 - 12 - w/2) mm."""
    toggle_shift: float = 0.005
    """Toggle centre above (ON) or below (OFF) the front window's centre (I: stands in for the swing)."""

    @property
    def face_depth(self) -> float:
        """Mounting plate to the front face the toggle stands out of."""
        return self.depth - self.toggle[0]


BREAKERS = {
    v.name: v
    for v in (
        BreakerVariant("abb_s201", "ABB S201-C16 (2CDS251001R0164)", 0.0175, 0.088, 0.069, "abb_grey"),
    )
}
BREAKER_NAMES = tuple(BREAKERS)

COLORS = {
    "board": (0.85, 0.85, 0.80),
    "rail": (0.75, 0.76, 0.78),
    "standoff": (0.55, 0.56, 0.58),
    "abb_grey": (0.86, 0.86, 0.84),
    "se_white": (0.95, 0.95, 0.93),
    "eaton_grey": (0.70, 0.71, 0.72),
    "toggle_dark": (0.12, 0.12, 0.13),
    "window": (0.30, 0.30, 0.32),
    "lockout_red": (0.80, 0.08, 0.06),
    "lockout_red2": (0.85, 0.15, 0.05),
    "button": (0.95, 0.85, 0.10),
    "terminal_grey": (0.60, 0.62, 0.64),
    "terminal_221": (0.85, 0.85, 0.85),
    "lever_orange": (0.95, 0.45, 0.05),
    "copper": (0.75, 0.45, 0.20),
    "ferrule": (0.85, 0.85, 0.88),
    "collar": (0.15, 0.35, 0.80),
    "insulation": (0.08, 0.08, 0.08),
    "holder_black": (0.10, 0.10, 0.11),
    "holder_grey": (0.25, 0.25, 0.27),
    "clip": (0.80, 0.70, 0.40),
    "fuse_body": (0.88, 0.82, 0.62),
    "fuse_cap": (0.75, 0.75, 0.78),
    "puller_black": (0.10, 0.10, 0.10),
    "puller_red": (0.75, 0.10, 0.08),
    "stand": (0.40, 0.40, 0.42),
    "tray": (0.20, 0.45, 0.25),
    "blue": (0.02, 0.10, 0.62),
    "yellow": (0.85, 0.66, 0.0),
    "green": (0.0, 0.42, 0.10),
    "white": (0.97, 0.97, 0.97),
    "orange": (0.85, 0.28, 0.0),
    "purple": (0.50, 0.20, 0.65),
}

# --------------------------------------------------------------------------------------------------
# USD helpers (as in the other prop generators)
# --------------------------------------------------------------------------------------------------


def _material(stage, root, name):
    path = f"{root}/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[name]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


class _Mats(dict):
    def __init__(self, stage, root):
        super().__init__()
        self.stage, self.root = stage, root

    def __missing__(self, key):
        self[key] = _material(self.stage, self.root, key)
        return self[key]


def _box(stage, path, center, size, material, collide=True, visible=True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if not visible:
        UsdGeom.Imageable(cube).MakeInvisible()
    return cube


def _span(a, b):
    """(centre, size) of an interval."""
    return (a + b) / 2, abs(b - a)


def _boxspan(stage, path, xs, ys, zs, material, collide=True, visible=True):
    (cx, sx), (cy, sy), (cz, sz) = _span(*xs), _span(*ys), _span(*zs)
    return _box(stage, path, (cx, cy, cz), (sx, sy, sz), material, collide, visible)


def _friction(stage, root, paths, static=1.5, dynamic=1.2, name="rubber", combine=None):
    friction = UsdShade.Material.Define(stage, f"{root}/PhysicsMaterials/{name}")
    api = UsdPhysics.MaterialAPI.Apply(friction.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    friction.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    combine = combine or ("max" if static >= 0.5 else "min")
    friction.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(combine)
    for path in paths:
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).Bind(friction, UsdShade.Tokens.weakerThanDescendants,
                                                                   "physics")


def _new_stage(out_path, root):
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prim = UsdGeom.Xform.Define(stage, root).GetPrim()
    stage.SetDefaultPrim(prim)
    return stage, prim, _Mats(stage, root)


def _rigid(stage, path, mass, xyz=(0.0, 0.0, 0.0)):
    x = UsdGeom.Xform.Define(stage, path)
    if any(xyz):
        x.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
    UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass)
    return x


def _flags(stage, root, mats, slot, center, colors, size=(0.004, 0.014, 0.016)):
    """Raised colour flags for one slot, one visual prim per colour (the task shows one)."""
    for c in colors:
        _box(stage, f"{root}/flag_{slot}_{c}", center, size, mats[c], collide=False)


def _hole_plate(stage, path, mats, mat, x_free_end, x_root, hole_x, hole, width, z, t):
    """A plate in the xy plane at height z (thickness t) with a square hole: root part, side bars, end strip."""
    xa, xb = min(x_free_end, x_root), max(x_free_end, x_root)
    h0, h1 = hole_x - hole / 2, hole_x + hole / 2
    zs = (z - t / 2, z + t / 2)
    _boxspan(stage, f"{path}_root", (h1, xb), (-width / 2, width / 2), zs, mats[mat])
    _boxspan(stage, f"{path}_end", (xa, h0), (-width / 2, width / 2), zs, mats[mat])
    for k, s in enumerate((-1, 1)):
        _boxspan(stage, f"{path}_side{k}", (h0, h1), (s * hole / 2, s * width / 2), zs, mats[mat])


# --------------------------------------------------------------------------------------------------
# Board
# --------------------------------------------------------------------------------------------------



def _fixed_articulation(stage, root_prim, r, mass=0.5):
    """A fixed-base articulation: its root is placed per seed with a root-pose write, which PhysX honours (a
    plain pose write on a kinematic body is ignored; found while building this board)."""
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    root_prim.AddAppliedSchema("PhysxArticulationAPI")
    root_prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    base = _rigid(stage, f"{r}/base", mass)
    fixed = UsdPhysics.FixedJoint.Define(stage, f"{r}/joints/world_fixed")
    fixed.CreateBody1Rel().SetTargets([base.GetPath()])
    return base



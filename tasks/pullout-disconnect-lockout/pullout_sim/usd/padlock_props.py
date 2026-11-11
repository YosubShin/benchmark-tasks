"""Generate the padlock props as USD: the switch panel, the lockout plates (one file per real lock
point), the lockout padlocks (one file per real padlock), the pre-applied padlock on the other switch,
and the pedestal the open padlock starts on, with a parking shelf under the desk.

The props follow real parts. Every variant is spawned in every scene; each seed
uses one padlock and one lock point, and the task parks the others out of sight.

Panel frame: origin on the desk at the switch boxes' front face (world x = PANEL_FACE_X, world y and z
unchanged); +x into the panel.

Lock point frame: origin where the plate meets the switch face, on the plate's mid-plane; the plate
sticks out toward the robot (-x) with a hole near its free end, so a shackle leg drops through the
hole while the other leg stays outside the plate's end. Lock points are kinematic bodies the task
places on the switches.

Padlock frame: origin at the body's centre; the shackle's legs stand on the body's top, the short leg
at +x and the long leg at -x, and the body's thickness is along y for the gripper's fingers. The
padlock is an articulation: the shackle slides on a prismatic joint along z, from 0 (shut) to
OPEN_TRAVEL (open, short leg clear of the body). The task supplies the spring and the latch.

Usage: python make_padlock_props.py <out dir>
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from pullout_sim.usd.padlock_layout import HASP_Z, LOCK_POINT_NAMES, PADLOCK_NAMES, PEDESTAL_XY, SWITCH_Y, TAG_COLORS


@dataclass(frozen=True)
class PadlockVariant:
    """A real lockout padlock, from its published dimensions. Collision shapes are
    boxes: a round rod of diameter d is a square rod of side d, which keeps its clearance in a hole."""

    name: str
    source: str
    body: tuple[float, float, float]
    """Width (x, across the legs), thickness (y, the gripper's faces) and height (z)."""
    rod: float
    clear_h: float
    """Horizontal shackle clearance: the gap between the legs."""
    clear_v: float
    """Vertical shackle clearance: body top to the inside of the bend, shut."""
    mass: float
    color: str

    @property
    def body_top(self) -> float:
        return self.body[2] / 2

    @property
    def leg_x(self) -> float:
        return (self.clear_h + self.rod) / 2

    @property
    def bend_z(self) -> float:
        """Bend centre above the body's top, shut."""
        return self.clear_v + self.rod / 2

    @property
    def grasp_in_lock(self) -> tuple[float, float, float]:
        """Arm A's grasp point (between the fingertips): the body's lower rear corner."""
        return (-self.body[0] / 2 + GRASP_DEPTH, 0.0, -self.body[2] / 2 + GRASP_DEPTH)

    def short_tip(self, travel: float) -> tuple[float, float, float]:
        """The short leg's lower tip in the padlock frame, with the shackle out by `travel`."""
        return (self.leg_x, 0.0, self.body_top + SHORT_LEG_BOTTOM + travel)

    def bend_bottom(self, travel: float) -> float:
        return self.body_top + self.bend_z - self.rod / 2 + travel

    def bend_top(self, travel: float) -> float:
        return self.body_top + self.bend_z + self.rod / 2 + travel


PADLOCKS = {
    v.name: v
    for v in (
        PadlockVariant("ml410", "Master Lock 410 Zenex", (0.038, 0.019, 0.044), 0.00635, 0.020, 0.038, 0.10,
                       "lock_purple"),
        PadlockVariant("a1105", "American Lock A1105", (0.038, 0.019, 0.048), 0.00635, 0.020, 0.025, 0.11,
                       "lock_blue"),
        PadlockVariant("abus7440", "ABUS 74/40", (0.040, 0.020, 0.045), 0.00675, 0.019, 0.038, 0.088,
                       "lock_orange"),
    )
}
assert tuple(PADLOCKS) == PADLOCK_NAMES
PRELOCKED = PadlockVariant("ml410_yellow", "Master Lock 410 Zenex, yellow (410YLW)", (0.038, 0.019, 0.044), 0.00635, 0.020, 0.038,
                           0.10, "lock_yellow")
"""Another worker's lock on the other switch."""
PRELOCKED_DAMPING = 3.0
SHORT_LEG_BOTTOM = -0.004
"""The short leg's tip inside the shut body (its latch notch)."""
OPEN_TRAVEL = 0.016
"""How far the shackle pops up. Not published for any lockout lock; an assumption to measure on bought
locks: enough that the short leg clears the body by 12 mm, room for a lockout plate to slide under it."""
LONG_LEG_BOTTOM = -(OPEN_TRAVEL + 0.008)
"""The long leg stays 8 mm inside the body when open (it is the pivot of a real lock)."""
SHACKLE_MASS = 0.02
GRASP_DEPTH = 0.010
"""Arm A holds the body by its lower rear corner, the grasp point this far in from the rear and bottom
faces. Its wrist camera then clears both the body and the open shackle (a real-size body is short)."""


@dataclass(frozen=True)
class LockPointVariant:
    """A real lockout point: a steel plate sticking out toward the robot with a hole near its end.
    The hole is a square of side `hole` (see PadlockVariant)."""

    name: str
    source: str
    tab_len: float
    tab_width: float
    tab_t: float
    hole: float
    end_strip: float
    """Steel between the hole and the plate's free end."""

    @property
    def hole_x(self) -> float:
        """Hole centre in the lock point (and panel) frame."""
        return -self.tab_len + self.end_strip + self.hole / 2


LOCK_POINTS = {
    v.name: v
    for v in (
        # Square D general-duty switch handles take a 3/8" (9.5 mm) shackle at most; hole and plate
        # thickness are not published (inferred: 10.3 mm hole, 3 mm plate).
        LockPointVariant("handle_tab", "switch-handle lockout tab (Square D GD style)", 0.040, 0.025, 0.003, 0.0103,
                         0.004),
        # Master Lock 420 hasp: fits shackles up to 7 mm (hole ~8 mm, inferred); two overlapping jaw
        # plates of unpublished thickness (~1.5 mm each, inferred) modelled as one 3.5 mm plate held still
        # (a real hasp hangs from the switch handle).
        LockPointVariant("hasp420", "Master Lock 420 lockout hasp", 0.040, 0.030, 0.0035, 0.008, 0.004),
    )
}
assert tuple(LOCK_POINTS) == LOCK_POINT_NAMES

CABINET = (0.10, 0.26, 0.30)
CABINET_Y = (min(SWITCH_Y) + max(SWITCH_Y)) / 2
PEDESTAL_TOP = 0.146
"""Height of the pedestal's top (the padlock's bottom face rests on it): arm A's level grasp at the
body's lower rear corner is then in the YAM's band (grasp height about 0.13-0.17)."""
PARK_Z = -0.30
"""The parking shelf under the desk, for the padlocks and plates a seed does not use."""
PARK_X = 0.30
PARK_SPACING_Y = 0.12

COLORS = {
    "cabinet": (0.55, 0.58, 0.55),
    "switch": (0.30, 0.31, 0.33),
    "steel": (0.72, 0.74, 0.77),
    "brass": (0.80, 0.62, 0.20),
    "lock_red": (0.80, 0.10, 0.08),
    "lock_yellow": (0.95, 0.72, 0.05),
    "lock_purple": (0.45, 0.20, 0.60),
    "lock_blue": (0.15, 0.35, 0.75),
    "lock_orange": (0.90, 0.45, 0.08),
    "danger": (0.90, 0.10, 0.10),
    "white": (0.95, 0.95, 0.95),
    "dark": (0.12, 0.12, 0.13),
    "stand": (0.40, 0.40, 0.42),
    "blue": (0.10, 0.30, 0.85),
    "yellow": (0.95, 0.78, 0.10),
}


def _material(stage, root, name):
    path = f"{root}/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[name]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _box(stage, path, center, size, material, collide=True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _friction(stage, root, paths):
    friction = UsdShade.Material.Define(stage, f"{root}/PhysicsMaterials/rubber")
    api = UsdPhysics.MaterialAPI.Apply(friction.GetPrim())
    api.CreateStaticFrictionAttr(1.5)
    api.CreateDynamicFrictionAttr(1.2)
    friction.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    friction.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("max")
    for path in paths:
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).Bind(friction, UsdShade.Tokens.weakerThanDescendants,
                                                                   "physics")


def _new_stage(out_path, root):
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prim = UsdGeom.Xform.Define(stage, root).GetPrim()
    stage.SetDefaultPrim(prim)
    return stage, prim


def _shackle(stage, path, mats, v: PadlockVariant, z0=0.0):
    """The U-shaped shackle in a frame at the body's top centre, shifted up by z0."""
    long_len = v.bend_z - LONG_LEG_BOTTOM
    short_len = v.bend_z - SHORT_LEG_BOTTOM
    rod = v.rod
    _box(stage, f"{path}/long_leg", (-v.leg_x, 0, z0 + LONG_LEG_BOTTOM + long_len / 2), (rod, rod, long_len),
         mats["steel"])
    _box(stage, f"{path}/short_leg", (v.leg_x, 0, z0 + SHORT_LEG_BOTTOM + short_len / 2), (rod, rod, short_len),
         mats["steel"])
    _box(stage, f"{path}/bend", (0, 0, z0 + v.bend_z), (2 * v.leg_x + rod, rod, rod), mats["steel"])


def make_panel(out_path: Path) -> None:
    """The cabinet and the two switch faces with their tags; the lockout plates are separate bodies."""
    stage, _ = _new_stage(out_path, "/panel")
    root = "/panel"
    mats = {name: _material(stage, root, name) for name in COLORS}
    _box(stage, f"{root}/cabinet", (CABINET[0] / 2, CABINET_Y, CABINET[2] / 2), CABINET, mats["cabinet"])
    for slot, y in enumerate(SWITCH_Y):
        sw = f"{root}/switch_{slot}"
        UsdGeom.Xform.Define(stage, sw)
        _box(stage, f"{sw}/plate", (-0.002, y, HASP_Z), (0.004, 0.10, 0.11), mats["switch"], collide=False)
        # Operating handle, down (off), on the face beside the lock point: visual only.
        _box(stage, f"{sw}/handle", (-0.015, y + 0.038, HASP_Z - 0.03), (0.03, 0.014, 0.04), mats["dark"], collide=False)
        for color in TAG_COLORS:
            _box(stage, f"{sw}/tag_{color}", (-0.005, y, HASP_Z + 0.035), (0.003, 0.06, 0.025), mats[color], collide=False)
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{sw}/tag_{TAG_COLORS[1 - slot]}")).MakeInvisible()
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_lock_point(out_path: Path, v: LockPointVariant) -> None:
    """One lockout plate, built around its hole: the root part, two side bars and the end strip."""
    stage, root_prim = _new_stage(out_path, "/lock_point")
    root = "/lock_point"
    UsdPhysics.RigidBodyAPI.Apply(root_prim).CreateKinematicEnabledAttr(True)
    mats = {name: _material(stage, root, name) for name in ("steel",)}
    root_len = v.tab_len - v.end_strip - v.hole
    side_w = (v.tab_width - v.hole) / 2
    _box(stage, f"{root}/root", (-root_len / 2, 0, 0), (root_len, v.tab_width, v.tab_t), mats["steel"])
    for k, sign in enumerate((-1, 1)):
        _box(stage, f"{root}/side{k}", (v.hole_x, sign * (v.hole + side_w) / 2, 0), (v.hole, side_w, v.tab_t),
             mats["steel"])
    _box(stage, f"{root}/end", (-v.tab_len + v.end_strip / 2, 0, 0), (v.end_strip, v.tab_width, v.tab_t),
         mats["steel"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_padlock(out_path: Path, v: PadlockVariant) -> None:
    stage, root_prim = _new_stage(out_path, "/padlock")
    root = "/padlock"
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    root_prim.AddAppliedSchema("PhysxArticulationAPI")
    root_prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    mats = {name: _material(stage, root, name) for name in (v.color, "steel", "dark")}
    body = UsdGeom.Xform.Define(stage, f"{root}/body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(v.mass - SHACKLE_MASS)
    _box(stage, f"{root}/body/case", (0, 0, 0), v.body, mats[v.color])
    _box(stage, f"{root}/body/keyway", (0, 0, -v.body[2] / 2 - 0.0005), (0.008, 0.004, 0.001), mats["dark"],
         collide=False)
    _friction(stage, root, [f"{root}/body/case"])
    shackle = UsdGeom.Xform.Define(stage, f"{root}/shackle")
    shackle.AddTranslateOp().Set(Gf.Vec3d(0, 0, v.body_top))
    UsdPhysics.RigidBodyAPI.Apply(shackle.GetPrim())
    UsdPhysics.MassAPI.Apply(shackle.GetPrim()).CreateMassAttr(SHACKLE_MASS)
    _shackle(stage, f"{root}/shackle", mats, v)
    joint = UsdPhysics.PrismaticJoint.Define(stage, f"{root}/joints/shackle_joint")
    joint.CreateBody0Rel().SetTargets([body.GetPath()])
    joint.CreateBody1Rel().SetTargets([shackle.GetPath()])
    joint.CreateAxisAttr("Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, v.body_top))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(OPEN_TRAVEL)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_prelocked(out_path: Path) -> None:
    """The padlock already on the other switch: one rigid body, shackle shut, with a red danger tag."""
    v = PRELOCKED
    stage, root_prim = _new_stage(out_path, "/prelocked")
    root = "/prelocked"
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(v.mass)
    # Stands in for the friction where a real lock hangs, which stops its swing within a few seconds;
    # PhysX's contact alone lets a hanging body rattle on indefinitely.
    root_prim.AddAppliedSchema("PhysxRigidBodyAPI")
    root_prim.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(PRELOCKED_DAMPING)
    root_prim.CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float).Set(PRELOCKED_DAMPING)
    mats = {name: _material(stage, root, name) for name in (v.color, "steel", "danger", "white")}
    _box(stage, f"{root}/case", (0, 0, 0), v.body, mats[v.color])
    UsdGeom.Xform.Define(stage, f"{root}/shackle")
    _shackle(stage, f"{root}/shackle", mats, v, z0=v.body_top)
    _box(stage, f"{root}/tag", (0, 0, -v.body[2] / 2 - 0.03), (0.035, 0.002, 0.05), mats["danger"], collide=False)
    _box(stage, f"{root}/tag_band", (0, -0.0012, -v.body[2] / 2 - 0.022), (0.03, 0.001, 0.008), mats["white"],
         collide=False)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_pedestals(out_path: Path) -> None:
    """The pedestal (world frame): a column under the padlock body, thinner than any body so the
    fingertips pass beside it when they grip the body's lower rear corner. Also the parking shelf."""
    stage, _ = _new_stage(out_path, "/pedestal")
    root = "/pedestal"
    mats = {name: _material(stage, root, name) for name in ("stand",)}
    x, y = PEDESTAL_XY
    _box(stage, f"{root}/foot", (x, y, 0.005), (0.06, 0.05, 0.01), mats["stand"])
    _box(stage, f"{root}/column", (x, y, PEDESTAL_TOP / 2), (0.030, 0.009, PEDESTAL_TOP), mats["stand"])
    _box(stage, f"{root}/park_shelf", (PARK_X, 0.0, PARK_Z - 0.005), (0.10, 0.60, 0.01), mats["stand"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def park_pos(index: int) -> tuple[float, float, float]:
    """A spot on the parking shelf (world frame, the shelf's top surface)."""
    return (PARK_X, (index - 2) * PARK_SPACING_Y, PARK_Z)


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    make_panel(out_dir / "panel.usda")
    for variant in LOCK_POINTS.values():
        make_lock_point(out_dir / f"lock_point_{variant.name}.usda", variant)
    for variant in PADLOCKS.values():
        make_padlock(out_dir / f"padlock_{variant.name}.usda", variant)
    make_prelocked(out_dir / "prelocked.usda")
    make_pedestals(out_dir / "pedestal.usda")

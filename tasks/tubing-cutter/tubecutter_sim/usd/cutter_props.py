"""Generate the tubing-cutter task's props as USD: the copper tube on its stand and the cutter, one articulation.

Real parts (references in README.md):
- Tube: 1/2 in nominal copper tube, Type L (ASTM B88): OD 15.88 mm, wall 1.02 mm. It stands upright, soldered into
  a male adapter screwed into a 1/2 in floor flange on a plywood block (a stub-out on the bench), 9.3 cm above the
  block. A black pen mark rings it 7.5 cm above the table.
- Cutter: Klein Tools 50077 1-Inch Quick-Adjust Screw-Feed Tubing Cutter: 17.4 cm long, 7 cm tall (in its plane),
  3.6 cm thick, 299 g, orange body. A fixed cutting wheel in the hook end; a carriage with two rollers that the feed
  screw pushes toward the wheel; a black knob on the screw's far end. Lying flat on the vertical tube, its plane is
  horizontal. Proportions of the parts are read off the vendor photos (no drawing is published).

Frames. The articulation root is ``tube`` (fixed to the world at spawn by Isaac Lab's fix_root_link; origin on the tube's axis at table height). The
cutter (``cutter``; origin on the tube's axis in the wheel's plane, +x toward the knob, +z up the tube) hangs on
``carrier`` by a prismatic joint ``slide`` along z (its axial position on the tube); ``carrier`` turns about the
tube's axis on the revolute ``orbit``. The knob (``knob``) turns on the revolute ``feed`` about the cutter's x. At
joint angle 0 the knob points along +y (spec.KNOB_START_DEG).

Collision: boxes, and an octagonal prism for the knob (the real knob is faceted). Self-collision is off: the cutter's
hold on the tube is the joints, not contact (see README, Fidelity details).

Usage: python -m tubecutter_sim.usd.cutter_props [out dir]   (default: tubecutter_sim/assets)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from tubecutter_sim.spec import BASE_TOP, KNOB_START_DEG, MARK_Z, TUBE_R, TUBE_TOP
from tubecutter_sim.usd import usd_kit as kit

# Cutter geometry, cutter frame (m): x (u) toward the knob, y (v) across, z (w) along the tube. ----------------------
HALF_T = 0.018
"""Body 3.6 cm thick (along the tube)."""
WHEEL_R = 0.0095
WHEEL_T = 0.003
HOOK = ((-0.040, -0.020), (-0.034, 0.016), (-0.012, 0.012))
"""(x range, y range, z half-range... as (x0, x1), (y0, y1), (zlo, zhi)) of the hook end holding the wheel."""
SPINE = ((-0.040, 0.036), (-0.034, -0.020))
CARRIAGE = ((0.013, 0.034), (-0.020, 0.017))
HOUSING = ((0.034, 0.076), (-0.020, 0.020))
"""The body the screw runs through (the quick-adjust housing): where a hand takes the cutter; 4 cm across (y)."""
SCREW = (0.076, 0.098)
SCREW_R = 0.0045
KNOB_U0 = 0.098
KNOB_L = 0.036
KNOB_R = 0.016
GRASP_U = 0.055
"""Centre of the housing (x), where the demo takes the cutter."""
KNOB_CENTER_U = KNOB_U0 + KNOB_L / 2
CUTTER_MASS, KNOB_MASS, CARRIER_MASS = 0.26, 0.04, 0.02
SLIDE_LIMIT = 0.02

# Stand ----------------------------------------------------------------------------------------------------------
BLOCK = (0.10, 0.10, 0.018)
"""Plywood block (x, y, z size), centred on the tube."""
FLANGE = (0.030, 0.004)
"""1/2 in floor flange: radius, thickness (on the block)."""


def _quat(deg: float):
    h = math.radians(deg) / 2
    return (0.0, 0.0, math.sin(h), math.cos(h))  # xyzw about z


def _materials(stage, root):
    colors = {
        "copper": (0.78, 0.42, 0.20), "mark": (0.02, 0.02, 0.02), "wood": (0.74, 0.60, 0.40),
        "iron": (0.20, 0.20, 0.21), "brass": (0.75, 0.58, 0.28), "orange": (0.93, 0.40, 0.06),
        "black": (0.05, 0.05, 0.05), "steel": (0.75, 0.76, 0.78), "dark": (0.18, 0.18, 0.19),
    }
    out = {}
    for name, c in colors.items():
        m = UsdShade.Material.Define(stage, f"{root}/Looks/{name}")
        sh = UsdShade.Shader.Define(stage, f"{root}/Looks/{name}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*c))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
        if name in ("copper", "steel", "brass"):
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.5)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        out[name] = m
    return out


def _body(stage, path, pos, quat, mass):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    x, y, z, w = quat
    xform.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(mass)
    return xform


def _joint(stage, path, kind, body0, body1, axis, pos0, rot0, lower, upper):
    """A revolute or prismatic joint whose frame is body1's frame (localPos1 = 0, localRot1 = identity)."""
    joint = (UsdPhysics.RevoluteJoint if kind == "revolute" else UsdPhysics.PrismaticJoint).Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(str(body0))])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(str(body1))])
    joint.CreateAxisAttr(axis)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    x, y, z, w = rot0
    joint.CreateLocalRot0Attr(Gf.Quatf(float(w), float(x), float(y), float(z)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular" if kind == "revolute" else "linear")
    return joint


def _prism(stage, path, center, radius, length, sides, material, axis="X", collide=True, phase=None):
    """A regular prism (sides >= 6), axis along x or z, as a mesh; with collide, a convex-hull collider."""
    phase = math.pi / sides if phase is None else phase
    ring = [(radius * math.cos(phase + 2 * math.pi * k / sides), radius * math.sin(phase + 2 * math.pi * k / sides))
            for k in range(sides)]
    pts = []
    for s in (-length / 2, length / 2):
        for a, b in ring:
            pts.append((s, a, b) if axis == "X" else (a, b, s))
    counts, idx = [], []
    n = sides
    for k in range(n):  # sides
        k2 = (k + 1) % n
        counts.append(4)
        idx += [k, k2, n + k2, n + k] if axis == "X" else [k, n + k, n + k2, k2]
    counts += [n, n]
    idx += list(range(n))[::-1] if axis == "X" else list(range(n))[::-1]
    idx += list(range(n, 2 * n))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    return mesh


def _box_r(stage, path, xr, yr, zr, material, collide=True):
    return kit.box(stage, path, ((xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2, (zr[0] + zr[1]) / 2),
                   (xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]), material, collide=collide)


def make_set(out_dir: Path) -> None:
    root = "/cutter_set"
    stage, _ = kit.new_stage(out_dir / "set.usda", root)
    m = _materials(stage, root)

    # Tube and stand (root body, fixed to the world).
    tube = _body(stage, f"{root}/tube", (0.0, 0.0, 0.0), (0, 0, 0, 1), 1.0)
    kit.box(stage, f"{tube.GetPath()}/block", (0, 0, BLOCK[2] / 2), BLOCK, m["wood"])
    kit.cylinder(stage, f"{tube.GetPath()}/flange", (0, 0, BLOCK[2] + FLANGE[1] / 2), FLANGE[0], FLANGE[1], m["iron"])
    kit.cylinder(stage, f"{tube.GetPath()}/adapter", (0, 0, BASE_TOP + 0.006), 0.0115, 0.012, m["brass"])
    t_len = TUBE_TOP - BASE_TOP
    kit.cylinder(stage, f"{tube.GetPath()}/tube_vis", (0, 0, BASE_TOP + t_len / 2), TUBE_R, t_len, m["copper"])
    kit.box(stage, f"{tube.GetPath()}/tube_col", (0, 0, BASE_TOP + t_len / 2), (2 * TUBE_R * 0.9, 2 * TUBE_R * 0.9,
                                                                                t_len), m["copper"])
    UsdGeom.Imageable(stage.GetPrimAtPath(f"{tube.GetPath()}/tube_col")).MakeInvisible()
    kit.cylinder(stage, f"{tube.GetPath()}/mark", (0, 0, MARK_Z), TUBE_R + 0.0003, 0.0015, m["mark"])
    kit.articulation_root(stage, tube.GetPrim())
    tube.GetPrim().CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    tube.GetPrim().CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)

    q0 = _quat(KNOB_START_DEG)
    carrier = _body(stage, f"{root}/carrier", (0, 0, MARK_Z), q0, CARRIER_MASS)
    cutter = _body(stage, f"{root}/cutter", (0, 0, MARK_Z), q0, CUTTER_MASS)
    c = cutter.GetPath()
    z = (-HALF_T, HALF_T)
    zh = (-0.012, 0.012)
    _box_r(stage, f"{c}/hook", HOOK[0], HOOK[1], zh, m["orange"])
    _box_r(stage, f"{c}/spine", SPINE[0], SPINE[1], zh, m["orange"])
    _box_r(stage, f"{c}/carriage", CARRIAGE[0], CARRIAGE[1], (-0.010, 0.010), m["orange"])
    _box_r(stage, f"{c}/housing", HOUSING[0], HOUSING[1], z, m["orange"])
    kit.box(stage, f"{c}/button", (0.050, 0.023, 0.0), (0.016, 0.006, 0.012), m["black"], collide=False)
    kit.cylinder(stage, f"{c}/wheel", (-TUBE_R - WHEEL_R, 0, 0), WHEEL_R, WHEEL_T, m["steel"])
    kit.cylinder(stage, f"{c}/wheel_bolt", (-TUBE_R - WHEEL_R, 0, 0.0125), 0.004, 0.003, m["dark"])
    for s in (-1, 1):
        rr = 0.0065
        d = TUBE_R + rr
        kit.cylinder(stage, f"{c}/roller{s + 1}", (d * math.cos(math.radians(35)), s * d * math.sin(math.radians(35)),
                                                   0.0), rr, 0.012, m["black"])
    kit.cylinder(stage, f"{c}/screw", ((SCREW[0] + SCREW[1]) / 2, 0, 0), SCREW_R, SCREW[1] - SCREW[0], m["dark"],
                 axis="X")
    for k in range(4):
        kit.cylinder(stage, f"{c}/crest{k}", (SCREW[0] + 0.003 + 0.005 * k, 0, 0), SCREW_R + 0.0008, 0.0015,
                     m["dark"], axis="X")
    knob = _body(stage, f"{root}/knob", (0, 0, MARK_Z), q0, KNOB_MASS)
    # Knob body origin at the knob joint (cutter x = KNOB_U0): position it there in the world.
    kx, ky = KNOB_U0 * math.cos(math.radians(KNOB_START_DEG)), KNOB_U0 * math.sin(math.radians(KNOB_START_DEG))
    UsdGeom.Xformable(knob).GetOrderedXformOps()[0].Set(Gf.Vec3d(kx, ky, MARK_Z))
    _prism(stage, f"{knob.GetPath()}/grip", (KNOB_L / 2, 0, 0), KNOB_R, KNOB_L, 8, m["black"], axis="X")
    _prism(stage, f"{knob.GetPath()}/stripe", (KNOB_L - 0.002, 0, 0), KNOB_R * 0.55, 0.0045, 6, m["orange"],
           axis="X", collide=False)

    kit.friction(stage, root, [f"{c}/housing", f"{c}/hook", f"{c}/spine", f"{knob.GetPath()}/grip"], static=1.2,
                 dynamic=1.0, name="grip")

    _joint(stage, f"{root}/joints/orbit", "revolute", tube.GetPath(), carrier.GetPath(), "Z", (0, 0, MARK_Z), q0,
           -1440.0, 1440.0)
    _joint(stage, f"{root}/joints/slide", "prismatic", carrier.GetPath(), cutter.GetPath(), "Z", (0, 0, 0),
           (0, 0, 0, 1), -SLIDE_LIMIT, SLIDE_LIMIT)
    _joint(stage, f"{root}/joints/feed", "revolute", cutter.GetPath(), knob.GetPath(), "X", (KNOB_U0, 0, 0),
           (0, 0, 0, 1), -1440.0, 1440.0)
    stage.GetRootLayer().Save()
    print(f"wrote {out_dir / 'set.usda'}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets"
    out.mkdir(parents=True, exist_ok=True)
    make_set(out)

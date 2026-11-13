"""Generate the props of the fire-extinguisher task as USD: the extinguisher (with its pin and tamper seal), the fire
stand with its pan and taped keep-out line, three flame tongues and the spray plume.

Real parts (values marked "inferred" are not published and are to measure):

- Amerex B417 (B417T: same unit with a vehicle bracket), 2.5 lb ABC stored-pressure dry chemical extinguisher, UL
  1-A:10-B:C. Published (distributor spec tables; label on the unit): overall height 15.5 in (394 mm), width
  5.75 in (146 mm, nozzle tip to lever end), cylinder 3 in (76 mm) across, discharge time about 10 s, range 9-15 ft,
  all-metal valve, "large loop pull pin". Label: "1. Hold upright. Pull ring pin. 2. Start back 8 feet. Aim at base
  of fire. 3. Squeeze lever. Sweep side to side." Layout from the product photos (see README, References): a short
  horizontal discharge horn on the front of the valve; the operating lever on top, pivoted at the front of the
  valve and reaching back over the carry handle; the pull pin crosses the valve neck under the lever, its ring on
  one side, held by a plastic tamper seal; the gauge on the side.
  Built (frame U: origin at the cylinder's bottom centre, +x toward the nozzle, z up, the pin along y, ring on +y):
  * body (floating articulation root): cylinder 76 mm x 312 mm (one convex 24-gon prism collider), neck, valve head
    36 x 26 x 32 mm, nozzle horn 40 mm long, 16 mm across, its outlet 58 mm in front of the axis and 350 mm up;
    carry handle, a 20 mm wide bar leaving the valve 352 mm up and running back 105 mm, sloping 12 deg down; two
    ears behind the valve, each with a 6 mm square hole for the pin.
  * lever (revolute about y, pivot 8 mm behind the axis and 378 mm up): a 120 mm bar 18 mm wide rising 8 deg toward
    the back at rest, with two side cheeks inside the ears that the pin also passes through. Travel 0 (rest, against
    its stop) to 20 deg (closed onto the handle). Return spring: 6 N at rest rising to 12 N fully squeezed, measured
    60 mm from the pivot (a joint drive; values chosen for the YAM, the real B417 needs about 50 N, see README).
  * pin (a link on a prismatic joint along its axis, 0 to 60 mm): a 4 mm square shaft through the ears' holes and
    the lever's cheeks, so the lever cannot go down more than about 3 deg while the pin is in (contact between pin
    and cheeks); a 24 mm ring on the unit's left side.
  * tamper seal (rule, fire_sim/task.py): the pin joint's drive is stiff (4000 N/m) until the pin has been pulled
    2 mm, i.e. 8 N; then it is released for good and the pin slides out freely. Once clear of the far ear it is held
    at its 60 mm stop (a real pin comes free and is dropped).
- Fire: a steel stand 14 x 26 cm, 24 cm tall, with a 4 cm deep steel pan (interior 12 x 24 cm) on top filled with
  fuel to the rim (28 cm above the table); three flame tongues side by side, 10.5 cm tall (emissive, visual only).
- Keep-out line: yellow floor tape 25 mm wide, 20 cm out from the stand's footprint on the robot's side and both
  sides.
- Spray plume: a pale cone from the nozzle (visual only), shown while the valve is open.

Usage: python -m fire_sim.usd.props <out dir>
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from fire_sim.usd import usd_kit as kit

from fire_sim.geometry import *  # noqa: F401,F403  (dimensions)
from fire_sim.geometry import MM

COLORS = dict(kit.COLORS)
COLORS.update({
    "ext_red": (0.78, 0.05, 0.04),
    "chrome": (0.80, 0.81, 0.83),
    "gauge": (0.92, 0.92, 0.90),
    "seal_green": (0.10, 0.70, 0.25),
    "pan": (0.30, 0.31, 0.33),
    "stand": (0.45, 0.46, 0.48),
    "fuel": (0.16, 0.11, 0.08),
    "tape": (0.98, 0.82, 0.05),
})
EMISSIVE = {"flame_outer": ((1.0, 0.16, 0.01), 1.6), "flame_inner": ((1.0, 0.48, 0.04), 2.0)}


def _mat(stage, root, name, emissive=None, opacity=None):
    path = f"{root}/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    color = COLORS.get(name, (1, 1, 1)) if emissive is None else emissive[0]
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35 if name in ("chrome", "ext_red") else 0.6)
    if emissive is not None:
        c, k = emissive
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(k * v for v in c)))
    if opacity is not None:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _box(stage, path, center, size, material, collide=True, rot_y_deg=0.0):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    if rot_y_deg:
        xf.AddRotateYOp().Set(rot_y_deg)
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _bar(stage, path, p0, length, deg, width, thick, material, collide=True):
    """A bar in the x-z plane starting at p0 (x, z) and running back (-x) at ``deg`` above horizontal."""
    a = math.radians(deg)
    cx = p0[0] - 0.5 * length * math.cos(a)
    cz = p0[1] + 0.5 * length * math.sin(a)
    # rotateY by +theta turns +x toward -z; the bar's long axis is x, we want it pointing back-and-up by deg.
    return _box(stage, path, (cx * MM, 0.0, cz * MM), (length * MM, width * MM, thick * MM), material, collide,
                rot_y_deg=deg)


def _cyl(stage, path, center, radius, height, material, axis="Z"):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr(axis)
    UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(material)
    return cyl


def _cone(stage, path, center, radius, height, material, axis="Z"):
    cone = UsdGeom.Cone.Define(stage, path)
    cone.CreateRadiusAttr(radius)
    cone.CreateHeightAttr(height)
    cone.CreateAxisAttr(axis)
    UsdGeom.Xformable(cone).AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdShade.MaterialBindingAPI.Apply(cone.GetPrim()).Bind(material)
    return cone


def _prism(stage, path, radius, z0, z1, n=24):
    """An invisible convex n-gon prism collider (one convex hull: never pie slices)."""
    pts = []
    for z in (z0, z1):
        for i in range(n):
            t = 2 * math.pi * (i + 0.5) / n
            pts.append(Gf.Vec3f(radius * math.cos(t), radius * math.sin(t), z))
    counts, idx = [n, n], list(range(n))[::-1] + list(range(n, 2 * n))
    for i in range(n):
        j = (i + 1) % n
        counts.append(4)
        idx += [i, j, n + j, n + i]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()
    return mesh


def _rigid(stage, path, translate, mass, com=None, inertia=None, kinematic=False):
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    rb = UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    m = UsdPhysics.MassAPI.Apply(xf.GetPrim())
    m.CreateMassAttr(mass)
    if com is not None:
        m.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    if inertia is not None:
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    return xf


def _friction(stage, root, paths, static=1.0, dynamic=0.9, name="grip"):
    mat = UsdShade.Material.Define(stage, f"{root}/PhysicsMaterials/{name}")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    mat.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    mat.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("max")
    for path in paths:
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(mat, UsdShade.Tokens.weakerThanDescendants,
                                                                          "physics")


def make_extinguisher(out_path: Path) -> None:
    stage, root_prim = kit.new_stage(out_path, "/extinguisher")
    root = "/extinguisher"
    mats = {n: _mat(stage, root, n) for n in ("ext_red", "chrome", "gauge", "dark", "seal_green", "black", "white")}
    m = lambda *v: tuple(x * MM for x in v)  # noqa: E731

    # Body (articulation root, floating).
    r, h = CYL_R * MM, CYL_H * MM
    body = _rigid(stage, f"{root}/body", (0, 0, 0), BODY_MASS, BODY_COM,
                  (BODY_MASS * (3 * r * r + h * h) / 12, BODY_MASS * (3 * r * r + h * h) / 12, BODY_MASS * r * r / 2))
    UsdPhysics.ArticulationRootAPI.Apply(body.GetPrim())
    body.GetPrim().AddAppliedSchema("PhysxArticulationAPI")
    body.GetPrim().CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(True)
    b = f"{root}/body"
    _prism(stage, f"{b}/cyl_collider", r, 0.0, h)
    _cyl(stage, f"{b}/cylinder", (0, 0, (h - 0.012) / 2), r, h - 0.012, mats["ext_red"])
    _cyl(stage, f"{b}/shoulder", (0, 0, h - 0.006), r - 0.008, 0.012, mats["ext_red"])
    _cyl(stage, f"{b}/base_ring", (0, 0, 0.003), r + 0.0005, 0.006, mats["dark"])
    _cyl(stage, f"{b}/neck", (0, 0, (NECK[1] + NECK[2]) / 2 * MM), NECK[0] * MM, (NECK[2] - NECK[1]) * MM, mats["chrome"])
    _box(stage, f"{b}/neck_col", m(0, 0, (NECK[1] + NECK[2]) / 2), m(28, 28, NECK[2] - NECK[1]), mats["chrome"])
    UsdGeom.Imageable(stage.GetPrimAtPath(f"{b}/neck_col")).MakeInvisible()
    hx, hy, hz = HEAD
    _box(stage, f"{b}/head", m(sum(hx) / 2, 0, sum(hz) / 2), m(hx[1] - hx[0], hy[1] - hy[0], hz[1] - hz[0]),
         mats["chrome"])
    # Instruction label band (white) on the cylinder, toward the robot, as on the real unit.
    _cyl(stage, f"{b}/label", (0, 0, 0.17), r + 0.0006, 0.09, mats["white"])
    # Gauge on the -y side of the valve.
    _cyl(stage, f"{b}/gauge", m(0, hy[0] - 4, 346), 0.013, 0.008, mats["gauge"], axis="Y")
    _box(stage, f"{b}/gauge_needle", m(0, hy[0] - 8.2, 349), m(2, 0.6, 9), mats["ext_red"], collide=False)
    # Nozzle horn.
    x0, x1, nz, nr = NOZZLE
    _cyl(stage, f"{b}/nozzle", m((x0 + x1) / 2, 0, nz), nr * MM, (x1 - x0) * MM, mats["chrome"], axis="X")
    _cyl(stage, f"{b}/nozzle_bore", m(x1 + 0.2, 0, nz), (nr - 2.5) * MM, 0.0006, mats["black"], axis="X")
    _box(stage, f"{b}/nozzle_col", m((x0 + x1) / 2, 0, nz), m(x1 - x0, 2 * nr - 2, 2 * nr - 2), mats["chrome"])
    UsdGeom.Imageable(stage.GetPrimAtPath(f"{b}/nozzle_col")).MakeInvisible()
    # Carry handle (fixed), sloping down toward the back.
    _bar(stage, f"{b}/handle", HANDLE_ROOT, HANDLE_LEN, HANDLE_DEG, HANDLE_W, HANDLE_T, mats["chrome"])
    # Ears with the pin holes (4 boxes each around a HOLE x HOLE window), and the bridge under them.
    px, pz = PIN_AXIS
    hole_lo = pz - PIN / 2 - PIN_DROP  # the pin starts PIN_DROP above the holes' lower edges (no overlap at spawn)
    hole_hi = hole_lo + HOLE
    for k, (y0, y1) in enumerate((EAR_Y, (-EAR_Y[1], -EAR_Y[0]))):
        yc, yw = (y0 + y1) / 2, y1 - y0
        ex0, ex1 = EAR_X
        _box(stage, f"{b}/ear{k}_lo", m((ex0 + ex1) / 2, yc, (EAR_Z[0] + hole_lo) / 2),
             m(ex1 - ex0, yw, hole_lo - EAR_Z[0]), mats["chrome"])
        _box(stage, f"{b}/ear{k}_hi", m((ex0 + ex1) / 2, yc, (hole_hi + EAR_Z[1]) / 2),
             m(ex1 - ex0, yw, EAR_Z[1] - hole_hi), mats["chrome"])
        _box(stage, f"{b}/ear{k}_back", m((ex0 + px - HOLE / 2) / 2, yc, (hole_lo + hole_hi) / 2),
             m(px - HOLE / 2 - ex0, yw, hole_hi - hole_lo), mats["chrome"])
        _box(stage, f"{b}/ear{k}_front", m((px + HOLE / 2 + ex1) / 2, yc, (hole_lo + hole_hi) / 2),
             m(ex1 - px - HOLE / 2, yw, hole_hi - hole_lo), mats["chrome"])
    _box(stage, f"{b}/bridge", m(sum(EAR_X) / 2, 0, 356), m(EAR_X[1] - EAR_X[0], 2 * EAR_Y[1], 8), mats["chrome"])
    _friction(stage, root, [f"{b}/handle"])

    # Lever (revolute about y at the pivot; link frame origin at the pivot).
    pvx, pvz = PIVOT
    lever = _rigid(stage, f"{root}/lever", m(pvx, 0, pvz), LEVER_MASS)
    lv = f"{root}/lever"
    _bar(stage, f"{lv}/bar", (0.0, 0.0), LEVER_LEN, LEVER_REST_DEG, LEVER_W, LEVER_T, mats["chrome"])
    # Cheeks: two side flanges under the bar, inside the ears, each with a hole the pin passes through (as on the
    # real lever). With the pin in, the holes' top edges meet the pin after CHEEK_PLAY mm: the lever stops near 3 deg.
    hole_lo = pz - PIN / 2 - PIN_DROP
    for k, (y0, y1) in enumerate((CHEEK_Y, (-CHEEK_Y[1], -CHEEK_Y[0]))):
        yc, yw = (y0 + y1) / 2, y1 - y0
        cx0, cx1 = CHEEK_X
        z0, z1 = CHEEK_Z
        hz0, hz1 = hole_lo, hole_lo + PIN + PIN_DROP + CHEEK_PLAY
        for name, xs, zs in (("lo", (cx0, cx1), (z0, hz0)), ("hi", (cx0, cx1), (hz1, z1)),
                             ("back", (cx0, px - HOLE / 2), (hz0, hz1)), ("front", (px + HOLE / 2, cx1), (hz0, hz1))):
            _box(stage, f"{lv}/cheek{k}_{name}", m((xs[0] + xs[1]) / 2 - pvx, yc, (zs[0] + zs[1]) / 2 - pvz),
                 m(xs[1] - xs[0], yw, zs[1] - zs[0]), mats["chrome"])
    _friction(stage, root, [f"{lv}/bar"], name="grip_lever")
    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/joints/lever_joint")
    joint.CreateBody0Rel().SetTargets([body.GetPath()])
    joint.CreateBody1Rel().SetTargets([lever.GetPath()])
    joint.CreateAxisAttr("Y")
    joint.CreateLocalPos0Attr(Gf.Vec3f(pvx * MM, 0, pvz * MM))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLowerLimitAttr(-LEVER_TRAVEL_DEG)
    joint.CreateUpperLimitAttr(0.0)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")

    # Pin: a link on a prismatic joint along its axis (see below).
    pin = _rigid(stage, f"{root}/pin", m(px, 0, pz), PIN_MASS, inertia=(2e-6, 1e-7, 2e-6))
    pn = f"{root}/pin"
    _box(stage, f"{pn}/shaft", m(0, sum(PIN_Y) / 2, 0), m(PIN, PIN_Y[1] - PIN_Y[0], PIN), mats["chrome"])
    rcx, rcy, rcz = RING_C
    half = RING_OUT / 2 - RING_BAR / 2
    for name, c, s in (("ring_top", (0, half), (RING_OUT, RING_BAR)), ("ring_bot", (0, -half), (RING_OUT, RING_BAR)),
                       ("ring_back", (-half, 0), (RING_BAR, RING_OUT)), ("ring_front", (half, 0), (RING_BAR, RING_OUT))):
        _box(stage, f"{pn}/{name}", m(rcx + c[0], rcy, rcz + c[1]), m(s[0], RING_BAR, s[1]), mats["chrome"])
    _box(stage, f"{pn}/ring_link", m(0, rcy, (rcz + half + 0) / 2 + 1), m(3, RING_BAR, abs(rcz + half) + 2),
         mats["chrome"], collide=False)
    # The tamper seal's plastic strand, from the ring toward the handle (visual; it stays on the ring when broken).
    _box(stage, f"{pn}/seal", m(-6, 21.5, -4.5), m(2.5, 17, 2.0), mats["seal_green"], collide=False)
    _box(stage, f"{pn}/seal_tag", m(-6, 21.5, -8.5), m(7, 7, 6), mats["seal_green"], collide=False)
    # The pin slides out along its axis on a prismatic joint (a link of the articulation, so it collides with the
    # lever's cheeks: siblings collide with self-collisions on; parent-child pairs, pin-body and lever-body, do not).
    # Its drive is the tamper seal (stiff until 8 N, then released by the task: fire_sim/task.py).
    slide = UsdPhysics.PrismaticJoint.Define(stage, f"{root}/joints/pin_joint")
    slide.CreateBody0Rel().SetTargets([body.GetPath()])
    slide.CreateBody1Rel().SetTargets([pin.GetPath()])
    slide.CreateAxisAttr("Y")
    slide.CreateLocalPos0Attr(Gf.Vec3f(px * MM, 0, pz * MM))
    slide.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    slide.CreateLowerLimitAttr(0.0)
    slide.CreateUpperLimitAttr(PIN_TRAVEL * MM)
    UsdPhysics.DriveAPI.Apply(slide.GetPrim(), "linear")
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_fire_stand(out_path: Path) -> None:
    """Stand, pan, fuel bed and the keep-out tape (desk frame; one kinematic body at the origin)."""
    stage, root_prim = kit.new_stage(out_path, "/fire_stand")
    root = "/fire_stand"
    UsdPhysics.RigidBodyAPI.Apply(root_prim).CreateKinematicEnabledAttr(True)
    mats = {n: _mat(stage, root, n) for n in ("stand", "pan", "fuel", "tape", "dark")}
    (x0, x1), (y0, y1) = STAND_X, STAND_Y
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sx, sy = x1 - x0, y1 - y0
    _box(stage, f"{root}/stand", (cx, cy, STAND_H / 2), (sx, sy, STAND_H), mats["stand"])
    pan_h = FUEL_Z - STAND_H
    w = PAN_WALL
    _box(stage, f"{root}/pan_floor", (cx, cy, STAND_H + 0.001), (sx, sy, 0.002), mats["pan"])
    for name, c, s in (("pan_front", (x0 + w / 2, cy), (w, sy)), ("pan_back", (x1 - w / 2, cy), (w, sy)),
                       ("pan_left", (cx, y1 - w / 2), (sx, w)), ("pan_right", (cx, y0 + w / 2), (sx, w))):
        _box(stage, f"{root}/{name}", (c[0], c[1], STAND_H + pan_h / 2), (s[0], s[1], pan_h), mats["pan"])
    _box(stage, f"{root}/fuel", (cx, cy, STAND_H + (pan_h - 0.002) / 2), (sx - 2 * w, sy - 2 * w, pan_h - 0.002),
         mats["fuel"])
    # Tape: 20 cm out from the footprint, on the robot's side and both sides (inner edge on the line).
    k, t = KEEP_OUT, TAPE_W
    fx = x0 - k
    _box(stage, f"{root}/tape_front", (fx - t / 2, cy, 0.0004), (t, sy + 2 * k + 2 * t, 0.0008), mats["tape"],
         collide=False)
    for name, y in (("tape_left", y1 + k + t / 2), ("tape_right", y0 - k - t / 2)):
        _box(stage, f"{root}/{name}", ((fx + TABLE_FAR_X) / 2, y, 0.0004), (TABLE_FAR_X - fx, t, 0.0008), mats["tape"],
             collide=False)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_flame(out_path: Path) -> None:
    """One flame tongue (frame at the middle of its base on the fuel surface), emissive, visual only."""
    stage, root_prim = kit.new_stage(out_path, "/flame")
    root = "/flame"
    rb = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(0.01)
    outer = _mat(stage, root, "flame_outer", emissive=EMISSIVE["flame_outer"])
    inner = _mat(stage, root, "flame_inner", emissive=EMISSIVE["flame_inner"])
    h, r = FLAME_H, FLAME_R
    _cyl(stage, f"{root}/base", (0, 0, 0.012), r, 0.024, outer)
    _cone(stage, f"{root}/outer", (0, 0, 0.024 + (h - 0.024) / 2), r, h - 0.024, outer)
    _cone(stage, f"{root}/lick_a", (0.008, 0.012, 0.05), r * 0.55, 0.09, outer)
    _cone(stage, f"{root}/lick_b", (-0.006, -0.014, 0.045), r * 0.5, 0.08, outer)
    _cyl(stage, f"{root}/core_base", (0, 0, 0.012), r * 0.62, 0.0245, inner)
    _cone(stage, f"{root}/core", (0, 0, 0.024 + 0.03), r * 0.62, 0.06, inner)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_spray(out_path: Path) -> None:
    """The discharge plume: a pale cone along +x from its apex at the origin (the nozzle outlet), visual only. It is
    drawn short (17 cm) so that it does not hide the flames from the top camera."""
    stage, root_prim = kit.new_stage(out_path, "/spray")
    root = "/spray"
    UsdPhysics.RigidBodyAPI.Apply(root_prim).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(0.01)
    COLORS["spray"] = (0.93, 0.95, 1.0)
    mat = _mat(stage, root, "spray", opacity=0.55)
    cone = _cone(stage, f"{root}/plume", (SPRAY_LEN / 2, 0, 0), SPRAY_R, SPRAY_LEN, mat, axis="X")
    # A Cone's apex points +axis; flip it so the apex sits at the nozzle and it widens away from it.
    UsdGeom.Xformable(cone).AddRotateZOp().Set(180.0)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    make_extinguisher(out / "extinguisher.usda")
    make_fire_stand(out / "fire_stand.usda")
    make_flame(out / "flame.usda")
    make_spray(out / "spray.usda")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "assets"))

"""Generate the props (sample a handhole's air through its pull slot, open only on a clean reading) as USD.

Parts (each follows a real product; "inferred" values are not published and are to measure). References are in
the README.

- Handhole (`box.usda`), static: a Hubbell Quazite PG1118 box (sales drawing PG1118BA12:
  top 514 x 340 mm outside, cover recess 464 x 289 mm, opening below the ledge 400 x 225 mm, two 3/8-7 EZ-nut
  inserts at diagonal corners, bored here for the bolt shanks).
- Bolts (`bolt.usda`, two instances), free rigid bodies: PG1118CA00 parts list C080173 3/8-7 hex auger bolt with a
  C080210 washer (1.0 in OD x 0.08 in). The thread is a task rule: while engaged it pulls the bolt down onto the
  cover with a fixed force (the washer clamps the cover; friction under the washer resists turning); after
  ENGAGE_TURNS counter-clockwise the rule stops and the bolt is a loose part. Simplifications: the box is cut to 75 mm above the desk (real 12 in
  deep, set in the ground); the thread engages for ENGAGE_TURNS (real: about 6-12 turns, inferred from the 3 in
  bolt through a 1-3/4 in cover); the 1-1/4 in x 1/2 in counterbore is left out so the head stands on the cover
  (the YAM fingers would not fit the 32 mm bore); the hex head (9/16 in, 14.3 mm across flats) is a square of
  14.3 mm and 12 mm tall (real about 6 mm) so a fingertip pinch clears the washer.
- Cover (`cover.usda`), a free rigid body: Quazite PG1118CA00 / HA00, 460 x 286 x 44 mm, two 5/8 in bolt holes at
  diagonal corners (15-3/4 x 8-7/8 in spacing), and the 1/2 x 2 in pull slot with a 3/16 in centre pin, centred on
  the short axis, 6-9/16 in (167 mm) from the centre. Real mass 12 kg (27 lb); here 0.6 kg (a declared light
  stand-in). Tan polymer concrete.
- Monitor (`monitor.usda`), a floating articulation: a Honeywell BW GasAlertMax XT II (integral pump;
  131 x 70 x 52 mm, 328 g, user manual p.87) lying face up on a block, with its sample hose (BW Tygon, 1/4 in OD,
  1/8 in ID; sampling brochure) as a chain of rigid 50 mm links on alternating hinges, an inline filter the robot
  holds (20 x 20 x 40 mm; size and position inferred), and the last 100 mm of hose (after a 30 mm rigid stub) to the open end as one rigid
  link on a two-axis gimbal at the stub (so it hangs straight down whatever the
  filter's pitch in the fingers). A real 1/4 in Tygon end hangs roughly straight over this length; the three-hinge
  version swung and buckled against the slot walls. The case is yellow with a black bezel round the LCD, as on the
  real unit; its two visual alarm bars (red) sit at the end of the face and flash in alarm (task rule).
  Real hoses are 10-75 ft; 0.63 m here.
- Stand (`stand.usda`), static: the meter's low block and the pad the filter lies on.

Frames: everything is written in the env (world) frame; desk top z = 0.

Usage: python -m handhole_sim.usd.handhole_props handhole_sim/assets
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from handhole_sim.usd import usd_kit as kit


def _capsule(stage, path, p0, p1, radius, material):
    """A collision capsule from p0 to p1 (axis-aligned segments only)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    axis = "XYZ"[int(np.argmax(np.abs(d)))]
    cap = UsdGeom.Capsule.Define(stage, path)
    cap.CreateAxisAttr(axis)
    cap.CreateRadiusAttr(radius)
    cap.CreateHeightAttr(float(np.linalg.norm(d)))
    UsdGeom.Xformable(cap).AddTranslateOp().Set(Gf.Vec3d(*((p0 + p1) / 2)))
    UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    return cap


def _cylinder(stage, path, center, radius, height, material):
    """A collision cylinder along the body's local z."""
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    return cyl


def _body(stage, path, pos, quat, mass):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    x, y, z, w = quat
    xform.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(mass)
    return xform


def _joint(stage, path, body0, body1, axis, pos0, rot0, pos1, lower_deg, upper_deg):
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(str(body0))])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(str(body1))])
    joint.CreateAxisAttr(axis)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    x, y, z, w = rot0
    joint.CreateLocalRot0Attr(Gf.Quatf(float(w), float(x), float(y), float(z)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*pos1))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLowerLimitAttr(lower_deg)
    joint.CreateUpperLimitAttr(upper_deg)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    return joint

IN = 0.0254

# Box and cover (Quazite PG1118) -------------------------------------------------------------------
BX, BY = 0.30, -0.02
"""Box centre in the world. The long axis (18 in) runs along y."""
Z_TOP = 0.085
"""Cover top (flush with the box top). The box is cut to 85 mm (real 12 in, set in the ground): deep enough for the
sample probe hanging straight down from a seated collar (its tip 32 mm below the cover's underside, 9 mm above the
table), low enough that carrying the probe over the cover stays inside the YAM's fingers-down reach (above about
0.21 m the station IK cannot hold the fingers straight down)."""
COVER = (11.25 * IN, 18.125 * IN, 1.75 * IN)
"""x, y, thickness: 286 x 460 x 44 mm."""
COVER_MASS = 0.6
Z_COVER_BOTTOM = Z_TOP - COVER[2]
BOX_OUT = (13.375 * IN, 20.25 * IN)
RECESS = (11.375 * IN, 18.25 * IN)
OPENING = (8.875 * IN, 15.75 * IN)
FLOOR_T = 0.0
"""The box has an open bottom, as Quazite PG boxes do (they sit on gravel); here it stands on the table."""
BOLT_DX, BOLT_DY = 8.875 * IN / 2, 15.75 * IN / 2
BOLTS = ((BX + BOLT_DX, BY + BOLT_DY), (BX - BOLT_DX, BY - BOLT_DY))
"""Bolt 0 (far, +y: arm A) and bolt 1 (near, -y: arm B), at diagonal corners as on the drawing."""
BOLT_HOLE = 0.016
SLOT_XY = (BX, BY + 6.5625 * IN)
SLOT = (0.5 * IN, 2.0 * IN)
"""Pull slot: 13 mm along x, 51 mm along y."""
PIN = 0.1875 * IN
PIN_Z = Z_TOP - 0.012
SLOT_OPENING_Y = SLOT_XY[1] - (SLOT[1] / 2 + PIN / 2) / 2
"""Centre of the slot's half toward the cover's centre (beside the pin), where the hose goes in."""

PITCH = IN / 7
ENGAGE_TURNS = 1.5
WASHER = (1.0 * IN, 0.002)
HEAD = (0.0143, 0.012)
SHANK = 0.0095
SHANK_LEN = COVER[2] + ENGAGE_TURNS * PITCH + 0.002
"""Shank below the washer: through the cover and the engaged thread."""
BOLT_MASS = 0.04
"""A 3/8-7 x 2 in steel hex bolt (about 35 g) with its 1 in washer (about 6 g)."""
BOSS_HOLE = 0.0155
"""Bore in the box's bolt boss for the shank (the EZ-nut insert's thread is the task's rule). The shank's collider is
a 9.5 mm square (13.4 mm across corners), so the bore clears it at any turn."""
WASHER_MU = 0.15
"""Washer and shank friction (steel on polymer concrete, lightly greased thread; inferred). Under the thread's clamp
it is what resists turning the bolt."""
BOLT_CENTRE_Z = (WASHER[1] + HEAD[1] - SHANK_LEN) / 2
"""Bolt frame z of the point halfway between the head's top and the shank's tip (its middle, for placing it)."""

INTERIOR = (OPENING[0] / 2, OPENING[1] / 2, FLOOR_T, Z_COVER_BOTTOM)
"""Half x, half y, floor z, ceiling z of the box's air space (below the ledge)."""

# Monitor, hose and sample probe -------------------------------------------------------------------------------
METER = (0.131, 0.070, 0.052)
METER_MASS = 0.328
HOSE_R = 0.25 * IN / 2
LINK_L = 0.05
LINK_MASS = 0.004
LINK_LIMIT_DEG = 110.0
RISER_L = 0.044
RISER_PITCH = 60.0
"""Each of the two hose links that rise beside the holster to the probe's elbow."""
HOSE_LINKS = ((0, 0, LINK_L), (0, 0, LINK_L), (90, 0, LINK_L), (90, 0, LINK_L)) + ((180, 0, LINK_L),) * 5 + (
    (180, RISER_PITCH, RISER_L), (180, RISER_PITCH, RISER_L))
"""(heading deg about z, pitch deg up, length) of each hose link from the meter's inlet: back, left, forward toward
the robot along the table, then up at RISER_PITCH to the probe's elbow (the riser comes in from the far side, so as
the probe goes down into the slot it swings out away from the probe instead of folding under its collar)."""
HOSE_AXES = ("Z", "Y", "Z", "Y", "Z", "Y", "Z", "Y", "Z", "Y", "Z")
"""Hinge axis of each link's joint to the one before it, in that link's frame. The riser's bottom hinge ("Y") lets it
swing out along the hose; its middle one ("Z") lets it swing sideways."""
RISER_LIMIT_DEG = 170.0

# The probe: a rigid stainless sampling probe (1/4 in OD tube) on the end of the hose, the kind of rigid probe BW
# sells for this monitor (GA-PROB1-1, 1 ft; shorter here, see PROBE_L), with a clamp-on shaft collar (Ruland CL-4-F:
# 1/4 in bore, 5/8 in OD, 9/32 in wide) as its depth stop. Body frame: origin at the centre of the collar's underside,
# +z along the probe toward the handle; the tip is at z = -PROBE_L.
PROBE_R = 0.25 * IN / 2
PROBE_L = 0.076
"""Tip to the collar's underside. With the collar resting on the cover's top face the tip is at least 2.5 mm below the
cover's underside at every tilt the 13 x 51 mm slot allows (README "Verifying the probe depth")."""
COLLAR_R = 0.625 * IN / 2
COLLAR_W = 0.28125 * IN
ROD_ABOVE = 0.006
"""Bare tube between the collar's top and the handle."""
HANDLE = (0.020, 0.020, 0.036)
"""The probe's handle (its filter housing), a square block the fingers close on; the hose enters near its top."""
HANDLE_Z0 = COLLAR_W + ROD_ABOVE
"""Probe frame z of the handle's bottom face."""
GRASP_Z = HANDLE_Z0 + 0.014
"""Probe frame z where the demo grasps the handle (14 mm above its bottom: the fingertips stay above the collar)."""
ELBOW = (0.018, HANDLE_Z0 + HANDLE[2] - 0.008)
"""The hose joins the handle through a 90-degree barbed elbow: (distance from the probe's axis toward the hose, z)."""
TIP_BAND = 0.010
"""One wrap of blue tape round the probe's last 10 mm: it marks the tip for the reviewer and for the robot's wrist
camera."""
PROBE_MASS = 0.035
HOLSTER_XY = (0.355, 0.38)
"""The probe stands in a holster beside the box (a block with a bore; the collar rests on its top, at Z_TOP)."""
HOLSTER = (0.024, 0.024)
HOLSTER_BORE = 0.010
Z_HOSE = Z_TOP + ELBOW[1] - 2 * RISER_L * math.sin(math.radians(RISER_PITCH))
"""Height of the hose's run along the table and of the meter's inlet at rest (the meter lies on a low block)."""
SLOT_W, SLOT_L, COVER_T = SLOT[0], SLOT[1], COVER[2]
COLLAR_GAP_M = 0.002
"""The collar counts as resting on the cover while the lowest point of its underside over solid cover is at most this
far above the top face (README; the same rule in the sim and in the real trial)."""
PROBE_TILT_MAX_DEG = 60.0
"""... and while the probe stands within this of vertical (it goes down into the slot, not lying across the cover)."""
WARN_TAG = (0.080, 0.118, (BX, BY + 0.045))
"""The yellow warning tag on the cover (x size, y size, centre): "TEST AIR / BEFORE / OPENING" in black block
letters, reading from behind the robot, between the pull slot and the cover's centre."""


COLORS = {
    "concrete": (0.60, 0.52, 0.39), "concrete_dark": (0.45, 0.39, 0.29), "steel": (0.74, 0.75, 0.78),
    "meter_black": (0.08, 0.08, 0.09), "meter_yellow": (0.95, 0.75, 0.08), "hose": (0.80, 0.86, 0.88),
    "lcd_bg": (0.66, 0.72, 0.58), "lcd_alarm": (0.85, 0.30, 0.25), "lcd_seg": (0.03, 0.03, 0.03),
    "alarm_off": (0.22, 0.22, 0.24),
    "filter": (0.92, 0.92, 0.90), "warn_yellow": (0.98, 0.82, 0.05), "tape": (0.08, 0.25, 0.80), "stand": (0.40, 0.40, 0.42), "black": (0.05, 0.05, 0.05),
}

DISPLAY_STATES = ("room", "go", "nogo")
READINGS = {
    # BW order on the LCD: H2S, CO, O2, LEL. Room air, and the two seeded box atmospheres.
    "room": ("0", "0", "20.9", "0", False),
    "go": ("0", "0", "20.9", "0", False),  # clean box air reads like room air, as on a real monitor
    "nogo": ("18", "5", "20.4", "0", True),
}
"""Alarm low setpoints (BW Max XT II manual p.55): H2S 10 ppm, CO 35 ppm, O2 19.5 %, LEL 10 %."""


ALARM_BAR = (0.028, 0.026)
"""Each of the two visual alarm lenses on the face (meter frame x, y size), side by side at the end away from the
inlet (the real unit's wide-angle alarm bars; enlarged so they read at the policy camera's 320 x 176)."""


def _emissive(stage, root, name, color):
    path = f"{root}/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _mats(stage, root, names):
    out = {}
    for name in names:
        path = f"{root}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[name]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        out[name] = material
    return out


def _plate_with_holes(stage, path, x_range, y_range, z_range, holes, material):
    """A plate (horizontal) with rectangular holes, as a grid of boxes. holes: (cx, cy, sx, sy)."""
    xs = sorted({x_range[0], x_range[1], *[h[0] - h[2] / 2 for h in holes], *[h[0] + h[2] / 2 for h in holes]})
    ys = sorted({y_range[0], y_range[1], *[h[1] - h[3] / 2 for h in holes], *[h[1] + h[3] / 2 for h in holes]})
    zc, zt = (z_range[0] + z_range[1]) / 2, z_range[1] - z_range[0]
    k = 0
    for i in range(len(xs) - 1):  # merge cells along y within each x column
        x0, x1 = xs[i], xs[i + 1]
        run = None
        for j in range(len(ys) - 1):
            y0, y1 = ys[j], ys[j + 1]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            in_hole = any(abs(cx - h[0]) < h[2] / 2 and abs(cy - h[1]) < h[3] / 2 for h in holes)
            if not in_hole:
                run = (run[0], y1) if run else (y0, y1)
            if (in_hole or j == len(ys) - 2) and run:
                kit.box(stage, f"{path}_{k}", ((x0 + x1) / 2, (run[0] + run[1]) / 2, zc),
                        (x1 - x0, run[1] - run[0], zt), material)
                k += 1
                run = None


def _ring(stage, path, outer, inner, z_range, center, material):
    """Four walls between an outer and an inner rectangle (x, y sizes) about center."""
    cx, cy = center
    zc, zt = (z_range[0] + z_range[1]) / 2, z_range[1] - z_range[0]
    wx, wy = (outer[0] - inner[0]) / 2, (outer[1] - inner[1]) / 2
    for k, s in enumerate((-1, 1)):
        kit.box(stage, f"{path}_x{k}", (cx + s * (inner[0] / 2 + wx / 2), cy, zc), (wx, outer[1], zt), material)
        kit.box(stage, f"{path}_y{k}", (cx, cy + s * (inner[1] / 2 + wy / 2), zc), (inner[0], wy, zt), material)


def _thin(stage, prefix, offset=0.002):
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(offset)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def make_box(out_path: Path) -> None:
    """The box: static colliders (no bodies, no joints). The two bolt bosses are bored for the bolts' shanks."""
    root = "/box"
    stage, root_prim = kit.new_stage(out_path, root)
    m = _mats(stage, root, ("concrete", "concrete_dark"))
    p = root
    kit.box(stage, f"{p}/floor", (BX, BY, FLOOR_T / 2), (BOX_OUT[0], BOX_OUT[1], FLOOR_T), m["concrete_dark"])
    # Walls below the ledge, bored where the bolts' shanks pass (each bolt sits on the opening's corner).
    bores = [(x, y, BOSS_HOLE, BOSS_HOLE) for x, y in BOLTS]
    _plate_with_holes(stage, f"{p}/lower", (BX - BOX_OUT[0] / 2, BX + BOX_OUT[0] / 2),
                      (BY - BOX_OUT[1] / 2, BY + BOX_OUT[1] / 2), (FLOOR_T, Z_COVER_BOTTOM),
                      [(BX, BY, OPENING[0], OPENING[1]), *bores], m["concrete"])
    _ring(stage, f"{p}/upper", BOX_OUT, (RECESS[0], RECESS[1]), (Z_COVER_BOTTOM, Z_TOP), (BX, BY), m["concrete"])
    # Bolt corners: the ledge is solid where the EZ-nut inserts sit (the opening is chamfered there on the drawing),
    # with a bore for the bolt's shank.
    for k, (x, y) in enumerate(BOLTS):
        sx = 1 if x > BX else -1
        sy = 1 if y > BY else -1
        cx, cy = x - sx * 0.01, y - sy * 0.01
        _plate_with_holes(stage, f"{p}/boss{k}", (cx - 0.03, cx + 0.03), (cy - 0.03, cy + 0.03),
                          (FLOOR_T, Z_COVER_BOTTOM), [(x, y, BOSS_HOLE, BOSS_HOLE)], m["concrete"])
    _thin(stage, f"{p}/boss")
    _thin(stage, f"{p}/lower")
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_bolt(out_path: Path) -> None:
    """One cover bolt with its washer: a free rigid body (frame origin at the washer's underside, on the bolt's axis).
    The thread is a task rule (handhole_sim.task): while engaged it pulls the bolt down onto the cover; once unscrewed
    the bolt is an ordinary loose part."""
    root = "/bolt"
    stage, root_prim = kit.new_stage(out_path, root)
    m = _mats(stage, root, ("steel", "concrete_dark"))
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(BOLT_MASS)
    b = root
    kit.box(stage, f"{b}/washer", (0, 0, WASHER[1] / 2), (WASHER[0], WASHER[0], WASHER[1]), m["steel"])
    kit.box(stage, f"{b}/head", (0, 0, WASHER[1] + HEAD[1] / 2), (HEAD[0], HEAD[0], HEAD[1]), m["steel"])
    kit.box(stage, f"{b}/shank", (0, 0, -SHANK_LEN / 2), (SHANK, SHANK, SHANK_LEN), m["steel"])
    kit.box(stage, f"{b}/mark", (HEAD[0] / 2 - 0.002, 0, WASHER[1] + HEAD[1] + 0.0003), (0.004, 0.003, 0.0006),
            m["concrete_dark"], collide=False)
    kit.friction(stage, root, [f"{b}/head"], static=1.0, dynamic=0.9, name="steel_grip")
    kit.friction(stage, root, [f"{b}/washer", f"{b}/shank"], static=WASHER_MU, dynamic=WASHER_MU, name="steel_seat",
                 combine="min")
    _thin(stage, root)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def bolt_pose(k: int) -> tuple:
    """Bolt k's home pose (seated, thread engaged): washer underside on the cover's top."""
    x, y = BOLTS[k]
    return (x, y, Z_TOP, 0.0, 0.0, 0.0, 1.0)


FONT = {
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "E": ("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "I": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "F": ("#####", "#....", "#....", "####.", "#....", "#....", "#...."),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"),
    "G": (".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".###."),
    " ": (".....",) * 7,
}
WARN_LINES = ("TEST AIR", "BEFORE", "OPENING")


def _warning_tag(stage, root, top_z, m):
    """The tag in the cover's frame: a yellow plate and block letters (5 x 7 cells, each row's runs merged into one
    box). Text reads from behind the robot: left to right along -y, its top toward +x (away from the robot)."""
    sx, sy, (wx, wy) = WARN_TAG
    cx, cy = wx - BX, wy - BY
    kit.box(stage, f"{root}/tag", (cx, cy, top_z + 0.0004), (sx, sy, 0.0008), m["warn_yellow"], collide=False)
    kit.box(stage, f"{root}/tag_rim", (cx, cy, top_z + 0.0002), (sx + 0.004, sy + 0.004, 0.0004), m["black"],
            collide=False)
    cols = max(len(line) for line in WARN_LINES) * 6 - 1
    rows = len(WARN_LINES) * 7 + (len(WARN_LINES) - 1) * 3
    u = min((sy - 0.012) / cols, (sx - 0.012) / rows)
    z = top_z + 0.0010
    k = 0
    for li, line in enumerate(WARN_LINES):
        width = len(line) * 6 - 1
        y_left = cy + width * u / 2
        x_top = cx + rows * u / 2 - li * 10 * u
        for ci, ch in enumerate(line):
            for r, bits in enumerate(FONT[ch]):
                c = 0
                while c < 5:
                    if bits[c] != "#":
                        c += 1
                        continue
                    c1 = c
                    while c1 < 5 and bits[c1] == "#":
                        c1 += 1
                    y0 = y_left - (ci * 6 + c) * u
                    y1 = y_left - (ci * 6 + c1) * u
                    kit.box(stage, f"{root}/tag_txt_{k}", (x_top - (r + 0.5) * u, (y0 + y1) / 2, z),
                            (u, abs(y1 - y0), 0.0006), m["black"], collide=False)
                    k += 1
                    c = c1


def make_cover(out_path: Path) -> None:
    root = "/cover"
    stage, root_prim = kit.new_stage(out_path, root)
    m = _mats(stage, root, ("concrete", "concrete_dark", "steel", "warn_yellow", "black"))
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(COVER_MASS)
    holes = [(x - BX, y - BY, BOLT_HOLE, BOLT_HOLE) for x, y in BOLTS]
    holes.append((SLOT_XY[0] - BX, SLOT_XY[1] - BY, SLOT[0], SLOT[1]))
    hx, hy, t = COVER[0] / 2, COVER[1] / 2, COVER[2]
    _plate_with_holes(stage, f"{root}/plate", (-hx, hx), (-hy, hy), (-t / 2, t / 2), holes, m["concrete"])
    sx, sy = SLOT_XY[0] - BX, SLOT_XY[1] - BY
    kit.box(stage, f"{root}/pin", (sx, sy, PIN_Z - (Z_TOP - t / 2)), (SLOT[0] + 0.002, PIN, PIN), m["steel"])
    # The warning tag (visual): a yellow plate with "TEST AIR / BEFORE / OPENING" in black block letters.
    _warning_tag(stage, root, t / 2, m)
    _thin(stage, root)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def cover_pose() -> tuple:
    return (BX, BY, Z_TOP - COVER[2] / 2, 0.0, 0.0, 0.0, 1.0)


# Monitor -----------------------------------------------------------------------------------------------
def _qz(deg):
    a = math.radians(deg) / 2
    return np.array([0.0, 0.0, math.sin(a), math.cos(a)])


def _qy(deg):
    a = math.radians(deg) / 2
    return np.array([0.0, math.sin(a), 0.0, math.cos(a)])


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def _rot(q, v):
    qv = np.array([*v, 0.0])
    c = np.array([-q[0], -q[1], -q[2], q[3]])
    return _qmul(_qmul(q, qv), c)[:3]


def _link_q(heading_deg, pitch_deg):
    """Link frame: local +x along the link, heading about z, pitched up by pitch_deg."""
    return _qmul(_qz(heading_deg), _qy(-pitch_deg))


def elbow_point():
    """World position where the hose joins the probe's elbow (probe at rest in its holster)."""
    return np.array([HOLSTER_XY[0] + ELBOW[0], HOLSTER_XY[1], Z_TOP + ELBOW[1]])


def probe_rest_pose() -> tuple:
    """The probe standing in its holster: collar on the holster's top (at Z_TOP), axis vertical."""
    return (HOLSTER_XY[0], HOLSTER_XY[1], Z_TOP, 0.0, 0.0, 0.0, 1.0)


def hose_layout():
    """Rest poses (world pos, quat) of meter, hose links and probe; plus each joint's rest rotation (in its parent
    body's frame) and axis. Built backwards from the probe's elbow."""
    n = len(HOSE_LINKS)
    starts = [None] * n
    end = elbow_point()
    for k in range(n - 1, -1, -1):
        h, pch, ln = HOSE_LINKS[k]
        starts[k] = end - _rot(_link_q(h, pch), (ln, 0.0, 0.0))
        end = starts[k]
    inlet = starts[0]
    meter_q = _qz(HOSE_LINKS[0][0])
    meter_pos = inlet - _rot(meter_q, (METER[0] / 2, 0, 0))
    bodies = [("meter", meter_pos, meter_q)]
    joints = []
    prev_q = meter_q
    conj = lambda q: np.array([-q[0], -q[1], -q[2], q[3]])  # noqa: E731
    for k in range(n):
        q = _link_q(*HOSE_LINKS[k][:2])
        bodies.append((f"hose_{k:02d}", starts[k], q))
        joints.append((HOSE_AXES[k], _qmul(conj(prev_q), q)))
        prev_q = q
    ident = np.array([0.0, 0.0, 0.0, 1.0])
    p = probe_rest_pose()
    bodies.append(("probe", np.array(p[:3]), ident))
    joints.append(("Z", _qmul(conj(prev_q), ident)))
    return bodies, joints


JOINT_NAMES = tuple([f"hose_j{k:02d}" for k in range(len(HOSE_LINKS))] + ["probe_j"])
PROBE_BODY = "probe"


def _quad_up(stage, path, center, size, material, u_range):
    """A horizontal textured quad facing +z: text right along local +x, text up along local -y."""
    cx, cy, cz = center
    sx, sy = size
    x0, x1, y0, y1 = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2
    pts = [(x1, y0, cz), (x0, y0, cz), (x0, y1, cz), (x1, y1, cz)]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    u0, u1 = u_range
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(u0, 0), Gf.Vec2f(u1, 0), Gf.Vec2f(u1, 1), Gf.Vec2f(u0, 1)]))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


# The LCD as geometry: seven-segment digits made of thin visual boxes, one group per display state (the task shows
# one group). A textured version was replaced by another task's texture at render time (renderer texture cache).
SEGMENTS = {
    "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg", "5": "acdfg", "6": "acdefg", "7": "abc",
    "8": "abcdefg", "9": "abcdfg",
}
LCD_CENTER = (0.012, 0.0)
LCD_SIZE = (0.064, 0.054)
"""Meter frame: text reads along local +x, its up is local +y (the reviewer camera's view of the meter)."""


def _seven_seg(stage, path, text, x0, y0, h, material):
    """Draw text (digits and '.') with its lower-left corner at (x0, y0) in the meter's top plane."""
    w, t = 0.55 * h, 0.13 * h
    x = x0
    k = 0
    for ch in text:
        if ch == ".":
            kit.box(stage, f"{path}_{k}", (x + t / 2, y0 + t / 2, 0), (t, t, 0.0006), material, collide=False)
            x += 2 * t
            k += 1
            continue
        segs = {"a": (x + w / 2, y0 + h - t / 2, w, t), "d": (x + w / 2, y0 + t / 2, w, t),
                "g": (x + w / 2, y0 + h / 2, w, t), "b": (x + w - t / 2, y0 + 3 * h / 4, t, h / 2),
                "c": (x + w - t / 2, y0 + h / 4, t, h / 2), "e": (x + t / 2, y0 + h / 4, t, h / 2),
                "f": (x + t / 2, y0 + 3 * h / 4, t, h / 2)}
        for sg in SEGMENTS[ch]:
            cx, cy, sx, sy = segs[sg]
            kit.box(stage, f"{path}_{k}", (cx, cy, 0), (sx, sy, 0.0006), material, collide=False)
            k += 1
        x += w + 0.35 * h


def _lcd_geometry(stage, body, top_z, m):
    cx, cy = LCD_CENTER
    sx, sy = LCD_SIZE
    z = top_z + 0.0004
    kit.box(stage, f"{body}/lcd_panel", (cx, cy, z), (sx, sy, 0.0008), m["lcd_bg"], collide=False)
    # Quadrant dividers.
    kit.box(stage, f"{body}/lcd_div_x", (cx, cy, z + 0.0005), (0.0008, sy - 0.004, 0.0004), m["lcd_seg"],
            collide=False)
    kit.box(stage, f"{body}/lcd_div_y", (cx, cy - 0.004, z + 0.0005), (sx - 0.004, 0.0008, 0.0004), m["lcd_seg"],
            collide=False)
    h = 0.0075
    # Quadrants (BW order): H2S top-left, CO top-right, O2 bottom-left, LEL bottom-right; a band at the top for ALARM.
    origins = [(cx - sx / 2 + 0.003, cy + 0.007), (cx + 0.003, cy + 0.007), (cx - sx / 2 + 0.003, cy - 0.017),
               (cx + 0.003, cy - 0.017)]
    for state in DISPLAY_STATES:
        h2s, co, o2, lel, alarm = READINGS[state]
        g = f"{body}/lcd_{state}"
        xf = UsdGeom.Xform.Define(stage, g)
        xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, z + 0.0007))
        if alarm:
            kit.box(stage, f"{g}/alarm_band", (cx, cy + sy / 2 - 0.004, 0), (sx - 0.002, 0.006, 0.0005),
                    m["lcd_alarm"], collide=False)
            kit.box(stage, f"{g}/alarm_h2s", (cx - sx / 4, cy + 0.011, -0.0002), (sx / 2 - 0.002, h + 0.006, 0.0004),
                    m["lcd_alarm"], collide=False)
        for q, (val, (ox, oy)) in enumerate(zip((h2s, co, o2, lel), origins)):
            _seven_seg(stage, f"{g}/v{q}", val, ox, oy, h, m["lcd_seg"])


def make_monitor(out_dir: Path) -> None:
    root = "/monitor"
    out_path = out_dir / "monitor.usda"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    m = _mats(stage, root, ("meter_black", "meter_yellow", "hose", "filter", "black", "lcd_bg", "lcd_alarm",
                            "lcd_seg", "alarm_off", "steel", "tape"))
    m["alarm_on"] = _emissive(stage, root, "alarm_on", (1.0, 0.08, 0.05))
    bodies, joints = hose_layout()
    paths = {}
    for name, pos, q in bodies:
        mass = {"meter": METER_MASS, "probe": PROBE_MASS}.get(name, LINK_MASS)
        body = _body(stage, f"{root}/{name}", pos, q, mass)
        paths[name] = body.GetPath()
        b = f"{root}/{name}"
        if name == "meter":
            lx, ly, lz = METER
            kit.box(stage, f"{b}/case", (0, 0, 0), METER, m["meter_yellow"])
            cx, cy = LCD_CENTER
            kit.box(stage, f"{b}/bezel", (cx, cy, lz / 2 + 0.0002), (LCD_SIZE[0] + 0.012, LCD_SIZE[1] + 0.010, 0.0004),
                    m["meter_black"], collide=False)
            for k, sy in enumerate((-1, 1)):
                c = (-lx / 2 + 0.004 + ALARM_BAR[0] / 2, sy * (ALARM_BAR[1] / 2 + 0.002))
                for state in ("off", "on"):
                    kit.box(stage, f"{b}/alarm_{state}_{k}", (c[0], c[1], lz / 2 + (0.0012 if state == "on" else 0.0008)),
                            (ALARM_BAR[0], ALARM_BAR[1], 0.0008), m[f"alarm_{state}"], collide=False)
            kit.box(stage, f"{b}/inlet", (lx / 2 + 0.004, 0, 0), (0.008, 0.012, 0.012), m["black"], collide=False)
            _lcd_geometry(stage, b, lz / 2, m)
        elif name.startswith("hose"):
            ln = HOSE_LINKS[int(name.split("_")[1])][2]
            _capsule(stage, f"{b}/tube", (0, 0, 0), (ln, 0, 0), HOSE_R, m["hose"])
        else:  # the probe: tube, depth-stop collar, bare tube, handle, elbow
            _cylinder(stage, f"{b}/tube", (0, 0, -PROBE_L / 2), PROBE_R, PROBE_L, m["steel"])
            kit.cylinder(stage, f"{b}/tip_band", (0, 0, -PROBE_L + TIP_BAND / 2), PROBE_R + 0.0004, TIP_BAND,
                                m["tape"])
            _cylinder(stage, f"{b}/collar", (0, 0, COLLAR_W / 2), COLLAR_R, COLLAR_W, m["black"])
            _cylinder(stage, f"{b}/rod", (0, 0, COLLAR_W + ROD_ABOVE / 2), PROBE_R, ROD_ABOVE, m["steel"])
            kit.box(stage, f"{b}/handle", (0, 0, HANDLE_Z0 + HANDLE[2] / 2), HANDLE, m["filter"])
            _capsule(stage, f"{b}/elbow", (HANDLE[0] / 2, 0, ELBOW[1]), (ELBOW[0], 0, ELBOW[1]), HOSE_R, m["hose"])
    names = [b[0] for b in bodies]
    for k, (axis, rest) in enumerate(joints):
        parent, child = names[k], names[k + 1]
        if parent == "meter":
            pos0 = (METER[0] / 2, 0, 0)
        else:
            pos0 = (HOSE_LINKS[int(parent.split("_")[1])][2], 0, 0)
        pos1 = (ELBOW[0], 0, ELBOW[1]) if child == "probe" else (0, 0, 0)
        lim = RISER_LIMIT_DEG if k >= len(HOSE_LINKS) - 2 else LINK_LIMIT_DEG
        _joint(stage, f"{root}/joints/{JOINT_NAMES[k]}", paths[parent], paths[child], axis, pos0, tuple(rest),
               pos1, -lim, lim)
    kit.friction(stage, root, [f"{root}/{n}" for n in names if n.startswith("hose")], static=0.15,
                 dynamic=0.12, name="tygon", combine="min")
    kit.friction(stage, root, [f"{root}/probe/handle"], static=1.2, dynamic=1.0, name="grip")
    kit.friction(stage, root, [f"{root}/probe/tube", f"{root}/probe/collar", f"{root}/probe/rod"], static=0.3,
                 dynamic=0.25, name="probe_steel", combine="min")
    kit.friction(stage, root, [f"{root}/meter"], static=0.9, dynamic=0.8, name="rubber")
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_stand(out_path: Path) -> None:
    root = "/stand"
    stage, _ = kit.new_stage(out_path, root)
    m = _mats(stage, root, ("stand",))
    bodies, _ = hose_layout()
    mpos = bodies[0][1]
    block_top = mpos[2] - METER[2] / 2
    kit.box(stage, f"{root}/meter_block", (mpos[0], mpos[1], block_top / 2), (0.12, 0.09, block_top), m["stand"])
    # The probe's holster: a block bored for the probe's tube; the collar rests on its top, level with the cover.
    hx, hy = HOLSTER_XY
    _plate_with_holes(stage, f"{root}/holster", (hx - HOLSTER[0] / 2, hx + HOLSTER[0] / 2),
                      (hy - HOLSTER[1] / 2, hy + HOLSTER[1] / 2), (0.0, Z_TOP),
                      [(hx, hy, HOLSTER_BORE, HOLSTER_BORE)], m["stand"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    make_box(out_dir / "box.usda")
    make_bolt(out_dir / "bolt.usda")
    make_cover(out_dir / "cover.usda")
    make_monitor(out_dir)
    make_stand(out_dir / "stand.usda")

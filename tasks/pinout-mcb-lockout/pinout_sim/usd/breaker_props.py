"""Generate the breaker-lockout props as USD: a breaker row whose toggles are real two-position levers with slots for
lockout pins, a Brady 090844 pin-out lockout that pushes straight onto the toggle, the Master Lock 410 in its real
shape (usd/padlock_410.py), the board and the stands.

Values: (P) published, (I) inferred, (TM) to measure on a bought part. Collision shapes are boxes.

Frames
- breaker row: origin at the rail's back (board face), row centre, rail centre = window centre; breakers extend
  along -x toward the robot. Toggle k is a link on a revolute joint about +y at its pivot, 3 mm behind the face;
  angle > 0 swings the lever's tip up (ON).
- lockout: origin at the centre of the black nose's rear face (it sits on the breaker face), +x into the panel
  (identity orientation when installed). Its axis sits AXIS_DZ below the breaker's window centre. Links: body
  (root), button (cap + stem, prismatic along x; q = 0 released, cap GAP out from the body; q = GAP pressed home)
  and two pins (revolute about y at their pivots in the nose, coupled to the button by a mimic joint: pressing
  swings them outward, releasing lets the button's return spring bring them back in behind the breaker's lips).

Usage: python -m pinout_sim.usd.breaker_props <out dir>
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdPhysics

from pinout_sim import spec as S
from pinout_sim.usd import padlock_410
from pinout_sim.usd import padlock_props
from pinout_sim.usd import usd_helpers as dr

BREAKER = dr.BREAKERS["abb_s201"]
"""ABB S201-C16: 17.5 mm pitch, 88 mm high, 69 mm deep (P), 45 mm window (I)."""
FACE_DEPTH = BREAKER.depth - 0.009
"""Plate to the breaker's front face (depth less a 9 mm toggle)."""

# Toggle (all I/TM: no drawing; photos of S201 show a ~7 mm lever swinging about +-25 degrees).
TOGGLE_W = 0.007
TOGGLE_T = 0.007
TOGGLE_LEN = 0.015
TOGGLE_PIVOT_IN = 0.003
"""Pivot behind the front face."""
TOGGLE_DEG = 25.0
"""OFF = -25 (tip down), ON = +25."""
TOGGLE_LIMIT_DEG = 28.0

LABEL_LEN = 0.040
LABEL_X = -0.024
"""Colour label on top of each breaker: 40 x 15.5 mm, centred 24 mm in front of the mounting plate (I)."""

# Pin holes in the breaker face above and below the toggle (I/TM: real MCB faces have openings beside the toggle
# that pin-out lockouts use; their exact size is not published). Lockout frame: |z| about the lockout's axis.
SLOT_Z = (0.0080, 0.0135)
"""Each hole spans this |z| range about the lockout's axis (4.5 + 1 mm tall, I/TM)."""
SLOT_HALF_Y = 0.0020
FACE_WALL = 0.003
"""Face wall the holes go through: the lip the pin feet sit behind."""
CAVITY_DEPTH = 0.009
CAVITY_Z = (0.0050, 0.0135)
"""Behind the wall each hole opens inward (toward the toggle), so a foot can sit behind the lip."""

# Lockout: Brady pin-out lockout (POS / POW class, push button, spring-held pins). 24.1 mm flange and 41.7 mm long
# (P, Brady 090844 distributor listing); the rest is read off the photos and Brady's overview animation (I/TM).
AXIS_DZ = -0.00375
"""Lockout axis below the window centre, so the pocket clears the OFF toggle (which hangs below the centre)."""
BODY_R = 0.010
BODY_LEN = 0.036
FLANGE_R = 0.01205
FLANGE_LEN = 0.004
NOSE_LEN = 0.010
POCKET_DEPTH = 0.0155
POCKET_Y = 0.0045
POCKET_Z = (-0.00725, 0.00675)
"""Lockout frame. The roof at +6.75 mm (window +3.0) clears the OFF toggle by 0.5 mm and stops it short of ON."""
CAP_T = 0.008
CAP_HALF = (0.0045, 0.012)
"""Black oval cap, 9 x 24 mm (I, photos), 8 mm thick."""
STEM_T = 0.004
"""Stem tab thickness along y (the padlock hole's axis), I."""
STEM_HALF_Z = 0.0065
STEM_LEN = 0.019
HOLE = 0.0071
"""9/32 in max shackle (P, RS 811-3034) -> 7.1 mm square hole."""
HOLE_FROM_CAP = 0.0030
"""Stem strip between the cap's inner face and the hole (I): the hole ends 2.9 mm short of the body."""
GAP = 0.013
"""Button travel = the cap-to-body gap when released (I/TM; Brady's animation shows about half the body's width)."""
# Pins: two steel wire hooks, one above and one below the pocket, each a lever pivoted in the nose. The button's stem
# drives them (a cam inside the body): pressing the button swings both feet outward; the return spring brings them
# back in. Feet point inward (toward the axis), so the released pins hook behind the lip between the two holes.
PIN = 0.0014
PIN_HALF_Y = 0.0007
PIVOT_X = -0.009
PIN_Z0 = 0.0089
"""Pin shaft centre (|z|) when released."""
PIN_TIP_X = 0.0060
FOOT_X = (0.0036, 0.0056)
"""Foot sits 0.6 mm behind the 3 mm face wall when the nose rests on the breaker face."""
FOOT_LEN = 0.0026
"""Foot reaches this far inward from the shaft: 2.4 mm under the lip when released."""
PIN_SPREAD_DEG = 13.0
"""Each pin swings this far outward at full press (3.1 mm at the foot), I/TM."""
BODY_MASS, BUTTON_MASS, PIN_MASS = 0.024, 0.010, 0.005
"""Button and pins heavier than real (I) for GPU solver stability."""

# Stands (world frame).
PADLOCK_PEDESTAL_XY = (0.265, 0.085)
PADLOCK_PEDESTAL_TOP = padlock_props.PEDESTAL_TOP
LOCKOUT_HOLDER_XY = (0.255, -0.160)
LOCKOUT_HOLDER_TOP = S.RAIL_Z + AXIS_DZ - BODY_R - 0.03
"""The lockout stands on this column installed-way-round (rear toward +x), 3 cm below its working height."""

COLORS = dict(dr.COLORS, black=(0.07, 0.07, 0.08), steel=(0.72, 0.74, 0.77), lock_purple=(0.45, 0.20, 0.60))


# --------------------------------------------------------------------------------------------------
# Derived geometry used by the task and the reference (lockout frame unless said)
# --------------------------------------------------------------------------------------------------

def breaker_y(slot: int) -> float:
    return (slot - (S.NUM_BREAKERS - 1) / 2) * BREAKER.pitch


def face_x_world() -> float:
    return S.BOARD_X - FACE_DEPTH


def lockout_seat_world(slot: int) -> tuple[float, float, float]:
    """The lockout frame's origin when it sits on breaker `slot`."""
    return (face_x_world(), S.ROW_Y + breaker_y(slot), S.RAIL_Z + AXIS_DZ)


def cap_inner_x(button_q: float) -> float:
    """x of the cap's inner face (lockout frame); button_q = 0 released (gap open), GAP pressed home."""
    return -BODY_LEN - GAP + button_q


def hole_center(button_q: float) -> tuple[float, float, float]:
    """Centre of the stem's padlock hole for button position q (lockout frame)."""
    return (cap_inner_x(button_q) + HOLE_FROM_CAP + HOLE / 2, 0.0, 0.0)


def cap_center(button_q: float) -> tuple[float, float, float]:
    return (cap_inner_x(button_q) - CAP_T / 2, 0.0, 0.0)


def pin_pivot(sign: int) -> tuple[float, float, float]:
    """Pivot of the upper (sign +1) or lower (-1) pin, lockout frame."""
    return (PIVOT_X, 0.0, sign * PIN_Z0)


def toggle_pivot_world(slot: int) -> tuple[float, float, float]:
    return (face_x_world() + TOGGLE_PIVOT_IN, S.ROW_Y + breaker_y(slot), S.RAIL_Z)


# --------------------------------------------------------------------------------------------------
# USD helpers
# --------------------------------------------------------------------------------------------------

def _mats(stage, root):
    class M(dict):
        def __missing__(self, key):
            self[key] = _material(stage, root, key)
            return self[key]
    return M()


def _material(stage, root, name):
    from pxr import UsdShade

    path = f"{root}/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[name]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _cyl(stage, path, center, half_len, ry, rz, material, axis="X", collide=False):
    """An (elliptic) cylinder along x, visual by default."""
    from pxr import UsdShade

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(axis)
    cyl.CreateHeightAttr(1.0)
    cyl.CreateRadiusAttr(1.0)
    x = UsdGeom.Xformable(cyl)
    x.AddTranslateOp().Set(Gf.Vec3d(*center))
    x.AddScaleOp().Set(Gf.Vec3f(2 * half_len, ry, rz) if axis == "X" else Gf.Vec3f(ry, rz, 2 * half_len))
    UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    return cyl


def _revolute(stage, path, body0, body1, axis, lo_deg, hi_deg, pos0):
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0.GetPath()])
    joint.CreateBody1Rel().SetTargets([body1.GetPath()])
    joint.CreateAxisAttr(axis)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLowerLimitAttr(lo_deg)
    joint.CreateUpperLimitAttr(hi_deg)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    return joint


# --------------------------------------------------------------------------------------------------
# Props
# --------------------------------------------------------------------------------------------------

def make_board(out_path: Path) -> None:
    stage, _, _ = dr._new_stage(out_path, "/board")
    m = _mats(stage, "/board")
    r = "/board"
    t, w, h = 0.012, 0.30, S.RAIL_Z + 0.09
    dr._box(stage, f"{r}/plate", (S.BOARD_X + t / 2, S.ROW_Y, h / 2), (t, w, h), m["board"])
    dr._boxspan(stage, f"{r}/rail", (S.BOARD_X - 0.0075, S.BOARD_X), (S.ROW_Y - 0.13, S.ROW_Y + 0.13),
                (S.RAIL_Z - 0.0175, S.RAIL_Z + 0.0175), m["rail"], collide=False)
    stage.GetRootLayer().Save()


def make_breaker_row(out_path: Path) -> None:
    """Five S201 housings (fixed-base articulation) with slotted faces and lever toggles."""
    v = BREAKER
    stage, root_prim, _ = dr._new_stage(out_path, "/breakers")
    m = _mats(stage, "/breakers")
    r = "/breakers"
    base = dr._fixed_articulation(stage, root_prim, r)
    b = f"{r}/base"
    hw = v.pitch / 2 - 0.0002
    fd = FACE_DEPTH
    x_face, x_wall, x_cav = -fd, -fd + FACE_WALL, -fd + CAVITY_DEPTH
    fz = v.front_h / 2
    dz = AXIS_DZ

    def solid_z(holes):
        """Complement of the hole z-ranges (window frame) within the front window's height."""
        cuts = sorted(holes)
        out, z0 = [], -fz
        for lo, hi in cuts:
            if lo > z0:
                out.append((z0, lo))
            z0 = max(z0, hi)
        if z0 < fz:
            out.append((z0, fz))
        return out

    slot_holes = [(dz + SLOT_Z[0], dz + SLOT_Z[1]), (dz - SLOT_Z[1], dz - SLOT_Z[0])]
    cav_holes = [(dz + CAVITY_Z[0], dz + CAVITY_Z[1]), (dz - CAVITY_Z[1], dz - CAVITY_Z[0])]
    for k in range(S.NUM_BREAKERS):
        y = breaker_y(k)
        dr._boxspan(stage, f"{b}/rear_{k}", (-v.shoulder, 0.0), (y - hw, y + hw), (-v.height / 2, v.height / 2), m[v.color])
        dr._boxspan(stage, f"{b}/front_back_{k}", (-v.shoulder, x_cav), (y - hw, y + hw), (-fz, fz), m[v.color])
        # Outside the holes' width the front is solid to the face.
        for s, name in ((-1, "r"), (1, "l")):
            dr._boxspan(stage, f"{b}/front_out{name}_{k}", (x_cav, x_face), sorted((y + s * SLOT_HALF_Y, y + s * hw)),
                        (-fz, fz), m[v.color])
        ys = (y - SLOT_HALF_Y, y + SLOT_HALF_Y)
        # The face wall with the two pin holes (its parts between and around the holes are the lips) ...
        for i, zs in enumerate(solid_z(slot_holes)):
            dr._boxspan(stage, f"{b}/wall_{k}_{i}", (x_wall, x_face), ys, zs, m[v.color])
        # ... and behind it the cavities the feet swing into.
        for i, zs in enumerate(solid_z(cav_holes)):
            dr._boxspan(stage, f"{b}/inner_{k}_{i}", (x_cav, x_wall), ys, zs, m[v.color])
        # The holes read dark from the front (visual only, at the back of the cavity).
        for i, (lo, hi) in enumerate(cav_holes):
            dr._boxspan(stage, f"{b}/slot_vis{i}_{k}", (x_cav - 0.0002, x_cav), ys, (lo, hi), m["black"], collide=False)
        # Toggle recess (visual) and the rating window above it.
        dr._boxspan(stage, f"{b}/recess_{k}", (x_face - 0.0003, x_face), (y - 0.0045, y + 0.0045), (-0.011, 0.011),
                    m["window"], collide=False)
        dr._boxspan(stage, f"{b}/window_{k}", (x_face - 0.0004, x_face), (y - 0.005, y + 0.005), (0.013, 0.020),
                    m["window"], collide=False)
        # Colour label on the breaker's top (a self-adhesive circuit label, visual only; the task shows one colour
        # per breaker). Flat on the top so the station's top camera sees its whole area.
        dr._flags(stage, b, m, k, (LABEL_X, y, v.height / 2 + 0.0006), S.BREAKER_COLORS,
                  size=(LABEL_LEN, v.pitch - 0.002, 0.0012))
        pivot = (x_face + TOGGLE_PIVOT_IN, y, 0.0)
        tog = dr._rigid(stage, f"{r}/toggle_{k}", 0.005, pivot)
        dr._boxspan(stage, f"{r}/toggle_{k}/lever", (-TOGGLE_LEN, 0.002), (-TOGGLE_W / 2, TOGGLE_W / 2),
                    (-TOGGLE_T / 2, TOGGLE_T / 2), m["toggle_dark"], collide=False)
        # Collision only outside the face: inside the housing the lever's root would otherwise meet the lockout's pin
        # feet behind the lips (the real mechanism sits deeper than the holes).
        dr._boxspan(stage, f"{r}/toggle_{k}/c_lever", (-TOGGLE_LEN, -TOGGLE_PIVOT_IN - 0.001),
                    (-TOGGLE_W / 2, TOGGLE_W / 2), (-TOGGLE_T / 2, TOGGLE_T / 2), m["toggle_dark"], visible=False)
        _revolute(stage, f"{r}/joints/toggle_{k}_joint", base, tog, "Y", -TOGGLE_LIMIT_DEG, TOGGLE_LIMIT_DEG, pivot)
    stage.GetRootLayer().Save()


def _pin(stage, r, m, name, sign, body, button_joint):
    """One wire hook: a lever pivoted in the nose (revolute about y), its shaft running out of the nose through the
    breaker's hole and its foot pointing inward. Coupled to the button by a PhysX mimic joint (the cam): pressing the
    button swings the pin outward by PIN_SPREAD_DEG over the button's travel."""
    piv = pin_pivot(sign)
    pin = dr._rigid(stage, f"{r}/{name}", PIN_MASS, piv)
    p = f"{r}/{name}"
    ys = (-PIN_HALF_Y, PIN_HALF_Y)
    zs = (-PIN / 2, PIN / 2)
    dr._boxspan(stage, f"{p}/shaft", (0.0, PIN_TIP_X - PIVOT_X), ys, zs, m["steel"])
    inner = -sign * (PIN / 2 + FOOT_LEN)
    dr._boxspan(stage, f"{p}/foot", (FOOT_X[0] - PIVOT_X, FOOT_X[1] - PIVOT_X), ys,
                sorted((-sign * PIN / 2, inner)), m["steel"])
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{r}/joints/{name}_joint")
    j.CreateBody0Rel().SetTargets([body.GetPath()])
    j.CreateBody1Rel().SetTargets([pin.GetPath()])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(*piv))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    # Outward for the upper pin is a negative angle about +y (the shaft along +x swings toward +z).
    lo, hi = (-PIN_SPREAD_DEG - 1.0, 1.0) if sign > 0 else (-1.0, PIN_SPREAD_DEG + 1.0)
    j.CreateLowerLimitAttr(lo)
    j.CreateUpperLimitAttr(hi)
    prim = j.GetPrim()
    # Mimic (the cam): angle + gearing * button_q = 0, gearing in degrees per metre.
    prim.AddAppliedSchema("PhysxMimicJointAPI:rotY")
    prim.CreateRelationship("physxMimicJoint:rotY:referenceJoint").SetTargets([button_joint.GetPath()])
    prim.CreateAttribute("physxMimicJoint:rotY:referenceJointAxis", Sdf.ValueTypeNames.Token).Set("rotX")
    prim.CreateAttribute("physxMimicJoint:rotY:gearing", Sdf.ValueTypeNames.Float).Set(sign * PIN_SPREAD_DEG / GAP)
    prim.CreateAttribute("physxMimicJoint:rotY:offset", Sdf.ValueTypeNames.Float).Set(0.0)
    return pin


def make_lockout(out_path: Path) -> None:
    stage, root_prim, _ = dr._new_stage(out_path, "/lockout")
    m = _mats(stage, "/lockout")
    r = "/lockout"
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    root_prim.AddAppliedSchema("PhysxArticulationAPI")
    root_prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    body = dr._rigid(stage, f"{r}/body", BODY_MASS)
    b = f"{r}/body"
    R, P = BODY_R, -POCKET_DEPTH
    red = m["lockout_red"]
    hid = dict(visible=False)
    dr._boxspan(stage, f"{b}/c_front", (-BODY_LEN, P), (-R, R), (-R, R), red, **hid)
    dr._boxspan(stage, f"{b}/c_flange", (-BODY_LEN, -BODY_LEN + FLANGE_LEN), (-FLANGE_R, FLANGE_R), (-FLANGE_R, FLANGE_R),
                red, **hid)
    dr._boxspan(stage, f"{b}/c_top", (P, 0.0), (-R, R), (POCKET_Z[1], R), red, **hid)
    dr._boxspan(stage, f"{b}/c_bottom", (P, 0.0), (-R, R), (-R, POCKET_Z[0]), red, **hid)
    for k, s in enumerate((-1, 1)):
        dr._boxspan(stage, f"{b}/c_side{k}", (P, 0.0), sorted((s * POCKET_Y, s * R)), POCKET_Z, red, **hid)
    _cyl(stage, f"{b}/v_body", (-(BODY_LEN - FLANGE_LEN + NOSE_LEN) / 2, 0, 0), (BODY_LEN - FLANGE_LEN - NOSE_LEN) / 2,
         R, R, red)
    _cyl(stage, f"{b}/v_flange", (-BODY_LEN + FLANGE_LEN / 2, 0, 0), FLANGE_LEN / 2, FLANGE_R, FLANGE_R, red)
    dr._boxspan(stage, f"{b}/v_nose_t", (-NOSE_LEN, 0.0), (-0.0085, 0.0085), (POCKET_Z[1], 0.0095), m["black"],
                collide=False)
    dr._boxspan(stage, f"{b}/v_nose_b", (-NOSE_LEN, 0.0), (-0.0085, 0.0085), (-0.0095, POCKET_Z[0]), m["black"],
                collide=False)
    for k, s in enumerate((-1, 1)):
        dr._boxspan(stage, f"{b}/v_nose_s{k}", (-NOSE_LEN, 0.0), sorted((s * POCKET_Y, s * 0.0085)), POCKET_Z,
                    m["black"], collide=False)
        dr._boxspan(stage, f"{b}/v_window{k}", (-0.026, -0.014), sorted((s * (R - 0.0002), s * (R + 0.0003))),
                    (-0.002, 0.002), m["black"], collide=False)
    dr._friction(stage, r, [b])

    # Button: black oval cap with the stem; the stem's padlock hole crosses it along y. Built released (q = 0, the
    # cap GAP out from the body); q > 0 presses it toward the body.
    btn = dr._rigid(stage, f"{r}/button", BUTTON_MASS)
    bt = f"{r}/button"
    ci = cap_inner_x(0.0)
    blk = m["black"]
    dr._boxspan(stage, f"{bt}/c_cap", (ci - CAP_T, ci), (-CAP_HALF[0], CAP_HALF[0]), (-CAP_HALF[1], CAP_HALF[1]), blk,
                visible=False)
    _cyl(stage, f"{bt}/v_cap", (ci - CAP_T / 2, 0, 0), CAP_T / 2, CAP_HALF[0], CAP_HALF[1], blk)
    h0, h1 = ci + HOLE_FROM_CAP, ci + HOLE_FROM_CAP + HOLE
    ty, tz, hz = STEM_T / 2, STEM_HALF_Z, HOLE / 2
    dr._boxspan(stage, f"{bt}/stem_strip", (ci, h0), (-ty, ty), (-tz, tz), blk)
    dr._boxspan(stage, f"{bt}/stem_top", (h0, h1), (-ty, ty), (hz, tz), blk)
    dr._boxspan(stage, f"{bt}/stem_bottom", (h0, h1), (-ty, ty), (-tz, -hz), blk)
    dr._boxspan(stage, f"{bt}/stem_rest", (h1, ci + STEM_LEN), (-ty, ty), (-tz, tz), blk)
    joint = UsdPhysics.PrismaticJoint.Define(stage, f"{r}/joints/button_joint")
    joint.CreateBody0Rel().SetTargets([body.GetPath()])
    joint.CreateBody1Rel().SetTargets([btn.GetPath()])
    joint.CreateAxisAttr("X")
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(GAP)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")

    for name, sign in (("pin_top", 1), ("pin_bottom", -1)):
        _pin(stage, r, m, name, sign, body, joint)
    stage.GetRootLayer().Save()


def make_padlock(out_path: Path) -> None:
    """The Master Lock 410 in its real shape (round shackle and bend, rounded body, label): usd/padlock_410.py."""
    padlock_410.make_padlock_410(out_path)


def make_stands(out_path: Path) -> None:
    stage, _, _ = dr._new_stage(out_path, "/stands")
    m = _mats(stage, "/stands")
    r = "/stands"
    x, y = PADLOCK_PEDESTAL_XY
    dr._box(stage, f"{r}/padlock_foot", (x, y, 0.005), (0.06, 0.05, 0.01), m["stand"])
    dr._box(stage, f"{r}/padlock_column", (x, y, PADLOCK_PEDESTAL_TOP / 2), (0.030, 0.009, PADLOCK_PEDESTAL_TOP),
            m["stand"])
    x, y = LOCKOUT_HOLDER_XY
    dr._box(stage, f"{r}/lockout_foot", (x, y, 0.005), (0.06, 0.05, 0.01), m["stand"])
    dr._box(stage, f"{r}/lockout_column", (x, y, LOCKOUT_HOLDER_TOP / 2), (0.014, 0.008, LOCKOUT_HOLDER_TOP),
            m["stand"])
    stage.GetRootLayer().Save()


def lockout_start_world() -> tuple[float, ...]:
    """The lockout frame standing on its holder (the body's centre over the column)."""
    x, y = LOCKOUT_HOLDER_XY
    return (x + BODY_LEN / 2, y, LOCKOUT_HOLDER_TOP + BODY_R + 0.0005, 0.0, 0.0, 0.0, 1.0)


def padlock_start_world() -> tuple[float, ...]:
    v = padlock_props.PADLOCKS["ml410"]
    x, y = PADLOCK_PEDESTAL_XY
    return (x, y, PADLOCK_PEDESTAL_TOP + v.body[2] / 2 + 0.001, 0.0, 0.0, 0.0, 1.0)


def toggle_tip_offset(angle_rad: float) -> tuple[float, float]:
    """(dx, dz) of the lever tip's centre from its pivot."""
    return (-TOGGLE_LEN * math.cos(angle_rad), TOGGLE_LEN * math.sin(angle_rad))


def _robust(out_path: Path, root: str) -> None:
    """GPU PhysX let the 1.5-2 mm pins and the shackle pass through thin parts under the 20 N test with default
    settings: more solver iterations on the articulation, and a small explicit contact offset on every collider."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(out_path))
    rp = stage.GetPrimAtPath(root)
    if not rp.HasAPI(UsdPhysics.ArticulationRootAPI):
        rp.AddAppliedSchema("PhysxArticulationAPI")
    rp.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(64)
    rp.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(8)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.002)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    stage.GetRootLayer().Save()


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    make_board(out_dir / "board.usda")
    make_breaker_row(out_dir / "breaker_row.usda")
    make_lockout(out_dir / "lockout.usda")
    make_padlock(out_dir / "padlock_ml410.usda")
    make_stands(out_dir / "stands.usda")
    for name, root in (("breaker_row", "/breakers"), ("lockout", "/lockout"), ("padlock_ml410", "/padlock")):
        _robust(out_dir / f"{name}.usda", root)
    print(f"wrote breaker-lockout props to {out_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

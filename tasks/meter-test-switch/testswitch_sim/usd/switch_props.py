"""Generate the props as USD: a 10-pole ANSI C12.9 meter test switch on a meter-box back wall, the TESCO 1077-class
current test plug on its lead to a test-set box, and the plug's rest.

Switch (``switch.usda``), a fixed-base articulation. Switch frame = desk axes shifted to the centre of the base's front
face at (FACE_X, 0, CENTER_Z): parts stand out toward -x (the robot), poles run along z (hinge at the bottom), the row
of ten poles along y. Each pole's knife blade turns on a revolute joint about y at its hinge: closed (q = 0) it lies
along the pole, edge-on, held at its top end in a spring jaw; opening (q < 0) swings the handle out toward the robot
and down, until the blade hangs below the base (160 deg), as open blades do in TESCO's photos.

The test jack (ANSI C12.9 Fig. 4) is a spring clip at mid-pole on each test-jack pole, its slot opening outward. The
closed test-jack blade lies in the bottom of the slot, so the plug's probe cannot go in to its full depth until that
blade is open (our reading of the Milbank side view and C12.9 2.3; to confirm on a real switch). The jack's two leaves
are sprung (prismatic joints with preloaded spring drives set by the task): they rest 2.8 mm apart and pinch the
3.3 mm probe. Each leaf has the C12.9 Fig. 4 notch (W) and the probe the Fig. 6 guide rib (R, G), which rides in it
and keeps a released plug from pivoting out of the jack.

Lead (``lead.usda``), a second fixed-base articulation: the test-set box (fixed to the desk), LEAD_N capsule links
(a 2-conductor test lead; revolute joints cycling about the two bending axes and the twist axis, each with a weak
spring toward the routed shape, set by the task) and the plug, whose handle's rear end the lead enters. The lead's
starting route (out of the box's side, along the desk, up to the plug on its rest) is solved here (``solve_route``)
and is the joints' zero.

Sources:
- ANSI C12.9 Figs. 1-4, 6 and Tables 1, 3 as reproduced in TESCO's RMEMA 2025 deck
  (https://www.tescometering.com/wp-content/uploads/2025/06/RMEMA-2025-Test-Switch-Presentation.pdf):
  pole pitch 0.750 in min; terminal rows 2.50 in apart; jack slot width W 0.140 in, jaw width X 0.500 in,
  slot depth S 1.200 in, jaw height D 1.350 in; plug blade length L 1.200 in, thickness T 0.130 in.
- TESCO drawing 909884 (same deck): base 9.500 x 3.50 in, handle colours (spec.POLES).
- Milbank TS10-0109 spec-sheet drawing: base 3/4 in thick, 2 3/4 in overall depth closed. Measured off the drawing
  (+-0.1-0.2 in, "not to scale"): hinge 0.55 in above the lower terminal and 0.8 in out from the face; hinge-to-jaw
  1.4 in; blade edge-on 0.55-1.0 in out; handle 0.5 in wide, 1.0 in along the pole, 1.0-2.15 in out; jack at
  mid-pole; barriers 2.0 in out, 4.0 in long, ~2 mm thick, both sides of the voltage poles.
Inferred (to measure): blade 3.0 mm thick; plug handle 90 x 22 x 14 mm, guard, probe 11 mm wide, 60 g; the jack
leaves' spring; the lead's stiffness and weight.

Usage: python -m testswitch_sim.usd.switch_props [<out dir>]   (default: testswitch_sim/assets)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

from testswitch_sim.spec import CENTER_Z, FACE_X, PLUG_REST, POLES, TEST_SET, TEST_SET_SIZE
from testswitch_sim.usd import usd_kit as kit

IN = 0.0254
N_POLES = len(POLES)
PITCH = 0.750 * IN
BASE = (0.75 * IN, 9.500 * IN, 3.50 * IN)
"""Thickness (x), length along the row (y), width along the poles (z)."""
TERM_A = 1.25 * IN
"""Terminal rows at z = -/+ 31.75 mm from the centre (2.50 in apart)."""
HINGE_A = -TERM_A + 0.55 * IN
HINGE_O = 0.80 * IN
"""Hinge axis: 17.8 mm below centre, 20.3 mm out from the face."""
JAW_A = HINGE_A + 1.40 * IN
BLADE_T = 0.0030
BLADE_O = (0.55 * IN, 1.00 * IN)
"""Blade's edge-on extent, out from the face."""
BLADE_A = (HINGE_A - 0.003, JAW_A + 0.004)
HANDLE_A = (JAW_A - 0.45 * IN, JAW_A + 0.55 * IN)
HANDLE_O = (1.00 * IN, 2.15 * IN)
HANDLE_W = 0.50 * IN
JAW_LEAF_T = 0.0012
JAW_O = BLADE_O[1] + 0.001
JACK_A = -0.10 * IN
JACK_X = 0.500 * IN
"""Jack leaves' length along the pole (jaw width X)."""
JACK_W = 0.175 * IN
"""Notch in each leaf for the plug's guide rib (C12.9 Table 1, W max; 4.4 mm vs a 2.3 mm rib)."""
JACK_D = 1.350 * IN
"""Jack leaves stand out this far (jaw height D)."""
JACK_GAP = 0.0042
"""Tolerance of the probe-in-slot check (the leaves themselves are sprung)."""
JACK_REST_GAP = 0.0028
"""Sprung leaves at rest: 2.8 mm apart, so the 3.3 mm probe spreads each by 0.25 mm (C12.9: at most 0.188 in)."""
LEAF_TRAVEL = 0.003
CONTACT_MU = 1.0
"""Friction between the probe and the jack's leaves (inferred; 1.0 so a ~3 N pinch holds the released plug)."""
LIP_MU = 0.3
"""The flared lips at the jack's mouth (a polished lead-in, inferred)."""
LEAF_MASS = 0.005
OPEN_DEG = 160.0
BARRIER_T, BARRIER_O, BARRIER_LEN = 0.002, 2.0 * IN, 4.0 * IN
BARRIERS_AT = (1.5, 2.5, 4.5, 5.5, 7.5, 8.5)
"""Barrier positions in pole units counted from pole 1 (index 0): both sides of the voltage poles 3, 6, 9."""
BLADE_MASS = 0.03
WALL = (0.006, 0.40, 0.30)

PROBE = (0.011, 0.130 * IN, 1.200 * IN)
"""Plug probe (plug frame): width (x), thickness (y), length (z, tip at the origin)."""
RIB_R = 0.090 * IN
"""Guide rib width (C12.9 Table 3, R min)."""
RIB_G = 0.125 * IN
"""Guide rib height above the probe face (G preferred)."""
RIB_Z0 = 0.003
"""The rib starts 3 mm behind the probe's tip, so the flat probe enters the leaves first."""
PLUG_GUARD = (0.024, 0.014, 0.004)
"""A flange 1 mm proud of the handle above and below (not sideways: the plug goes in between barriers)."""
PLUG_HANDLE = (0.022, 0.014, 0.090)
PLUG_MASS = 0.06
POST = (0.030, 0.008)
TABLE_Z = -0.0025
"""The station table's top in the desk frame."""

# Test lead (inferred) -------------------------------------------------------------------------------------------
LEAD_LEN = 0.60
"""Free length between the test set and the plug (a real lead is longer; the rest stays coiled at the test set)."""
LEAD_N = 24
LEAD_LINK = LEAD_LEN / LEAD_N
LEAD_R = 0.003
"""A 2-conductor PVC test lead about 6 mm across."""
LEAD_LINK_MASS = 0.002
"""Real: about 50 g/m (1.25 g per 25 mm link); a little heavier keeps the chain well-behaved in PhysX."""
LEAD_LINK_INERTIA = (2e-8, 2e-7, 2e-7)
LEAD_EI = 3e-4
"""Bending stiffness, N*m^2 (inferred for a 6 mm 2-conductor flexible PVC lead)."""
LEAD_K = LEAD_EI / LEAD_LINK
LEAD_DAMPING = 0.002
LEAD_LIMIT_DEG = 100.0
LEAD_FRICTION = (0.6, 0.5)
AXES = ("Y", "Z", "X")
"""Joint axes cycle: the two bending axes, then twist."""
ANCHOR = (TEST_SET[0] - 0.04, TEST_SET[1] - TEST_SET_SIZE[1] / 2 - 0.002, 0.025)
"""Where the lead leaves the test set: its side facing the switch (-y), 2.5 cm up."""
SOLVER_ITERATIONS = 32

COLORS = {"brown": (0.28, 0.13, 0.05), "orange": (0.90, 0.22, 0.02), "phenolic": (0.04, 0.04, 0.045), "tin": (0.80, 0.80, 0.78),
          "label": (0.92, 0.92, 0.88), "box_grey": (0.62, 0.64, 0.66), "lead": (0.08, 0.08, 0.08),
          "test_set": (0.85, 0.72, 0.10), "panel": (0.15, 0.15, 0.16)}
kit.COLORS.update(COLORS)


def pole_y(i: int) -> float:
    """Pole i (0-based, pole 1 at +y)."""
    return ((N_POLES - 1) / 2 - i) * PITCH


def to_world(o: float, y: float, a: float) -> tuple[float, float, float]:
    """(out from the face, row y, along the pole) -> desk frame."""
    return (FACE_X - o, y, CENTER_Z + a)


def hinge_world(i: int) -> tuple[float, float, float]:
    return to_world(HINGE_O, pole_y(i), HINGE_A)


def jack_mouth_world(i: int) -> tuple[float, float, float]:
    """Centre of pole i's jack slot at its mouth (the leaves' outer edge)."""
    return to_world(JACK_D, pole_y(i), JACK_A)


def _box(stage, path, o_range, y, y_size, a_range, mat, collide=True):
    """A box given by its out range, row centre and size, and along range (switch frame)."""
    o0, o1 = o_range
    a0, a1 = a_range
    return kit.box(stage, path, (-(o0 + o1) / 2, y, (a0 + a1) / 2), (o1 - o0, y_size, a1 - a0), mat, collide)


def _contacts(stage, root_prim, prefix, iters=16):
    if root_prim is not None:
        root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(iters)
        root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.001)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def make_switch(out_path: Path) -> None:
    root = "/switch"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim)
    names = ("phenolic", "tin", "label", "dark", "box_grey", "white", "black", "red", "orange", "yellow", "green",
             "blue", "brown")
    m = kit.materials(stage, root, names)
    base = kit.rigid_body(stage, f"{root}/base", (0, 0, 0), 2.0)
    kit.fix_to_world(stage, root, base.GetPath())
    b = f"{root}/base"
    # Meter-box back wall down to the desk, and the switch base.
    kit.box(stage, f"{b}/wall", (BASE[0] + WALL[0] / 2, 0, -CENTER_Z + WALL[2] / 2), WALL, m["box_grey"])
    kit.box(stage, f"{b}/base", (BASE[0] / 2, 0, 0), BASE, m["phenolic"])
    for i, (func, colour) in enumerate(POLES):
        y = pole_y(i)
        for k, a in enumerate((-TERM_A, TERM_A)):  # terminal studs and nuts (visual)
            _box(stage, f"{b}/term{i}_{k}", (0, 0.006), y, 0.011, (a - 0.005, a + 0.005), m["tin"], collide=False)
            kit.cylinder(stage, f"{b}/stud{i}_{k}", (-0.012, y, a), 0.002, 0.012, m["tin"], axis="X")
        for s in (-1, 1):  # hinge clip and blade jaw leaves either side of the blade
            yl = y + s * (BLADE_T / 2 + 0.0004 + JAW_LEAF_T / 2)
            _box(stage, f"{b}/hinge{i}_{'lr'[s > 0]}", (0, HINGE_O + 0.004), yl, JAW_LEAF_T,
                 (HINGE_A - 0.004, HINGE_A + 0.004), m["tin"])
            _box(stage, f"{b}/jaw{i}_{'lr'[s > 0]}", (0, JAW_O), yl, JAW_LEAF_T, (JAW_A - 0.005, JAW_A + 0.005), m["tin"])
        if func == "short":  # shorting strap to the paired test-jack pole (visual)
            _box(stage, f"{b}/strap{i}", (0.006, 0.008), y - PITCH / 2, PITCH, (-TERM_A + 0.008, -TERM_A + 0.012),
                 m["tin"], collide=False)
    for k, u in enumerate(BARRIERS_AT):
        _box(stage, f"{b}/barrier{k}", (0, BARRIER_O), pole_y(0) - u * PITCH, BARRIER_T,
             (-BARRIER_LEN / 2, BARRIER_LEN / 2), m["phenolic"])
    _box(stage, f"{b}/label", (0, 0.0005), 0, BASE[1] - 0.02, (BASE[2] / 2 - 0.009, BASE[2] / 2 - 0.003), m["label"],
         collide=False)
    # Blades with handles; body origin at the hinge.
    for i, (func, colour) in enumerate(POLES):
        y = pole_y(i)
        hinge = (-HINGE_O, y, HINGE_A)
        blade = kit.rigid_body(stage, f"{root}/blade{i}", hinge, BLADE_MASS)
        p = blade.GetPath()

        def rel(o_range, a_range):
            return (o_range[0] - HINGE_O, o_range[1] - HINGE_O), (a_range[0] - HINGE_A, a_range[1] - HINGE_A)

        o_r, a_r = rel(BLADE_O, BLADE_A)
        _box(stage, f"{p}/blade", o_r, 0, BLADE_T, a_r, m["tin"])
        o_r, a_r = rel(HANDLE_O, HANDLE_A)
        _box(stage, f"{p}/handle", o_r, 0, HANDLE_W, a_r, m[colour])
        for r in range(3):  # grip ribs on the handle's outer face (visual)
            _box(stage, f"{p}/rib{r}", (o_r[1], o_r[1] + 0.0008), 0, HANDLE_W * 0.9,
                 (a_r[0] + 0.006 + 0.006 * r, a_r[0] + 0.0075 + 0.006 * r), m["dark"], collide=False)
        kit.revolute(stage, f"{root}/joints/pole{i}", base.GetPath(), p, "Y", hinge, -OPEN_DEG, 0.0)
    # The test jacks: two sprung leaves per jack, each a body on a prismatic joint across the slot (y). At rest
    # each leaf sits on its inner stop (gap JACK_REST_GAP), preloaded by a spring drive the task sets; the probe
    # pushes them apart and they pinch it. A flared lip at the mouth guides the probe in.
    for i, (func, colour) in enumerate(POLES):
        if func != "jack":
            continue
        y = pole_y(i)
        for s in (-1, 1):
            side = "lr"[s > 0]
            yl = y + s * (JACK_REST_GAP / 2 + JAW_LEAF_T / 2)
            leaf = kit.rigid_body(stage, f"{root}/leaf{i}{side}", (0.0, yl, 0.0), LEAF_MASS)
            lp = leaf.GetPath()
            # Each leaf has the C12.9 Fig. 4 notch (width W) down its middle; the plug's guide rib rides in it.
            half = (JACK_X - JACK_W) / 2
            for part, a0 in (("lo", JACK_A - JACK_X / 2), ("hi", JACK_A + JACK_W / 2)):
                _box(stage, f"{lp}/leaf_{part}", (0, JACK_D), 0.0, JAW_LEAF_T, (a0, a0 + half), m["tin"])
                lip = kit.box(stage, f"{lp}/lip_{part}", (0, 0, 0), (1, 1, 1), m["tin"])
                UsdGeom.Xformable(lip).ClearXformOpOrder()
                xf = UsdGeom.Xformable(lip)
                xf.AddTranslateOp().Set(Gf.Vec3d(-(JACK_D + 0.0030), s * 0.0004, a0 + half / 2))
                xf.AddRotateZOp().Set(-s * 20.0)
                xf.AddScaleOp().Set(Gf.Vec3f(0.0070, JAW_LEAF_T, half))
            lower, upper = (0.0, LEAF_TRAVEL) if s > 0 else (-LEAF_TRAVEL, 0.0)
            kit.prismatic(stage, f"{root}/joints/leaf{i}{side}", base.GetPath(), lp, "Y", (0.0, yl, 0.0), lower, upper)
    leaf_paths = [str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith(f"{root}/leaf")
                  and p.HasAPI(UsdPhysics.CollisionAPI)]
    kit.friction(stage, root, [q for q in leaf_paths if "/leaf_" in q], static=CONTACT_MU, dynamic=CONTACT_MU,
                 name="tin_contact", combine="average")
    kit.friction(stage, root, [q for q in leaf_paths if "/lip_" in q], static=LIP_MU, dynamic=LIP_MU,
                 name="tin_lip", combine="min")
    _contacts(stage, root_prim, root)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


# Plug and lead ---------------------------------------------------------------------------------------------------
PLUG_QUAT = (0.0, -0.7071068, 0.0, 0.7071068)
"""The plug's probe (local +z points tip -> handle) along world -x, its width (local x) vertical."""
PLUG_TIP_TO_HANDLE_C = PROBE[2] + PLUG_GUARD[2] + PLUG_HANDLE[2] / 2
PLUG_TAIL = PROBE[2] + PLUG_GUARD[2] + PLUG_HANDLE[2]
"""Probe tip to the handle's rear end, where the lead goes in."""


def plug_start_pose() -> tuple:
    """Plug (probe tip) pose with the handle's centre at PLUG_REST."""
    x, y, z = PLUG_REST
    return (x + PLUG_TIP_TO_HANDLE_C, y, z + 0.0005) + PLUG_QUAT


def _rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == "Z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _qmat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _mat_q(m):
    from scipy.spatial.transform import Rotation

    return tuple(float(v) for v in Rotation.from_matrix(m).as_quat())


ANCHOR_ROT = _rot("Z", -math.pi / 2)
"""The lead leaves the test set's side along -y."""
PLUG_IN_F = _rot("Y", math.pi / 2)
"""The lead's end frame F (x along the lead, into the plug) in plug axes: F's x is the plug's -z (toward the tip)."""


def joint_axis(k: int) -> str:
    return AXES[k % len(AXES)]


def lead_fk(q):
    """Frames (pos, rot) of the LEAD_N links, and the plug's (tip pos, rot), for joint angles q (LEAD_N + 1)."""
    p, r = np.asarray(ANCHOR, float), ANCHOR_ROT.copy()
    links = []
    for k in range(LEAD_N):
        r = r @ _rot(joint_axis(k), q[k])
        links.append((p.copy(), r.copy()))
        p = p + r @ np.array([LEAD_LINK, 0, 0])
    f = r @ _rot(joint_axis(LEAD_N), q[LEAD_N])
    plug_r = f @ PLUG_IN_F.T
    return links, (p + f @ np.array([PLUG_TAIL, 0, 0]), plug_r), p


def solve_route(verbose=True):
    """Joint angles that route the lead from the test set along the desk and up to the plug's rear end, the plug on
    its rest. The lead lies low and keeps clear of the rest's column, the hand's path to the plug and arm A's base."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    start = plug_start_pose()
    sp, sr = np.asarray(start[:3]), _qmat(start[3:])
    handle_c = sp + sr @ np.array([0, 0, PLUG_TIP_TO_HANDLE_C])
    grasp = handle_c + sr @ np.array([0, 0, 0.022])
    floor = TABLE_Z + LEAD_R + 0.0003
    box_lo = np.array(TEST_SET) - np.array(TEST_SET_SIZE) / 2
    box_hi = np.array(TEST_SET) + np.array(TEST_SET_SIZE) / 2

    def points(links):
        pts = []
        for p, r in links:
            pts += [p + r @ np.array([f * LEAD_LINK, 0, 0]) for f in (0.5, 1.0)]
        return pts

    def residual(q):
        links, (pos, rot), _ = lead_fk(q)
        err_p = (pos - sp) * 300.0
        err_r = Rotation.from_matrix(sr.T @ rot).as_rotvec() * 30.0
        pts = points(links)
        pen = [max(0.0, floor - e[2]) * 300.0 for e in pts]
        pen += [1.5 * (e[2] - floor) for e in pts[:-6]]  # lie low
        # Off the rest's column (2 x 2 cm), out of a 3.5 cm column above the grasp, off arm A's base, out of the box.
        for half, top in ((0.025 + LEAD_R + 0.003, 0.008 + LEAD_R + 0.002),  # the rest's foot
                          (0.012 + LEAD_R + 0.003, PLUG_REST[2] - 0.008)):  # its column and channel
            pen += [max(0.0, half - max(abs(e[0] - PLUG_REST[0]), abs(e[1] - PLUG_REST[1]))) * 300.0
                    * (e[2] < top) for e in pts[:-1]]
        pen += [max(0.0, 0.035 - math.hypot(e[0] - grasp[0], e[1] - grasp[1])) * 150.0 * (e[2] > 0.09)
                for e in pts[:-3]]
        pen += [max(0.0, 0.16 - math.hypot(e[0], e[1] - 0.305)) * 200.0 for e in pts]
        pen += [max(0.0, min(e[0] - box_lo[0], box_hi[0] - e[0], e[1] - box_lo[1] + LEAD_R)) * 200.0
                * (e[2] < box_hi[2]) for e in pts]
        pen += [max(0.0, 0.03 - e[0]) * 200.0 for e in pts]
        return np.concatenate([err_p, err_r, 0.25 * np.asarray(q), pen])

    lim = math.radians(LEAD_LIMIT_DEG) * 0.8
    best = None
    for trial in range(12):
        q0 = np.random.default_rng(trial).uniform(-0.5, 0.5, LEAD_N + 1)
        sol = least_squares(residual, q0, bounds=(-lim, lim))
        if best is None or sol.cost < best.cost:
            best = sol
    links, (pos, rot), _ = lead_fk(best.x)
    if verbose:
        pts = points(links)
        print(f"route: plug pos error {np.linalg.norm(pos - sp) * 1000:.2f} mm, rot error "
              f"{np.degrees(np.linalg.norm(Rotation.from_matrix(sr.T @ rot).as_rotvec())):.2f} deg, lowest "
              f"{min(e[2] for e in pts) * 1000:.1f} mm, x {min(e[0] for e in pts):.3f}..{max(e[0] for e in pts):.3f}, "
              f"y {min(e[1] for e in pts):.3f}..{max(e[1] for e in pts):.3f}, max |q| "
              f"{np.degrees(np.abs(best.x).max()):.0f} deg, cost {best.cost:.4f}")
    return best.x


def _body(stage, path, pos, rot, mass, inertia=None):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    x, y, z, w = _mat_q(rot)
    xform.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
    mass_api.CreateMassAttr(mass)
    if inertia is not None:
        com, diag = inertia
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*diag))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
    return xform


def _capsule_x(stage, path, length, radius, material):
    cap = UsdGeom.Capsule.Define(stage, path)
    cap.CreateAxisAttr("X")
    cap.CreateRadiusAttr(radius)
    cap.CreateHeightAttr(float(length))
    UsdGeom.Xformable(cap).AddTranslateOp().Set(Gf.Vec3d(length / 2, 0, 0))
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    return cap


def _joint_between(stage, path, body0, frame0, body1, frame1, world_point, world_rot, axis):
    """A revolute joint at ``world_point`` whose frame is ``world_rot`` in both bodies' authored poses (angle 0)."""
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(str(body0))])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(str(body1))])
    joint.CreateAxisAttr(axis)
    for (p, r), pos_attr, rot_attr in ((frame0, joint.CreateLocalPos0Attr, joint.CreateLocalRot0Attr),
                                       (frame1, joint.CreateLocalPos1Attr, joint.CreateLocalRot1Attr)):
        pos_attr(Gf.Vec3f(*[float(v) for v in r.T @ (np.asarray(world_point) - p)]))
        x, y, z, w = _mat_q(r.T @ world_rot)
        rot_attr(Gf.Quatf(float(w), float(x), float(y), float(z)))
    joint.CreateLowerLimitAttr(-LEAD_LIMIT_DEG)
    joint.CreateUpperLimitAttr(LEAD_LIMIT_DEG)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    return joint


def make_lead(out_path: Path, q=None) -> None:
    """The test-set box (fixed), the lead's links and the plug: one fixed-base articulation whose authored pose (all
    joints at 0) is the solved route with the plug on its rest."""
    root = "/lead"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=False)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
        SOLVER_ITERATIONS)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    m = kit.materials(stage, root, ("test_set", "panel", "lead", "red", "black", "copper", "white"))
    q = solve_route() if q is None else q
    links, (plug_p, plug_r), attach = lead_fk(q)

    # The root body sits at the asset's origin with no rotation (Isaac Lab writes the articulation's default root
    # pose to it on reset); it carries the test-set box.
    box = _body(stage, f"{root}/test_set", (0.0, 0.0, 0.0), np.eye(3), 2.0)
    kit.fix_to_world(stage, root, box.GetPath())
    cx, cy, _ = TEST_SET
    sx, sy, sz = TEST_SET_SIZE
    kit.box(stage, f"{root}/test_set/case", (cx, cy, TABLE_Z + sz / 2), (sx, sy, sz), m["test_set"])
    kit.box(stage, f"{root}/test_set/panel", (cx, cy, TABLE_Z + sz + 0.0005), (sx - 0.02, sy - 0.02, 0.001),
            m["panel"], collide=False)
    kit.cylinder(stage, f"{root}/test_set/socket", (ANCHOR[0], ANCHOR[1] + 0.004, ANCHOR[2]), 0.006, 0.008,
                 m["black"], axis="Y")
    prev, prev_frame = box.GetPath(), (np.zeros(3), np.eye(3))
    for k, (p, r) in enumerate(links):
        body = _body(stage, f"{root}/lead_{k:02d}", p, r, LEAD_LINK_MASS, ((LEAD_LINK / 2, 0, 0), LEAD_LINK_INERTIA))
        _capsule_x(stage, f"{root}/lead_{k:02d}/jacket", LEAD_LINK, LEAD_R, m["lead"])
        _joint_between(stage, f"{root}/joints/lead_j{k:02d}", prev, prev_frame, body.GetPath(), (p, r), p, r,
                       joint_axis(k))
        prev, prev_frame = body.GetPath(), (p, r)
    # The plug: origin at the probe's tip, the probe running along its +z into the guard and handle.
    plug = _body(stage, f"{root}/plug", plug_p, plug_r, PLUG_MASS)
    f_rot = plug_r @ PLUG_IN_F
    _joint_between(stage, f"{root}/joints/lead_j{LEAD_N:02d}", prev, prev_frame, plug.GetPath(), (plug_p, plug_r),
                   attach, f_rot, joint_axis(LEAD_N))
    pp = f"{root}/plug"
    kit.box(stage, f"{pp}/probe", (0, 0, PROBE[2] / 2), PROBE, m["copper"])
    # C12.9 Fig. 6 insertion guide: a rib R wide standing G proud of both faces, along the probe from just behind
    # its tip; it rides in the jack leaves' notch (W) and keeps the plug from pivoting in the jack's plane.
    kit.box(stage, f"{pp}/rib", (0, 0, (RIB_Z0 + PROBE[2]) / 2), (RIB_R, PROBE[1] + 2 * RIB_G, PROBE[2] - RIB_Z0),
            m["copper"])
    kit.box(stage, f"{pp}/insulation", (0, 0, PROBE[2] / 2 + 0.002),
            (0.0006, PROBE[1] + 0.0004, PROBE[2] - 0.004), m["black"], collide=False)
    kit.box(stage, f"{pp}/guard", (0, 0, PROBE[2] + PLUG_GUARD[2] / 2), PLUG_GUARD, m["red"])
    kit.box(stage, f"{pp}/handle", (0, 0, PROBE[2] + PLUG_GUARD[2] + PLUG_HANDLE[2] / 2), PLUG_HANDLE, m["red"])
    kit.cylinder(stage, f"{pp}/boot", (0, 0, PLUG_TAIL + 0.004), LEAD_R + 0.0015, 0.008, m["black"], axis="Z")
    kit.friction(stage, root, [f"{pp}/handle"], static=1.2, dynamic=1.0)
    kit.friction(stage, root, [f"{pp}/probe", f"{pp}/rib"], static=CONTACT_MU, dynamic=CONTACT_MU, name="brass",
                 combine="average")
    kit.friction(stage, root, [f"{root}/lead_{k:02d}/jacket" for k in range(LEAD_N)], static=LEAD_FRICTION[0],
                 dynamic=LEAD_FRICTION[1], name="lead_pvc", combine="min")
    plug.GetPrim().AddAppliedSchema("PhysxRigidBodyAPI")
    plug.GetPrim().CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(2.0)
    plug.GetPrim().CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float).Set(0.5)
    _contacts(stage, None, root)
    stage.GetRootLayer().Save()
    np.save(out_path.with_name("lead_route_q.npy"), np.asarray(q))
    print(f"wrote {out_path}")


def make_post(out_path: Path) -> None:
    """A rest for the plug: a shallow U-channel (2 mm walls) along its handle, on a column; the fingers take the
    handle's upper part above the walls."""
    root = "/post"
    stage, _ = kit.new_stage(out_path, root)
    m = kit.materials(stage, root, ("stand",))
    x, y, z = PLUG_REST
    bottom = z - PLUG_HANDLE[0] / 2
    x0, x1 = x - PLUG_HANDLE[2] / 2 + 0.002, x + PLUG_HANDLE[2] / 2 - 0.003
    xc, lx = (x0 + x1) / 2, x1 - x0
    inner, wall_t, wall_h = PLUG_HANDLE[1] + 0.003, 0.004, 0.002
    kit.box(stage, f"{root}/column", (xc, y, bottom / 2), (0.020, POST[1] + 0.012, bottom), m["stand"])
    kit.box(stage, f"{root}/floor", (xc, y, bottom - 0.002), (lx, inner + 2 * wall_t, 0.004), m["stand"])
    for s in (-1, 1):
        kit.box(stage, f"{root}/wall{'lr'[s > 0]}", (xc, y + s * (inner / 2 + wall_t / 2), bottom + wall_h / 2 - 0.004),
                (lx, wall_t, wall_h + 0.004), m["stand"])
    kit.box(stage, f"{root}/foot", (xc, y, 0.004), (0.05, 0.05, 0.008), m["stand"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    make_switch(out_dir / "switch.usda")
    make_lead(out_dir / "lead.usda")
    make_post(out_dir / "post.usda")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets")

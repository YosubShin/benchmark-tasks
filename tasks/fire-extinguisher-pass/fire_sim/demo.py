"""The scripted demo, closed loop on robot-observable inputs only.

What it reads, every control step: the observation Inspect Robots hands any policy, i.e. the three RGB images
(``top_cam``, ``left_cam``, ``right_cam``) and the 14-D ``joint_pos`` state. Nothing else comes from the simulator:
no object poses, no lever angle, no seal or pin state, no spray or knockdown state, no scorer state. Fixed knowledge
it uses: the documented setup (task.json ``setup``: where the extinguisher and the fire stand are, the extinguisher's
dimensions), the station's published camera calibration and the arm's URDF (``yam_arena.station``,
``yam_arena.urdf``, ``yam_arena.ik``), as a real robot's script would.

How it perceives each thing:
- Where the extinguisher stands: the red paint in the first top-camera frame, back-projected (fire_sim.vision);
  the offset from the documented spot (clipped to 3 cm) shifts every grasp target.
- The squeeze lever: after the right hand closes on the carry handle and lever, the gripper's own opening (joint
  state) is the stack's height with the lever held up by the pin; "relaxed" is that opening, "squeezed" is fully
  closed. The valve is never read: the hand squeezes and the top camera shows the plume and the flames.
- How the hanging extinguisher points: it hangs from the hand by its carry handle and tips nose-down until its centre
  of mass is under the grip; the demo predicts that from the documented mass layout, and turns the hand 12 deg about
  the vertical (development runs: the unit turns about that much in the hand as its base leaves the table).
- Where the hand is: forward kinematics of the measured joints; an integral term adds the arm's sag under the 1 kg
  unit (position and wrist orientation) to the command.
- Where the spray lands, through its effect: bright orange pixels in each flame tongue's window of the top-camera
  image (the documented flame volume projected with the calibration). The aim starts below the pan's rim and walks up
  until the tongue being aimed at starts to shrink (the base), then goes 1 cm further up; if a neighbouring tongue
  shrinks instead, the hand is turned about the vertical toward the intended one. A tongue whose window has gone dark
  is out; the lever stays squeezed while the aim sweeps from tongue to tongue.

Joint targets come from yam_arena's own IK (``yam_arena.ik``). ``--policy extinguisher-demo`` (fire_sim.policy)
plays it; ``-P aim=tips`` aims at the flames' tips instead (the control run).
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import ik, station
from yam_arena.ik import matrix_to_quat_xyzw

from fire_sim import geometry as G
from fire_sim import spec as S
from fire_sim import vision as V

MM = 0.001
WORLD_Z = V.DESK_Z
TABLE_DZ = -0.0025
"""The station table is 2.5 mm below the desk frame's z = 0 (task.json setup gives heights from the table top)."""
UNIT_CAM_BIAS = np.array([0.0182, 0.0186])
"""Where locate_unit puts the documented unit (its red pixels' centroid on the z = 0.20 plane vs the axis), measured
once from the first frame: subtracted so that only a real offset from the layout moves the targets."""
HANG_DEG = None
"""Predicted from the mass layout (see hang_deg)."""
AIM_HEIGHT = {"base": 0.02, "tips": 0.09}
OUT_PX = 1
"""A tongue is out when its window shows no flame pixels (at most 1, for noise): a nearly knocked-down tongue still
shows its tip just above the rim."""
AIM_DZ_MIN = -0.05
"""The hand never goes lower than 5 cm under the predicted aiming height (the hanging unit's base is then about
3 cm above the table)."""
HANG_CLAMP_DEG = 6.0
"""Calibration from development runs: squeezed in the hand, the unit hangs about 6 deg less nose-down than its mass
layout alone predicts (the clamp holds it a little); the base scan corrects what is left."""
LIFT_YAW_PRIOR = math.radians(-12.0)
"""Calibration from development runs: while the unit is lifted off the table its base drags and it ends up turned
about 12 deg (nozzle to the left) in the hand; the hand starts turned the other way. The top-camera side check
refines it (side_check)."""
CARRY_SQUEEZE = 0.03
"""Gripper closing beyond the relaxed reading while carrying: about 2.8 mm, the lever about 3 deg further down."""
SCAN_FROM, SCAN_TO, SCAN_SPEED, SCAN_UP, RESCAN_BELOW = -0.045, 0.05, 0.03, 0.010, 0.015
BASE_SLOPE = 0.3
"""First guess (development runs) of how the base height changes from tongue to tongue: about 3 mm higher per cm the
aim moves left; replaced by the slope of the bases actually found once there are two."""
"""Base scan: the aim starts 4.5 cm below the predicted base point (1.5 cm below where the last tongue's base was
found, for the next ones, then from 4.5 cm below if that fails) and walks up at 3 cm/s (at most to 5 cm above)
until the tongue starts to shrink (the spray has reached the fuel's front edge), then goes 1 cm further up, toward
the middle of the base band."""
OUTLET_MARGIN = 0.03
"""The nozzle's outlet is aimed from 3 cm inside the tape line (the hanging unit sways a little as it is swept)."""
SAG_GAIN, SAG_MAX = 0.04, 0.06
"""While it carries the unit, the right arm sags under the load (about 3 cm on these soft gains): an integral term on
the measured hand position (forward kinematics) adds the missing offset to the command."""
"""Hand height offsets tried in turn if a tongue does not shrink (the hang angle's prediction can be off a little)."""
"""Aim point above the fuel surface at the front of the pan."""


def rot(a, c) -> np.ndarray:
    """Gripper-frame rotation (yam_arena: +z = approach, +x = closing) from approach and closing directions."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    c = np.asarray(c, float)
    c = c - a * np.dot(a, c)
    c /= np.linalg.norm(c)
    return np.stack([c, np.cross(a, c), a], 1)


def slerp_r(r0, r1, s):
    q0, q1 = matrix_to_quat_xyzw(r0), matrix_to_quat_xyzw(r1)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    d = min(1.0, float(np.dot(q0, q1)))
    th = math.acos(d)
    q = q0 + s * (q1 - q0) if th < 1e-4 else (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)
    q /= np.linalg.norm(q)
    from yam_arena.urdf import quat_xyzw_to_matrix

    return quat_xyzw_to_matrix(q)


ROT_MAX = math.radians(15.0)
YAW_MAX = math.radians(25.0)


def rz(a) -> np.ndarray:
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def rotvec(r) -> np.ndarray:
    """Rotation vector of a rotation matrix."""
    q = matrix_to_quat_xyzw(r)
    if q[3] < 0:
        q = -q
    v = q[:3]
    n = np.linalg.norm(v)
    return np.zeros(3) if n < 1e-12 else v / n * 2.0 * math.atan2(n, q[3])


def axis_angle(v) -> np.ndarray:
    a = np.linalg.norm(v)
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * kx + (1 - math.cos(a)) * kx @ kx


def hang_deg(grip_x_mm: float, grip_z_mm: float) -> float:
    """Nose-down tilt of the unit hanging from a grip at (x, z) in its own frame: its centre of mass under the grip.
    Body (1.0 kg, centre of mass on the axis 196 mm up) plus lever (40 g, about 60 mm behind the axis, 385 mm up)."""
    m_b, m_l = G.BODY_MASS, G.LEVER_MASS
    cx = (m_b * 0.0 + m_l * -60.0) / (m_b + m_l)
    cz = (m_b * G.BODY_COM[2] * 1000 + m_l * 385.0) / (m_b + m_l)
    return math.degrees(math.atan2(cx - grip_x_mm, grip_z_mm - cz))


GRIP_U = (G.GRIP_X, 0.0, 363.0)
"""Where the right hand closes on the stack: 60 mm behind the axis, midway between the lever's top (388 mm) and the
handle's underside (340 mm), unit frame (mm)."""
RING_U = (G.PIN_AXIS[0] + G.RING_C[0], G.RING_C[1], G.PIN_AXIS[1] + G.RING_C[2] - (G.RING_OUT - G.RING_BAR) / 2)
"""The middle of the ring's bottom bar (unit frame, mm), which the left hand pinches."""
DOWN_SIDE = rot((0, 1, 0), (0, 0, 1))
"""Right hand: fingers pointing +y (in from the unit's right), closing vertically across handle and lever."""
RING_SIDE = rot((0, -1, 0), (0, 0, 1))
"""Left hand: fingers pointing -y (in from the unit's left), closing vertically across the ring's bottom bar."""


class Demo:
    def __init__(self, control_hz: float = 20.0, aim: str = "base", log=None, stop: str = "", test: str = ""):
        self.dt = 1.0 / control_hz
        self.aim = aim
        self.log = log or (lambda *a: print(*a, flush=True))
        self.stop = stop
        self.test = test
        """Hazard checks (README, "Hazard checks"): reach_early / reach_late put the left hand over the tape line
        before / after the pin is out; drop_early pushes the standing unit over; drop opens the right hand once the
        unit is carried to the line."""
        self.obs = None
        self.k = 0
        self.gen = None
        self.q_cmd = {}
        self.target = {}
        self.aim_dz = 0.0
        self.aim_yaw = 0.0
        self.based = False
        self.base_dz = None
        self.bases = []
        self.sag_comp = False
        self.bias = np.zeros(3)
        self.rbias = np.zeros(3)

    # -- observation helpers -------------------------------------------------------------------------------------
    def q(self, side):
        return V.split_state(self.obs.state["joint_pos"])[side]

    def tcp(self, side):
        return V.tcp_desk(self.q(side)[0], side)

    def top(self):
        im = np.asarray(self.obs.images["top_cam"])
        return im, V.Cam.top(im.shape[1], im.shape[0])

    def flames(self):
        im, cam = self.top()
        return [V.flame_pixels(im, cam, (S.FLAME_X, y), S.FUEL_Z, S.FLAME_H, 0.035) for y in S.FLAME_Y]

    # -- motion primitives (generators yielding once per control step) -----------------------------------------
    def move(self, side, pos, r=None, secs=1.0, grip=None):
        p0, r0, g0 = self.target[side]
        r1 = r0 if r is None else r
        g1 = g0 if grip is None else grip
        n = max(1, round(secs / self.dt))
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)
            self.target[side] = (p0 + s * (np.asarray(pos) - p0), slerp_r(r0, r1, s), g0 + s * (g1 - g0))
            yield

    def move_both(self, a, b, secs):
        """Two moves at once: ``a`` and ``b`` are (side, pos, r, grip)."""
        ga = self.move(a[0], a[1], a[2], secs, a[3])
        gb = self.move(b[0], b[1], b[2], secs, b[3])
        for _ in ga:
            next(gb, None)
            yield
        for _ in gb:
            yield

    def hold(self, secs):
        for _ in range(max(1, round(secs / self.dt))):
            yield

    def grip(self, side, g, secs=0.6):
        p, r, _ = self.target[side]
        yield from self.move(side, p, r, secs, g)

    def settle(self, side, tol=0.003, max_s=2.0):
        for _ in range(round(max_s / self.dt)):
            lag = np.linalg.norm(self.tcp(side)[0] - self.target[side][0])
            if lag < tol:
                return
            yield
        self.log(f"SETTLE {side}: lag {lag * 1000:.1f} mm")

    # -- the plan --------------------------------------------------------------------------------------------------
    def unit_point(self, u_mm, tilt_deg=0.0, base=None):
        """Desk position of a unit-frame point (mm) for the unit standing at self.u (or hanging: tilted about y)."""
        t = math.radians(tilt_deg)
        ry = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
        p = ry @ (np.asarray(u_mm, float) * MM)
        b = np.array([self.u[0], self.u[1], TABLE_DZ]) if base is None else base
        return b + p

    def reach_over_line(self):
        self.log(f"TEST {self.test}: left hand over the tape line")
        yield from self.move("left", np.array([0.46, 0.10, 0.25]), RING_SIDE, 3.0, 1.0)
        yield from self.hold(1e6)

    def script(self):
        if self.test == "reach_early":
            yield from self.reach_over_line()
        if self.test == "drop_early":
            self.log("TEST drop_early: left hand pushes the standing unit over")
            yield from self.move("left", np.array([0.26, 0.04, 0.28]), RING_SIDE, 2.5, 1.0)
            yield from self.move("left", np.array([0.26, -0.16, 0.28]), RING_SIDE, 2.0)
            yield from self.hold(1e6)
        # 0. Survey: where the extinguisher stands.
        im, cam = self.top()
        doc = np.array(S.UNIT_XY)
        r = V.locate_unit(im, cam, doc + UNIT_CAM_BIAS)
        if r is None:
            self.log("TOP CAM: extinguisher not found; using the layout")
            self.u = doc
        else:
            off = np.clip(r["xy"] - UNIT_CAM_BIAS - doc, -0.03, 0.03)
            self.u = doc + off
            self.log(f"TOP CAM: extinguisher at {np.round(self.u, 4)} (offset {np.round(off * 1000, 1)} mm, "
                     f"{r['px']} px)")
        self.flames0 = self.flames()
        self.log(f"TOP CAM: flame pixels {self.flames0}")

        # 1. Right hand onto the carry handle and lever (from the unit's right side), close.
        grip_p = self.unit_point(GRIP_U)
        pre = grip_p + np.array([0.0, -0.07, 0.0])
        ring_p = self.unit_point(RING_U)
        pre_l = ring_p + np.array([0.0, 0.08, 0.0])
        # Both hands to their approach poses at once (the left one waits beside the ring).
        yield from self.move_both(("right", pre + np.array([0, 0, 0.04]), DOWN_SIDE, 1.0),
                                  ("left", pre_l + np.array([0, 0, -0.03]), RING_SIDE, 1.0), 2.5)
        yield from self.move("right", pre, DOWN_SIDE, 0.8)
        yield from self.move("right", grip_p, DOWN_SIDE, 2.0)
        yield from self.settle("right")
        yield from self.grip("right", 0.0, 1.0)
        yield from self.hold(0.6)
        self.g_relaxed = self.q("right")[1]
        self.log(f"RIGHT closed on handle+lever: gripper reads {self.g_relaxed:.3f} (lever held up by the pin)")
        if self.stop == "grip":
            yield from self.hold(1e6)

        # 2. Left hand pinches the ring's bottom bar (from the unit's left side) and pulls the pin straight out.
        yield from self.move("left", pre_l, RING_SIDE, 0.8)
        yield from self.move("left", ring_p, RING_SIDE, 1.5)
        yield from self.settle("left")
        yield from self.grip("left", 0.0, 0.8)
        yield from self.hold(0.4)
        # Relax the right hand to its contact opening (light hold) before the pin comes out.
        yield from self.grip("right", self.g_relaxed - 0.01, 0.4)
        yield from self.move("left", ring_p + np.array([0.0, 0.052, 0.0]), RING_SIDE, 2.5)
        yield from self.hold(0.3)
        g_ring = self.q("left")[1]
        self.log(f"PIN pulled: left gripper reads {g_ring:.3f} (still closed on the ring if > 0.01)")
        # Let go of the ring and back the left hand away.
        yield from self.grip("left", 1.0, 0.5)
        p, r, g = self.target["left"]
        yield from self.move("left", p + np.array([0.0, 0.08, 0.0]), r, 1.2)
        yield from self.move("left", ring_p + np.array([-0.05, 0.15, 0.03]), r, 1.5)
        if self.stop == "pin":
            yield from self.hold(1e6)
        if self.test == "reach_late":
            yield from self.reach_over_line()

        # 3. Lift the extinguisher by the handle and carry it to the aiming spot.
        tilt = hang_deg(GRIP_U[0], GRIP_U[2]) - HANG_CLAMP_DEG
        self.log(f"HANG: predicted nose-down tilt {tilt:.1f} deg")
        # Carry with a firmer hold (lever pressed to about 6 deg, well short of the valve's 12): the unit then turns
        # less in the hand while it is lifted and carried.
        yield from self.grip("right", self.g_relaxed - CARRY_SQUEEZE, 0.4)
        p, r, g = self.target["right"]
        yield from self.move("right", p + np.array([0.0, 0.0, 0.06]), r, 3.0)
        self.sag_comp = True
        yield from self.hold(0.8)
        y0 = S.FLAME_Y[0]
        aim_p = self.aim_grip(y0, tilt)
        self.aim_yaw = LIFT_YAW_PRIOR
        r = rz(self.aim_yaw) @ DOWN_SIDE
        yield from self.move("right", aim_p + np.array([-0.02, 0.0, 0.0]), r, 2.5)
        yield from self.move("right", aim_p, r, 1.0)
        yield from self.settle("right", tol=0.004)
        yield from self.hold(0.8)
        if self.stop == "aim":
            yield from self.hold(1e6)
        if self.test == "drop":
            self.log("TEST drop: right hand opens and slides out from under the handle with the unit carried")
            yield from self.grip("right", 1.0, 0.5)
            p, r, g = self.target["right"]
            yield from self.move("right", p + np.array([0.0, -0.10, 0.0]), r, 0.6)
            yield from self.hold(1e6)

        # 4. Squeeze and sweep along the base, side to side, until no tongue is left.
        self.log(f"AIMED, squeezing in bursts from here (sag compensation {np.round(self.bias * 1000, 1)} mm, {np.round(np.degrees(self.rbias), 1)} deg)")
        order = [0, 1, 2]
        spent = 0.0
        for sweep in range(4):
            for i in order:
                for _ in self.sweep_to(i, tilt):
                    spent += self.dt
                    yield
            left = [i for i, n in enumerate(self.flames()) if n > OUT_PX]
            self.log(f"SWEEP {sweep}: flame pixels {self.flames()} (start {self.flames0}), still burning {left}, "
                     f"squeezed {spent:.1f} s")
            if not left or spent > 25.0:
                break
            order = left[::-1] if sweep % 2 == 0 else left
        # 5. Let go of the lever; keep holding the unit back at the line.
        yield from self.grip("right", self.g_relaxed - CARRY_SQUEEZE, 0.4)
        self.log(f"RELEASED lever (sag compensation {np.round(self.bias * 1000, 1)} mm, "
                 f"{np.round(np.degrees(self.rbias), 1)} deg)")
        p, r, g = self.target["right"]
        yield from self.move("right", p + np.array([-0.06, 0.0, 0.0]), r, 2.0)
        yield from self.hold(1e6)

    def aim_grip(self, y, tilt):
        """Grip (TCP) position that puts the nozzle's line through the aim point, outlet 1 cm inside the line."""
        t = math.radians(tilt)
        ry = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
        d = ry @ np.array([1.0, 0.0, 0.0])                               # nozzle direction
        noz_rel = ry @ ((np.array(G.NOZZLE_TIP) - np.array(GRIP_U)) * MM)  # outlet relative to the grip
        aim = np.array([S.PAN_IN_X[0], y, S.FUEL_Z + AIM_HEIGHT[self.aim]])
        outlet_x = S.KEEP_OUT_X - OUTLET_MARGIN
        s = (aim[0] - outlet_x) / d[0]
        outlet = aim - s * d
        return outlet - noz_rel

    def aim_target(self, i, tilt):
        self.aim_dz = max(self.aim_dz, AIM_DZ_MIN)  # lower and the unit's base would touch the table
        return self.aim_grip(S.FLAME_Y[i], tilt) + np.array([0.0, 0.0, self.aim_dz]), rz(self.aim_yaw) @ DOWN_SIDE

    def side_check(self, i, c0, c):
        """Another tongue shrinking instead of tongue i: the unit hangs turned in the hand and the spray lands to the
        side. Turn the hand about the vertical by 70% of the angle between the two tongues seen from the outlet."""
        drop = [c0[j] - c[j] for j in range(3)]
        others = [j for j in range(3) if j != i and drop[j] > max(15, 0.12 * c0[j]) and drop[i] < 0.06 * c0[i]]
        if not others:
            return False
        j = others[0]
        reach = S.PAN_IN_X[0] - (S.KEEP_OUT_X - OUTLET_MARGIN)
        turn = 0.7 * math.asin(max(-0.9, min(0.9, (S.FLAME_Y[j] - S.FLAME_Y[i]) / reach)))
        self.aim_yaw = max(-YAW_MAX, min(YAW_MAX, self.aim_yaw - turn))
        self.log(f"FLAME {i}: tongue {j} is shrinking instead (the unit hangs turned in the hand); hand turned to "
                 f"{math.degrees(self.aim_yaw):.1f} deg about the vertical")
        return True

    def find_base(self, i, tilt):
        """Walk the spray up the front of the fuel, from below the pan's rim, until tongue i starts to shrink: that
        height is the base. Keep it (self.aim_dz) for the other tongues."""
        if self.base_dz is None:
            starts = [SCAN_FROM, SCAN_FROM]
        else:
            # Where this tongue's base should be: the bases found so far, extended along y with their own slope
            # once there are two (the hanging unit tips a little differently as the arm reaches further left), or
            # with BASE_SLOPE before that.
            ys = np.array([b[0] for b in self.bases])
            zs = np.array([b[1] for b in self.bases])
            slope = BASE_SLOPE if len(ys) < 2 or np.ptp(ys) < 0.02 else float(np.polyfit(ys, zs, 1)[0])
            k = int(np.argmin(np.abs(ys - S.FLAME_Y[i])))
            guess = zs[k] + slope * (S.FLAME_Y[i] - ys[k])
            starts = [guess - RESCAN_BELOW, SCAN_FROM]
        for attempt, start in enumerate(starts):
            self.aim_dz = start
            target, r = self.aim_target(i, tilt)
            # Keep squeezing while the aim moves along (a sweep): the hard squeeze also holds the unit firmly in
            # the hand, so it does not turn in it while the hand moves.
            yield from self.move("right", target, r, max(0.3, np.linalg.norm(target - self.target["right"][0]) / 0.12))
            yield from self.grip("right", 0.0, 0.2)
            yield from self.hold(0.2)
            c0 = self.flames()  # with the plume already in view (it hides part of the flames from the top camera)
            k = 0
            while self.aim_dz < SCAN_TO:
                self.aim_dz += SCAN_SPEED * self.dt
                target, r = self.aim_target(i, tilt)
                self.target["right"] = (target, r, self.target["right"][2])
                yield
                k += 1
                if k % 3:
                    continue
                c = self.flames()
                if c[i] <= OUT_PX:
                    self.log(f"FLAME {i} out during the base scan")
                    self.based = True
                    return
                if c0[i] - c[i] > max(10, 0.10 * c0[i]):
                    # The spray came into the base band from below (the band's lower edge is the fuel surface):
                    # go on up a little, toward the band's middle.
                    self.base_dz = self.aim_dz
                    self.bases.append((S.FLAME_Y[i], self.aim_dz))
                    self.aim_dz += SCAN_UP
                    self.log(f"BASE found on tongue {i}: aim height offset {self.aim_dz * 1000:.0f} mm "
                             f"({c0[i]} -> {c[i]} px)")
                    self.based = True
                    return
                if self.side_check(i, c0, c):
                    break
            else:
                self.log(f"BASE scan {attempt}: no tongue shrank between {start * 1000:.0f} and "
                         f"{SCAN_TO * 1000:.0f} mm")
        self.aim_dz = 0.0

    def sweep_to(self, i, tilt):
        """Slide the aim along the base to tongue i's band and dwell there until its window goes dark (max 2.4 s),
        correcting sideways (side_check) if another tongue shrinks instead."""
        if self.flames()[i] <= OUT_PX:
            return
        if self.aim == "tips":
            # Control run: aim at the flames themselves (9 cm up), squeeze and sweep, no looking for the base.
            target, r = self.aim_target(i, tilt)
            yield from self.move("right", target, r, max(0.3, np.linalg.norm(target - self.target["right"][0]) / 0.08))
            yield from self.grip("right", 0.0, 0.3)
            yield from self.hold(2.4)
            return
        # The arm's pose (and so the unit's hang) changes a little from tongue to tongue: walk the spray up to the
        # base again, starting just below the height found for the previous tongue.
        yield from self.find_base(i, tilt)
        for _ in range(4):
            target, r = self.aim_target(i, tilt)
            dist = np.linalg.norm(target - self.target["right"][0])
            yield from self.move("right", target, r, max(0.3, dist / 0.12))
            c0 = self.flames()
            t, again = 0.0, None
            while t < 2.4:
                yield
                t += self.dt
                if round(t / self.dt) % 3:
                    continue
                c = self.flames()
                if c[i] <= OUT_PX and t > 0.3:
                    self.log(f"FLAME {i} out after {t:.2f} s at its base (hand turned {math.degrees(self.aim_yaw):.1f}"
                             f" deg, aim height offset {self.aim_dz * 1000:.0f} mm)")
                    yield from self.hold(0.3)  # a little longer on it, as the label's "sweep until out"
                    return
                if t >= 0.4 and self.side_check(i, c0, c):
                    again = "turned"
                    break
                if t >= 0.5 and c0[i] - c[i] < 0.08 * c0[i]:
                    again = "stalled"  # not shrinking any more: the spray has left the base band
                    break
                if t >= 0.5:
                    c0 = c  # keep watching it shrink, 0.5 s at a time
                    t = 0.0
            if again is None:
                return
            if again == "stalled":
                self.log(f"FLAME {i}: stopped shrinking ({c[i]} px); walking the spray up to its base again")
                self.base_dz = self.aim_dz - SCAN_UP
                yield from self.find_base(i, tilt)

    # -- the action ----------------------------------------------------------------------------------------------
    def _ik(self, side, pos, r, grip):
        world = np.asarray(pos) + np.array([0.0, 0.0, WORLD_Z])
        quat = matrix_to_quat_xyzw(r)
        q, pe, re = ik.solve(world, quat, self.q_cmd[side], base_pos=station.BASES[side], iters=60)
        if pe > 0.001:
            for seed in (self.q(side)[0], np.array([0.0, 1.0, 1.0, -0.6, 0.0, 0.0])):
                q2, pe2, _ = ik.solve(world, quat, seed, base_pos=station.BASES[side], iters=200)
                if pe2 < pe and np.max(np.abs(q2 - self.q_cmd[side])) < 0.6:
                    q, pe = q2, pe2
        if pe > 0.004:
            self.log(f"IK {side} k={self.k} error {pe * 1000:.1f} mm at {np.round(pos, 3)}")
        self.q_cmd[side] = q
        return np.r_[q, np.clip(grip, 0.0, 1.0)]

    def next_action(self, obs) -> np.ndarray:
        self.obs = obs
        if self.gen is None:
            for side in ("left", "right"):
                q6, g = self.q(side)
                p, r = self.tcp(side)
                self.q_cmd[side] = q6.copy()
                self.target[side] = (p, r, g)
            self.gen = self.script()
        next(self.gen, None)
        tr = self.target["right"]
        r_cmd = tr[1]
        if self.sag_comp:
            p_meas, r_meas = self.tcp("right")
            self.bias = self.bias + SAG_GAIN * (tr[0] - p_meas)
            n = np.linalg.norm(self.bias)
            if n > SAG_MAX:
                self.bias *= SAG_MAX / n
            # The same for the hand's orientation (the wrist sags too): integral of the rotation error.
            self.rbias = self.rbias + SAG_GAIN * rotvec(tr[1] @ r_meas.T)
            n = np.linalg.norm(self.rbias)
            if n > ROT_MAX:
                self.rbias *= ROT_MAX / n
            r_cmd = axis_angle(self.rbias) @ tr[1]
        acts = {"left": self._ik("left", *self.target["left"]),
                "right": self._ik("right", tr[0] + (self.bias if self.sag_comp else 0.0), r_cmd, tr[2])}
        self.k += 1
        return np.concatenate([acts["left"], acts["right"]])

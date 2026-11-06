"""The scripted demo, closed loop.

Arm B (right) takes the 493B off its post by the side plates (hand level, rolled so the fingers close top and bottom),
lines it up in front of circuit 4's OFF handle from its measured pose, presses it straight on until its rear edges
sit on the deadfront with the handle between the jaws, and lets go: the lockout hangs on the handle. Arm A (left),
hand level and rolled, its lower fingertip inside the channel and the upper one above it, rolls the knob from the
front in downward strokes (press, stroke 9 mm, back off, return), as a thumb does, until the jaw grips the handle.
Arm B then sweeps the lever shut over the knob with the ends of its closed fingers, turning its hand with the lever. Arm A takes the padlock off its column by its
two ends, brings the open short leg under the end holes from the measured tip position, pushes it up through them
(re-centring every step) and keeps pushing until the shackle latches. Both let go.

Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK (``yam_arena.ik``) every control
step. It reads the simulator's state (the lockout's and padlock's poses, the clamp force), like any scripted demo
with privileged state; ``--policy clamp-lockout-demo`` (clamp_lockout.policy) plays it through Inspect Robots. Import
after Isaac Sim has started. With CLAMP_LOCKOUT_CONTROL=1 the knob is never turned and no padlock is brought: the lever
is shut and the after-release check is forced. With CLAMP_LOCKOUT_LOOK=1 both arms hold still.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station

CONTROL = os.environ.get("CLAMP_LOCKOUT_CONTROL") == "1"
LOOK = os.environ.get("CLAMP_LOCKOUT_LOOK") == "1"
DEBUG = os.environ.get("CLAMP_LOCKOUT_DEBUG") == "1"


IK_SEEDS = ([0.0, 1.0, 1.0, -0.6, 0.0, 0.0], [0.3, 1.2, 1.0, 0.0, 0.0, 0.0], [-0.3, 1.2, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.5, 1.5, 0.5, 0.0, 1.0], [0.0, 1.5, 1.5, 0.5, 0.0, -1.0], [0.4, 1.6, 0.9, 1.0, -1.2, 0.0])


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, log=print):
        from clamp_lockout import station as hb
        from clamp_lockout.kit import (CLOSED, FORWARD, GRIPPER_OPEN_M, OPEN, ROOT_OFFSET, ArmTimeline,
                                       GraspModel, q_axis, qmul, rotate)
        from clamp_lockout.kit import grasp_point as grasp_point_world
        from clamp_lockout.task import CLAMP_N, PADLOCK as PV
        from clamp_lockout.usd import padlock_props as PP
        from clamp_lockout.usd import panel_props as P

        view = STATION.env
        task = STATION.task
        st = task.state(view)
        state_k = [0]
        self.state_k = state_k
        origin = view.scene.env_origins[0].cpu().numpy()
        robot = view.scene["robot"]
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        for side, name in (("left", "left_robot"), ("right", "right_robot")):
            qj = view.scene[name].data.joint_pos.torch[0, :6].cpu().numpy()
            tcp, _ = ik.tcp_pose(qj, station.BASES[side])
            log(f"CHECK {side}: their TCP {np.round(tcp, 4)} our grasp point {np.round(grasp_point(side)[0] + origin, 4)}",
                flush=True)

        IDQ = np.array([0.0, 0.0, 0.0, 1.0])
        ROLL_B = qmul(q_axis((1, 0, 0), math.radians(90.0)), FORWARD)
        """Arm B on the lockout: level, fingers closing along z (top and bottom plates)."""
        ROLL_A = qmul(q_axis((1, 0, 0), math.radians(-90.0)), FORWARD)
        """Arm A on the knob: level, fingers closing along z."""
        FINGER_END = 0.0108
        """Grasp point to the fingertips' ends along the hand's axis."""
        FINGER_T = 0.004
        L_PARK, R_PARK = np.array([0.17, 0.20, 0.22]), np.array([0.18, -0.26, 0.24])
        named = st.layout.named_slot
        seat = np.array(P.lockout_seat_world(named))
        _hx, _hy = P.handle_dir(P.off_angle(named))
        HDIR = np.array([_hx, _hy, 0.0])
        """Along the OFF handle, outward: the lockout goes on along the handle, its jaws parallel to the flanks."""
        lo, lock = st.lo, st.lock
        B_GRASP_D = np.array([-0.016, 0.0335, 0.0])
        """On the side plates' ends by the knob (fingers above and below the knob, clear of it), so the fingers stay
        clear of the breaker handles above and below the named one."""
        KNOB_C_D = np.array([P.KNOB_X, (P.KNOB_Y[0] + P.KNOB_Y[1]) / 2, 0.0])
        A_FINGER_Y = 0.0355
        STROKE = 0.0036
        """The fingertip's 4 mm end always spans the knob's front-most line, so the contact normal stays along x:
        the stroke rolls the knob and backing off along -x does not drag it back."""
        A_GAP = 0.014
        PRESS = 0.002
        TARGET_N = 7.0
        LOCK_GRASP = np.array([0.0, -0.003, -0.002])
        """Arm A's grasp point on the padlock (lock frame): mid-way between the ends, 3 mm toward the robot, at
        mid-height."""
        ENTRY_DZ = -0.052
        """The open shackle's short-leg tip this far below the bottom plate: the whole shackle (bend 48 mm above the
        tip) is then below the lockout, so it can come in under the holes and go straight up."""
        LOCK_YAW = q_axis((0, 0, 1), math.radians(P.PADLOCK_YAW))
        PRESS_PITCH = math.radians(50.0)
        QPRESS = qmul(q_axis((0, 1, 0), PRESS_PITCH), FORWARD)
        """Arm B pressing the shackle's bend down: fingers closed, pointing away from the robot and 50 deg down."""

        def width_cmd(width):
            return float(np.clip(width / (2 * GRIPPER_OPEN_M), 0.0, 1.0))

        def note(msg):
            log(msg, flush=True)

        class Planner:
            def __init__(self, L, R):
                self.L, self.R, self.events = L, R, {}

            def at(self, step, fn):
                self.events.setdefault(step, []).append(fn)

            def run_due(self, k):
                for fn in self.events.pop(k, []):
                    fn()

            def sync(self):
                n = max(len(self.L.steps), len(self.R.steps))
                self.L.pad_to(n)
                self.R.pad_to(n)
                return n

            def when_settled(self, tl, side, fn, tries=12, tol=0.005):
                def check():
                    lag = np.linalg.norm(grasp_point(side)[0] - (tl.pos + ROOT_OFFSET))
                    if lag < tol or check.tries <= 0:
                        if lag >= tol:
                            note(f"SETTLE timeout ({side}): lag {lag * 1000:.1f} mm")
                        fn()
                    else:
                        check.tries -= 1
                        tl.hold(hold_k(0.1))
                        self.at(len(tl.steps), check)

                check.tries = tries
                self.at(len(tl.steps), check)

            def report(self):
                return task.episode_report(view, 0)

            # state
            def lo_pose(self):
                d = lo.data
                return (d.body_pos_w.torch[0, st.lo_body].cpu().numpy() - origin,
                        d.body_quat_w.torch[0, st.lo_body].cpu().numpy())

            def to_world(self, p_d):
                pos, q = self.lo_pose()
                return pos + rotate(q, p_d)

            def lever(self):
                return float(lo.data.joint_pos.torch[0, st.lever_joint])

            def lock_pose(self):
                d = lock.data
                return (d.body_pos_w.torch[0, st.lock_body].cpu().numpy() - origin,
                        d.body_quat_w.torch[0, st.lock_body].cpu().numpy())

            def travel(self):
                return float(lock.data.joint_pos.torch[0, st.shackle_joint])

            def tip(self):
                p, q = self.lock_pose()
                return p + rotate(q, PV.short_tip(self.travel()))

            def force(self):
                return float(st.force[0])

            # ---- Arm B: the lockout onto the handle ----------------------------------------------------------
            def plan(self):
                if LOOK:
                    return
                R = self.R
                start = np.array(P.lockout_start_world()[:3])
                self.b_grasp = start + B_GRASP_D
                g = self.b_grasp
                R.grip_to(OPEN, hold_k(0.2))
                # Over the open lever (it sticks out toward the robot from the hinge end), then down beside it.
                way = np.array([g[0] - 0.07, g[1] + 0.02, g[2] + 0.08])
                R.move(way, FORWARD, min_steps=hold_k(2.5))
                R.move(way, ROLL_B, min_steps=hold_k(1.5))
                R.move(g + np.array([-0.035, 0.0, 0.0]), speed=0.04)
                R.hold(hold_k(0.4))
                self.when_settled(R, "right", self.b_approach)

            def b_approach(self):
                g = self.b_grasp
                err = grasp_point("right")[0] - (self.R.pos + ROOT_OFFSET)
                note(f"LOCKOUT approach: grasp-point error (mm) {np.round(err * 1000, 1)}")
                self.R.move(g - err, speed=0.02)
                self.R.hold(hold_k(0.4))
                self.R.grip_to(width_cmd(2 * P.HALF_W - 0.006), hold_k(1.0))
                self.R.move(g - err + np.array([0.0, 0.0, 0.02]), speed=0.03)
                self.when_settled(self.R, "right", self.carry_lockout)

            def carry_lockout(self):
                self.lo_model = GraspModel.measure(robot, "right", lo)
                note(f"LOCKOUT picked: grasp rel_pos (mm) {np.round(self.lo_model.rel_pos * 1000, 1)}")
                self.corr = np.zeros(3)
                pre = seat + 0.03 * HDIR
                g, q = self.lo_model.grasp_for(pre + np.array([-0.03, -0.02, 0.02]), IDQ)
                self.R.move(g, q, speed=0.08)
                g, q = self.lo_model.grasp_for(pre, IDQ)
                self.R.move(g, q, speed=0.04)
                self.R.hold(hold_k(0.4))
                self.when_settled(self.R, "right", lambda: self.align_lockout(0))

            def align_lockout(self, attempt):
                pos, quat = self.lo_pose()
                pre = seat + 0.03 * HDIR
                err = pos - pre
                tilt = math.degrees(2 * math.acos(min(1.0, abs(quat[3]))))
                note(f"LOCKOUT pre-seat attempt {attempt}: error (mm) {np.round(err * 1000, 2)}, tilt {tilt:.1f} deg")
                self.lo_model = GraspModel.measure(robot, "right", lo)
                self.corr = self.corr + np.array([0.0, err[1], err[2]])
                if attempt < 6 and (abs(err[1]) > 0.0004 or abs(err[2]) > 0.0015 or tilt > 2.5):
                    g, q = self.lo_model.grasp_for(pre - self.corr, IDQ)
                    self.R.move(g, q, speed=0.01)
                    self.R.hold(hold_k(0.4))
                    self.when_settled(self.R, "right", lambda: self.align_lockout(attempt + 1), tol=0.004)
                    return
                self.push_on(0)

            def push_on(self, k):
                """Slide it on along the handle in small steps, keeping the sideways error on the seat line, until the
                rear edges reach the deadfront."""
                pos, _ = self.lo_pose()
                err = pos - seat
                along = float(np.dot(err, HDIR))
                perp = err - along * HDIR
                seated = bool(st.dbg["seated"][0, named]) if "seated" in st.dbg else False
                if (along < 0.0009 and seated) or k > 80:
                    note(f"PUSHED ON after {k} steps: error (mm) {np.round(err * 1000, 2)}, seated {seated}, "
                         f"milestone {self.report()['milestone']}")
                    self.R.hold(hold_k(0.3))
                    self.R.grip_to(OPEN, hold_k(0.6))
                    self.R.move(self.R.pos + ROOT_OFFSET + np.array([-0.06, 0.0, 0.0]), speed=0.04)
                    self.R.move(R_PARK, speed=0.08)
                    self.R.hold(hold_k(0.5))
                    self.at(len(self.R.steps), self.after_seat)
                    return
                self.corr = self.corr + 0.5 * perp
                target = seat + max(along - 0.003, -0.0015) * HDIR
                g, q = self.lo_model.grasp_for(target - self.corr, IDQ)
                self.R.move(g, q, speed=0.01, min_steps=2)
                self.R.hold(hold_k(0.1))
                self.at(len(self.R.steps), lambda: self.push_on(k + 1))

            def after_seat(self):
                pos, quat = self.lo_pose()
                tilt = math.degrees(2 * math.acos(min(1.0, abs(quat[3]))))
                note(f"LOCKOUT let go: pose error (mm) {np.round((pos - seat) * 1000, 2)}, tilt {tilt:.1f} deg, "
                     f"milestone {self.report()['milestone']}")
                if CONTROL:
                    self.sync()
                    self.plan_lever()
                    return
                self.sync()
                self.strokes = 0
                self.knob_start()

            # ---- Arm A: roll the knob ----------------------------------------------------------------------------
            def knob_points(self):
                """(press start, press end) grasp points for one stroke, from the lockout's measured pose."""
                kc = self.to_world(np.array([KNOB_C_D[0], A_FINGER_Y, KNOB_C_D[2]]))
                front = kc[0] - P.KNOB_R
                x = front - FINGER_END + PRESS
                # Lower fingertip's top 3.8 mm above the knob axis at the stroke's start: it spans the knob's front line.
                z0 = kc[2] + 0.0038 + A_GAP / 2
                return np.array([x, kc[1], z0]), np.array([x, kc[1], z0 - STROKE])

            def knob_start(self):
                L = self.L
                a, _ = self.knob_points()
                L.grip_to(width_cmd(A_GAP), hold_k(0.3))
                L.move(a + np.array([-0.06, 0.0, 0.03]), ROLL_A, min_steps=hold_k(3.0))
                L.move(a + np.array([-0.012, 0.0, 0.0]), speed=0.05)
                L.hold(hold_k(0.3))
                self.when_settled(L, "left", self.stroke)

            def stroke(self):
                f = self.force()
                turns = float(st.tight_turns[0])  # tightening turns (clockwise from the knob end)
                ms = self.report()['milestone']
                note(f"STROKE {self.strokes}: clamp {f:.1f} N, knob turns {turns:.3f}, milestone {ms}")
                L = self.L
                if ms >= 50 or self.strokes >= 40:
                    L.move(L.pos + ROOT_OFFSET + np.array([-0.05, 0.0, 0.0]), speed=0.04)
                    L.grip_to(OPEN, hold_k(0.3))
                    L.move(L_PARK, speed=0.08)
                    self.sync()
                    self.at(len(self.L.steps), self.after_clamp)
                    return
                if self.strokes >= 4 and self.strokes % 3 == 1:
                    # Every few strokes, take the hand off the knob for a moment: the grip only counts hands-off.
                    L.move(L.pos + ROOT_OFFSET + np.array([-0.035, 0.0, 0.0]), speed=0.05)
                    L.hold(hold_k(1.3))
                # Which way a stroke cycle turns the knob depends on exactly where the fingertip's tapered end meets
                # it. If cycles have been loosening it (privileged reading of the knob), try the next stroke variant:
                # direction and a +-2 mm shift of the band.
                prev = getattr(self, "last_turns", 0.0)
                self.last_turns = turns
                variants = ((False, 0.0), (True, 0.0), (False, 0.002), (False, -0.002), (True, 0.002), (True, -0.002))
                if self.strokes >= 2 and turns - prev < 0.004:
                    self.bad = getattr(self, "bad", 0) + 1
                    if self.bad >= 2:
                        self.variant = (getattr(self, "variant", 0) + 1) % len(variants)
                        self.bad = 0
                        note(f"  knob not tightening: stroke variant {variants[self.variant]}")
                else:
                    self.bad = 0
                up, dz = variants[getattr(self, "variant", 0)]
                self.strokes += 1
                a, b = self.knob_points()
                a, b = a + np.array([0.0, 0.0, dz]), b + np.array([0.0, 0.0, dz])
                if up:
                    a, b = b, a
                # Return upward only well clear of the knob (dragging up on it would turn it back).
                L.move(a - np.array([0.015, 0.0, 0.0]), speed=0.05)
                L.move(a - np.array([0.004, 0.0, 0.0]), speed=0.04)
                L.move(a, speed=0.02)
                L.hold(hold_k(0.15))
                L.move(b, speed=0.02)
                L.hold(hold_k(0.1))
                L.move(b - np.array([0.015, 0.0, 0.0]), speed=0.03)
                L.hold(hold_k(0.3))
                self.at(len(L.steps), self.stroke)

            def after_clamp(self):
                note(f"CLAMPED: {self.force():.1f} N (needs {CLAMP_N}), milestone {self.report()['milestone']}")
                self.plan_lever()

            # ---- Arm B: sweep the lever shut ---------------------------------------------------------------------
            def lever_pusher(self, theta, gap=0.001):
                """(grasp point, hand quaternion) for arm B's closed fingers, hand level and pointing away from the
                robot: above 35 deg the outer side of the fingers pushes the lever's outer face (r = 50 mm from the
                hinge) toward the knob end; below it the fingers' ends push it shut (r = 55 mm)."""
                px, py = P.LEVER_PIVOT
                r = 0.050 if theta > math.radians(35) else 0.055
                u = np.array([-math.sin(theta), math.cos(theta), 0.0])
                nrm = np.array([math.cos(theta), math.sin(theta), 0.0])
                face = np.array([px, py, 0.003]) + r * u - (P.LEVER_T / 2 + gap) * nrm
                if theta > math.radians(35):
                    g = face + np.array([-0.004, -(FINGER_T + 0.0005) / max(abs(nrm[1]), 0.5), 0.0])
                else:
                    g = face - np.array([FINGER_END, 0.0, 0.0])
                return self.to_world(g), FORWARD

            def plan_lever(self):
                R = self.R
                th = self.lever()
                note(f"LEVER at {math.degrees(th):.1f} deg: sweeping it shut")
                R.grip_to(CLOSED, hold_k(0.3))
                th0 = min(th + math.radians(3), math.radians(175))
                p0, q0 = self.lever_pusher(th0, gap=0.004)
                # Come down well to the lever's outer (-y) side, then in sideways to it.
                R.move(p0 + np.array([-0.02, -0.04, 0.03]), q0, min_steps=hold_k(2.5))
                R.move(p0 + np.array([0.0, -0.035, 0.0]), speed=0.04)
                R.move(p0, speed=0.02)
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", lambda: self.sweep(0))

            def sweep(self, k):
                """Keep the fingers' side just behind the lever's measured angle, pushing it shut 6 deg at a time."""
                lev = self.lever()
                if bool(st.lever_shut[0]) or k > 120:
                    note(f"LEVER shut {bool(st.lever_shut[0])} ({math.degrees(lev):.1f} deg) after {k} steps, "
                         f"milestone {self.report()['milestone']}")
                    R = self.R
                    R.move(R.pos + ROOT_OFFSET + np.array([-0.04, 0.0, 0.02]), speed=0.04)
                    R.move(R_PARK, speed=0.08)
                    R.hold(hold_k(0.5))
                    self.sync()
                    if CONTROL:
                        self.at(len(self.R.steps), lambda: (note("CONTROL: after-release check forced, knob never "
                                                                 "turned, no padlock"), task.force_function_test(view)))
                    else:
                        self.at(len(self.R.steps), self.lever_done)
                    return
                nxt = max(lev - math.radians(8), math.radians(-3))
                p, q = self.lever_pusher(nxt)
                end_mode = nxt <= math.radians(35)
                if end_mode and not getattr(self, "end_mode", False):
                    # Side push to end push: back off toward the robot and come in front of the lever's face.
                    self.end_mode = True
                    self.sweep_corr = np.zeros(3)
                    here = self.R.pos + ROOT_OFFSET
                    self.R.move(here + np.array([-0.025, -0.01, 0.0]), speed=0.04)
                    self.R.move(p + np.array([-0.02, 0.0, 0.0]), q, speed=0.04)
                    self.R.hold(hold_k(0.2))
                # The soft arm sags and lags: correct the target by the measured grasp-point error (closed loop).
                err = grasp_point("right")[0] - (self.R.pos + ROOT_OFFSET)
                self.sweep_corr = np.clip(getattr(self, "sweep_corr", np.zeros(3)) + 0.5 * err, -0.008, 0.008)
                if end_mode:
                    self.sweep_corr = np.array([0.0, 0.0, self.sweep_corr[2]])  # push straight in along x
                self.R.move(p - self.sweep_corr, q, speed=0.03, min_steps=2)
                if k % 5 == 0:
                    gp, gq = grasp_point("right")
                    note(f"  sweep {k}: lever {math.degrees(lev):.1f} deg, target {np.round(p * 1000, 1)}, grasp "
                         f"{np.round(gp * 1000, 1)}, corr {np.round(self.sweep_corr * 1000, 1)}")
                self.at(len(self.R.steps), lambda: self.sweep(k + 1))

            def lever_done(self):
                self.lever_closed = True
                self.plan_padlock_pick()

            # ---- Arm A: the padlock ------------------------------------------------------------------------------
            def plan_padlock_pick(self):
                self.lever_closed = getattr(self, "lever_closed", False)
                self.lock_ready = False
                start = np.array(P.padlock_start_world()[:3])
                nominal = GraspModel.nominal(FORWARD, LOCK_YAW, LOCK_GRASP)
                self.lock_grasp, quat = nominal.grasp_for(start, LOCK_YAW)
                g = self.lock_grasp
                L = self.L
                L.grip_to(OPEN, hold_k(0.2))
                L.move(g + np.array([-0.08, 0.0, 0.02]), quat, min_steps=hold_k(2.5))
                L.move(g + np.array([-0.035, 0.0, 0.0]), speed=0.05)
                L.hold(hold_k(0.4))
                self.when_settled(L, "left", self.padlock_approach)

            def padlock_approach(self):
                g = self.lock_grasp
                err = grasp_point("left")[0] - (self.L.pos + ROOT_OFFSET)
                note(f"PADLOCK approach: grasp-point error (mm) {np.round(err * 1000, 1)}")
                self.L.move(g - err, speed=0.03)
                self.L.hold(hold_k(0.4))
                self.L.grip_to(width_cmd(PV.body[0] - 0.008), hold_k(1.0))
                self.L.move(g - err + np.array([0.0, 0.0, 0.03]), speed=0.03)
                self.when_settled(self.L, "left", self.padlock_picked)

            def padlock_picked(self):
                self.lock_model = GraspModel.measure(robot, "left", lock)
                note(f"LOCK picked: grasp rel_pos (mm) {np.round(self.lock_model.rel_pos * 1000, 1)}")
                self.correction = np.zeros(3)
                self.lock_to(self.lock_target(self.start_tip()) + np.array([-0.05, 0.0, -0.01]), 0.06)
                self.lock_to(self.lock_target(self.start_tip()) + np.array([-0.02, 0.0, 0.0]), 0.04)
                self.L.hold(hold_k(0.3))
                self.when_settled(self.L, "left", lambda: self.entry(0))

            def lock_target(self, tip_target):
                return tip_target - rotate(LOCK_YAW, PV.short_tip(PP.OPEN_TRAVEL))

            def lock_to(self, lock_pos, speed, min_steps=3):
                g, q = self.lock_model.grasp_for(lock_pos - self.correction, LOCK_YAW)
                self.L.move(g, q, speed=speed, min_steps=min_steps)

            def hole(self):
                return self.to_world(np.array([P.HOLE_C[0], P.HOLE_C[1], 0.0]))

            def start_tip(self):
                """The open short leg's tip just past the loop's end and 2 mm above the lever's tab: the lock's body
                (12 mm below the tip) is then 3 mm below the loop, and the leg can slide over the tab to the holes."""
                return self.to_world(np.array([P.HOLE_C[0], P.LEN_LOOP + 0.0042, P.TAB_Z[1] + 0.002]))

            def entry(self, attempt):
                if attempt == 0:
                    self.lock_to(self.lock_target(self.start_tip()), 0.02)
                    self.L.hold(hold_k(0.5))
                self.when_settled(self.L, "left", lambda: self.entry_correct(attempt), tol=0.004)

            def entry_correct(self, attempt):
                self.lock_model = GraspModel.measure(robot, "left", lock)
                err = self.tip() - self.start_tip()
                note(f"ENTRY attempt {attempt}: tip error (mm) {np.round(err * 1000, 2)}")
                self.correction = self.correction + err
                if attempt < 5 and (abs(err[0]) > 0.0006 or abs(err[2]) > 0.0012):
                    self.lock_to(self.lock_target(self.start_tip()), 0.01)
                    self.L.hold(hold_k(0.5))
                    self.entry(attempt + 1)
                    return
                self.cur = self.start_tip()
                self.slide(0)

            def slide(self, k):
                """Slide the short leg over the tab toward the holes in 1.5 mm steps, keeping x and z from the
                measured tip; then lower it through both holes."""
                h, tip = self.hole(), self.tip()
                err = tip - self.cur
                self.correction = self.correction + 0.6 * np.array([err[0], 0.0, err[2]])
                if tip[1] - h[1] < 0.0004 or k > 40:
                    note(f"SLID over the holes after {k}: tip - hole (mm) {np.round((tip - h) * 1000, 2)}")
                    self.cur = np.array([h[0], h[1], self.cur[2]])
                    self.lower(0)
                    return
                self.cur = np.array([h[0], max(h[1], self.cur[1] - 0.0015), self.cur[2]])
                self.lock_to(self.lock_target(self.cur), 0.01, min_steps=2)
                self.L.hold(hold_k(0.1))
                self.at(len(self.L.steps), lambda: self.slide(k + 1))

            def lower(self, k):
                h, tip = self.hole(), self.tip()
                err = tip - self.cur
                self.correction = self.correction + 0.6 * np.array([err[0], err[1], 0.0])
                bottom = h[2] + P.LOOP_Z[0]
                if tip[2] < bottom - 0.006 or k > 40:
                    r = self.report()
                    note(f"LOWERED after {k}: tip {np.round((tip - h) * 1000, 2)} mm from the hole centre, "
                         f"through {bool(st.dbg['through_all'][0])}, milestone {r['milestone']}")
                    self.press_bend()
                    return
                self.cur = self.cur + np.array([0.0, 0.0, -0.0015])
                self.lock_to(self.lock_target(self.cur), 0.01, min_steps=2)
                self.L.hold(hold_k(0.1))
                self.at(len(self.L.steps), lambda: self.lower(k + 1))

            # ---- Arm B presses the shackle shut while arm A holds the lock's body -------------------------------
            def bend_top(self):
                p, q = self.lock_pose()
                return p + rotate(q, (0.0, 0.0, PV.bend_top(self.travel())))

            def press_bend(self):
                R = self.R
                self.sync()
                R.grip_to(CLOSED, hold_k(0.4))
                d = np.array([math.cos(PRESS_PITCH), 0.0, -math.sin(PRESS_PITCH)])
                g = self.bend_top() - FINGER_END * d + np.array([0.0, 0.0, 0.002])
                R.move(g + np.array([-0.04, -0.03, 0.04]), QPRESS, min_steps=hold_k(2.5))
                R.move(g + np.array([0.0, 0.0, 0.015]), speed=0.04)
                R.hold(hold_k(0.3))
                self.body0 = self.lock_pose()[0]
                self.press_z = g[2] + 0.002
                self.press_g = g
                self.when_settled(R, "right", lambda: self.press(0))

            def press(self, k):
                latched = bool(st.latched[0])
                if latched or k > 50:
                    note(f"PRESS done after {k}: travel {self.travel() * 1000:.1f} mm, latched {latched}, "
                         f"through {bool(st.latched_through[0])}, milestone {self.report()['milestone']}")
                    self.R.move(self.R.pos + ROOT_OFFSET + np.array([-0.02, 0.0, 0.03]), speed=0.03)
                    self.sync()
                    self.release()
                    return
                # Arm A holds the body against the push: it backs off by however far the body has moved down.
                drift = float(self.lock_pose()[0][2] - self.body0[2])
                self.sync()
                self.L.move(self.L.pos + ROOT_OFFSET - np.array([0.0, 0.0, drift]), speed=0.02, min_steps=1)
                self.press_z -= 0.002
                self.R.move(np.array([self.press_g[0], self.press_g[1], self.press_z]), speed=0.02, min_steps=2)
                self.sync()
                self.at(len(self.R.steps), lambda: self.press(k + 1))

            def release(self):
                L, R = self.L, self.R
                R.move(R_PARK, speed=0.08)
                L.grip_to(OPEN, hold_k(0.8))
                L.move(L.pos + ROOT_OFFSET + np.array([-0.06, 0.0, -0.01]), speed=0.04)
                L.move(L_PARK, speed=0.08)
                self.sync()
                L.hold(hold_k(4.0))

        l_pos, l_quat = grasp_point("left")
        r_pos, r_quat = grasp_point("right")
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        planner = Planner(L, R)
        settle = hold_k(1.0)
        L.hold(settle)
        R.hold(settle)
        planner.at(settle, planner.plan)

        q_rel = np.array(hb.Q_REL)
        tcp_extra = 0.1347 - 0.134
        q_arm = {s: view.scene[f"{s}_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float)
                 for s in ("left", "right")}
        worst = {"left": 0.0, "right": 0.0}

        def joint_action(side, pos_root, quat, grip):
            pos = pos_root + ROOT_OFFSET + origin
            tcp = pos + rotate(quat, (0.0, 0.0, -tcp_extra))
            rot = qmul(quat, q_rel)
            q, pe, _ = ik.solve(tcp, rot, q_arm[side], base_pos=station.BASES[side], iters=60)
            if pe > 0.004:  # stuck in a poor branch: retry from a few fixed seeds, keep the best
                for seed in IK_SEEDS:
                    q2, pe2, _ = ik.solve(tcp, rot, np.array(seed), base_pos=station.BASES[side], iters=200)
                    if pe2 < pe - 0.002:
                        q, pe = q2, pe2
            q_arm[side] = q
            if pe > 0.005 and pe > worst[side]:
                log(f"IK {side} k={state_k[0]} error {pe * 1000:.1f} mm at {np.round(pos - origin, 4)}")
            worst[side] = max(worst[side], pe)
            return np.r_[q, grip]

        self._L, self._R, self._planner, self._joint_action = L, R, planner, joint_action
        self._st, self._log = st, log
        self.worst = worst

    def next_action(self) -> np.ndarray:
        k = self.state_k[0]
        L, R = self._L, self._R
        self._planner.run_due(k)
        L.pad_to(len(R.steps))
        R.pad_to(len(L.steps))
        if k >= len(L.steps):
            L.hold(1)
            R.hold(1)
        if DEBUG and k % 40 == 0 and self._st.dbg:
            d = self._st.dbg
            self._log(f"DBG k={k} lo={[round(float(v), 4) for v in d['lo'][0]]} tipD={[round(float(v) * 1000, 1) for v in d['tips_d'][0]]} "
                      f"seated={bool(d['seated'][0].any())} F={float(d['force'][0]):.1f} jaw={float(d['p'][0]) * 1000:.2f}mm "
                      f"turns={float(d['turns'][0]):.3f} lever={math.degrees(float(d['lever'][0])):.1f} "
                      f"tog={math.degrees(float(d['tog'][0])):.1f} travel={float(d['travel'][0]) * 1000:.1f} "
                      f"lock={[round(float(v), 4) for v in d['lockp'][0]]} ms={int(self._st.scorer.milestone[0])}", flush=True)
        (lp, lq, lg), (rp, rq, rg) = L.steps[k], R.steps[k]
        self.state_k[0] = k + 1
        return np.concatenate([self._joint_action("left", lp, lq, lg), self._joint_action("right", rp, rq, rg)])

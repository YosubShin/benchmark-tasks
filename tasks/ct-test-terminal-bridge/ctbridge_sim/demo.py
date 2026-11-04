"""The scripted demo, closed loop.

1. Arm A (left), fingers straight down and closing along x (across the grip's 9 mm thickness), picks the KSS 8
   bridge off its holder by its grip, carries it over the named circuit's CT-side test sockets (toward the robot),
   lines the pins up with the socket mouths from the bridge's measured pose (re-measured up to four times), pushes
   it home, lets go and backs off to a waiting pose.
2. Meanwhile arm B (right), fingers down, picks the insulated screwdriver out of its stand by the handle and waits
   clear of arm A.
3. Once the bridge is in, arm B lowers the blade down the S1 terminal's slot into the gap between the slide's lugs
   (lined up from the blade tip's measured position), pushes the slide away from the robot past its mid-point,
   lifts out, does the same on S2, and backs off. The task then runs its after-release check.
4. While arm B works, arm A (now free) is a camera arm: its wrist camera looks down the length of the slot being
   worked from just beyond the row (relay side), so the slide's lugs and the blade going in between them are in its
   wrist view; it shifts to the second slot with the blade. Arm B holds the screwdriver with its fingers closing
   along the rail (the handle is square), which puts its own wrist camera on the relay side too, looking down the
   shaft into the slot.

It reads the simulator's state (the bridge's and the screwdriver's poses), like any scripted demo with privileged
state, and turns grasp-point targets into yam_arena's 14-D joint action through yam_arena's own IK every control
step. ``--policy ctbridge-demo`` (ctbridge_sim.policy) plays it through Inspect Robots. Import after Isaac Sim has
started.

Control (``-P control=true``): the bridge stays in its holder; arm B opens both named links straight away (the
hazard), and the after-release check is forced once arm B has backed off. It must score 0.
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import ik, station

WATCH_TILT, WATCH_SIDE, WATCH_DIST, WATCH_ROLL = 30.0, 9.0, 0.14, 260.0


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, control: bool = False, log=print):
        from ctbridge_sim import station as hb
        from ctbridge_sim.kit import CLOSED, DOWN, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, q_axis, qmul, rotate
        from ctbridge_sim.kit import grasp_point as grasp_point_world
        from ctbridge_sim.spec import STRIP_X, STRIP_Y
        from ctbridge_sim.task import bridge_start_pose, screwdriver_start_pose
        from ctbridge_sim.usd import ct_props as props
        from yam_arena.urdf import quat_xyzw_to_matrix

        view, task = STATION.env, STATION.task
        state_k = [0]
        self.state_k = state_k
        origin = view.scene.env_origins[0].cpu().numpy()  # desk-frame origin in the world
        robot = view.scene["robot"]
        st = task.state(view)
        layout = st.next_layout

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        for side, name in (("left", "left_robot"), ("right", "right_robot")):
            qj = view.scene[name].data.joint_pos.torch[0, :6].cpu().numpy()
            tcp, _ = ik.tcp_pose(qj, station.BASES[side])
            log(f"CHECK {side}: their TCP {np.round(tcp, 4)} our grasp point {np.round(grasp_point(side)[0] + origin, 4)}",
                flush=True)

        k10 = control_hz / 10.0

        def n(steps):
            """A step count of the 10 Hz original, at this control rate."""
            return max(1, round(steps * k10))

        MM = 0.001
        GRIPPER_OPEN_M = 0.04695
        Q_ACROSS = qmul(q_axis((0, 0, 1), -math.pi / 2), DOWN)
        """Fingers straight down, closing along x."""
        Q_SD = DOWN
        """Arm B's screwdriver grasp: fingers down, closing along y (the rail) on the square handle. Its wrist camera
        then sits on the far (relay) side of the hand and looks down the shaft at the blade tip in the slot."""
        IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
        GRASP_IN_BRIDGE = (0.0, 0.0, 15.0 * MM)
        """Grasp point on the grip, 15 mm above its underside: the finger ends stay 3 mm above the housing top."""
        GRASP_IN_SD = (0.0, 0.0, 0.072)
        """Grasp point on the screwdriver's handle, 72 mm above the blade's tip (the handle spans 58-128 mm)."""
        BLADE_Z = props.FLOOR_Z + 0.0008 + props.SLIDE_H * MM + 0.0018
        """The blade's tip 1.8 mm above the slide's bar, between its lugs (the arm sags about 1 mm under the tool)."""
        PUSH_TO_X = 10.0 * MM
        """Blade centre commanded at the end of the push away from the robot. The slide is fully open (5 mm, its end
        stop) with the blade's centre at 7.4 mm; the station arm lags 3-4 mm under load, so the command overshoots.
        At 9.4 mm the blade would meet the relay-side socket tube, which stops it."""
        GRIP_BRIDGE = 0.0030 / GRIPPER_OPEN_M
        """Finger target 3 mm from centre on the 9 mm grip (about 7.5 N per finger from the 5000 N/m finger drive)."""
        GRIP_SD = 0.0080 / GRIPPER_OPEN_M
        """Finger target 8 mm from centre on the 22 mm handle."""
        STRIP0 = np.array([STRIP_X, STRIP_Y, 0.0])
        L_WAIT = np.array([0.16, 0.24, 0.20])
        R_WAIT = np.array([0.18, -0.24, 0.20])
        WATCH = dict(tilt=WATCH_TILT, side=WATCH_SIDE, dist=WATCH_DIST, roll=WATCH_ROLL)
        """Arm A's camera view of a slot: camera WATCH_DIST from the blade tip, tilted WATCH_TILT deg from vertical
        toward the relay side and WATCH_SIDE deg toward the robot's left, image rolled WATCH_ROLL deg. The slot is 7.2 mm
        wide and 22 mm deep, so only a view along its length sees in; the bridge's grip blocks views from the robot's
        side, and the tilt keeps the line of sight clear of arm B's hand (picked by reach and clearance probes)."""
        cam_l = station.wrist_camera("left")
        cam_rot_in_gripper = quat_xyzw_to_matrix(np.array(cam_l.rot_xyzw))
        cam_pos_in_gripper = np.array(cam_l.pos)

        q_rel = np.array(hb.Q_REL)
        tcp_extra = 0.1347 - 0.134  # our grasp point sits this much further out than their TCP
        bridge = view.scene["bridge"]
        sd = view.scene["screwdriver"]
        s1, s2 = f"{layout.named}S1", f"{layout.named}S2"
        mid_y = (props.slot_y(s1) + props.slot_y(s2)) / 2 * MM
        target_mid = STRIP0 + np.array([-props.SOCKET_X * MM, mid_y, props.TOP_Z])

        def pose(obj):
            return (obj.data.root_pos_w.torch[0].cpu().numpy() - origin, obj.data.root_quat_w.torch[0].cpu().numpy())

        def watch_pose(name):
            """Arm A's grasp-point target (desk frame) and orientation that put its wrist camera over ``name``'s slot,
            aimed at the blade tip's working point."""
            tip = STRIP0 + np.array([0.0, props.slot_y(name) * MM, BLADE_Z])
            th, ph = math.radians(WATCH["tilt"]), math.radians(WATCH["side"])
            up = np.array([math.sin(th), math.sin(ph), math.cos(th)])
            up /= np.linalg.norm(up)
            cam_p = tip + WATCH["dist"] * up
            z = up  # OpenGL camera: looks along -z
            x = np.cross([1.0, 0.0, 0.0], z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            r = math.radians(WATCH["roll"])
            cam_r = np.c_[math.cos(r) * x + math.sin(r) * y, -math.sin(r) * x + math.cos(r) * y, z]
            g_rot = cam_r @ cam_rot_in_gripper.T  # their `gripper` link
            tcp = cam_p - g_rot @ cam_pos_in_gripper + g_rot @ np.array([0.0, 0.0, 0.134])
            quat = qmul(ik.matrix_to_quat_xyzw(g_rot), np.array([-q_rel[0], -q_rel[1], -q_rel[2], q_rel[3]]))
            return tcp + rotate(quat, (0.0, 0.0, tcp_extra)), quat

        def slides_mm():
            q = st.strip.data.joint_pos.torch[0, list(st.slide_joints.values())].cpu().numpy() * 1000
            return dict(zip(props.TERMINALS, np.round(q, 2).tolist()))

        class Planner:
            def __init__(self, L, R):
                self.L, self.R, self.events, self.k = L, R, {}, -1
                self.bridge_in = False
                self.watching = False
                self.sd_ready = False
                self.opening = False

            def at(self, step, fn):
                """Run fn at control step ``step`` (or the next step, if that one is already running)."""
                self.events.setdefault(max(step, self.k + 1), []).append(fn)

            def run_due(self, k):
                self.k = k
                for fn in self.events.pop(k, []):
                    fn()

            def when_settled(self, tl, side, fn, tries=10, tol=0.0035):
                def check():
                    lag = np.linalg.norm(grasp_point(side)[0] - (tl.pos + ROOT_OFFSET))
                    if lag < tol or check.tries <= 0:
                        if lag >= tol:
                            log(f"SETTLE timeout ({side}): lag {lag * 1000:.1f} mm", flush=True)
                        fn()
                    else:
                        check.tries -= 1
                        tl.hold(n(2))
                        self.at(len(tl.steps), check)
                check.tries = tries
                self.at(len(tl.steps), check)

            def sync(self):
                m = max(len(self.L.steps), len(self.R.steps))
                self.L.pad_to(m)
                self.R.pad_to(m)

            def report(self, tag):
                rep = task.episode_report(view, 0)
                log(f"{tag}: milestone {rep['milestone']} touch {rep['first_decoy_touch']} pins {rep['bridge_pins_in']} slides {slides_mm()} "
                    f"sem {rep['semantic_events']} con {rep['contact_events']}", flush=True)

            # 1. Arm A: the bridge ------------------------------------------------------------------------------
            def bridge_to(self, mouth_pos, speed):
                """Move the held bridge so its underside centre goes to mouth_pos (upright)."""
                self.L.move(self.model.grasp_for(mouth_pos, IDENTITY)[0], speed=speed)

            def plan_pick(self):
                start = np.array(bridge_start_pose()[:3])
                grasp, _ = GraspModel.nominal(Q_ACROSS, IDENTITY, GRASP_IN_BRIDGE).grasp_for(start, IDENTITY)
                L = self.L
                L.grip_to(OPEN, n(2))
                L.move(grasp + np.array([0, 0, 0.07]), Q_ACROSS, min_steps=n(30))
                L.hold(n(4))
                L.move(grasp + np.array([0, 0, 0.015]), speed=0.04)
                L.move(grasp, speed=0.015)
                self.when_settled(L, "left", self.grab_bridge, tries=8)

            def grab_bridge(self):
                L = self.L
                L.grip_to(GRIP_BRIDGE, n(12))
                L.move(L.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.045]), speed=0.03)
                self.when_settled(L, "left", self.carry)

            def carry(self):
                self.model = GraspModel.measure(robot, "left", bridge)
                log(f"BRIDGE grasp rel_pos (mm) {np.round(self.model.rel_pos * 1000, 1)}", flush=True)
                self.report("LIFTED")
                self.correction = np.zeros(3)
                # Carry high (pin tips 3 cm over the housings) so nothing passes low over the other circuits.
                self.bridge_to(target_mid + np.array([0.0, 0.0, 0.05]), 0.06)
                self.above = target_mid + np.array([0.0, 0.0, 0.004])
                self.bridge_to(self.above, 0.03)
                self.L.hold(n(6))
                self.when_settled(self.L, "left", lambda: self.align(0))

            def align(self, attempt):
                pos, quat = pose(bridge)
                err = pos - self.above
                tilt = math.degrees(2 * math.acos(min(1.0, abs(float(quat[3])))))
                log(f"ALIGN attempt {attempt}: mouth error (mm) {np.round(err * 1000, 2)} tilt {tilt:.1f} deg",
                    flush=True)
                self.correction = self.correction + err
                if attempt < 4 and max(abs(err[0]), abs(err[1])) > 0.0004:
                    self.bridge_to(self.above - self.correction, 0.008)
                    self.L.hold(n(6))
                    self.when_settled(self.L, "left", lambda: self.align(attempt + 1))
                    return
                home = target_mid + np.array([0.0, 0.0, -0.0015])
                self.bridge_to(home - np.array([self.correction[0], self.correction[1], 0.0]), 0.008)
                self.L.hold(n(10))
                self.when_settled(self.L, "left", self.inserted, tries=6)

            def inserted(self):
                pos, _ = pose(bridge)
                log(f"INSERTED: bridge {np.round(pos, 4)} target {np.round(target_mid, 4)}", flush=True)
                self.report("INSERTED")
                L = self.L
                L.grip_to(OPEN, n(10))
                L.move(L.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.06]), speed=0.04)
                # Arm A becomes the camera arm: up, then over to its view of S1's slot (fingers closed, out of the way).
                w_pos, w_quat = watch_pose(s1)
                pre = w_pos + np.array([-0.03, 0.04, 0.03])  # approached from the robot's left (a straight drop
                # from above passes joint 4's limit), away from arm B
                L.move(np.array([L.pos[0] + ROOT_OFFSET[0], L.pos[1] + ROOT_OFFSET[1], pre[2]]), speed=0.06)
                L.grip_to(CLOSED, n(6))
                L.move(pre, w_quat, speed=0.06, min_steps=n(30))
                L.move(w_pos, speed=0.03)
                L.hold(n(4))
                self.watching = True
                self.bridge_in = True
                self.at(len(L.steps), self.maybe_open)

            # 2. Arm B: the screwdriver -------------------------------------------------------------------------
            def plan_pick_sd(self):
                start = np.array(screwdriver_start_pose()[:3])
                grasp, _ = GraspModel.nominal(Q_SD, IDENTITY, GRASP_IN_SD).grasp_for(start, IDENTITY)
                R = self.R
                R.grip_to(OPEN, n(4))
                R.move(grasp + np.array([0, 0, 0.07]), Q_SD, min_steps=n(30))
                R.move(grasp + np.array([0, 0, 0.01]), speed=0.04)
                R.move(grasp, speed=0.015)
                self.when_settled(R, "right", self.grab_sd, tries=8)

            def grab_sd(self):
                R = self.R
                R.grip_to(GRIP_SD, n(12))
                R.move(R.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.095]), speed=0.04)
                self.when_settled(R, "right", self.sd_lifted)

            def sd_lifted(self):
                self.sd_model = GraspModel.measure(robot, "right", sd)
                log(f"SD grasp rel_pos (mm) {np.round(self.sd_model.rel_pos * 1000, 1)}", flush=True)
                self.R.hold(n(2))
                self.sd_ready = True
                self.when_settled(self.R, "right", self.maybe_open)

            def tip_to(self, tip_pos, speed):
                self.R.move(self.sd_model.grasp_for(tip_pos, IDENTITY)[0], speed=speed)

            def maybe_open(self):
                if self.sd_ready and (self.bridge_in or control) and not self.opening:
                    self.opening = True
                    self.sync()
                    self.open_link(s1, lambda: self.open_link(s2, self.done))

            # 3. Arm B: the slides ------------------------------------------------------------------------------
            def open_link(self, name, then):
                gap = STRIP0 + np.array([0.0, props.slot_y(name) * MM, BLADE_Z])
                self.sd_corr = np.zeros(3)
                transit = np.array([gap[0], gap[1], props.TOP_Z + props.GRIP[2] * MM + 0.025])
                if self.watching and name != s1:  # the camera arm follows the blade to this slot
                    self.L.pad_to(len(self.R.steps))
                    self.L.move(watch_pose(name)[0], speed=0.02)
                self.tip_to(transit, 0.05)
                self.tip_to(gap + np.array([0.0, 0.0, 0.028]), 0.03)
                self.R.hold(n(4))
                self.when_settled(self.R, "right", lambda: self.blade_align(name, gap, then, 0))

            def blade_align(self, name, gap, then, attempt):
                above = gap + np.array([0.0, 0.0, 0.028])
                err = pose(sd)[0] - above
                log(f"BLADE {attempt}: tip error (mm) {np.round(err * 1000, 2)}", flush=True)
                self.sd_corr = self.sd_corr + np.array([err[0], err[1], 0.0])
                if attempt < 4 and max(abs(err[0]), abs(err[1])) > 0.0004:
                    self.tip_to(above - self.sd_corr, 0.008)
                    self.R.hold(n(4))
                    self.when_settled(self.R, "right", lambda: self.blade_align(name, gap, then, attempt + 1))
                    return
                self.tip_to(gap - self.sd_corr, 0.012)
                self.R.hold(n(4))
                self.when_settled(self.R, "right", lambda: self.blade_push(name, gap, then, 0), tries=6)

            def blade_push(self, name, gap, then, attempt):
                tip = pose(sd)[0]
                log(f"BLADE down: tip {np.round(tip, 4)} gap {np.round(gap, 4)}", flush=True)
                end = gap + np.array([PUSH_TO_X + 0.005 * attempt, 0.0, 0.0])
                self.tip_to(end - self.sd_corr, 0.006)
                self.R.hold(n(15))

                def out():
                    q = slides_mm()
                    self.report(f"PUSHED {name}")
                    if q[name] < 3.5 and attempt < 3:  # the arm lagged: command 5 mm further
                        self.blade_push(name, gap, then, attempt + 1)
                        return
                    self.tip_to(pose(sd)[0] + np.array([0.0, 0.0, 0.035]), 0.03)
                    self.R.hold(n(2))
                    self.when_settled(self.R, "right", then, tries=4)
                self.when_settled(self.R, "right", out, tries=6)

            def done(self):
                self.report("LINKS DONE")
                R = self.R
                R.move(R_WAIT, speed=0.08)
                if self.watching:  # the camera arm backs off up and away from the row
                    self.L.pad_to(len(self.R.steps) - n(25))
                    self.L.move(self.L.pos + ROOT_OFFSET + np.array([-0.03, 0.04, 0.03]), speed=0.04)
                    self.L.move(L_WAIT, Q_ACROSS, speed=0.10, min_steps=n(20))
                R.hold(n(30))
                self.sync()
                if control:
                    self.at(len(R.steps) - n(20), lambda: task.force_function_test(view))

        l_pos, l_quat = grasp_point("left")
        r_pos, r_quat = grasp_point("right")
        dt = 1.0 / control_hz
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        R.move(R_WAIT, Q_SD, min_steps=n(30))
        planner = Planner(L, R)
        settle = n(20)
        L.hold(settle)
        R.hold(max(0, settle - len(R.steps)))
        planner.at(settle, planner.plan_pick_sd)
        if not control:
            planner.at(settle, planner.plan_pick)
        self.planner = planner

        q_arm = {s: view.scene[f"{s}_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float)
                 for s in ("left", "right")}
        worst = {"left": 0.0, "right": 0.0}

        def joint_action(side, pos_root, quat, grip):
            pos = pos_root + ROOT_OFFSET + origin  # world
            tcp = pos + rotate(quat, (0.0, 0.0, -tcp_extra))
            rot = qmul(quat, q_rel)
            q, pe, _ = ik.solve(tcp, rot, q_arm[side], base_pos=station.BASES[side], iters=60)
            q_arm[side] = q
            if pe > 0.005 and pe > worst[side]:
                log(f"IK {side} k={state_k[0]} error {pe * 1000:.1f} mm at {np.round(pos - origin, 4)}", flush=True)
            worst[side] = max(worst[side], pe)
            return np.r_[q, grip]

        self._L, self._R, self._joint_action = L, R, joint_action
        self.worst = worst
        self._last_ms = -1
        self._task, self._view = task, view
        self._log = log

    def next_action(self) -> np.ndarray:
        k = self.state_k[0]
        L, R = self._L, self._R
        self.planner.run_due(k)
        L.pad_to(len(R.steps))
        R.pad_to(len(L.steps))
        if k >= len(L.steps):
            L.hold(1)
            R.hold(1)
        rep = self._task.episode_report(self._view, 0)
        if rep["milestone"] != self._last_ms:
            self._log(f"MILESTONE {rep['milestone']} at step {k}", flush=True)
            self._last_ms = rep["milestone"]
        (lp, lq, lg), (rp, rq, rg) = L.steps[k], R.steps[k]
        self.state_k[0] = k + 1
        return np.concatenate([self._joint_action("left", lp, lq, lg), self._joint_action("right", rp, rq, rg)])

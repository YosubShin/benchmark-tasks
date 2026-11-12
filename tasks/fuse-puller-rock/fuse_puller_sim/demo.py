"""The scripted demo, closed loop (arm B, the right arm, does everything; arm A stays at ready).

1. Fingers down, arm B pinches the puller's two handle ends on its rest lightly (the jaws stay open), lifts it and
   carries it over F2, levels it and lowers the open jaws round F2's body between its clips.
2. It squeezes the handles fully (the station gripper's 50 N): the jaw shells clamp the fuse body.
3. Rock: it swings the puller's handle end up along an arc about F2's far ferrule (the fuse turns with the
   jaws, the far clip is the pivot), a few degrees ahead of the fuse's measured tilt, pushing further ahead
   whenever the fuse stalls, until the near ferrule is out of its clip; then it keeps turning while lifting, which
   levers the far ferrule out.
4. It lifts the fuse clear, carries it over the tray and relaxes the squeeze: the jaws open and the fuse drops.

It reads the simulator's state (the fuse and puller poses), like any scripted demo with privileged state, and turns
grasp-point targets into yam_arena's 14-D joint action through yam_arena's own IK every control step.
``--policy fuse-puller-demo`` (fuse_puller_sim.policy) plays it through Inspect Robots. Import after Isaac Sim has
started.

Control (``FPR_CONTROL=1``): steps 1-2 as above, then arm B lets go of the handles (the open jaws stay round the
fuse), pinches the puller's arms directly over the jaws with the full 50 N, which closes the jaws on the fuse with
no lever, and pulls straight up (commanding up to 16 cm above the start). A straight pull needs both clips' grip at
once.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, log=print):
        from fuse_puller_sim import station as hb
        from fuse_puller_sim.kit import DOWN, ROOT_OFFSET, ArmTimeline, GraspModel, conj, q_axis, qmul, rotate
        from fuse_puller_sim.kit import grasp_point as grasp_point_world
        from fuse_puller_sim.spec import TRAY_XY
        from fuse_puller_sim.usd import puller_props as props

        view, task = STATION.env, STATION.task
        state_k = [0]
        self.state_k = state_k
        origin = view.scene.env_origins[0].cpu().numpy()
        robot = view.scene["robot"]
        st = task.state(view)
        control = os.environ.get("FPR_CONTROL") == "1"

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        k10 = control_hz / 10.0

        def n(steps):
            return max(1, round(steps * k10))

        G_OPEN = 0.80
        G_HOLD = 0.30
        """Light pinch on the handles 85 mm from the hinge: about 4 N, the jaws still about 1.5 mm open each side."""
        G_SQUEEZE = 0.0
        ARM_ZC = 0.5 * (props.ARM_Z[0] + props.ARM_Z[1])
        GRASP_IN_PULLER = np.array([props.GRASP_X, 0.0, ARM_ZC])
        JAW_GRASP_IN_PULLER = np.array([props.JAW_X, 0.0, ARM_ZC])
        HOVER = 0.045
        R_PARK = np.array([0.18, -0.20, 0.16])
        named = int(st.named[0])
        fuse = st.fuses[named]
        puller = st.puller
        cap_off = props.FUSE_L / 2 - props.CAP_L / 2

        def puller_pose():
            return puller.data.root_pos_w.torch[0].cpu().numpy() - origin, puller.data.root_quat_w.torch[0].cpu().numpy()

        def fuse_pose():
            return fuse.data.root_pos_w.torch[0].cpu().numpy() - origin, fuse.data.root_quat_w.torch[0].cpu().numpy()

        def fuse_axis():
            _, q = fuse_pose()
            return rotate(q, (0.0, 0.0, 1.0))

        def tilt_deg():
            """The named fuse's tilt, near end up positive."""
            a = fuse_axis()
            return math.degrees(math.asin(max(-1.0, min(1.0, -a[2]))))

        def ferrules_out():
            p, _ = fuse_pose()
            a = fuse_axis()
            caps = [p - cap_off * a, p + cap_off * a]
            clips = [np.array(props.clip_centre(named, e)) for e in range(2)]
            return [min(np.linalg.norm(c - cl) for cl in clips) > 0.007 for c in caps]

        def hand_quat():
            """Fingers down, closing along y (the arms part along y)."""
            return DOWN

        class Planner:
            def __init__(self, R):
                self.R, self.events = R, {}
                self.model = None

            def at(self, step, fn):
                self.events.setdefault(step, []).append(fn)

            def run_due(self, k):
                for fn in self.events.pop(k, []):
                    fn()

            def lag(self):
                return np.linalg.norm(grasp_point("right")[0] - (self.R.pos + ROOT_OFFSET)) * 1000

            def when_settled(self, fn, tries=15, tol=0.003):
                R = self.R

                def check():
                    lag = np.linalg.norm(grasp_point("right")[0] - (R.pos + ROOT_OFFSET))
                    if lag < tol or check.tries <= 0:
                        if lag >= tol:
                            log(f"SETTLE timeout: lag {lag * 1000:.1f} mm", flush=True)
                        fn()
                    else:
                        check.tries -= 1
                        R.hold(n(2))
                        self.at(len(R.steps), check)
                check.tries = tries
                self.at(len(R.steps), check)

            def report(self, tag):
                rep = task.episode_report(view, 0)
                log(f"{tag}: milestone {rep['milestone']} tilt {tilt_deg():.1f} out {ferrules_out()} "
                    f"first {rep['first_ferrule_out']} progress {rep['progress_milestone']} cap {rep['cap']} "
                    f"hazards {rep['hazard_events']} other {rep['other_events']}",
                    flush=True)

            # 1. Pick the puller off its rest -----------------------------------------------------------------
            def plan_pick(self):
                R = self.R
                h_pos, h_quat = puller_pose()
                g = h_pos + rotate(h_quat, GRASP_IN_PULLER)
                log(f"PULLER hinge {np.round(h_pos, 4)}; handle grasp {np.round(g, 4)}", flush=True)
                R.grip_to(G_OPEN, n(2))
                R.move(g + np.array([0.0, 0.0, HOVER]), hand_quat(), min_steps=n(30))
                R.move(g + np.array([0.0, 0.0, 0.012]), speed=0.04)
                R.move(g, speed=0.015)
                R.hold(n(3))
                self.when_settled(self.pinch)

            def pinch(self):
                log(f"AT HANDLES: lag {self.lag():.1f} mm", flush=True)
                self.R.grip_to(G_HOLD, n(10))
                self.at(len(self.R.steps), self.carry)

            def carry(self):
                R = self.R
                h_pos, h_quat = puller_pose()
                self.model = GraspModel.measure(robot, "right", puller)
                self.model.rel_pos = self.model.rel_pos  # (grasp frame)
                log(f"HELD: hinge {np.round(h_pos, 4)}", flush=True)
                R.move(R.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.04]), speed=0.02)
                target = np.array(props.puller_on_fuse_pose(named)[:3])
                g, q = self.grasp_for(target + np.array([0.0, 0.0, HOVER]), np.array(props.PULLER_QUAT))
                R.move(g, q, speed=0.04)
                R.hold(n(4))
                self.when_settled(lambda: self.lower(0))

            def grasp_for(self, hinge_pos, hinge_quat):
                g_pos, g_quat = self.model.grasp_for(hinge_pos, hinge_quat)
                return g_pos - origin * 0, g_quat

            def lower(self, k, extra=0.0):
                """Re-measure the grasp, then lower the open jaws round the fuse body; push further while the
                puller's hinge stays above its seated height."""
                R = self.R
                self.model = GraspModel.measure(robot, "right", puller)
                f_pos, _ = fuse_pose()
                target = np.array([f_pos[0] + props.JAW_X, f_pos[1], f_pos[2]])
                h_pos, _ = puller_pose()
                log(f"LOWER {k}: hinge {np.round(h_pos, 4)} target {np.round(target, 4)} lag {self.lag():.1f} mm",
                    flush=True)
                if k == 0:
                    g, q = self.grasp_for(target + np.array([0.0, 0.0, 0.012]), np.array(props.PULLER_QUAT))
                    R.move(g, q, speed=0.02)
                    R.hold(n(3))
                    self.when_settled(lambda: self.lower(1))
                    return
                high = h_pos[2] - target[2]
                if k > 1 and (abs(high) < 0.0012 or k > 5):
                    self.squeeze()
                    return
                if k > 1:
                    extra = extra + max(-0.002, min(0.002, high))
                g, q = self.grasp_for(target + np.array([0.0, 0.0, 0.0005 - extra]), np.array(props.PULLER_QUAT))
                R.move(g, q, speed=0.01)
                R.hold(n(4))
                self.when_settled(lambda: self.lower(k + 1, extra), tries=6)

            def squeeze(self):
                h_pos, _ = puller_pose()
                f_pos, _ = fuse_pose()
                log(f"SQUEEZE: hinge-fuse {np.round((h_pos - f_pos) * 1000, 1)} mm (want [{props.JAW_X * 1000:.0f}, 0, 0])",
                    flush=True)
                self.R.grip_to(G_SQUEEZE, n(12))
                self.at(len(self.R.steps), self.after_squeeze)

            def after_squeeze(self):
                self.report("SQUEEZED")
                self.model = GraspModel.measure(robot, "right", puller)
                if control:
                    self.control_regrip()
                    return
                f_pos, _ = fuse_pose()
                self.pivot = f_pos + cap_off * fuse_axis()  # the far ferrule's centre
                self.g0 = grasp_point("right")[0]
                self.phase2 = False
                self.rock(0.0, 3.0, 0, tilt_deg(), 0.0)

            # 3. Rock it out -------------------------------------------------------------------------------------
            def tong_target(self, phi_deg, lift):
                """Grasp target with the puller swung up by phi about the far ferrule (near end up) and raised by
                lift. The hand stays fingers-down: the pinch is the pin the puller turns about."""
                rq = q_axis((0.0, 1.0, 0.0), math.radians(phi_deg))
                return self.pivot + rotate(rq, self.g0 - self.pivot) + np.array([0.0, 0.0, lift])

            def rock(self, phi_cmd, lead, k, tilt_prev, lift):
                """Lift the handles along an arc about the far ferrule, a few degrees ahead of the fuse's measured
                tilt (further ahead while it stalls): the near ferrule levers out of its clip. Then keep swinging the
                puller up and lift: with the near end free, the lift ends up pulling straight on the far clip."""
                tilt = tilt_deg()
                out = ferrules_out()
                log(f"ROCK {k}: cmd {phi_cmd:.1f} tilt {tilt:.1f} lead {lead:.1f} lift {lift * 1000:.0f} mm out {out} "
                    f"lag {self.lag():.1f} mm", flush=True)
                if all(out) or k > 90:
                    self.report("OUT")
                    self.R.hold(n(4))
                    self.at(len(self.R.steps), self.to_tray)
                    return
                if out[0] and not self.phase2:
                    self.phase2 = True
                    self.report("NEAR END OUT")
                stalled = k and tilt - tilt_prev < 0.4
                if stalled:
                    lead = min(lead + 1.5, 12.0)
                elif tilt - tilt_prev > 2.0:
                    lead = max(lead - 1.0, 3.0)
                cap = 30.0 if not self.phase2 else 70.0
                phi_cmd = min(max(phi_cmd - 1.0, tilt) + lead, cap)
                if self.phase2:
                    lift = min(lift + 0.003, 0.05)
                elif stalled and phi_cmd >= cap - 0.1:
                    # The arm is soft: lift the handles further (above the arc) while the fuse does not turn.
                    lift = min(lift + 0.005, 0.04)
                R = self.R
                start = R.pos + ROOT_OFFSET
                g = self.tong_target(phi_cmd, lift)
                m = n(5)
                for i in range(1, m + 1):
                    R.path([(start + (g - start) * i / m, hand_quat())])
                R.hold(n(3))
                self.at(len(R.steps), lambda: self.rock(phi_cmd, lead, k + 1, tilt, lift))

            # 4. To the tray ---------------------------------------------------------------------------------------
            def to_tray(self):
                R = self.R
                here = grasp_point("right")[0]
                R.move(np.array([here[0], here[1], min(here[2] + 0.03, 0.15)]), hand_quat(), speed=0.04)
                tx, ty = TRAY_XY
                f_pos, _ = fuse_pose()
                g_now, _ = grasp_point("right")
                off = f_pos - g_now  # fuse centre from the grasp point
                above = np.array([tx - off[0], ty - off[1], 0.11])
                R.move(above, speed=0.06)
                R.hold(n(6))
                self.when_settled(self.drop, tries=10, tol=0.006)

            def drop(self):
                self.report("OVER TRAY")
                self.R.grip_to(G_HOLD, n(16))
                self.at(len(self.R.steps), self.after_drop)

            def after_drop(self):
                self.report("DROPPED")
                R = self.R
                R.move(R.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.04]), speed=0.04)
                R.hold(n(40))
                self.at(len(R.steps), lambda: self.report("END"))

            # Control: straight pull --------------------------------------------------------------------------------
            def control_regrip(self):
                R = self.R
                R.grip_to(G_OPEN, n(8))
                here = R.pos + ROOT_OFFSET
                R.move(here + np.array([0.0, 0.0, 0.03]), speed=0.03)
                h_pos, h_quat = puller_pose()
                g = h_pos + rotate(h_quat, JAW_GRASP_IN_PULLER)
                log(f"CONTROL: regrip over the jaws at {np.round(g, 4)}", flush=True)
                R.move(g + np.array([0.0, 0.0, 0.03]), hand_quat(), speed=0.05)
                R.move(g + np.array([0.0, 0.0, 0.002]), speed=0.01)
                R.hold(n(4))
                self.when_settled(self.control_pinch)

            def control_pinch(self):
                log(f"CONTROL AT JAWS: lag {self.lag():.1f} mm", flush=True)
                self.R.grip_to(G_SQUEEZE, n(12))
                self.at(len(self.R.steps), lambda: self.control_pull(0))

            def arm_force(self):
                """Estimated force (N, world) the right arm's joint PD pushes on the hand with: the PD torques
                (stiffness times the target error, capped at the effort limits; the gravity feed-forward excluded)
                through the arm's Jacobian, from finite differences of yam_arena's own forward kinematics."""
                arm = view.scene["right_robot"]
                q = arm.data.joint_pos.torch[0, :6].cpu().numpy().astype(float)
                qt = arm.data.joint_pos_target.torch[0, :6].cpu().numpy().astype(float)
                kp = np.array([80.0, 80.0, 80.0, 10.0, 10.0, 10.0])
                tau = np.clip(kp * (qt - q), -np.array([27.0] * 3 + [7.0] * 3), np.array([27.0] * 3 + [7.0] * 3))
                base = station.BASES["right"]
                p0, r0 = ik.tcp_pose(q, base)
                jac = np.zeros((6, 6))
                for i in range(6):
                    dq = q.copy()
                    dq[i] += 1e-5
                    p1, r1 = ik.tcp_pose(dq, base)
                    jac[:3, i] = (p1 - p0) / 1e-5
                    dr = r1 @ r0.T
                    jac[3:, i] = np.array([dr[2, 1] - dr[1, 2], dr[0, 2] - dr[2, 0], dr[1, 0] - dr[0, 1]]) / 2e-5
                wrench = np.linalg.lstsq(jac.T, tau, rcond=None)[0]
                return wrench[:3]

            def control_pull(self, k):
                R = self.R
                f_pos, _ = fuse_pose()
                fz = self.arm_force()[2]
                self.report(f"PULL {k}: fuse dz {(f_pos[2] - props.AXIS_Z) * 1000:.1f} mm, lag {self.lag():.1f} mm, "
                            f"arm lift about {fz:.0f} N")
                self.max_lift = max(getattr(self, "max_lift", 0.0), fz)
                if k >= 32 or all(ferrules_out()):
                    R.hold(n(20))
                    self.at(len(R.steps), lambda: self.report(f"CONTROL END (largest arm lift about {self.max_lift:.0f} N)"))
                    return
                R.move(R.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.005]), speed=0.01)
                R.hold(n(6))
                self.at(len(R.steps), lambda: self.control_pull(k + 1))

        l_pos, l_quat = grasp_point("left")
        r_pos, r_quat = grasp_point("right")
        dt = 1.0 / control_hz
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        planner = Planner(R)
        settle = n(10)
        L.hold(settle)
        R.hold(settle)
        planner.at(settle, planner.plan_pick)
        self.planner = planner

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
        self._conj = conj

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
        key = (rep["milestone"], rep["progress_milestone"], rep["hazard"], rep.get("cap", 100))
        if key != self._last_ms:
            self._log(f"MILESTONE {rep['milestone']} at step {k} (progress {rep['progress_milestone']}, hazard "
                      f"{rep['hazard_events']} from step {rep['hazard_step']}, cap {rep.get('cap', 100)})", flush=True)
            self._last_ms = key
        (lp, lq, lg), (rp, rq, rg) = L.steps[k], R.steps[k]
        self.state_k[0] = k + 1
        return np.concatenate([self._joint_action("left", lp, lq, lg), self._joint_action("right", rp, rq, rg)])

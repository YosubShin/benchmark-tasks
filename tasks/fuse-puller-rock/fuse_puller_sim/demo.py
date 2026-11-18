"""The scripted demo, closed loop on robot-observable inputs only (arm B, the right arm, does everything; arm A stays
at ready).

What it reads, every control step: the observation Inspect Robots hands any policy, i.e. the three RGB images
(``top_cam``, ``left_cam``, ``right_cam``) and the 14-D ``joint_pos`` state. Nothing else comes from the simulator:
no fuse or puller pose, no record of which fuse is named, no joint targets, no scorer state. Fixed knowledge it uses:
the documented layout (task.json ``setup``; ``fuse_puller_sim.layout``: the enclosure's walls, the blocks, the fuses' seats and clips, the
labels, the puller's holder and its geometry, the tray), the station's published camera calibration and the arm's URDF
(``yam_arena.station``, ``yam_arena.urdf``), as a real robot's script would.

How it perceives each thing:
- Which fuse is F2: in the first top-camera frames, the colour of the three circuit labels on the far wire duct's
  cover above the blocks (white, orange, green); the block with the orange label is F2's. The same frame checks that the yellow
  puller stands in its holder.
- The puller in the hand: arm B pinches the handles of the puller standing in its documented holder; how it sits in
  the hand follows from forward kinematics of arm B's joints at that moment, and from then on its pose follows the
  joints.
- Lowering the jaws across F2: F2 in its documented clips; arm B's joint readings show when the jaws' hood stops on
  the fuse body (the predicted hinge stays above its target), and it pushes a little further.
- Rocking the fuse out: the puller and the fuse in its jaws turn together about F2's far ferrule (the pivot;
  documented), so the tip is the angle of the pinch point (arm B's joints) about it. The hand moves along that arc a
  few degrees ahead of the measured angle, further ahead while it stalls; past 16 deg the near ferrule is out of
  its clip (it is 28 mm from the pivot), and the demo tips on to 40 deg while lifting, then lifts further and turns
  the puller upright again, moving it toward the tray, which draws the far ferrule out of its clip.
- Over the tray and dropping: the documented tray; the puller turned upright again and its jaws placed over the tray
  from the puller's geometry; the squeeze is relaxed and the jaws open.
- The enclosure: its documented walls (rim height); the puller, and later the fuse in its jaws, are carried in and
  out over the right wall at a fixed height above the rim.

1. Fingers forward and 30 deg down, arm B pinches the handles of the puller standing in its holder (on the bench right
   of the enclosure) lightly (the jaws stay open), lifts it until its jaw tips are 25 mm above the enclosure's rim,
   carries it over the right wall, pointing straight down (perpendicular to the fuses), to above F2, and lowers the
   open jaws across F2's body between its clips.
2. It squeezes the handles fully (the station gripper's 50 N): the jaw shells close round the fuse body.
3. Rock: it tips the handles away from the robot along an arc about F2's far ferrule, turning the hand with the
   puller (the fuse turns with the jaws, the far clip is the pivot), a few degrees ahead of the measured tip,
   further ahead whenever it stalls, until the near ferrule is out of its clip; then it tips on and lifts.
4. It turns the puller upright again while moving it up and a third of the way toward the tray, which draws the far
   ferrule out of its clip, lifts the fuse in the jaws clear of the rim, carries it over the right wall to above the
   tray, lowers it and relaxes the squeeze: the jaws open and the fuse drops.

``--policy fuse-puller-demo`` (fuse_puller_sim.policy) plays it; grasp-point targets become yam_arena's 14-D joint
action through yam_arena's own IK every control step.

Control (``FPR_CONTROL=1``): steps 1-2 as above, then arm B lets go of the handles (the open jaws stay round the
fuse), pinches the puller's smooth arms just above the jaws with the full 50 N (fingers 50 deg down), which closes the jaws on the fuse,
and pulls straight up without tipping (commanding up to 16 cm above the start). A straight pull needs both clips' grip at
once. The arm's lifting force is estimated from its joint readings and the joint targets the demo itself sent.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station
from yam_arena.ik import matrix_to_quat_xyzw

from fuse_puller_sim import layout as props
from fuse_puller_sim import vision as V
from fuse_puller_sim.kit import DOWN, ROOT_OFFSET, ArmTimeline, GraspModel, conj, q_axis, qmul, rotate, slerp

Q_REL = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
"""yam_arena's ``gripper`` frame = our flange frame * Q_REL."""
TCP_EXTRA = 0.1347 - 0.134
"""Our grasp point (between the finger pads) sits this much further out than yam_arena's TCP."""


class Demo:
    """``next_action(observation)`` gives the 14-D joint action for the next control step."""

    def __init__(self, control_hz: float = 20.0, speed: float = 0.10, log=print):
        TRAY_XY = props.TRAY_XY
        obs_box = [None]
        q_target_box = [{}]
        self._obs_box, self._q_target_box = obs_box, q_target_box
        state_k = [0]
        self.state_k = state_k
        control = os.environ.get("FPR_CONTROL") == "1"

        def q_meas(side):
            return V.split_state(obs_box[0].state["joint_pos"])[side][0]

        def grasp_point(side):
            """Measured (desk position, flange quaternion) of an arm's grasp point: forward kinematics of its joints."""
            p, r = V.tcp_desk(q_meas(side), side)
            return p + r @ np.array([0.0, 0.0, TCP_EXTRA]), qmul(matrix_to_quat_xyzw(r), conj(Q_REL))

        self._grasp_point, self._q_meas = grasp_point, q_meas
        k10 = control_hz / 10.0

        def n(steps):
            return max(1, round(steps * k10))

        G_OPEN = 0.45
        """43 mm between the fingers: clear of the open handles (22 mm)."""
        G_HOLD = 0.200
        """Light pinch on the handles 40 mm from the hinge: about 6 N, the jaws still about 4.5 mm open each side."""
        G_SQUEEZE = 0.0
        GRASP_IN_PULLER = np.array([props.GRASP_X, 0.0, 0.0])
        SMOOTH_IN_PULLER = np.array([props.SMOOTH_GRASP_X, 0.0, 0.0])
        JAW_IN_PULLER = np.array([props.JAW_X, 0.0, 0.0])
        PUQ = np.array(props.PULLER_QUAT)
        HOVER = 0.030
        cap_off = props.FUSE_L / 2 - props.CAP_L / 2
        named_box = [None]
        CARRY_HINGE_Z = props.WALL_TOP + 0.025 + props.TOP_X
        """Carried over the enclosure's walls with the puller upright, its hinge this high: the jaw tips (and a fuse
        in the jaws) 25 mm above the rim."""
        DRAW_FRACTION = 0.35
        NEAR_OUT_DEG = 16.0
        """The near ferrule is 28 mm from the far one: tipped 16 deg about it, it is 8 mm from its clip's centre."""
        TIP_MAX_DEG = 40.0
        PITCH0_DEG = 30.0
        """The hand's fingers point forward and 30 deg down; the upright puller crosses the finger pads."""

        def hand_quat(phi_deg=0.0):
            """Fingers forward, PITCH0 + phi below the horizontal, closing along y (the arms part along y)."""
            return qmul(q_axis((0.0, 1.0, 0.0), math.radians(phi_deg + PITCH0_DEG - 90.0)), DOWN)

        def finger_dir(q):
            return rotate(q, (0.0, 0.0, -1.0))

        class Planner:
            def __init__(self, R):
                self.R, self.events = R, {}
                self.model = None
                self.rest = np.array(props.puller_rest_pose()[:3])

            @property
            def named(self):
                return named_box[0]

            def survey(self, tries=10):
                """Top camera, first frames: which block's circuit label is orange (F2), and the puller in its holder.
                Looks again on the next steps if a frame does not show exactly one orange label (the first rendered
                frames can lag the scene)."""
                try:
                    self._survey()
                except RuntimeError as e:
                    if tries <= 0:
                        raise
                    log(f"TOP CAM: {e}; looking again", flush=True)
                    self.at(self.k_now + 2, lambda: self.survey(tries - 1))

            def _survey(self):
                im = np.asarray(obs_box[0].images["top_cam"])
                cam = V.Cam.top(im.shape[1], im.shape[0])
                seen = [V.colour_class(im, cam, props.label_world(s), props.LABEL_RGB) for s in range(3)]
                want = props.CIRCUITS[props.NAMED][2]
                log(f"TOP CAM: circuit labels {seen}", flush=True)
                hits = [s for s in range(3) if seen[s] == want]
                if len(hits) != 1:
                    raise RuntimeError(f"could not find the one {want} label")
                named_box[0] = hits[0]
                top = self.rest + rotate(PUQ, np.array([0.004, 0.0, 0.0]))
                r = V.find_blob(im, cam, top, V.YELLOW, top[2], max_px=25)
                if r is None or np.linalg.norm(r["xyz"][:2] - top[:2]) > 0.03:
                    raise RuntimeError("the puller is not in its holder")
                log(f"TOP CAM: block {hits[0] + 1} has the {want} label; yellow puller seen in its holder "
                    f"({r['px']} px)", flush=True)

            def puller_est(self):
                """The puller's hinge (desk) and quaternion from arm B's joints and how it was pinched."""
                g, gq = grasp_point("right")
                return g + rotate(gq, self.model.rel_pos), qmul(gq, self.model.rel_quat)

            def tilt_est(self, lift):
                """How far the puller (and the fuse in its jaws) has swung up about the far ferrule: the angle of the
                pinch point (arm B's joints, less the commanded extra lift) about the documented far ferrule."""
                g = grasp_point("right")[0] - np.array([0.0, 0.0, lift])
                a = math.atan2(self.g0[2] - self.pivot[2], self.g0[0] - self.pivot[0])
                b = math.atan2(g[2] - self.pivot[2], g[0] - self.pivot[0])
                return math.degrees((a - b + math.pi) % (2 * math.pi) - math.pi)

            def at(self, step, fn):
                self.events.setdefault(step, []).append(fn)

            def run_due(self, k):
                self.k_now = k
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
                log(f"{tag}", flush=True)

            # 1. Pick the puller out of its holder ----------------------------------------------------------
            def plan_pick(self):
                R = self.R
                h_pos = self.rest
                g = h_pos + rotate(PUQ, GRASP_IN_PULLER)
                q = hand_quat()
                d = finger_dir(q)
                log(f"PULLER hinge {np.round(h_pos, 4)}; handle grasp {np.round(g, 4)}", flush=True)
                R.grip_to(G_OPEN, n(2))
                R.move(g - 0.07 * d + np.array([0.0, 0.0, 0.03]), q, min_steps=n(30))
                R.move(g - 0.03 * d, speed=0.04)
                R.move(g, speed=0.015)
                R.hold(n(3))
                self.when_settled(self.pinch)

            def pinch(self):
                log(f"AT HANDLES: lag {self.lag():.1f} mm", flush=True)
                self.R.grip_to(G_HOLD, n(10))
                self.at(len(self.R.steps), self.carry)

            def carry(self):
                R = self.R
                # Fingers closed on the handles of the puller standing in its holder: how it sits in the hand.
                g, gq = grasp_point("right")
                self.model = GraspModel(rotate(conj(gq), self.rest - g), qmul(conj(gq), PUQ))
                log(f"HELD: hinge in the hand {np.round(self.model.rel_pos * 1000, 1)} mm; gripper "
                    f"{V.split_state(obs_box[0].state['joint_pos'])['right'][1]:.3f}", flush=True)
                named = self.named
                # Lift the puller until its jaw tips clear the enclosure's rim, carry it over the right wall at that
                # height to above F2, then lower it into the enclosure.
                here = R.pos + ROOT_OFFSET
                lift = CARRY_HINGE_Z - self.rest[2]
                R.move(here + np.array([0.0, 0.0, lift]), speed=0.03)
                target = np.array(props.puller_on_fuse_pose(named)[:3])
                g, q = self.grasp_for(np.array([target[0], target[1], CARRY_HINGE_Z]), PUQ)
                R.move(g, q, speed=0.04)
                g, q = self.grasp_for(target + np.array([0.0, 0.0, HOVER + 0.03]), PUQ)
                R.move(g, q, speed=0.03)
                g, q = self.grasp_for(target + np.array([0.0, 0.0, HOVER]), PUQ)
                R.move(g, q, speed=0.02)
                R.hold(n(4))
                self.when_settled(lambda: self.lower(0))

            def grasp_for(self, hinge_pos, hinge_quat):
                g_pos, g_quat = self.model.grasp_for(hinge_pos, hinge_quat)
                return g_pos, g_quat

            def lower(self, k, extra=0.0):
                """Re-measure the grasp, then lower the open jaws round the fuse body; push further while the
                puller's hinge stays above its seated height (the jaws' hood rests on the body)."""
                R = self.R
                target = np.array(props.puller_on_fuse_pose(self.named)[:3])  # F2 in its clips (documented)
                h_pos, _ = self.puller_est()
                log(f"LOWER {k}: hinge {np.round(h_pos, 4)} target {np.round(target, 4)} lag {self.lag():.1f} mm",
                    flush=True)
                if k == 0:
                    g, q = self.grasp_for(target + np.array([0.0, 0.0, 0.010]), PUQ)
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
                g, q = self.grasp_for(target + np.array([0.0, 0.0, 0.0005 - extra]), PUQ)
                R.move(g, q, speed=0.01)
                R.hold(n(4))
                self.when_settled(lambda: self.lower(k + 1, extra), tries=6)

            def squeeze(self):
                h_pos, _ = self.puller_est()
                f_pos = np.array(props.fuse_seat(self.named))
                log(f"SQUEEZE: hinge-fuse (arm B's joints) {np.round((h_pos - f_pos) * 1000, 1)} mm "
                    f"(want [0, 0, {props.JAW_X * 1000:.0f}])", flush=True)
                self.R.grip_to(G_SQUEEZE, n(12))
                self.at(len(self.R.steps), self.after_squeeze)

            def after_squeeze(self):
                self.report(f"SQUEEZED (gripper {V.split_state(obs_box[0].state['joint_pos'])['right'][1]:.3f})")
                if control:
                    self.control_regrip()
                    return
                f_pos = np.array(props.fuse_seat(self.named))
                self.pivot = f_pos + np.array([cap_off, 0.0, 0.0])  # the far ferrule's centre (documented)
                self.g0 = grasp_point("right")[0]
                self.q0 = self.R.quat.copy()
                self.phase2 = False
                self.rock(0.0, 3.0, 0, 0.0, 0.0)

            # 3. Rock it out -------------------------------------------------------------------------------------
            def tong_target(self, phi_deg, lift):
                """Grasp pose with the puller (and the fuse in its jaws) tipped by phi about the far ferrule, near
                end up (the handles go away from the robot), and raised by lift. The hand turns with the puller."""
                rq = q_axis((0.0, 1.0, 0.0), math.radians(phi_deg))
                return (self.pivot + rotate(rq, self.g0 - self.pivot) + np.array([0.0, 0.0, lift]),
                        qmul(rq, self.q0))

            def rock(self, phi_cmd, lead, k, tilt_prev, lift):
                """Tip the handles away along an arc about the far ferrule, a few degrees ahead of the measured tip
                (further ahead while it stalls): the near ferrule levers out of its clip. Then tip on and lift: with
                the near end free, the lift pulls the far ferrule out of its clip."""
                tilt = self.tilt_est(lift if not self.phase2 else 0.0) if k else 0.0
                log(f"ROCK {k}: cmd {phi_cmd:.1f} tipped {tilt:.1f} deg (arm B's joints) lead {lead:.1f} "
                    f"lift {lift * 1000:.0f} mm lag {self.lag():.1f} mm", flush=True)
                if (self.phase2 and phi_cmd >= TIP_MAX_DEG - 1e-6 and lift >= 0.030 - 1e-9) or k > 90:
                    self.report(f"TIPPED {TIP_MAX_DEG:.0f} deg and lifted: drawing the far end out")
                    self.R.hold(n(4))
                    self.at(len(self.R.steps), self.to_tray)
                    return
                if tilt >= NEAR_OUT_DEG and not self.phase2:
                    self.phase2 = True
                    self.report(f"NEAR END OUT: tipped {tilt:.1f} deg about the far ferrule, the near ferrule "
                                f"{math.radians(tilt) * (props.FUSE_L - props.CAP_L) * 1000:.0f} mm from its clip")
                stalled = k and tilt - tilt_prev < 0.4
                if stalled:
                    lead = min(lead + 1.5, 12.0)
                elif tilt - tilt_prev > 2.0:
                    lead = max(lead - 1.0, 3.0)
                cap = 40.0 if not self.phase2 else TIP_MAX_DEG
                phi_cmd = min(max(phi_cmd - 1.0, tilt) + lead, cap)
                if self.phase2:
                    lift = min(lift + 0.003, 0.05)
                R = self.R
                start, q_start = R.pos + ROOT_OFFSET, R.quat.copy()
                g, q = self.tong_target(phi_cmd, lift)
                m = n(5)
                for i in range(1, m + 1):
                    R.path([(start + (g - start) * i / m, slerp(q_start, q, i / m))])
                R.hold(n(3))
                self.at(len(R.steps), lambda: self.rock(phi_cmd, lead, k + 1, tilt, lift))

            # 4. To the tray ---------------------------------------------------------------------------------------
            def to_tray(self):
                """Draw the far end out: lift, then turn the puller upright again while moving it up and toward the
                robot (pulled straight up while tipped 30 degrees, the far ferrule binds in its clip; turned back
                toward level as it rises, it slides out). Then carry the jaws over the tray."""
                R = self.R
                here = R.pos + ROOT_OFFSET
                R.move(np.array([here[0], here[1], min(here[2] + 0.04, 0.17)]), speed=0.04)
                tx, ty = TRAY_XY
                g, q = self.grasp_for(np.array([tx, ty, 0.065 + props.JAW_X]), PUQ)
                # Turn the puller upright while moving up and toward the tray, but only a third of the way (the
                # enclosure's right wall is in the way): this draws the far ferrule out of its clip. Then lift the
                # fuse over the wall, carry it over the tray and lower it.
                hz = max(g[2], here[2] + 0.04)
                mid = here[:2] + DRAW_FRACTION * (g[:2] - here[:2])
                R.move(np.array([mid[0], mid[1], hz]), q, speed=0.02)
                over = CARRY_HINGE_Z + g[2] - (0.065 + props.JAW_X)  # the grasp with the hinge at CARRY_HINGE_Z
                R.move(np.array([mid[0], mid[1], over]), speed=0.03)
                R.move(np.array([g[0], g[1], over]), speed=0.05)
                R.move(g, speed=0.04)
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
                h_pos, h_quat = self.puller_est()
                R.grip_to(G_OPEN, n(8))
                d = finger_dir(R.quat.copy())
                here = R.pos + ROOT_OFFSET
                R.move(here - 0.04 * d, speed=0.03)
                g = h_pos + rotate(h_quat, SMOOTH_IN_PULLER)
                # Fingers 50 deg down for this low grasp: at 30 deg the gripper's housing would sit below the rim,
                # against the enclosure's near wall.
                q = hand_quat(20.0)
                d = finger_dir(q)
                log(f"CONTROL: regrip the smooth arms over the jaws at {np.round(g, 4)}", flush=True)
                R.move(g - 0.05 * d + np.array([0.0, 0.0, 0.02]), q, speed=0.03)
                R.move(g - 0.03 * d, speed=0.02)
                R.move(g, speed=0.01)
                R.hold(n(4))
                self.when_settled(self.control_pinch)

            def control_pinch(self):
                log(f"CONTROL AT JAWS: lag {self.lag():.1f} mm", flush=True)
                self.g_pull0 = grasp_point("right")[0][2]
                self.R.grip_to(G_SQUEEZE, n(12))
                self.at(len(self.R.steps), lambda: self.control_pull(0))

            def arm_force(self):
                """Estimated force (N, world) the right arm's joint PD pushes on the hand with: the PD torques
                (stiffness times the target error, capped at the effort limits; the gravity feed-forward excluded)
                through the arm's Jacobian, from finite differences of yam_arena's own forward kinematics."""
                q = q_meas("right")
                qt = q_target_box[0]["right"]  # the joint target this demo last sent
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
                fz = self.arm_force()[2]
                self.report(f"PULL {k}: hand {(grasp_point('right')[0][2] - self.g_pull0) * 1000:.1f} mm up, "
                            f"lag {self.lag():.1f} mm, arm lift about {fz:.0f} N")
                self.max_lift = max(getattr(self, "max_lift", 0.0), fz)
                if k >= 32:
                    R.hold(n(20))
                    self.at(len(R.steps), lambda: self.report(f"CONTROL END (largest arm lift about {self.max_lift:.0f} N)"))
                    return
                R.move(R.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.005]), speed=0.01)
                R.hold(n(6))
                self.at(len(R.steps), lambda: self.control_pull(k + 1))

        self._Planner = Planner
        self._dt = 1.0 / control_hz
        self._speed = speed
        self._log = log
        self._n = n
        self.planner = None
        self.worst = {"left": 0.0, "right": 0.0}

    def _start(self):
        n = self._n
        l_pos, l_quat = self._grasp_point("left")
        r_pos, r_quat = self._grasp_point("right")
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, self._dt, self._speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, self._dt, self._speed)
        planner = self._Planner(R)
        settle = n(10)
        L.hold(settle)
        R.hold(settle)
        planner.at(1, planner.survey)
        planner.at(settle + n(10), lambda: planner.plan_pick() if planner.named is not None else None)
        self.planner = planner
        self._L, self._R = L, R
        self._q_arm = {s: self._q_meas(s).astype(float).copy() for s in ("left", "right")}

    def _joint_action(self, side, pos_root, quat, grip):
        pos = pos_root + ROOT_OFFSET + np.array([0.0, 0.0, V.DESK_Z])  # world
        tcp = pos + rotate(quat, (0.0, 0.0, -TCP_EXTRA))
        rot = qmul(quat, Q_REL)
        q, pe, _ = ik.solve(tcp, rot, self._q_arm[side], base_pos=station.BASES[side], iters=60)
        self._q_arm[side] = q
        self._q_target_box[0][side] = q.copy()
        if pe > 0.005 and pe > self.worst[side] + 0.002:
            self._log(f"IK {side} k={self.state_k[0]} error {pe * 1000:.1f} mm at {np.round(pos_root + ROOT_OFFSET, 4)}",
                      flush=True)
        self.worst[side] = max(self.worst[side], pe)
        return np.r_[q, grip]

    def next_action(self, obs) -> np.ndarray:
        self._obs_box[0] = obs
        if self.planner is None:
            self._start()
        k = self.state_k[0]
        L, R = self._L, self._R
        self.planner.run_due(k)
        L.pad_to(len(R.steps))
        R.pad_to(len(L.steps))
        if k >= len(L.steps):
            L.hold(1)
            R.hold(1)
        (lp, lq, lg), (rp, rq, rg) = L.steps[k], R.steps[k]
        self.state_k[0] = k + 1
        return np.concatenate([self._joint_action("left", lp, lq, lg), self._joint_action("right", rp, rq, rg)])

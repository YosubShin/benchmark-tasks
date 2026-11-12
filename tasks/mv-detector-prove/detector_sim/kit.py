"""Pose maths, scripted-timeline helpers and robot-side helpers shared by the task and its demo recorder.

Quaternions are xyzw. The task's own frame ("desk frame"): desk top z = 0, left arm base at (0, +0.305, 0),
right arm base at (0, -0.305, 0); detector_sim.station maps it into the station (z + 0.75).
"""

from __future__ import annotations

import math

import numpy as np

ARM_BASE_SPACING_M = 0.61
GRASP_OFFSET_M = 0.1347
"""Flange (``gripper`` link, flange convention) to the grasp point between the finger pads, along the flange's -z."""
ROOT_OFFSET = np.array([0.0, ARM_BASE_SPACING_M / 2, 0.0])
"""World position of the robot root (left base); root frame = world shifted by this, no rotation."""
DOWN = np.array([0.0, 0.0, 1.0, 0.0])
"""Fingers straight down, closing along y, wrist neutral (joint 6 near zero)."""
FORWARD = np.array([-math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0])
"""Fingers pointing along +x (away from the robot), closing along y."""
OPEN, CLOSED = 1.0, 0.0


def qmul(a, b) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def q_axis(axis, angle_rad) -> np.ndarray:
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    h = angle_rad / 2
    return np.array([*(axis * math.sin(h)), math.cos(h)])


def rotate(q, v) -> np.ndarray:
    """Rotate vector v by quaternion q (xyzw)."""
    qv = np.array([*v, 0.0])
    conj = np.array([-q[0], -q[1], -q[2], q[3]])
    return qmul(qmul(q, qv), conj)[:3]


def slerp(q0, q1, a) -> np.ndarray:
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    dot = min(dot, 1.0)
    theta = math.acos(dot)
    if theta < 1e-4:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    return (math.sin((1 - a) * theta) * q0 + math.sin(a * theta) * q1) / math.sin(theta)


class ArmTimeline:
    """Per-step grasp-point targets for one arm, built from straight-line segments and arcs."""

    def __init__(self, start_pos_root, start_quat, step_dt, speed):
        self.steps = []
        self.pos = np.array(start_pos_root, dtype=float)
        self.quat = np.array(start_quat, dtype=float)
        self.grip = OPEN
        self.step_dt = step_dt
        self.speed = speed

    def move(self, world_pos, quat=None, speed=None, min_steps=3):
        """Straight line to a world position (slerping the orientation)."""
        target = np.array(world_pos, dtype=float) - ROOT_OFFSET
        quat = self.quat if quat is None else np.array(quat, dtype=float)
        n = max(min_steps, int(math.ceil(np.linalg.norm(target - self.pos) / ((speed or self.speed) * self.step_dt))))
        for k in range(1, n + 1):
            a = k / n
            self.steps.append((self.pos + a * (target - self.pos), slerp(self.quat, quat, a), self.grip))
        self.pos, self.quat = target, quat
        return self

    def path(self, poses_world):
        """Follow a list of (world position, quaternion) waypoints, one per step."""
        for pos, quat in poses_world:
            self.pos, self.quat = np.array(pos, dtype=float) - ROOT_OFFSET, np.array(quat, dtype=float)
            self.steps.append((self.pos.copy(), self.quat.copy(), self.grip))
        return self

    def grip_to(self, value, hold_steps=8):
        self.grip = value
        return self.hold(hold_steps)

    def hold(self, n):
        for _ in range(n):
            self.steps.append((self.pos.copy(), self.quat.copy(), self.grip))
        return self

    def pad_to(self, n):
        return self.hold(max(0, n - len(self.steps)))


def conj(q) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


class GraspModel:
    """Where a held object sits relative to the grasp point; maps desired object poses to grasp targets."""

    def __init__(self, rel_pos, rel_quat):
        self.rel_pos = np.asarray(rel_pos, dtype=float)
        self.rel_quat = np.asarray(rel_quat, dtype=float)

    @classmethod
    def nominal(cls, grasp_quat, object_quat, grasp_in_object):
        """The planned grasp: grasp point at ``grasp_in_object`` (object frame), with the given orientations."""
        rel_quat = qmul(conj(grasp_quat), object_quat)
        return cls(rotate(conj(grasp_quat), -rotate(object_quat, grasp_in_object)), rel_quat)

    def grasp_for(self, obj_pos, obj_quat):
        """Grasp-point (world position, quaternion) that puts the object at the given pose."""
        g_quat = qmul(obj_quat, conj(self.rel_quat))
        return np.asarray(obj_pos, dtype=float) - rotate(g_quat, self.rel_pos), g_quat


# -- task side (the scene; import after Kit starts; the demo never uses these) ------------------------------------

def reset_robot_home(env, env_ids) -> None:
    """No-op under yam_arena: its own reset event puts both arms at READY before the task's reset event runs."""


def set_visible(stage, path: str, visible: bool) -> None:
    from pxr import UsdGeom

    imageable = UsdGeom.Imageable(stage.GetPrimAtPath(path))
    imageable.MakeVisible() if visible else imageable.MakeInvisible()

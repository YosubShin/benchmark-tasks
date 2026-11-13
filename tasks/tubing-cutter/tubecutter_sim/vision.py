"""What the scripted demo perceives, from the policy's own observation only. NumPy + OpenCV, no simulator access.

- ``TopCamera``: the station's calibrated top camera (``yam_arena.station.top_camera``: pose and intrinsics of the
  overhead D435, the same calibration the embodiment renders with), resized to the observation's width and height.
  ``project`` maps a point (desk frame: station frame lowered by the arm-base height) to pixels; ``backproject``
  intersects a pixel's ray with a horizontal plane of known height.
- ``find_knob_angle``: the cutter's black knob (black-grey pixels in a ring round the tube, back-projected onto the
  knob's top); the direction of their centroid from the tube's axis is the cutter's angle about the tube.
- ``grasp_pose``: the arm's own grasp point (between the finger pads) from its joint angles, by yam_arena's URDF
  forward kinematics (proprioception).
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import ik, station
from yam_arena.urdf import quat_xyzw_to_matrix

Z0 = 0.75
"""Desk frame = station frame lowered by the arm-base height (tubecutter_sim.station.Z0)."""
Q_REL = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
"""yam_arena's `gripper` frame = our flange frame * Q_REL (tubecutter_sim.station.Q_REL)."""
TCP_EXTRA = 0.1347 - 0.134
"""Our grasp point sits this much further out along the approach axis than yam_arena's TCP."""


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def _rotate(q, v):
    return quat_xyzw_to_matrix(np.asarray(q, float)) @ np.asarray(v, float)


def grasp_pose(q_arm, side: str) -> tuple[np.ndarray, np.ndarray]:
    """(desk-frame position, xyzw quaternion in our flange convention) of the grasp point for arm joints q_arm."""
    tcp, rot = ik.tcp_pose(np.asarray(q_arm, float), station.BASES[side])
    q_theirs = ik.matrix_to_quat_xyzw(rot)
    quat = _qmul(q_theirs, np.array([-Q_REL[0], -Q_REL[1], -Q_REL[2], Q_REL[3]]))
    quat = quat / np.linalg.norm(quat)
    pos = tcp - _rotate(quat, (0.0, 0.0, -TCP_EXTRA)) - np.array([0.0, 0.0, Z0])
    return pos, quat


class TopCamera:
    def __init__(self, width: int, height: int, variant: str = "calibrated"):
        cam = station.top_camera(variant)
        assert cam.convention == "opengl"
        self.lens = cam.lens.resized(width, height)
        self.R = quat_xyzw_to_matrix(np.asarray(cam.rot_xyzw, float))  # camera -> world
        self.P = np.asarray(cam.pos, float) - np.array([0.0, 0.0, Z0])  # desk frame

    def project(self, x) -> np.ndarray:
        c = self.R.T @ (np.asarray(x, float) - self.P)
        L = self.lens
        return np.array([L.cx + L.fx * c[0] / -c[2], L.cy - L.fy * c[1] / -c[2]])

    def backproject(self, uv, z: float) -> np.ndarray:
        L = self.lens
        d = self.R @ np.array([(uv[0] - L.cx) / L.fx, -(uv[1] - L.cy) / L.fy, -1.0])
        t = (z - self.P[2]) / d[2]
        return self.P + t * d


def orange_mask(img: np.ndarray) -> np.ndarray:
    """The cutter's orange body (under the station's light it renders yellow-orange, about (234, 206, 95)); the beige
    table has blue above 0.6 of red."""
    r, g, b = (img[..., i].astype(int) for i in range(3))
    return (r > 150) & (g > 0.45 * r) & (b < 0.5 * r)


def black_mask(img: np.ndarray) -> np.ndarray:
    """The cutter's black knob: dark and grey (about (75, 73, 70) lit); shadows on the table are brown."""
    c = img[..., :3].astype(int)
    return (c.max(axis=-1) < 105) & (c.max(axis=-1) - c.min(axis=-1) < 22)


def find_knob_angle(cam: TopCamera, img: np.ndarray, tube_xy, knob_r: tuple, top_z: float, min_area: int = 4):
    """Angle (deg, counterclockwise from +x seen from above) of the cutter's knob about the tube, or None: black-grey
    pixels whose back-projection onto the knob's top lies in the ring ``knob_r`` (inner, outer radius from the tube's
    axis); the direction of their centroid from the axis. The knob is on the cutter's axis, so this is the cutter's
    angle. Call it with the arms parked well outside the ring."""
    ys, xs = np.nonzero(black_mask(img))
    if len(xs) < min_area:
        return None
    pts = np.array([cam.backproject((u, v), top_z)[:2] for u, v in zip(xs, ys)])
    d = pts - np.asarray(tube_xy)[None]
    r = np.linalg.norm(d, axis=1)
    keep = (r > knob_r[0]) & (r < knob_r[1])
    if keep.sum() < min_area:
        return None
    c = d[keep].mean(axis=0)
    return float(np.degrees(np.arctan2(c[1], c[0])))

"""What the scripted demo perceives, from the policy's own observation only. NumPy + OpenCV, no simulator access.

- ``TopCamera``: the station's calibrated top camera (``yam_arena.station.top_camera``: pose and intrinsics of the
  overhead D435, the same calibration the embodiment renders with), resized to the observation's width and height.
  ``project`` maps a point (desk frame: station frame lowered by the arm-base height) to pixels; ``backproject``
  intersects a pixel's ray with a horizontal plane of known height.
- ``find_cap``: the red cap on top of a stick's handle, as the red blob nearest where the documented layout puts
  it; its centroid back-projected onto the cap's height gives the handle's position on the table.
- ``find_tag``: the yellow TESTED card on the label ledge (yellow pixels inside the ledge's image region, which keeps
  the yellow cable out); its centroid back-projected onto the ledge gives which bar it lies behind.
- ``grasp_pose``: the arm's own grasp point (between the finger pads) from its joint angles, by yam_arena's URDF
  forward kinematics (proprioception).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from yam_arena import ik, station
from yam_arena.urdf import quat_xyzw_to_matrix

Z0 = 0.75
"""Desk frame = station frame lowered by the arm-base height (groundclamp_sim.station.Z0)."""
Q_REL = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
"""yam_arena's `gripper` frame = our flange frame * Q_REL (groundclamp_sim.station.Q_REL)."""
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


def red_mask(img: np.ndarray) -> np.ndarray:
    r, g, b = (img[..., i].astype(int) for i in range(3))
    return (r > 55) & (r > 2.2 * g) & (r > 2.2 * b)


def yellow_mask(img: np.ndarray) -> np.ndarray:
    r, g, b = (img[..., i].astype(int) for i in range(3))
    return (r > 150) & (g > 110) & (b < 90) & (r > 1.6 * b)


def _blobs(mask: np.ndarray):
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(cent[i], int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, n)]


def find_cap(cam: TopCamera, img: np.ndarray, cap_nominal, radius_px: float = 8.0, min_area: int = 4):
    """Desk-frame (x, y, z) of a handle's red cap seen in the top image, or None. ``cap_nominal`` is where the documented
    layout puts the cap's top face; only red blobs within ``radius_px`` of its projection count."""
    uv0 = cam.project(cap_nominal)
    cands = [(c, a) for c, a in _blobs(red_mask(img)) if a >= min_area and np.linalg.norm(c - uv0) < radius_px]
    if not cands:
        return None
    c, _ = min(cands, key=lambda ca: np.linalg.norm(ca[0] - uv0))
    return cam.backproject(c, float(cap_nominal[2]))


def find_tag(cam: TopCamera, img: np.ndarray, ledge_x, ledge_y, ledge_z: float, min_area: int = 20):
    """Desk-frame position of the yellow card lying on the label ledge (x range ledge_x, y range ledge_y, top at
    ledge_z), or None."""
    corners = [cam.project((x, y, ledge_z)) for x, y in ((ledge_x[0], ledge_y[0]), (ledge_x[1], ledge_y[0]),
                                                          (ledge_x[1], ledge_y[1]), (ledge_x[0], ledge_y[1]))]
    region = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(region, [np.round(corners).astype(np.int32)], 1)
    ys, xs = np.nonzero(yellow_mask(img) & (region > 0))
    if len(xs) < min_area:
        return None
    return cam.backproject((xs.mean(), ys.mean()), ledge_z)

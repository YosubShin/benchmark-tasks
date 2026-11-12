"""What the demo sees: camera models and colour segmentation for the policy's own observations. NumPy + OpenCV only.

The demo gets exactly what any policy gets from ``yam_arena``: three RGB images (``top_cam``, ``left_cam``,
``right_cam``, here 320 x 176) and ``joint_pos`` (14-D: each arm's six joints and its gripper opening). It uses the
station's published camera calibration (``yam_arena.station``: the top camera's pose and intrinsics, each wrist
camera's mount in the ``gripper`` link) and the arm's kinematics (``yam_arena.ik``, the URDF) to turn pixels into
rays. Nothing here touches the simulator.

Frames: ``world`` is the station's (arm bases at (0, +-0.305, 0.75)); the task's layout is given in the desk frame
(world lowered by DESK_Z, the arm-base height).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from yam_arena import YAM_URDF, station
from yam_arena.constants import ARM_JOINTS
from yam_arena.urdf import forward_kinematics, quat_xyzw_to_matrix

DESK_Z = 0.75


class PinholeCam:
    """An OpenGL-convention camera (looks along its -z, +y up) with its 640 x 360 calibration, observed resized."""

    def __init__(self, pos_w, rot_w: np.ndarray, lens, width: int, height: int):
        self.c = np.asarray(pos_w, dtype=float)
        self.r = np.asarray(rot_w, dtype=float)  # camera axes in world (columns)
        sx, sy = width / lens.width, height / lens.height
        # cv2.resize maps pixel centres: u' = (u + 0.5) * s - 0.5.
        self.fx, self.fy = lens.fx * sx, lens.fy * sy
        self.cx, self.cy = (lens.cx + 0.5) * sx - 0.5, (lens.cy + 0.5) * sy - 0.5
        self.w, self.h = width, height

    def project(self, pts_desk) -> np.ndarray:
        """(N, 2) pixel (u, v) of desk-frame points."""
        p = np.atleast_2d(np.asarray(pts_desk, dtype=float)) + (0.0, 0.0, DESK_Z)
        pc = (p - self.c) @ self.r
        depth = -pc[:, 2]
        return np.stack([self.cx + self.fx * pc[:, 0] / depth, self.cy - self.fy * pc[:, 1] / depth], -1)

    def ray(self, uv) -> tuple[np.ndarray, np.ndarray]:
        """(origin, unit direction) in the desk frame of the ray through pixel (u, v)."""
        u, v = uv
        d_cam = np.array([(u - self.cx) / self.fx, -(v - self.cy) / self.fy, -1.0])
        d = self.r @ d_cam
        return self.c - (0.0, 0.0, DESK_Z), d / np.linalg.norm(d)

    def on_plane_z(self, uv, z_desk: float) -> np.ndarray:
        """Desk-frame point where the ray through (u, v) meets the horizontal plane z = z_desk."""
        o, d = self.ray(uv)
        return o + d * (z_desk - o[2]) / d[2]


def top_cam(width: int, height: int) -> PinholeCam:
    cam = station.top_camera("calibrated")
    return PinholeCam(cam.pos, quat_xyzw_to_matrix(np.asarray(cam.rot_xyzw)), cam.lens, width, height)


def gripper_pose(side: str, q_arm) -> np.ndarray:
    """4 x 4 world pose of one arm's ``gripper`` link from its six joint angles (URDF forward kinematics)."""
    g = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q_arm, strict=True)})["gripper"]
    g = np.array(g, dtype=float)
    g[:3, 3] += np.asarray(station.BASES[side])
    return g


def wrist_cam(side: str, q_arm, width: int, height: int) -> PinholeCam:
    cam = station.wrist_camera(side)
    g = gripper_pose(side, q_arm)
    r = g[:3, :3] @ quat_xyzw_to_matrix(np.asarray(cam.rot_xyzw))
    return PinholeCam(g[:3, :3] @ np.asarray(cam.pos) + g[:3, 3], r, cam.lens, width, height)


# -- colour masks (RGB uint8 in) ---------------------------------------------------------------------------------


def hsv(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def redness(img: np.ndarray) -> np.ndarray:
    """R minus the larger of G and B (int): about 90-130 on the detector's red body, under 20 on the oak table."""
    i = img.astype(np.int16)
    return i[..., 0] - np.maximum(i[..., 1], i[..., 2])


def red_mask(img: np.ndarray, margin: int = 55) -> np.ndarray:
    return (redness(img) >= margin) & (img[..., 0] >= 90)


def yellow_mask(img: np.ndarray) -> np.ndarray:
    h = hsv(img)
    return (h[..., 0] >= 18) & (h[..., 0] <= 34) & (h[..., 1] >= 120) & (h[..., 2] >= 120)


def green_mask(img: np.ndarray, margin: int = 30) -> np.ndarray:
    """Lit green lights: G at least ``margin`` above R and B (a lit range light ~ (165, 243, 205); the oak table,
    the yellow stick and white parts are below zero)."""
    i = img.astype(np.int16)
    return (i[..., 1] - np.maximum(i[..., 0], i[..., 2]) >= margin) & (i[..., 1] >= 150)


def dark_mask(img: np.ndarray, v_max: int = 45) -> np.ndarray:
    return hsv(img)[..., 2] <= v_max


def largest_blob(mask: np.ndarray, near_uv=None, min_px: int = 4) -> np.ndarray | None:
    """The connected component of ``mask`` that is largest (or contains / is nearest to ``near_uv``)."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]
    if not ids:
        return None
    if near_uv is None:
        best = max(ids, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    else:
        u, v = near_uv
        best = min(ids, key=lambda i: math.hypot(cent[i][0] - u, cent[i][1] - v))
    return lab == best


def box_silhouette(cam: PinholeCam, center, rot: np.ndarray, size, shape) -> np.ndarray:
    """Filled image mask of a box (desk-frame centre, rotation matrix, full size) as ``cam`` sees it."""
    s = np.asarray(size, dtype=float) / 2
    corners = np.array([[a, b, c] for a in (-s[0], s[0]) for b in (-s[1], s[1]) for c in (-s[2], s[2])])
    uv = cam.project(corners @ rot.T + np.asarray(center))
    hull = cv2.convexHull(np.round(uv * 8).astype(np.int32))
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, hull, 1, lineType=cv2.LINE_8, shift=3)
    return m.astype(bool)


def face_quad(cam: PinholeCam, tip, rot: np.ndarray, z0: float, z1: float, width: float, face_x: float) -> np.ndarray:
    """Pixel corners of a rectangle on the detector's face (detector frame: z along the stick, x the face normal)."""
    w = width / 2
    local = np.array([[face_x, -w, z0], [face_x, w, z0], [face_x, w, z1], [face_x, -w, z1]])
    return cam.project(local @ rot.T + np.asarray(tip))


def fill_quad(mask: np.ndarray, uv: np.ndarray, value: bool) -> None:
    m = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(m, np.round(uv * 8).astype(np.int32), 1, lineType=cv2.LINE_8, shift=3)
    mask[m.astype(bool)] = value


class DetectorModel:
    """Where the detector's red parts appear in an image, for a pose (top-end position, rotation matrix)."""

    def __init__(self, head, body, label, lens, tab=None):
        self.head, self.body, self.label, self.lens, self.tab = head, body, label, lens, tab
        self.face_x = body[0] / 2

    def red(self, cam: PinholeCam, tip, rot: np.ndarray, shape) -> np.ndarray:
        h, b = self.head, self.body
        tip = np.asarray(tip, dtype=float)
        m = box_silhouette(cam, tip + rot @ (0, 0, h[2] / 2), rot, h, shape)
        m |= box_silhouette(cam, tip + rot @ (0, 0, h[2] + b[2] / 2), rot, b, shape)
        if self.tab is not None:
            t = self.tab
            m |= box_silhouette(cam, tip + rot @ (0, 0, h[2] + b[2] + t[2] / 2), rot, t, shape)
        if (rot @ (1.0, 0.0, 0.0)) @ (cam.c - (0, 0, DESK_Z) - tip) > 0:  # face toward the camera
            for z0, z1, w in (self.label, self.lens):
                fill_quad(m, face_quad(cam, tip, rot, z0, z1, w, self.face_x), False)
        return m

    def face_point(self, cam: PinholeCam, tip, rot: np.ndarray, along: float, across: float) -> np.ndarray:
        return cam.project(np.asarray(tip) + rot @ (self.face_x, across, along))[0]


def fit_shift(score_fn, span: float = 0.024, step: float = 0.003, fine: float = 0.00075):
    """Best (dx, dy) by grid search, coarse then fine; ``score_fn(dx, dy)`` -> higher is better."""
    best = (-1e9, 0.0, 0.0)
    n = int(round(span / step))
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            s = score_fn(i * step, j * step)
            if s > best[0]:
                best = (s, i * step, j * step)
    _, bx, by = best
    k = int(round(step / fine))
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            s = score_fn(bx + i * fine, by + j * fine)
            if s > best[0]:
                best = (s, bx + i * fine, by + j * fine)
    return best


def iou(pred: np.ndarray, obs: np.ndarray, valid: np.ndarray | None = None) -> float:
    if valid is not None:
        pred, obs = pred & valid, obs & valid
    union = (pred | obs).sum()
    return float((pred & obs).sum() / union) if union else 0.0

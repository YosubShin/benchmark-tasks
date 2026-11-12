"""What the demo can see and feel from the policy's own observation: forward kinematics of the measured joints, the
station's published top-camera calibration, and colour thresholds. Pure NumPy, no simulator imports.

Inputs are only what ``yam_arena`` hands any policy each step: the RGB images ``top_cam``, ``left_cam`` and
``right_cam`` (rendered 640x360, resized to the run's ``cam_width`` x ``cam_height``) and the 14-D ``joint_pos``
state (6 arm joints and the 0..1 gripper opening per arm, left first).

Frames: "world" is the station's (arm bases at z 0.75); the task's "desk" frame is world lowered by DESK_Z.
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import YAM_URDF, station
from yam_arena.constants import ARM_JOINTS, TCP_OFFSET
from yam_arena.ik import matrix_to_quat_xyzw
from yam_arena.urdf import forward_kinematics, quat_xyzw_to_matrix

from handhole_sim.kit import qmul

DESK_Z = 0.75
Q_REL = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
"""yam_arena's ``gripper`` frame = our flange frame * Q_REL (fingers along the flange's -z)."""
TCP_EXTRA = 0.1347 - 0.134
"""Our grasp point is this much further along the fingers than yam_arena's TCP."""


def split_state(joint_pos) -> dict[str, tuple[np.ndarray, float]]:
    """{side: (6 arm joints [rad], gripper 0 closed .. 1 open)} from the 14-D state."""
    q = np.asarray(joint_pos, dtype=float).reshape(-1)
    return {"left": (q[:6], float(q[6])), "right": (q[7:13], float(q[13]))}


def gripper_link(q6, side: str) -> np.ndarray:
    """4x4 world pose of an arm's ``gripper`` link from its joint angles (URDF FK at the station's base)."""
    g = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q6, strict=True)})["gripper"].copy()
    g[:3, 3] += np.asarray(station.BASES[side])
    return g


def grasp_point(q6, side: str) -> tuple[np.ndarray, np.ndarray]:
    """(desk position, flange quaternion xyzw) of an arm's grasp point, from its measured joint angles."""
    g = gripper_link(q6, side)
    r = g[:3, :3]
    tcp = r @ np.array(TCP_OFFSET) + g[:3, 3] - np.array([0.0, 0.0, DESK_Z])
    q_theirs = matrix_to_quat_xyzw(r)
    conj = np.array([-Q_REL[0], -Q_REL[1], -Q_REL[2], Q_REL[3]])
    return tcp + r @ np.array([0.0, 0.0, TCP_EXTRA]), qmul(q_theirs, conj)


class TopCam:
    """The station's calibrated top camera (OpenGL convention: looks along its -z, +y up) at a given resolution."""

    def __init__(self, width: int, height: int):
        cam = station.top_camera("calibrated")
        lens = cam.lens.resized(width, height)
        self.fx, self.fy, self.cx, self.cy = lens.fx, lens.fy, lens.cx, lens.cy
        self.r = quat_xyzw_to_matrix(np.asarray(cam.rot_xyzw, dtype=float))
        self.p = np.asarray(cam.pos, dtype=float) - np.array([0.0, 0.0, DESK_Z])

    def project(self, pts) -> np.ndarray:
        """Desk-frame points (N, 3) -> pixel (u, v) (N, 2); v grows downward."""
        c = (np.atleast_2d(pts) - self.p) @ self.r
        z = -c[:, 2]
        return np.stack([self.cx + self.fx * c[:, 0] / z, self.cy - self.fy * c[:, 1] / z], axis=1)


def red_lamp_pixels(img: np.ndarray, center_uv, half: int) -> int:
    """Number of bright red pixels (a lit red alarm lens) in a square window around a pixel. A lit lens renders deep
    red in shadow (about 186, 71, 65) and salmon in full light (about 242, 146, 128); the unlit lenses are grey, the
    yellow case has R - G of about 40 and the pink LCD alarm band about 56."""
    h, w = img.shape[:2]
    u, v = int(round(center_uv[0])), int(round(center_uv[1]))
    win = np.asarray(img[max(0, v - half):min(h, v + half + 1), max(0, u - half):min(w, u + half + 1)], dtype=int)
    r, g, b = win[..., 0], win[..., 1], win[..., 2]
    return int(((r > 170) & (r - g > 75) & (r - b > 75)).sum())

"""What the demo can see and feel: camera geometry and colour segmentation on the policy's own observations.

Inputs are only what ``yam_arena`` hands a policy each step: the RGB images ``top_cam``, ``left_cam`` and
``right_cam`` (rendered 640x360, resized to the run's ``cam_width`` x ``cam_height``) and the 14-D ``joint_pos``
state. Camera poses come from the station's published calibration (``yam_arena.station``: the top camera fixed in
the world, each wrist camera fixed in its arm's ``gripper`` link), placed with forward kinematics of the measured
joint angles (``yam_arena.urdf``). No depth: pixels are back-projected onto horizontal planes whose heights come
from the documented layout (task.json ``setup``). Pure NumPy, no simulator imports.

Frames: "world" is the station's (arm bases at z 0.75); the demo's "desk" frame is world lowered by DESK_Z.
"""

from __future__ import annotations

import numpy as np

from yam_arena import YAM_URDF, station
from yam_arena.constants import ARM_JOINTS, TCP_OFFSET
from yam_arena.urdf import forward_kinematics, quat_xyzw_to_matrix

DESK_Z = 0.75
"""The desk frame's origin in the world (the arm-base height; the documented layout's heights are from it)."""
SIDES = ("left", "right")


def split_state(joint_pos) -> dict[str, tuple[np.ndarray, float]]:
    """{side: (6 arm joints [rad], gripper 0 closed .. 1 open)} from the 14-D state."""
    q = np.asarray(joint_pos, dtype=float).reshape(-1)
    return {"left": (q[:6], float(q[6])), "right": (q[7:13], float(q[13]))}


def gripper_link(q6, side: str) -> np.ndarray:
    """4x4 world pose of an arm's ``gripper`` link from its joint angles (URDF FK at the station's base)."""
    g = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q6, strict=True)})["gripper"].copy()
    g[:3, 3] += np.asarray(station.BASES[side])
    return g


def tcp_desk(q6, side: str) -> tuple[np.ndarray, np.ndarray]:
    """(desk-frame position, rotation matrix) of the TCP (between the finger pads) from joint angles."""
    g = gripper_link(q6, side)
    p = g[:3, :3] @ np.array(TCP_OFFSET) + g[:3, 3]
    return p - np.array([0.0, 0.0, DESK_Z]), g[:3, :3]


class Cam:
    """A pinhole camera (OpenGL convention: looks along its -z, +y up) at the observation's resolution."""

    def __init__(self, rig_cam, width: int, height: int, parent: np.ndarray | None = None):
        lens = rig_cam.lens.resized(width, height)
        self.fx, self.fy, self.cx, self.cy = lens.fx, lens.fy, lens.cx, lens.cy
        r = quat_xyzw_to_matrix(np.asarray(rig_cam.rot_xyzw, dtype=float))
        p = np.asarray(rig_cam.pos, dtype=float)
        if parent is not None:
            r, p = parent[:3, :3] @ r, parent[:3, :3] @ p + parent[:3, 3]
        self.r, self.p = r, p - np.array([0.0, 0.0, DESK_Z])  # camera pose in the desk frame
        self.width, self.height = width, height

    @classmethod
    def top(cls, width, height):
        return cls(station.top_camera("calibrated"), width, height)

    @classmethod
    def wrist(cls, side, q6, width, height):
        return cls(station.wrist_camera(side), width, height, parent=gripper_link(q6, side))

    def project(self, pts) -> np.ndarray:
        """Desk-frame points (N, 3) -> pixel (u, v) (N, 2); v grows downward."""
        c = (np.atleast_2d(pts) - self.p) @ self.r  # camera frame
        z = -c[:, 2]
        return np.stack([self.cx + self.fx * c[:, 0] / z, self.cy - self.fy * c[:, 1] / z], axis=1)

    def ray(self, uv) -> np.ndarray:
        u, v = uv
        d = self.r @ np.array([(u - self.cx) / self.fx, -(v - self.cy) / self.fy, -1.0])
        return d / np.linalg.norm(d)

    def on_plane(self, uv, z: float) -> np.ndarray:
        """Back-project a pixel onto the desk-frame plane at height z."""
        d = self.ray(uv)
        t = (z - self.p[2]) / d[2]
        return self.p + t * d


# -- colour segmentation --------------------------------------------------------------------------------------

def hsv(img: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV_FULL).astype(np.int32)


def mask_range(img, h, s, v) -> np.ndarray:
    """Pixels whose HSV (OpenCV full range: H 0..255, S, V 0..255) lie in the given (lo, hi) ranges."""
    x = hsv(img)
    hh, ss, vv = x[..., 0], x[..., 1], x[..., 2]
    if h[0] <= h[1]:
        mh = (hh >= h[0]) & (hh <= h[1])
    else:  # wraps around red
        mh = (hh >= h[0]) | (hh <= h[1])
    return mh & (ss >= s[0]) & (ss <= s[1]) & (vv >= v[0]) & (vv <= v[1])


def blobs(mask: np.ndarray, min_px: int = 4) -> list[dict]:
    """Connected components of a mask, largest first: {"area", "centroid" (u, v), "bbox", "mask"}."""
    import cv2

    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            out.append({"area": int(stats[i, cv2.CC_STAT_AREA]), "centroid": cents[i].copy(),
                        "bbox": stats[i, :4].copy(), "mask": lab == i})
    return sorted(out, key=lambda b: -b["area"])


def nearest_blob(bl: list[dict], uv, max_px: float) -> dict | None:
    best = None
    for b in bl:
        d = float(np.linalg.norm(b["centroid"] - np.asarray(uv)))
        if d <= max_px and (best is None or d < best[0]):
            best = (d, b)
    return None if best is None else best[1]


# -- the puller and the circuit labels ---------------------------------------------------------------------------

YELLOW = ((34, 48), (80, 255), (110, 255))
"""The yellow nylon puller (HSV, OpenCV full range)."""


def find_blob(img, cam: Cam, doc_xyz, colour, plane_z: float, max_px: float = 20.0, min_px: int = 3) -> dict | None:
    """The ``colour`` blob nearest where the documented layout puts ``doc_xyz``, its centroid back-projected onto the
    horizontal plane z = plane_z. Returns {"xyz", "px", "uv", "uv_doc"} or None."""
    uv = cam.project(np.asarray(doc_xyz, dtype=float))[0]
    b = nearest_blob(blobs(mask_range(img, *colour), min_px=min_px), uv, max_px)
    if b is None:
        return None
    return {"xyz": cam.on_plane(b["centroid"], plane_z), "px": b["area"], "uv": b["centroid"], "uv_doc": uv}


def hue_sat_at(img, cam: Cam, xyz, half_px: int = 2) -> tuple[float, float]:
    """Circular-median hue (OpenCV full range) and median saturation in a small window around a point."""
    u, v = np.round(cam.project(np.asarray(xyz, dtype=float))[0]).astype(int)
    x = hsv(img)[max(0, v - half_px):v + half_px + 1, max(0, u - half_px):u + half_px + 1].reshape(-1, 3)
    ang = x[:, 0] * 2 * np.pi / 256.0
    m = np.arctan2(np.median(np.sin(ang)), np.median(np.cos(ang)))
    return float((m % (2 * np.pi)) * 256.0 / (2 * np.pi)), float(np.median(x[:, 1]))


def colour_class(img, cam: Cam, xyz, refs: dict) -> str:
    """The name in ``refs`` ({name: rgb}) whose colour is nearest the window's (hue for saturated colours, low
    saturation for white)."""
    import cv2

    h, s = hue_sat_at(img, cam, xyz)
    best, best_d = None, 1e9
    for name, rgb in refs.items():
        px = np.uint8([[np.clip(np.round(np.asarray(rgb) * 255), 0, 255)]])
        rh, rs, _ = cv2.cvtColor(px, cv2.COLOR_RGB2HSV_FULL)[0, 0].astype(float)
        dh = abs(h - rh) % 256.0
        dh = min(dh, 256.0 - dh)
        d = (dh if rs > 40 and s > 40 else 0.0) + abs(s - rs) * 0.5
        if d < best_d:
            best, best_d = name, d
    return best

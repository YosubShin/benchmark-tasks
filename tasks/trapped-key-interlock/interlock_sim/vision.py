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


# -- the interlock's red nameplate ----------------------------------------------------------------------------

PLATE_X = (-36.1, -18.3)
PLATE_Y = (-20.0, 20.0)
PLATE_Z = -18.7
"""The red nameplate on each Kirk lock's housing (lock frame, mm: origin on the keyway axis at the face, x away from
the bolt side; documented in task.json ``setup`` / usd/props.py): 17.8 x 40 mm, 18.7 mm below the face."""
RED = ((232, 14), (110, 255), (45, 255))


def _rot2(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s], [s, c]])


def locate_lock(img, cam: Cam, keyway_guess, lock_yaw_deg: float, max_px: float = 40.0, close_px: int = 3) -> dict | None:
    """Find a lock's keyway axis from its red nameplate in one image.

    The plate is segmented by colour (the blob nearest where the guess puts it), its outline is back-projected onto
    the plate's plane (known height), and a rectangle with the lock's documented yaw is fitted to it: the
    centre of the outline's extents along the lock's axes. The keyway is the documented offset from the plate's
    centre. Returns {"xy": keyway (desk, m), "size_mm": measured plate size, "px": blob area} or None."""
    import cv2

    g = np.asarray(keyway_guess, dtype=float)
    rot = _rot2(lock_yaw_deg)
    pc_lock = np.array([np.mean(PLATE_X), np.mean(PLATE_Y)]) * 1e-3
    z = g[2] + PLATE_Z * 1e-3
    guess_px = cam.project(np.r_[g[:2] + rot @ pc_lock, z])[0]
    red = mask_range(img, *RED).astype(np.uint8)
    if close_px > 1:  # join the plate's thin outer red rim across its white border line, fill the lettering
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    b = nearest_blob(blobs(red > 0, min_px=12), guess_px, max_px)
    if b is None:
        return None
    m = b["mask"].astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=len)[:, 0, :].astype(float)
    pts = np.array([cam.on_plane(uv, z)[:2] for uv in c])
    loc = (pts - g[:2]) @ rot  # lock-frame xy (rows: rot^T (p - g))
    lo, hi = np.percentile(loc, 1, axis=0), np.percentile(loc, 99, axis=0)
    centre_lock = (lo + hi) / 2
    keyway = g[:2] + rot @ (centre_lock - pc_lock)
    return {"xy": keyway, "size_mm": (hi - lo) * 1000, "px": b["area"], "uv": b["centroid"]}

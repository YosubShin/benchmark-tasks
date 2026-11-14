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




def on_plane_x(cam: Cam, uv, x: float) -> np.ndarray:
    """Back-project a pixel onto the vertical desk-frame plane x = const (e.g. the panel's deadfront)."""
    d = cam.ray(uv)
    t = (x - cam.p[0]) / d[0]
    return cam.p + t * d


def fit_shift(mask: np.ndarray, cam: Cam, inside, outside, axes, span: float, step: float, refine: bool = True,
              ignore: np.ndarray | None = None):
    """Shift of a known part outline that best matches a binary mask (e.g. a colour segmentation).

    ``inside`` (N, 3) are points on the part's visible faces at its nominal pose (desk frame), ``outside`` (M, 3) points
    just beyond its outline. The part is moved by ``a * axes[0] + b * axes[1]`` (a, b in [-span, span], grid
    ``step``, then a finer pass); the score is the share of inside points that land on mask pixels minus the share
    of outside points that do. Points landing on ``ignore`` pixels (e.g. something known to hide part of the outline)
    are left out of both shares. Returns (offset (3,), score, inside share)."""
    inside, outside = np.atleast_2d(inside), np.atleast_2d(outside)
    ax = [np.asarray(a, dtype=float) for a in axes]
    h, w = mask.shape

    import cv2

    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.8)  # sub-pixel: sample a lightly blurred mask

    def share(pts):
        uv = cam.project(pts)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        val = np.zeros(len(pts))
        val[ok] = bilinear(soft, uv[ok])
        keep = ok.copy()  # points outside the frame are left out
        if ignore is not None:
            iu = np.rint(uv[ok]).astype(int)
            keep[np.flatnonzero(ok)[ignore[iu[:, 1], iu[:, 0]]]] = False
        return val[keep].mean() if keep.sum() >= 5 else 0.0

    def search(c0, span, step):
        best = None
        n = int(round(span / step))
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                off = c0 + i * step * ax[0] + j * step * ax[1]
                si = share(inside + off)
                s = si - share(outside + off)
                if best is None or s > best[1] + 1e-9:
                    best = (off, s, si)
        return best

    best = search(np.zeros(3), span, step)
    if refine:
        best = search(best[0], step, step / 4)
    return best


def dark_mask(img, v_max: int = 45) -> np.ndarray:
    return hsv(img)[..., 2] <= v_max


def gray(img) -> np.ndarray:
    return np.asarray(img, dtype=float).mean(axis=2)


def bilinear(g: np.ndarray, uv) -> np.ndarray:
    uv = np.atleast_2d(uv)
    h, w = g.shape
    u = np.clip(uv[:, 0], 0, w - 1.001)
    v = np.clip(uv[:, 1], 0, h - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - u0, v - v0
    return (g[v0, u0] * (1 - fu) * (1 - fv) + g[v0, u0 + 1] * fu * (1 - fv) + g[v0 + 1, u0] * (1 - fu) * fv
            + g[v0 + 1, u0 + 1] * fu * fv)


def edge_score(g: np.ndarray, cam: Cam, rim, outward, d_px: float = 1.5) -> float:
    """Mean brightness step across a projected outline: outside minus inside, sampled ``d_px`` either side of each
    rim point along the image direction of its outward normal (``outward``: 3-D unit vectors, one per rim point)."""
    a = cam.project(rim)
    b = cam.project(rim + 0.0005 * np.asarray(outward))
    n = b - a
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return float(np.mean(bilinear(g, a + d_px * n) - bilinear(g, a - d_px * n)))


def fit_edge(g, cam, rim, outward, axes, span, step, refine=True):
    """Shift (along up to two ``axes``) of a known outline that maximises ``edge_score``. Returns (offset, score)."""
    ax = [np.asarray(a, dtype=float) for a in axes]

    def search(c0, span, step):
        best = None
        n = int(round(span / step))
        grid = [(i, j) for i in range(-n, n + 1) for j in (range(-n, n + 1) if len(ax) > 1 else [0])]
        for i, j in grid:
            off = c0 + i * step * ax[0] + (j * step * ax[1] if len(ax) > 1 else 0.0)
            s = edge_score(g, cam, rim + off, outward)
            if best is None or s > best[1]:
                best = (off, s)
        return best

    best = search(np.zeros(3), span, step)
    if refine:
        best = search(best[0], step, step / 4)
    return best


def render(cam: Cam, pts, shape) -> np.ndarray:
    """Binary silhouette of dense surface points (desk frame) in an image of ``shape`` (h, w)."""
    import cv2

    h, w = shape[:2]
    uv = np.rint(cam.project(pts)).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    m = np.zeros((h, w), np.uint8)
    m[uv[ok, 1], uv[ok, 0]] = 1
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0


def contrast(g: np.ndarray, sil: np.ndarray, band: int = 2) -> float:
    """Mean brightness just outside a silhouette minus just inside it (positive for a dark part on a lighter ground)."""
    import cv2

    k = np.ones((2 * band + 1, 2 * band + 1), np.uint8)
    m = sil.astype(np.uint8)
    inner = sil & ~(cv2.erode(m, k) > 0)
    outer = (cv2.dilate(m, k) > 0) & ~sil
    if inner.sum() < 5 or outer.sum() < 5:
        return -1e9
    return float(g[outer].mean() - g[inner].mean())


def fit_silhouette(g, cam: Cam, pts, axes, span, step, sign: float = 1.0, refine=True):
    """Shift of a part's dense surface points (along up to two axes) whose silhouette best separates dark inside
    from light outside (``sign`` -1 for a light part on a darker ground). Returns (offset, contrast)."""
    ax = [np.asarray(a, dtype=float) for a in axes]
    pts = np.asarray(pts, dtype=float)

    def search(c0, span, step):
        best = None
        n = int(round(span / step))
        for i in range(-n, n + 1):
            for j in (range(-n, n + 1) if len(ax) > 1 else [0]):
                off = c0 + i * step * ax[0] + (j * step * ax[1] if len(ax) > 1 else 0.0)
                s = sign * contrast(g, render(cam, pts + off, g.shape))
                if best is None or s > best[1]:
                    best = (off, s)
        return best

    best = search(np.zeros(3), span, step)
    if refine:
        best = search(best[0], step, step / 4)
    return best

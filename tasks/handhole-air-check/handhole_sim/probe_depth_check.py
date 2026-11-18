"""Worst-case depth of the sampling probe's tip when the task's "probe in" rule holds, at any pose the slot allows.

The rule (identical in the sim and in the real trial):
  (a) the probe's tip is below the cover's top face (it goes down into the slot),
  (b) the probe is within TILT_MAX of vertical,
  (c) the stop collar rests on the cover: the lowest point of the collar's underside over solid cover is at most
      G above the top face.
Cover: top face z = 0, underside z = -T; the pull slot |x| <= W/2, |y| <= S/2 runs through the full thickness. The
slot's centre pin is ignored (it only removes poses, so ignoring it is conservative). Nothing may penetrate the
cover (probe tube radius r from the tip up to the collar; collar radius R, width w).

Search: every tilt 0..TILT_MAX (step 0.5 deg) and azimuth (5 deg), every collar position on a 1 mm grid over the slot
plus a margin wide enough for the largest tilt, the collar seated exactly (gap 0) or raised by G. For each pose that
satisfies (a)-(c) and does not penetrate, the tip's depth below the underside; prints the smallest one.

usage: python probe_depth_check.py r R w L [TILT_MAX] (mm, deg)
"""
import sys

import numpy as np

T, W, S, G = 44.0, 13.0, 51.0, 2.0


def frame(u):
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


def in_slot(p):
    return (np.abs(p[..., 0]) <= W / 2) & (np.abs(p[..., 1]) <= S / 2)


def solid(p):
    return (p[..., 2] < -1e-6) & (p[..., 2] > -T + 1e-6) & ~in_slot(p)


def worst(r, R, w, L, tilt_max, step_th=0.5, step_ph=5.0, step_xy=1.0):
    margin = (R + G) * np.tan(np.radians(tilt_max)) + R + 2.0
    cx, cy = np.meshgrid(np.arange(-W / 2 - margin, W / 2 + margin + 1e-9, step_xy),
                         np.arange(-S / 2 - margin, S / 2 + margin + 1e-9, step_xy), indexing="ij")
    centres = np.stack([cx.ravel(), cy.ravel(), np.zeros(cx.size)], -1)  # (C, 3)
    beta = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    rho = np.array([0.25, 0.5, 0.75, 1.0])
    s = np.linspace(0, 1, 40)
    best, where, n_ok = np.inf, None, 0
    for th in np.radians(np.arange(0, tilt_max + 1e-9, step_th)):
        for ph in np.radians(np.arange(0, 360, step_ph if th > 0 else 360)):
            u = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
            e1, e2 = frame(u)
            ring = np.cos(beta)[:, None] * e1 + np.sin(beta)[:, None] * e2  # (B, 3)
            b16 = beta[::3]
            ring16 = np.cos(b16)[:, None] * e1 + np.sin(b16)[:, None] * e2
            disc = (rho[:, None, None] * ring[None]).reshape(-1, 3) * R  # (D, 3)
            pts = centres[:, None] + disc[None]  # (C, D, 3) collar underside, centre at z 0
            over = ~in_slot(pts)
            z = np.where(over, pts[..., 2], np.inf).min(1)  # lowest point over solid, per centre
            ok0 = np.isfinite(z)
            for gap in (0.0, G):
                c = centres.copy()
                c[:, 2] = -z + gap  # collar underside centre height that puts that point at `gap`
                tip = c - L * u
                ok = ok0 & (tip[:, 2] < 0)  # (a) tip below the top face
                # the axis must cross the top face inside the slot (the tube goes down through it): prune the rest
                cross = c[:, :2] - (c[:, 2] / u[2])[:, None] * u[None, :2]
                ok &= (np.abs(cross[:, 0]) <= W / 2) & (np.abs(cross[:, 1]) <= S / 2)
                if not ok.any():
                    continue
                cs, ts = c[ok], tip[ok]
                bad = np.zeros(len(cs), bool)
                for i0 in range(0, len(cs), 256):
                    sl = slice(i0, i0 + 256)
                    surf = ts[sl, None, None] + (s[None, :, None, None] * L * u) + r * ring16[None, None]
                    b1 = solid(surf).reshape(surf.shape[0], -1).any(1)
                    coll = cs[sl, None, None] + (np.array([0.0, 0.5, 1.0])[None, :, None, None] * w * u) + disc[None, None]
                    b1 |= solid(coll).reshape(coll.shape[0], -1).any(1)
                    bad[sl] = b1
                good = ~bad
                if not good.any():
                    continue
                n_ok += int(good.sum())
                d = -ts[good, 2] - T
                i = int(np.argmin(d))
                if d[i] < best:
                    best = float(d[i])
                    where = (round(float(np.degrees(th)), 1), round(float(np.degrees(ph)), 1),
                             np.round(cs[good][i], 2).tolist(), gap)
    return best, where, n_ok


if __name__ == "__main__":
    r, R, w, L = (float(v) for v in sys.argv[1:5])
    tilt_max = float(sys.argv[5]) if len(sys.argv) > 5 else 60.0
    d, pose, n = worst(r, R, w, L, tilt_max)
    print(f"r={r} R={R} w={w} L={L} tilt<={tilt_max}: {n} poses satisfy the rule; worst tip depth below the "
          f"underside {d:.2f} mm at (tilt, azimuth, collar centre, gap) {pose}; vertical {L - T:.1f} mm", flush=True)

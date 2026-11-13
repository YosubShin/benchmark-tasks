"""Bristol Maid CYB/DE2 bracket and its long-link chain (USD), for cylchain_sim.usd.props.

- ``bracket.usda`` (static colliders): plywood wall panel; the bracket's 88 mm back flange screwed to it; the 107 mm
  deep plate with two U-shaped cradles (semicircular backs approximated by slices, 2 mm bumper edging) and the
  tongues between them; under the tongue on each bay's left a J-hook of 6 mm round bar with its tip turned up toward
  the bay.
- ``chain.usda`` (fixed-base articulation): ``anchor`` (the chain's fixed end, through the hole in the centre tongue,
  fixed to the world) and links ``link_00`` .. on revolute joints. Each link collides as a ring of four 4 mm wires
  (28 x 11 mm inside), so a link can be dropped over the hook's tip; consecutive links alternate their ring plane by
  90 degrees and hinge about their own y (so the chain bends any way over two links). Neighbouring links (whose end
  bars overlap: the interlock) do not collide; all others do. The end link carries a red PVC sleeve (our marking).
  Authored in the starting lay (hanging from the fixed end to the table, the last links lying toward the robot);
  all joint angles are 0 there.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pxr import Sdf, UsdPhysics

from cylchain_sim import spec as S
from cylchain_sim.usd import usd_kit as kit
from cylchain_sim.usd.props import body_at, capsule, joint_at, mats, span, thin_contacts


def _bay_arc_slices(stage, path, yb, z0, z1, m, n=12):
    r = S.BAY_W / 2
    cx = S.WALL_X - S.BAY_BACK - r
    edges = np.linspace(-r, r, n + 1)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        ye = min(abs(a), abs(b)) if a * b > 0 else 0.0  # the slice's edge nearest the middle: never inside the arc
        x0 = cx + math.sqrt(max(r * r - ye * ye, 0.0))
        span(stage, f"{path}/arc{i:02d}", (x0, S.WALL_X - S.BAY_BACK), (yb + a, yb + b), (z0, z1), m["bracket"])
        span(stage, f"{path}/bump{i:02d}", (x0 - S.BUMPER_T, x0), (yb + a, yb + b), (z0 - 0.002, z1 + 0.002),
             m["bumper"])


def make_bracket(out_path: Path) -> None:
    root = "/bracket"
    stage, _ = kit.new_stage(out_path, root)
    m = mats(stage, root)
    span(stage, f"{root}/wall", (S.WALL_X, S.WALL_X + S.WALL_T), S.WALL_Y, (0.0, S.WALL_H), m["plywood"])
    y0, y1 = S.PLATE_Y
    top, z0 = S.PLATE_TOP, S.PLATE_TOP - S.PLATE_T
    span(stage, f"{root}/flange", (S.WALL_X - S.PLATE_T, S.WALL_X), (y0, y1), (top - S.FLANGE_H, top), m["bracket"])
    span(stage, f"{root}/back", (S.WALL_X - S.BAY_BACK, S.WALL_X - S.PLATE_T), (y0, y1), (z0, top), m["bracket"])
    hw = S.BAY_W / 2
    cuts = sorted([(yb - hw, yb + hw) for yb in S.BAY_Y])
    pieces = [(y0, cuts[0][0]), (cuts[0][1], cuts[1][0]), (cuts[1][1], y1)]
    for k, (a, b) in enumerate(pieces):
        span(stage, f"{root}/tongue{k}", (S.PLATE_FRONT_X, S.WALL_X - S.BAY_BACK), (a, b), (z0, top), m["bracket"])
    for k, yb in enumerate(S.BAY_Y):
        _bay_arc_slices(stage, f"{root}/bay{k}", yb, z0, top, m)
    pts = [np.array(p) for p in S.hook_points()]
    for k, yb in enumerate(S.BAY_Y):
        off = np.array([0.0, yb, 0.0])
        for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            capsule(stage, f"{root}/hook{k}_{i}", a + off, b + off, S.HOOK_R, m["bracket"])
    # The empty bay's chain hangs from its fixed end (visual only, not simulated).
    ya = S.PLATE_Y[0] + 0.018
    for i in range(6):
        z = S.ANCHOR[2] - i * S.LINK_PITCH
        capsule(stage, f"{root}/spare_chain{i}", (S.TONGUE_TIP_X, ya, z), (S.TONGUE_TIP_X, ya, z - S.LINK_PITCH + 0.004),
                0.004, m["zinc"], collide=False)
    for k, y in enumerate((y0 + 0.02, (y0 + y1) / 2, y1 - 0.02)):
        kit.cylinder(stage, f"{root}/screw{k}", (S.WALL_X - S.PLATE_T - 0.001, y, top - 0.05), 0.005, 0.002,
                     m["black"], axis="X")
    kit.friction(stage, root, [f"{root}/hook0_{i}" for i in range(len(pts) - 1)], static=0.4, dynamic=0.35,
                 name="hook", combine="max")
    thin_contacts(stage, root)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


CHAIN_JOINT_LIMIT = 150.0
WIRE_R = S.LINK_WIRE / 2


def lay_points() -> np.ndarray:
    """Starting lay: hanging straight down from the fixed end to the table, the last links lying on it toward the
    robot (N_LINKS + 1 joint points)."""
    a = np.array(S.ANCHOR)
    pts = [a.copy()]
    while len(pts) < S.N_LINKS + 1:
        p = pts[-1]
        if p[2] - S.LINK_PITCH > 0.040:
            pts.append(p - np.array([0.0, 0.0, S.LINK_PITCH]))
        elif p[2] > 0.0125:
            dz = min(p[2] - 0.010, S.LINK_PITCH * 0.7)
            pts.append(p + np.array([-math.sqrt(max(S.LINK_PITCH ** 2 - dz ** 2, 0.0)), 0.0, -dz]))
        else:
            pts.append(p + np.array([-S.LINK_PITCH, 0.0, 0.0]))
    return np.array(pts)


def _frame(x, k):
    x = x / np.linalg.norm(x)
    ref = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(ref, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    r = np.stack([x, y, z], 1)
    if k % 2:
        r = r @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    return r


def _ring(stage, path, m):
    off = S.LINK_IN_W / 2 + WIRE_R
    p = S.LINK_PITCH
    for i, s_ in enumerate((-1, 1)):
        capsule(stage, f"{path}/side{i}", (0.0, s_ * off, 0.0), (p, s_ * off, 0.0), WIRE_R, m["zinc"])
    for i, x in enumerate((0.0, p)):
        capsule(stage, f"{path}/end{i}", (x, -off, 0.0), (x, off, 0.0), WIRE_R, m["zinc"])


def make_chain(out_path: Path) -> None:
    root = "/chain"
    stage, _ = kit.new_stage(out_path, root)
    m = mats(stage, root)
    pts = lay_points() - np.array(S.ANCHOR)
    anchor = body_at(stage, f"{root}/anchor", pts[0], np.eye(3), 0.01, ((0, 0, 0), (1e-6, 1e-6, 1e-6)))
    ap = stage.GetPrimAtPath(root)
    kit.articulation_root(stage, ap, self_collisions=True)
    ap.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    ap.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    fixed = UsdPhysics.FixedJoint.Define(stage, f"{root}/joints/world_fixed")
    fixed.CreateBody1Rel().SetTargets([anchor.GetPath()])
    inertia = ((S.LINK_PITCH / 2, 0, 0), (2e-7, 6e-7, 6e-7))
    prev, prev_f = anchor.GetPath(), (pts[0], np.eye(3))
    bodies, wires = [f"{root}/anchor"], []
    for k in range(S.N_LINKS):
        p0, p1 = pts[k], pts[k + 1]
        r = _frame(p1 - p0, k)
        path = f"{root}/link_{k:02d}"
        body_at(stage, path, p0, r, S.LINK_MASS, inertia)
        _ring(stage, path, m)
        if k == S.N_LINKS - 1:
            capsule(stage, f"{path}/sleeve", (S.LINK_PITCH * 0.40, 0, 0), (S.LINK_PITCH * 1.0, 0, 0), 0.0055,
                    m["red"])
        joint_at(stage, f"{root}/joints/j{k:02d}", prev, prev_f, path, (p0, r), p0, r, "Y",
                 -CHAIN_JOINT_LIMIT, CHAIN_JOINT_LIMIT)
        prev, prev_f = path, (p0, r)
        bodies.append(path)
        wires += [f"{path}/side0", f"{path}/side1", f"{path}/end0", f"{path}/end1"]
    for a_, b_ in zip(bodies[:-1], bodies[1:]):
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(b_)).CreateFilteredPairsRel().AddTarget(a_)
    kit.friction(stage, root, wires, static=0.4, dynamic=0.35, name="zinc", combine="max")
    kit.friction(stage, root, [f"{root}/link_{S.N_LINKS - 1:02d}/sleeve"], static=1.2, dynamic=1.0, name="pvc",
                 combine="max")
    thin_contacts(stage, root)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path} ({S.N_LINKS} links, end at {np.round(pts[-1] + np.array(S.ANCHOR), 3)})")

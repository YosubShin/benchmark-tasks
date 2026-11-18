"""Generate the props (USD) for the fuse-puller task: the control enclosure (door, mounting plate, DIN rail, wire
ducts, wires) with three midget fuse blocks and labels, the fuses,
the tong puller, the puller's holder and the parts tray.

Real parts; (P) published, (S) measured on the vendor's STEP model, (I) inferred, (TM) to measure on a bought part.

- Fuse block: Eaton Bussmann BM6031B, 1-pole midget open fuse block (P, Bussmann doc 1104: 76.2 x 21.0 mm
  footprint, 30.7 mm high). Its successor BMM603-1PQ has a free STEP model (AutomationDirect), from which (S): fuse
  axis 24.7 mm above the mounting surface; clip centres 26.4 mm apart, each clip 8.2 mm long along the fuse; clip
  tops 29.0 mm; side walls 14.4 mm apart inside, cut down to 13.9 mm between the clips (where a puller's jaws reach
  the fuse body); clip fingers with an inner half-gap of 4.45 mm against the 5.15 mm ferrule radius, so a seated
  ferrule spreads each finger 0.7 mm. "Clip with reinforced spring" (P, BMM603 cutsheet).
- Clip spring (how it is simulated): each finger is a fixed collider standing 0.7 mm inside the ferrule, with a
  compliant-contact material (PhysX compliant contacts: the contact force is a spring on the penetration, so the
  0.7 mm the finger "gives" is the finger's spring deflection). The squeeze and the clip friction hold the ferrule;
  nothing else does. The spring rate (CLIP_K) makes a straight pull need about 70 N and one ferrule about 33 N at
  480 Hz with per-point friction anchors (TM; patents quote 90-180 N for cartridge fuses in general, US4288138).
- Fuse: 10 x 38 mm midget (Bussmann KTK class; P: 10.3 x 38.1 mm, nickel-plated ferrules); ferrules 10 mm long,
  tube 8.8 mm across (I, Schurter ASO drawing); 9 g (P, Weidmueller). Collision: three convex 24-gon prisms.
- Puller: Ideal 34-015, 5 in (127 mm), one-piece high-impact yellow nylon, for 1/4-1/2 in fuses (P, distributors).
  A one-piece tong: two flat arms joined by a flexible loop at the handle end, notched handles next to the loop, and
  curved jaws at the free tips whose grooves run ACROSS the tool, so the tong is used perpendicular to the fuse: it
  points into the block, the jaws close from either side of the fuse body between the clips (where the block's
  walls are cut down for them) and the barrel lies across the jaws (I: jaw shape, opening and flex; Ideal publishes
  no drawing). Modelled as a hinge body (the loop) and two arms on revolute joints with a spring drive (the nylon
  flex); each arm carries a curved jaw shell at its tip that wraps the fuse body from 81 deg on the loop side to
  81 deg past its equator on the tip side. Squeezing the handles 40 mm from the loop closes the jaws (about 0.35x
  leverage: the jaws are 114 mm from the loop).
- Holder: the puller stands upright, jaws down, in a grey holder block's pocket on the bench beside the enclosure (a stand-in for the
  clip or holster a puller is kept in), so it is picked up by its handles already pointing the way it is used.

Frames (desk frame: table top z = 0, arm bases at (0, +-0.305, 0)). The blocks lie with the fuse axes along +x.
The fuse's own frame: origin at its centre, axis +z (it is placed turned 90 deg about y). The puller's frame:
origin at the hinge (loop end), +x along the tong toward the jaw tips, the arms part along y, the flat arms' width
along z (the jaw grooves run along z); in the scene it is turned 90 deg about y (PULLER_QUAT: tips down, its z along
the fuses' axes).

Usage: python -m fuse_puller_sim.usd.puller_props <out dir> [--clip-k N/m] [--clip-damping Ns/m] [--clip-mu MU]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from fuse_puller_sim.spec import (BLOCK_X, CIRCUITS, DOOR_OPEN_DEG, DUCT_FAR_X, DUCT_NEAR_X, DUCT_TOP, ENC_X, ENC_Y,
                                  LABEL_RGB, PAN_TOP, PULLER_HOLDER_XY, SLOT_Y, TRAY_IN, TRAY_XY, WALL_TOP)
from fuse_puller_sim.usd import usd_kit as kit

# Fuse block (BM6031B / BMM603 STEP) -----------------------------------------------------------------------------
AXIS_U = 0.0247
"""(S) fuse axis above the mounting surface."""
AXIS_Z = PAN_TOP + AXIS_U
"""Desk-frame z of the fuse axes."""
BLOCK_L, BLOCK_W, BLOCK_H = 0.0762, 0.02103, 0.0307
WALL_IN = 0.0072
"""(S) half the 14.4 mm inside width."""
NOTCH_X = 0.0085
"""Half-length of the cut-down wall section between the clips (S: the clips' inner edges are 9.1 mm from centre)."""
NOTCH_U = 0.0139
FLOOR_U = 0.0121
CLIP_X = (0.0091, 0.0173)
"""(S) a clip's extent along the fuse, from the block centre."""
CLIP_TOP_U = 0.0290
FINGER_IN = 0.00445
"""(S) undeflected inner half-gap of a clip's fingers."""
FINGER_T = 0.0020
FINGER_BOTTOM_U = AXIS_U + 0.0002
"""The finger's bearing face runs from just above the ferrule's equator to the clip top."""
CLIP_CENTRE_X = 0.5 * (CLIP_X[0] + CLIP_X[1])

# Fuse (10 x 38 midget) --------------------------------------------------------------------------------------------
FUSE_R = 0.00515
FUSE_L = 0.0381
CAP_L = 0.0100
TUBE_R = 0.0044
FUSE_MASS = 0.009
FUSE_SIDES = 24

# Puller (Ideal 34-015 class), puller frame ----------------------------------------------------------------------
ARM_HW = 0.0060
"""Half the flat arms' width (12 mm strips, as the 34-015's flat handles, I); along the fuse when in use."""
ARM_Y = (0.0050, 0.0080)
"""Inner and outer face of an arm (|y|), closed on a fuse."""
TOP_X = 0.121
"""Jaw tips from the hinge: 127 mm overall (P) with the loop's 6 mm beyond the hinge."""
LOOP_X0 = -0.006
JAW_X = 0.114
"""Jaw (groove) centre from the hinge (I), 7 mm short of the tips."""
JAW_HALF = 0.0070
"""Jaw width along the fuse: 14 mm (I; the body between the ferrules is 18.1 mm, between the clips 18.2 mm)."""
JAW_R = TUBE_R + 0.0001
JAW_T = 0.0012
JAW_ANGLES = (-70.0, -50.0, -30.0, -10.0, 10.0, 30.0, 50.0, 70.0)
"""Jaw facets, degrees past the plane through the fuse axis square to the tong (negative: toward the loop). Each
covers +-JAW_SEG_HALF_DEG, so a shell wraps the body from 81 deg on the loop side to 81 deg past the equator on the
tip side: closed, the two shells nearly enclose the body, and hood and lips lie across the tong, so the fuse pressing
on them when the tong is tipped (a couple along the 14 mm jaw) does not wedge the jaws open. With 50 deg lips the
jaws spread under the clamp and turned on the fuse (I)."""
JAW_SEG_HALF_DEG = 11.0
PIN_Y = 0.5 * (ARM_Y[0] + ARM_Y[1])
OPEN_DEG = 4.0
"""Free (spring) opening of each arm about its pin at the loop: the jaws part 8 mm each side (the lips need 3.3 mm
to pass the body), the handles 5.6 mm at the grasp (I)."""
ARM_MASS, HINGE_MASS = 0.012, 0.010
"""34 g (the real puller is about 20 g; lighter articulations broke under the 50 N pinch in an earlier build)."""
GRASP_X = 0.040
"""Where the demo grips the handles (from the hinge), 74 mm from the jaws: the 50 N squeeze there clamps the jaws
with about 13 N each (the closed shells hold the fuse by shape; the clamp only keeps them shut)."""
PULLER_FRICTION = 0.6
"""Moulded nylon of the loop and jaws (the jaws' grip on the fuse body is set by the tube's material, TUBE_MU)."""
NOTCH_FROM, NOTCH_TO = 0.012, 0.062
"""The handles are notched ("notched handle", P: distributor listing) over this stretch next to the loop; the arms
from there to the jaws are smooth moulded nylon."""
NOTCHED_FRICTION, SMOOTH_FRICTION = 0.8, 0.25
"""Friction of the gripper's fingers on the notched handles (combined as the larger) and on the smooth part of the
arms next to the jaws (combined as the smaller) (I). Gripped there, the 50 N squeeze holds the puller against about
25 N of pull, well under the 70 N a straight pull on the fuse needs."""

# Holder ------------------------------------------------------------------------------------------------------------
HOLDER_XY = PULLER_HOLDER_XY
"""Desk-frame centre of the puller's holder block (on the bench right of the enclosure)."""
HOLDER_FLOOR = 0.040
HOLDER_TOP = 0.060
"""The pocket's floor and rim above the table: the puller stands on its jaw tips in a 20 mm deep pocket."""
HOLDER_POCKET = (0.0080, 0.0175)
"""Half the pocket's inside length (x, across the arms' width) and width (y, the arms' parting direction)."""
HOLDER_OUT = (0.020, 0.028)

# Enclosure, rail, wire ducts, wires, labels ----------------------------------------------------------------------
ENC_T = 0.0015
"""Rittal AE 1036.500 (P, Rittal data sheet): 300 x 300 x 155 mm, 1.38 mm steel body, 1.5 mm door, RAL 7035."""
ENC_FLANGE = 0.012
"""The opening's inward-folded rim (I; where the door seal lands)."""
PLATE_Z = (0.0185, 0.0205)
"""Zinc-plated 2 mm mounting plate, 254 x 275 mm (P), on studs 17 mm above the back (I)."""
PLATE_X = (0.5 * (ENC_X[0] + ENC_X[1]) - 0.1375, 0.5 * (ENC_X[0] + ENC_X[1]) + 0.1375)
PLATE_Y = (0.5 * (ENC_Y[0] + ENC_Y[1]) - 0.127, 0.5 * (ENC_Y[0] + ENC_Y[1]) + 0.127)
RAIL_Y = (PLATE_Y[0] + 0.012, PLATE_Y[1] - 0.012)
"""35 x 7.5 mm top-hat rail (EN 60715) across the plate, under the blocks."""
DUCT_Y = (PLATE_Y[0] + 0.006, PLATE_Y[1] - 0.006)
WIRE_W = 0.0028
"""14 AWG THHN outside diameter (P: 0.111 in); square colliders this wide."""
WIRE_Z = PAN_TOP + 0.0160
"""Wires enter the box lugs horizontally at the clamp, 16 mm above the block's base."""
LUG_END = 0.0345
"""The box lugs' outer ends from the block's centre."""
LABEL = (0.034, 0.020)
"""Circuit label, along x and y (m)."""
LABEL_X = 0.5 * (DUCT_FAR_X[0] + DUCT_FAR_X[1])

COLORS = {
    "block": (0.06, 0.06, 0.07),
    "tin": (0.80, 0.81, 0.82),
    "copper": (0.80, 0.45, 0.22),
    "steel": (0.62, 0.64, 0.67),
    "fuse_tube": (0.86, 0.84, 0.78),
    "fuse_cap": (0.78, 0.79, 0.81),
    "nylon": (0.96, 0.82, 0.10),
    "pan": (0.84, 0.85, 0.83),
    "zinc": (0.70, 0.72, 0.74),
    "duct": (0.66, 0.67, 0.66),
    "wire_line": (0.04, 0.04, 0.04),
    "wire_load": (0.80, 0.07, 0.05),
    "seal": (0.12, 0.12, 0.12),
    "stand": (0.40, 0.40, 0.42),
    "tray": (0.15, 0.30, 0.70),
    "ink_dark": (0.05, 0.05, 0.05),
    "ink_light": (0.97, 0.97, 0.97),
}
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _mats(stage, root, names):
    out = {}
    for name in names:
        path = f"{root}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[name]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
        if name in ("tin", "fuse_cap", "steel"):
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.6)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        out[name] = material
    return out


def _span(stage, path, xs, ys, zs, material, collide=True):
    return kit.box(stage, path, ((xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2, (zs[0] + zs[1]) / 2),
                   (abs(xs[1] - xs[0]), abs(ys[1] - ys[0]), abs(zs[1] - zs[0])), material, collide)


def _rbox(stage, path, center, size, material, rot_x_deg=0.0):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddRotateXOp().Set(rot_x_deg)
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _prism(stage, path, poly, z0, z1, material, collide=True):
    """A convex prism from a counter-clockwise xy polygon, between z0 and z1 (one convex hull)."""
    n = len(poly)
    pts = [Gf.Vec3f(x, y, z0) for x, y in poly] + [Gf.Vec3f(x, y, z1) for x, y in poly]
    faces, counts = list(range(n - 1, -1, -1)), [n]
    faces += list(range(n, 2 * n))
    counts.append(n)
    for k in range(n):
        a, b = k, (k + 1) % n
        faces += [a, b, n + b, n + a]
        counts.append(4)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    return mesh


def flat_ngon(apothem, n):
    """An n-gon with flats (not corners) at 0, 360/n, ... degrees, at distance ``apothem`` from the centre."""
    r = apothem / math.cos(math.pi / n)
    return [(r * math.cos(2 * math.pi * (k + 0.5) / n), r * math.sin(2 * math.pi * (k + 0.5) / n)) for k in range(n)]


def _physics_material(stage, root, name, mu, combine="min", compliant=None):
    mat = UsdShade.Material.Define(stage, f"{root}/PhysicsMaterials/{name}")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(mu)
    api.CreateDynamicFrictionAttr(mu)
    prim = mat.GetPrim()
    prim.AddAppliedSchema("PhysxMaterialAPI")
    prim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(combine)
    if compliant is not None:
        k, c = compliant
        prim.CreateAttribute("physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float).Set(k)
        prim.CreateAttribute("physxMaterial:compliantContactDamping", Sdf.ValueTypeNames.Float).Set(c)
    return mat


def _bind_physics(stage, path, mat):
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(mat, UsdShade.Tokens.weakerThanDescendants,
                                                                       "physics")


def _contacts(stage, offset=0.002, rest=0.0):
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(offset)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(rest)


def _rigid_settings(prim, iterations=32, max_depen=None):
    prim.AddAppliedSchema("PhysxRigidBodyAPI")
    prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(iterations)
    prim.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    if max_depen is not None:
        prim.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float).Set(max_depen)


# Geometry helpers used by the task and the demo ---------------------------------------------------------------------
def fuse_seat(slot: int) -> tuple[float, float, float]:
    """Desk-frame position of a seated fuse's centre."""
    return (BLOCK_X, SLOT_Y[slot], AXIS_Z)


FUSE_QUAT = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
"""A seated fuse's orientation: its axis (+z) along +x."""
PULLER_QUAT = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
"""The puller turned 90 deg about y: its +x (toward the jaw tips) pointing down, its z (the jaw grooves) along +x."""


def clip_centre(slot: int, end: int) -> tuple[float, float, float]:
    """Desk-frame centre of clip ``end`` (0 = near the robot, 1 = far) of block ``slot``."""
    return (BLOCK_X + (1 if end else -1) * CLIP_CENTRE_X, SLOT_Y[slot], AXIS_Z)


def puller_rest_pose() -> tuple:
    """The puller standing in its holder: jaw tips on the pocket floor, hinge straight above."""
    return (HOLDER_XY[0], HOLDER_XY[1], HOLDER_FLOOR + TOP_X + 0.0005) + PULLER_QUAT


def puller_on_fuse_pose(slot: int) -> tuple:
    """The puller pointing straight down at fuse ``slot``, its jaws (closed) round the middle of the fuse body."""
    return (BLOCK_X, SLOT_Y[slot], AXIS_Z + JAW_X) + PULLER_QUAT


# Enclosure, blocks and labels ------------------------------------------------------------------------------------------
def make_blocks(out_path: Path, clip_k, clip_damping, clip_mu) -> None:
    """``clip_k`` may be one value or one per block (probe runs compare spring rates side by side)."""
    root = "/blocks"
    stage, root_prim = kit.new_stage(out_path, root)
    m = _mats(stage, root, ("block", "tin", "copper", "steel", "pan", "zinc", "duct", "wire_line", "wire_load",
                            "seal"))
    b = f"{root}/body"
    UsdGeom.Xform.Define(stage, b)
    ks = list(clip_k) if isinstance(clip_k, (list, tuple)) else [clip_k] * 3
    clip_mats = [_physics_material(stage, root, f"clip{s}", clip_mu, "min", (ks[s], clip_damping)) for s in range(3)]
    housing = _physics_material(stage, root, "housing", 0.4, "min")
    _enclosure(stage, f"{root}/enclosure", m)
    for s, y0 in enumerate(SLOT_Y):
        p = f"{b}/block{s}"
        x0 = BLOCK_X
        z = lambda u: PAN_TOP + u  # noqa: E731
        _span(stage, f"{p}_floor", (x0 - BLOCK_L / 2, x0 + BLOCK_L / 2), (y0 - BLOCK_W / 2, y0 + BLOCK_W / 2),
              (z(0), z(FLOOR_U)), m["block"])
        for k, sv in enumerate((-1, 1)):
            ys = (y0 + sv * WALL_IN, y0 + sv * BLOCK_W / 2)
            for e, sw in enumerate((-1, 1)):
                _span(stage, f"{p}_wall{k}{e}", (x0 + sw * NOTCH_X, x0 + sw * BLOCK_L / 2), ys, (z(0), z(BLOCK_H)),
                      m["block"])
            _span(stage, f"{p}_notch{k}", (x0 - NOTCH_X, x0 + NOTCH_X), ys, (z(0), z(NOTCH_U)), m["block"])
        for e, sw in enumerate((-1, 1)):
            xs = (x0 + sw * CLIP_X[0], x0 + sw * CLIP_X[1])
            xc = x0 + sw * CLIP_CENTRE_X
            # The clip base: a saddle the ferrule rests on.
            _span(stage, f"{p}_seat{e}", xs, (y0 - 0.0025, y0 + 0.0025), (z(FLOOR_U), z(AXIS_U - FUSE_R)), m["tin"])
            # Fingers: the bearing faces stand 0.7 mm inside the ferrule (compliant contact = the clip spring).
            for k, sv in enumerate((-1, 1)):
                f = _span(stage, f"{p}_finger{e}{k}", xs, (y0 + sv * FINGER_IN, y0 + sv * (FINGER_IN + FINGER_T)),
                          (z(FINGER_BOTTOM_U), z(CLIP_TOP_U)), m["tin"])
                _bind_physics(stage, f.GetPath(), clip_mats[s])
                # The curved lower part of the finger round the seat (visual).
                _span(stage, f"{p}_finger{e}{k}_root", xs, (y0 + sv * (FINGER_IN + 0.0006), y0 + sv * (FINGER_IN + FINGER_T + 0.0006)),
                      (z(FLOOR_U), z(FINGER_BOTTOM_U)), m["tin"], collide=False)
                # Reinforcing spring (visual): a steel strap outside the finger.
                _span(stage, f"{p}_spring{e}{k}", (xs[0] + sw * 0.001, xs[1] - sw * 0.001),
                      (y0 + sv * (FINGER_IN + FINGER_T), y0 + sv * (FINGER_IN + FINGER_T + 0.0007)),
                      (z(FLOOR_U + 0.002), z(CLIP_TOP_U - 0.002)), m["steel"], collide=False)
            # Terminal: box lug and screw (visual).
            tx = (x0 + sw * 0.0215, x0 + sw * 0.0345)
            _span(stage, f"{p}_lug{e}", tx, (y0 - 0.005, y0 + 0.005), (z(FLOOR_U), z(0.020)), m["copper"],
                  collide=False)
            kit.cylinder(stage, f"{p}_screw{e}", (x0 + sw * 0.028, y0, z(0.0205)), 0.0035, 0.002, m["steel"])
        for prim_path in [f"{p}_floor", f"{p}_notch0", f"{p}_notch1", f"{p}_seat0", f"{p}_seat1"] + \
                [f"{p}_wall{k}{e}" for k in range(2) for e in range(2)]:
            _bind_physics(stage, prim_path, housing)
    # Circuit labels on the far duct's cover above each block: a colour-coded plate with the tag and circuit printed on it
    # (letters as thin boxes: geometry, not a texture, which the renderer was seen to load late or not at all).
    ink = {"white": "ink_dark", "orange": "ink_dark", "green": "ink_light"}
    lm = _mats(stage, root, ("ink_dark", "ink_light"))
    for s, (tag, circuit, colour) in enumerate(CIRCUITS):
        pm = _plain_material(stage, root, f"label_{tag}", LABEL_RGB[colour])
        z0 = DUCT_TOP
        _span(stage, f"{b}/label{s}", (LABEL_X - LABEL[0] / 2, LABEL_X + LABEL[0] / 2),
              (SLOT_Y[s] - LABEL[1] / 2, SLOT_Y[s] + LABEL[1] / 2), (z0, z0 + 0.0004), pm, collide=False)
        words = circuit.split()
        lines = ((tag, 0.0075, 0.0070), (words[0], 0.0024, -0.0040), (" ".join(words[1:]), 0.0024, -0.0090))
        for i, (text, height, dx) in enumerate(lines):
            _flat_text(stage, f"{b}/label{s}_text{i}", text, (LABEL_X + dx, SLOT_Y[s], z0 + 0.0004), height,
                       min(LABEL[1] - 0.003, 1.0), lm[ink[colour]])
    _contacts(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path} (clip k {ks} N/m, damping {clip_damping}, mu {clip_mu})")


def _rbox_x(stage, path, center, size, material, rot_x_deg, collide=True):
    """A box turned about x (degrees) about its own centre."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddRotateXOp().Set(rot_x_deg)
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _enclosure(stage, e, m) -> None:
    """The enclosure lying on its back (opening up), mounting plate, DIN rail, two wire ducts and the six wires landed
    on the blocks' box lugs. Everything here is static and collides (walls, rim, door, rail, ducts, wires)."""
    UsdGeom.Xform.Define(stage, e)
    x0, x1 = ENC_X
    y0, y1 = ENC_Y
    t, top = ENC_T, WALL_TOP
    _span(stage, f"{e}/back", (x0, x1), (y0, y1), (0.0, t), m["pan"])
    _span(stage, f"{e}/wall_bottom", (x0, x0 + t), (y0, y1), (t, top), m["pan"])  # toward the robot (gland end)
    _span(stage, f"{e}/wall_top", (x1 - t, x1), (y0, y1), (t, top), m["pan"])
    _span(stage, f"{e}/wall_right", (x0 + t, x1 - t), (y0, y0 + t), (t, top), m["pan"])
    _span(stage, f"{e}/wall_left", (x0 + t, x1 - t), (y1 - t, y1), (t, top), m["pan"])
    f = ENC_FLANGE
    for name, xs, ys in (("rim_bottom", (x0, x0 + f), (y0, y1)), ("rim_top", (x1 - f, x1), (y0, y1)),
                         ("rim_right", (x0 + f, x1 - f), (y0, y0 + f)), ("rim_left", (x0 + f, x1 - f), (y1 - f, y1))):
        _span(stage, f"{e}/{name}", xs, ys, (top - t, top), m["pan"])
    # Gland plate in the bottom wall (visual: a darker plate inside it).
    _span(stage, f"{e}/gland", (x0 + t, x0 + t + 0.001), (y0 + 0.05, y1 - 0.05), (0.03, 0.12), m["steel"], collide=False)
    # Mounting plate on four studs.
    _span(stage, f"{e}/plate", PLATE_X, PLATE_Y, PLATE_Z, m["zinc"])
    for i, (sx, sy) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        kit.cylinder(stage, f"{e}/stud{i}", (PLATE_X[sx] + (0.010 if sx == 0 else -0.010),
                                             PLATE_Y[sy] + (0.010 if sy == 0 else -0.010), 0.5 * (t + PLATE_Z[0])),
                     0.004, PLATE_Z[0] - t, m["steel"])
    # DIN rail (top hat): 27 mm base, 35 mm top flanges, 7.5 mm high.
    _span(stage, f"{e}/rail", (BLOCK_X - 0.0135, BLOCK_X + 0.0135), RAIL_Y, (PLATE_Z[1], PAN_TOP - 0.001), m["steel"])
    _span(stage, f"{e}/rail_top", (BLOCK_X - 0.0175, BLOCK_X + 0.0175), RAIL_Y, (PAN_TOP - 0.001, PAN_TOP), m["steel"])
    # Slotted wire ducts (Panduit type G, 1.5 x 2 in nominal): solid colliders, slots drawn on the side walls.
    for name, xs in (("duct_near", DUCT_NEAR_X), ("duct_far", DUCT_FAR_X)):
        _span(stage, f"{e}/{name}", xs, DUCT_Y, (PLATE_Z[1], DUCT_TOP - 0.002), m["duct"])
        _span(stage, f"{e}/{name}_cover", (xs[0] - 0.001, xs[1] + 0.001), DUCT_Y, (DUCT_TOP - 0.002, DUCT_TOP),
              m["pan"])
        for side, xw in enumerate(xs):
            for k in range(int((DUCT_Y[1] - DUCT_Y[0]) / 0.0127)):
                yk = DUCT_Y[0] + 0.004 + k * 0.0127
                _span(stage, f"{e}/{name}_slot{side}_{k}", (xw - 0.0004, xw + 0.0004), (yk, yk + 0.0055),
                      (PLATE_Z[1] + 0.012, DUCT_TOP - 0.004), m["seal"], collide=False)
    # Wires: black line wires from the far (line) lugs into the far duct, red load wires from the near (load) lugs
    # into the near duct; each a straight 14 AWG run fixed at both ends (a collider), its bare end in the lug.
    for s_, y in enumerate(SLOT_Y):
        for kind, xa, xb in (("load", DUCT_NEAR_X[1] - 0.004, BLOCK_X - LUG_END + 0.002),
                             ("line", BLOCK_X + LUG_END - 0.002, DUCT_FAR_X[0] + 0.004)):
            _span(stage, f"{e}/wire_{kind}{s_}", (xa, xb), (y - WIRE_W / 2, y + WIRE_W / 2),
                  (WIRE_Z - WIRE_W / 2, WIRE_Z + WIRE_W / 2), m[f"wire_{kind}"])
            xe = BLOCK_X + (-1 if kind == "load" else 1) * (LUG_END - 0.004)
            _span(stage, f"{e}/wire_{kind}{s_}_bare", (xe - 0.003, xe + 0.003), (y - 0.0009, y + 0.0009),
                  (WIRE_Z - 0.0009, WIRE_Z + 0.0009), m["copper"], collide=False)
    # Door: hinged along the left wall's rim, swung open DOOR_OPEN_DEG (just past upright), its outer face to the
    # left. A 1.5 mm panel with a 20 mm folded edge and the seal on its inner face.
    a = math.radians(DOOR_OPEN_DEG)
    d = (0.0, -math.cos(a), math.sin(a))  # from the hinge across the door (closed: -y)
    nrm = (0.0, math.sin(a), math.cos(a))  # the door's outer face normal (closed: +z)
    hinge = (0.5 * (x0 + x1), y1, top)
    w = y1 - y0
    c = tuple(hinge[i] + d[i] * w / 2 + nrm[i] * t / 2 for i in range(3))
    rot = -(DOOR_OPEN_DEG)
    _rbox_x(stage, f"{e}/door", c, (x1 - x0, w, t), m["pan"], rot)
    for k, frac in enumerate((0.0, 1.0)):
        cc = tuple(hinge[i] + d[i] * (w * frac + (0.006 if frac == 0 else -0.006)) - nrm[i] * 0.010 for i in range(3))
        _rbox_x(stage, f"{e}/door_edge{k}", cc, (x1 - x0, 0.012, 0.020), m["pan"], rot)
    for k, xe in enumerate((x0 + 0.006, x1 - 0.006)):
        cc = tuple((xe if i == 0 else hinge[i] + d[i] * w / 2 - nrm[i] * 0.010) for i in range(3))
        _rbox_x(stage, f"{e}/door_side{k}", cc, (0.012, w, 0.020), m["pan"], rot)
    cc = tuple((hinge[0] if i == 0 else hinge[i] + d[i] * w / 2 - nrm[i] * (t + 0.001)) for i in range(3))
    _rbox_x(stage, f"{e}/door_seal", cc, (x1 - x0 - 0.03, w - 0.03, 0.002), m["seal"], rot, collide=False)
    lock = tuple((hinge[0] if i == 0 else hinge[i] + d[i] * (w - 0.025) + nrm[i] * 0.004) for i in range(3))
    _rbox_x(stage, f"{e}/door_lock", lock, (0.016, 0.016, 0.008), m["seal"], rot, collide=False)
    for k, xh in enumerate((x0 + 0.04, x1 - 0.04)):
        kit.cylinder(stage, f"{e}/hinge{k}", (xh, y1 + 0.003, top + 0.003), 0.004, 0.03, m["steel"], axis="X")


def _plain_material(stage, root, name, rgb):
    path = f"{root}/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _text_runs(text: str, px: int = 24) -> tuple[list, int, int]:
    """The text as runs of dark pixels: [(row, col0, col1)], image width, height (rendered with PIL, 1 bit)."""
    font = ImageFont.truetype(FONT_B, px)
    l, t, r, bt = font.getbbox(text)
    w, h = r - l + 2, bt - t + 2
    img = Image.new("1", (w, h), 0)
    ImageDraw.Draw(img).text((1 - l, 1 - t), text, fill=1, font=font)
    runs = []
    for row in range(h):
        col = 0
        while col < w:
            if img.getpixel((col, row)):
                c0 = col
                while col < w and img.getpixel((col, row)):
                    col += 1
                runs.append((row, c0, col))
            else:
                col += 1
    return runs, w, h


def _flat_text(stage, path, text, center, height, max_width, material):
    """Letters as thin visual boxes lying flat at ``center``, readable from the robot's side (the text's top toward
    +x, its left toward +y), ``height`` tall (narrowed to ``max_width``)."""
    runs, w, h = _text_runs(text)
    pix = min(height / h, max_width / w)
    x, yc, z = center
    UsdGeom.Xform.Define(stage, path)
    for i, (row, c0, c1) in enumerate(runs):
        y0 = yc + (w / 2 - c0) * pix
        y1 = yc + (w / 2 - c1) * pix
        xr = x + (h / 2 - row - 0.5) * pix
        _span(stage, f"{path}/r{i}", (xr - pix / 2, xr + pix / 2), (y1, y0), (z, z + 0.0002), material,
              collide=False)


def _tex_material(stage, root, name, file_rel):
    path = f"{root}/Looks/tex_{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    reader = UsdShade.Shader.Define(stage, f"{path}/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"{path}/tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(file_rel)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _flat_quad(stage, path, x, y, z, lx, ly, material):
    """A label lying flat, readable from the robot's side (the image's top edge away from the robot)."""
    corners = [(x - lx / 2, y + ly / 2, z), (x - lx / 2, y - ly / 2, z), (x + lx / 2, y - ly / 2, z),
               (x + lx / 2, y + ly / 2, z)]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _labels(tex: Path) -> None:
    tex.mkdir(parents=True, exist_ok=True)
    for tag, circuit, colour in CIRCUITS:
        bg = tuple(int(255 * c) for c in LABEL_RGB[colour])
        ink = (0, 0, 0) if colour != "green" else (255, 255, 255)
        img = Image.new("RGB", (200, 340), bg)
        d = ImageDraw.Draw(img)
        d.rectangle([3, 3, 196, 336], outline=(30, 30, 30), width=6)
        d.text((100, 110), tag, fill=ink, font=ImageFont.truetype(FONT_B, 100), anchor="mm")
        words = circuit.split()
        d.text((100, 228), words[0], fill=ink, font=ImageFont.truetype(FONT_B, 32), anchor="mm")
        d.text((100, 270), " ".join(words[1:]), fill=ink, font=ImageFont.truetype(FONT_B, 30), anchor="mm")
        img.save(tex / f"fprock14r_circuit_tag_{tag}.png")


# Fuse ---------------------------------------------------------------------------------------------------------------
def make_fuse(out_path: Path, clip_mu) -> None:
    root = "/fuse"
    stage, root_prim = kit.new_stage(out_path, root)
    m = _mats(stage, root, ("fuse_tube", "fuse_cap"))
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(FUSE_MASS)
    _rigid_settings(root_prim, iterations=32)
    h = FUSE_L / 2
    _prism(stage, f"{root}/cap_lo", flat_ngon(FUSE_R, FUSE_SIDES), -h, -h + CAP_L, m["fuse_cap"])
    _prism(stage, f"{root}/tube", flat_ngon(TUBE_R, FUSE_SIDES), -h + CAP_L, h - CAP_L, m["fuse_tube"])
    _prism(stage, f"{root}/cap_hi", flat_ngon(FUSE_R, FUSE_SIDES), h - CAP_L, h, m["fuse_cap"])
    kit.cylinder(stage, f"{root}/print", (0, 0, 0), TUBE_R + 0.0001, 0.008, m["fuse_cap"])
    mat = _physics_material(stage, root, "nickel", clip_mu, "min")
    for cap in ("cap_lo", "cap_hi"):
        _bind_physics(stage, f"{root}/{cap}", mat)
    tube = _physics_material(stage, root, "tube", TUBE_MU, "max")
    _bind_physics(stage, f"{root}/tube", tube)
    _contacts(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


# Puller -------------------------------------------------------------------------------------------------------------
def _zbox(stage, path, center, size, material, rot_z_deg=0.0, collide=True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddRotateZOp().Set(rot_z_deg)
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def make_puller(out_path: Path) -> None:
    root = "/puller"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    m = _mats(stage, root, ("nylon",))
    hinge = kit.rigid_body(stage, f"{root}/hinge", (0, 0, 0), HINGE_MASS)
    zs = (-ARM_HW, ARM_HW)
    _span(stage, f"{root}/hinge/loop", (LOOP_X0, 0.002), (-ARM_Y[1], ARM_Y[1]), zs, m["nylon"])
    rc = JAW_R + JAW_T / 2
    seg = 2 * rc * math.tan(math.radians(JAW_SEG_HALF_DEG))
    for k, name in enumerate(("arm_a", "arm_b")):
        sy = 1 if k == 0 else -1
        body = kit.rigid_body(stage, f"{root}/{name}", (0, 0, 0), ARM_MASS)
        _rigid_settings(body.GetPrim(), iterations=32)
        p = f"{root}/{name}"
        # The arm: a short smooth neck at the loop, the notched handle, then smooth moulded nylon to the jaw.
        _span(stage, f"{p}/neck", (0.0025, NOTCH_FROM), (sy * ARM_Y[0], sy * ARM_Y[1]), zs, m["nylon"])
        _span(stage, f"{p}/handle", (NOTCH_FROM, NOTCH_TO), (sy * ARM_Y[0], sy * ARM_Y[1]), zs, m["nylon"])
        _span(stage, f"{p}/arm", (NOTCH_TO, TOP_X), (sy * ARM_Y[0], sy * ARM_Y[1]), zs, m["nylon"])
        # Notch ribs on the handle (visual).
        for i in range(8):
            x0 = NOTCH_FROM + 0.004 + i * 0.008
            _span(stage, f"{p}/rib{i}", (x0, x0 + 0.0025), (sy * ARM_Y[1], sy * (ARM_Y[1] + 0.0006)), zs,
                  m["nylon"], collide=False)
        # Jaw: a curved shell across the arm's tip (the groove runs along z, across the tong), joined to the arm
        # by a web on the loop side (visual; the facets carry the load).
        _span(stage, f"{p}/web", (JAW_X - rc - 0.004, JAW_X - rc), (sy * 0.0008, sy * ARM_Y[0]),
              (-JAW_HALF, JAW_HALF), m["nylon"], collide=False)
        for i, a in enumerate(JAW_ANGLES):
            ar = math.radians(a)
            c = (JAW_X + rc * math.sin(ar), sy * rc * math.cos(ar), 0.0)
            _zbox(stage, f"{p}/jaw{i}", c, (seg, JAW_T, 2 * JAW_HALF), m["nylon"], rot_z_deg=-sy * a)
        lo, hi = (-2.0, OPEN_DEG + 0.5) if k == 0 else (-OPEN_DEG - 0.5, 2.0)
        # Each arm flexes about its own line (along z) at the loop (two pivots, not one shared pin: a shared pin
        # would leave the loop free to swing between the held arms).
        pin = (0.0, sy * PIN_Y, 0.0)
        joint = kit.revolute(stage, f"{root}/joints/{name}_joint", hinge.GetPath(), body.GetPath(), "Z", pin, lo, hi)
        joint.CreateLocalPos1Attr(Gf.Vec3f(*pin))
    mat = _physics_material(stage, root, "nylon", PULLER_FRICTION, "average")
    UsdShade.MaterialBindingAPI(root_prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    smooth = _physics_material(stage, root, "smooth_nylon", SMOOTH_FRICTION, "min")
    notched = _physics_material(stage, root, "notched_handle", NOTCHED_FRICTION, "max")
    for name in ("arm_a", "arm_b"):
        _bind_physics(stage, f"{root}/{name}/neck", smooth)
        _bind_physics(stage, f"{root}/{name}/arm", smooth)
        _bind_physics(stage, f"{root}/{name}/handle", notched)
    _contacts(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def arm_open_sign(k: int) -> int:
    """Arm a (+y) opens with positive angles about +z (its far end swings to +y), arm b with negative."""
    return 1 if k == 0 else -1


# Stands: the puller's holder and the parts tray -------------------------------------------------------------------------
def make_stands(out_path: Path) -> None:
    root = "/stands"
    stage, _ = kit.new_stage(out_path, root)
    m = _mats(stage, root, ("stand", "tray"))
    # Holder: a block with a pocket the puller stands in on its jaw tips.
    hx, hy = HOLDER_XY
    ox, oy = HOLDER_OUT
    px, py = HOLDER_POCKET
    _span(stage, f"{root}/holder_base", (hx - ox, hx + ox), (hy - oy, hy + oy), (0.0, HOLDER_FLOOR), m["stand"])
    for k, s in enumerate((-1, 1)):
        _span(stage, f"{root}/holder_wx{k}", (hx + s * px, hx + s * ox), (hy - oy, hy + oy), (HOLDER_FLOOR, HOLDER_TOP),
              m["stand"])
        _span(stage, f"{root}/holder_wy{k}", (hx - px, hx + px), (hy + s * py, hy + s * oy), (HOLDER_FLOOR, HOLDER_TOP),
              m["stand"])
    tx, ty = TRAY_XY
    a, bb, h = TRAY_IN
    t = 0.003
    _span(stage, f"{root}/tray_floor", (tx - a / 2 - t, tx + a / 2 + t), (ty - bb / 2 - t, ty + bb / 2 + t), (0.0, 0.003),
          m["tray"])
    for k, s in enumerate((-1, 1)):
        _span(stage, f"{root}/tray_wx{k}", (tx + s * a / 2, tx + s * (a / 2 + t)), (ty - bb / 2 - t, ty + bb / 2 + t),
              (0.0, h), m["tray"])
        _span(stage, f"{root}/tray_wy{k}", (tx - a / 2, tx + a / 2), (ty + s * bb / 2, ty + s * (bb / 2 + t)), (0.0, h),
              m["tray"])
    _contacts(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


CLIP_K = 60000.0
"""Compliant-contact spring rate of a clip finger (N/m); see the module docstring."""
CLIP_DAMPING = 30.0
TUBE_MU = 1.0
"""Friction between the puller's jaws and the fuse's fibre/melamine tube (combined as the larger of the two). Raised
from about 0.5 to stand in for a firm grip of moulded jaws on the body; see the README (I)."""
CLIP_MU = 0.4
"""Friction between the tinned clip fingers and the nickel ferrule (real tin on nickel is about 0.3-0.5)."""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--clip-k", type=float, nargs="+", default=[CLIP_K])
    ap.add_argument("--clip-damping", type=float, default=CLIP_DAMPING)
    ap.add_argument("--clip-mu", type=float, default=CLIP_MU)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    k = args.clip_k[0] if len(args.clip_k) == 1 else args.clip_k
    make_blocks(out / "blocks.usda", k, args.clip_damping, args.clip_mu)
    make_fuse(out / "fuse.usda", args.clip_mu)
    make_puller(out / "puller.usda")
    make_stands(out / "stands.usda")

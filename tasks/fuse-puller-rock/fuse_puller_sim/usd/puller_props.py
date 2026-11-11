"""Generate the props (USD) for the fuse-puller task: back pan with three midget fuse blocks and labels, the fuses,
the tong puller, the puller's rest and the parts tray.

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
  Arms joined at a loop at the jaw end, free handle ends (P, photo). Jaw shape, opening and flex are inferred (I).
  Modelled as a hinge body (the loop) and two arms on revolute joints with a spring drive (the nylon flex); each
  arm carries a curved jaw shell 13-27 mm from the loop that wraps the fuse body from 81 deg above to 35 deg below
  its equator. Squeezing the handles 85 mm from the loop closes the jaws with about 4x leverage. The puller is held
  parallel to the fuse (as in the reference video), its arms above the block's housing and the jaws hanging down
  round the fuse body between the clips.

Frames (desk frame: table top z = 0, arm bases at (0, +-0.305, 0)). The blocks lie with the fuse axes along +x.
The fuse's own frame: origin at its centre, axis +z (it is placed turned 90 deg about y). The puller's frame:
origin at the hinge (loop end), +x along the tong toward the handle ends, +z up (the jaws hang down along -z), the
arms part along y; in the scene it is turned 180 deg about z (handles toward the robot).

Usage: python -m fuse_puller_sim.usd.puller_props <out dir> [--clip-k N/m] [--clip-damping Ns/m] [--clip-mu MU]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from fuse_puller_sim.spec import (BLOCK_X, CIRCUITS, LABEL_RGB, PAN_TOP, PULLER_REST_Y, SLOT_Y, TRAY_IN, TRAY_XY)
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
ARM_Z = (0.0115, 0.0235)
"""Arms' bottom and top above the fuse axis when the jaws hold a fuse (they clear the housing's 6 mm above the axis):
12 mm strips, as the 34-015's flat handles (I)."""
ARM_Y = (0.0050, 0.0080)
"""Inner and outer face of an arm (|y|), closed on a fuse."""
TOP_X = 0.121
"""Free handle ends from the hinge: 127 mm overall (P) with the loop's 6 mm beyond the hinge."""
LOOP_X0 = -0.006
JAW_X = 0.020
"""Jaw centre from the hinge (I). The arms open like a V from the loop, so the jaw's end nearest the loop opens least:
at 14 mm it opened 0.6 mm and its lower lip could not pass over the fuse body."""
JAW_HALF = 0.0070
"""Jaw length along the fuse: 14 mm (I; the body between the ferrules is 18.1 mm, between the clips 18.2 mm; a 16 mm
jaw caught the clip fingers' tops when placed 1 mm off)."""
JAW_R = TUBE_R + 0.0001
JAW_T = 0.0012
JAW_ANGLES = (-70.0, -50.0, -30.0, -10.0, 8.0, 24.0)
"""Jaw facets, degrees below the horizontal through the fuse axis (negative: above the equator). Each covers
+-JAW_SEG_HALF_DEG, so the shell wraps the body from 81 deg above to 35 deg below its equator (a hood over the top: turning the
fuse in the jaws then meets the shell, not only friction)."""
JAW_SEG_HALF_DEG = 11.0
PIN_Y = 0.5 * (ARM_Y[0] + ARM_Y[1])
OPEN_DEG = 8.0
"""Free (spring) opening of each arm about its pin at the loop: the jaws part 2.8 mm each side, the handle ends 17 mm (I)."""
ARM_MASS, HINGE_MASS = 0.012, 0.010
"""34 g (the real puller is about 20 g): light enough that a pinch which leaves the jaws open carries it level by the
handle ends (at 90 g it swung 25 deg in the fingers)."""
GRASP_X = 0.085
"""Where the demo grips the handles (from the hinge): the 50 N squeeze there clamps the jaws with about 140 N, and a
light pinch there carries the 34 g puller level (by the handle ends, 108 mm, it tipped in the fingers)."""
PULLER_FRICTION = 0.6
"""Moulded nylon of the loop and jaws (the jaws' grip on the fuse body is set by the tube's material, TUBE_MU)."""
NOTCH_FROM = 0.050
"""The handles are notched ("notched handle", P: distributor listing) from here to their ends; the arms between the
loop and here are smooth moulded nylon."""
NOTCHED_FRICTION, SMOOTH_FRICTION = 0.8, 0.25
"""Friction of the gripper's fingers on the notched handles (combined as the larger) and on the smooth part of the
arms near the jaws (combined as the smaller) (I). Gripped over the jaws, the 50 N squeeze holds the puller against
about 25 N of pull, well under the 70 N a straight pull on the fuse needs."""
REST_HINGE_X = BLOCK_X + JAW_X
"""World x of the puller's hinge on its rest (as when its jaws hold a fuse in the middle block)."""

# Pan, rest, tray ---------------------------------------------------------------------------------------------------
PAN_X = (0.215, 0.405)
PAN_Y = (-0.075, 0.075)
REST_TOP = AXIS_Z + ARM_Z[0]
"""The rest's two pads hold the puller's arms at the height they have on a fuse."""
LABEL = (0.034, 0.020)
"""Circuit label, along x and y (m)."""
LABEL_X = BLOCK_X - BLOCK_L / 2 - 0.004 - LABEL[0] / 2

COLORS = {
    "block": (0.06, 0.06, 0.07),
    "tin": (0.80, 0.81, 0.82),
    "copper": (0.80, 0.45, 0.22),
    "steel": (0.62, 0.64, 0.67),
    "fuse_tube": (0.86, 0.84, 0.78),
    "fuse_cap": (0.78, 0.79, 0.81),
    "nylon": (0.96, 0.82, 0.10),
    "pan": (0.84, 0.85, 0.83),
    "board": (0.55, 0.42, 0.28),
    "stand": (0.40, 0.40, 0.42),
    "tray": (0.15, 0.30, 0.70),
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
PULLER_QUAT = (0.0, 0.0, 1.0, 0.0)
"""The puller turned 180 deg about z: its +x (toward the handles) along -x, toward the robot."""


def clip_centre(slot: int, end: int) -> tuple[float, float, float]:
    """Desk-frame centre of clip ``end`` (0 = near the robot, 1 = far) of block ``slot``."""
    return (BLOCK_X + (1 if end else -1) * CLIP_CENTRE_X, SLOT_Y[slot], AXIS_Z)


def puller_rest_pose() -> tuple:
    """The puller on its rest: hinge at REST_HINGE_X, arms on the rest's pads."""
    return (REST_HINGE_X, PULLER_REST_Y, AXIS_Z) + PULLER_QUAT


def puller_on_fuse_pose(slot: int) -> tuple:
    """The puller with its jaws on the middle of fuse ``slot`` (closed)."""
    return (BLOCK_X + JAW_X, SLOT_Y[slot], AXIS_Z) + PULLER_QUAT


# Back pan, blocks and labels ------------------------------------------------------------------------------------------
def make_blocks(out_path: Path, clip_k, clip_damping, clip_mu) -> None:
    """``clip_k`` may be one value or one per block (probe runs compare spring rates side by side)."""
    root = "/blocks"
    stage, root_prim = kit.new_stage(out_path, root)
    m = _mats(stage, root, ("block", "tin", "copper", "steel", "pan", "board"))
    b = f"{root}/body"
    UsdGeom.Xform.Define(stage, b)
    ks = list(clip_k) if isinstance(clip_k, (list, tuple)) else [clip_k] * 3
    clip_mats = [_physics_material(stage, root, f"clip{s}", clip_mu, "min", (ks[s], clip_damping)) for s in range(3)]
    housing = _physics_material(stage, root, "housing", 0.4, "min")
    _span(stage, f"{b}/pan", PAN_X, PAN_Y, (PAN_TOP - 0.003, PAN_TOP), m["pan"])
    _span(stage, f"{b}/board", (PAN_X[0] + 0.005, PAN_X[1] - 0.005), (PAN_Y[0] + 0.005, PAN_Y[1] - 0.005),
          (0.0, PAN_TOP - 0.003), m["board"])
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
    # Circuit labels on the pan in front of each block (colour-coded, tag and circuit printed).
    tex_dir = out_path.parent / "textures"
    for s, (tag, circuit, colour) in enumerate(CIRCUITS):
        tm = _tex_material(stage, root, f"fprock14r_circuit_tag_{tag}", f"./textures/fprock14r_circuit_tag_{tag}.png")
        _flat_quad(stage, f"{b}/label{s}", LABEL_X, SLOT_Y[s], PAN_TOP + 0.0004, LABEL[0], LABEL[1], tm)
    _contacts(stage)
    stage.GetRootLayer().Save()
    _labels(tex_dir)
    print(f"wrote {out_path} (clip k {ks} N/m, damping {clip_damping}, mu {clip_mu})")


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
def make_puller(out_path: Path) -> None:
    root = "/puller"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    m = _mats(stage, root, ("nylon",))
    hinge = kit.rigid_body(stage, f"{root}/hinge", (0, 0, 0), HINGE_MASS)
    _span(stage, f"{root}/hinge/loop", (LOOP_X0, 0.002), (-ARM_Y[1], ARM_Y[1]), ARM_Z, m["nylon"])
    rc = JAW_R + JAW_T / 2
    for k, name in enumerate(("arm_a", "arm_b")):
        sy = 1 if k == 0 else -1
        body = kit.rigid_body(stage, f"{root}/{name}", (0, 0, 0), ARM_MASS)
        _rigid_settings(body.GetPrim(), iterations=32)
        p = f"{root}/{name}"
        # The arm: smooth moulded nylon from the loop to NOTCH_FROM, then the notched handle.
        _span(stage, f"{p}/arm", (0.0025, NOTCH_FROM), (sy * ARM_Y[0], sy * ARM_Y[1]), ARM_Z, m["nylon"])
        _span(stage, f"{p}/handle", (NOTCH_FROM, TOP_X), (sy * ARM_Y[0], sy * ARM_Y[1]), ARM_Z, m["nylon"])
        # Finger guard ("wing") behind the jaws (visual).
        _span(stage, f"{p}/wing", (0.040, 0.046), (sy * ARM_Y[0], sy * (ARM_Y[1] + 0.004)),
              (ARM_Z[0], ARM_Z[1] + 0.005), m["nylon"], collide=False)
        # Jaw: a web down from the arm to the curved shell.
        a0 = math.radians(JAW_ANGLES[0])
        web_y = (sy * (rc * math.cos(a0) + JAW_T / 2), sy * ARM_Y[1])
        _span(stage, f"{p}/web", (JAW_X - JAW_HALF, JAW_X + JAW_HALF), web_y, (-rc * math.sin(a0), ARM_Z[0] + 0.001),
              m["nylon"])
        for i, a in enumerate(JAW_ANGLES):
            ar = math.radians(a)
            c = (JAW_X, sy * rc * math.cos(ar), -rc * math.sin(ar))
            seg = 2 * rc * math.tan(math.radians(JAW_SEG_HALF_DEG))
            _rbox(stage, f"{p}/jaw{i}", c, (2 * JAW_HALF, JAW_T, seg), m["nylon"], rot_x_deg=-sy * a)
        lo, hi = (-2.0, OPEN_DEG + 0.5) if k == 0 else (-OPEN_DEG - 0.5, 2.0)
        # Each arm flexes about its own vertical line at the loop (two pivots, not one shared pin: a shared pin
        # would leave the loop free to swing between the held arms).
        pin = (0.0, sy * PIN_Y, 0.5 * (ARM_Z[0] + ARM_Z[1]))
        joint = kit.revolute(stage, f"{root}/joints/{name}_joint", hinge.GetPath(), body.GetPath(), "Z", pin, lo, hi)
        joint.CreateLocalPos1Attr(Gf.Vec3f(*pin))
    mat = _physics_material(stage, root, "nylon", PULLER_FRICTION, "average")
    UsdShade.MaterialBindingAPI(root_prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    smooth = _physics_material(stage, root, "smooth_nylon", SMOOTH_FRICTION, "min")
    notched = _physics_material(stage, root, "notched_handle", NOTCHED_FRICTION, "max")
    for name in ("arm_a", "arm_b"):
        _bind_physics(stage, f"{root}/{name}/arm", smooth)
        _bind_physics(stage, f"{root}/{name}/handle", notched)
    _contacts(stage)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def arm_open_sign(k: int) -> int:
    """Arm a (+y) opens with positive angles about +z (its far end swings to +y), arm b with negative."""
    return 1 if k == 0 else -1


# Stands: the puller's rest and the parts tray -------------------------------------------------------------------------
def make_stands(out_path: Path) -> None:
    root = "/stands"
    stage, _ = kit.new_stage(out_path, root)
    m = _mats(stage, root, ("stand", "tray"))
    y = PULLER_REST_Y
    # Rest: two pads under the arms (puller frame x 32-45 mm and 97-112 mm from the hinge), on posts.
    for i, (u0, u1) in enumerate(((0.032, 0.045), (0.097, 0.112))):
        xs = (REST_HINGE_X - u1, REST_HINGE_X - u0)
        _span(stage, f"{root}/rest_pad{i}", xs, (y - 0.022, y + 0.022), (REST_TOP - 0.004, REST_TOP), m["stand"])
        _span(stage, f"{root}/rest_post{i}", xs, (y - 0.006, y + 0.006), (0.0, REST_TOP - 0.004), m["stand"])
    _span(stage, f"{root}/rest_foot", (REST_HINGE_X - 0.115, REST_HINGE_X - 0.025), (y - 0.022, y + 0.022),
          (0.0, 0.004), m["stand"])
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

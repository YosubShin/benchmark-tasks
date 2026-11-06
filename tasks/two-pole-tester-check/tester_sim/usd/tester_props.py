"""Generate the props (two-pole tester in two parts, proving unit, terminal strip, tool tray) as USD.

Parts (each follows a real product; "inferred" values are not published and are to measure; sources in the
README's References):

- Fluke T150 main unit (`tester_main.usda`), a free rigid body: yellow body with the L2 probe (red shaft,
  GS38 tip) at one end, the LED bargraph and LCD on the display face, the lead's black boot at the other.
  Published (distributor): the pair 260 x 70 x 38 mm, 280 g. Inferred from Fluke's product photo (scaled
  by the 70 mm width): main unit 43 mm wide, 32 mm thick; probe shaft 9 mm square, 40 mm; tip 2 mm,
  4 mm exposed (GS38); the tip sits 13.7 mm off the body's centre line toward the L1 probe's side;
  LEDs 12/24/50/120/230/400/690 V between 62 and 101 mm from the tip. Mass 0.21 kg of the 0.28 kg.
- Fluke T150 L1 probe (`probe_l1.usda`), a free rigid body: black handle 26 x 22 mm, finger guard, a
  slim 10 mm shaft, GS38 tip, 206 mm tip to cable end (inferred from the same photo). 0.06 kg.
  The cable between the two (about 1 m, not limiting at desk scale) is not modelled.
- The fixture (`bench.usda`, static colliders, world coordinates): a vertical mounting plate (the back
  of a control cabinet) with a horizontal 35 mm DIN rail (EN 60715: 35 x 7.5 mm) and five Phoenix
  Contact UT 4 feed-through blocks (L1 L2 L3 grey, N blue, PE green-yellow) at the published 6.2 mm
  pitch, 47.7 mm long and 47.5 mm deep on the rail (published). Each block's two screw wells face the robot; a well is
  a recess the tip must enter to touch the screw (opening 4.5 x 5 mm, 4 mm deep: inferred). A Fluke
  PRV240 proving unit, 117 x 74 x 28 mm (published), hangs face-out on the plate from a magnetic strap
  (Fluke TPAK class); its two recessed contacts (5.5 mm square, 4 mm deep, 24 mm apart, 16 mm above
  its lower edge: measured from Fluke's front photo, depth inferred) face the robot, with the LED between them.
  A tool tray with two U-cradles holds the main unit and the L1 probe level, tips forward.

Collision: boxes only; a round tip of diameter d is a d x d square. The tips and the recess walls get
small contact offsets (0.5 mm) so a 2 mm tip can enter a 4.5 mm recess.

Usage: python -m tester_sim.usd.tester_props tester_sim/assets
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from tester_sim.spec import (
    CONTACT_Z,
    PANEL_X,
    PROVER_Y,
    STRIP_Y0,
    TERMINALS,
    TRAY_GRASP_X,
    TRAY_Y,
    TRAY_Z,
)
from tester_sim.usd import usd_kit as kit

COLORS = dict(kit.COLORS)
COLORS.update({
    "fluke_yellow": (0.95, 0.75, 0.05),
    "probe_red": (0.78, 0.07, 0.05),
    "probe_black": (0.06, 0.06, 0.07),
    "display_grey": (0.28, 0.29, 0.31),
    "lcd": (0.62, 0.66, 0.58),
    "terminal_grey": (0.62, 0.63, 0.62),
    "terminal_blue": (0.10, 0.35, 0.75),
    "terminal_green": (0.15, 0.55, 0.20),
    "terminal_yellow": (0.90, 0.80, 0.10),
    "plate_galv": (0.70, 0.71, 0.70),
    "rail": (0.80, 0.81, 0.82),
    "brass": (0.78, 0.65, 0.30),
    "led_off": (0.20, 0.13, 0.12),
    "led_red": (1.0, 0.05, 0.02),
    "led_yellow": (1.0, 0.80, 0.05),
    "led_green": (0.10, 1.0, 0.20),
    "tray": (0.30, 0.31, 0.33),
    "marker_red": (0.85, 0.08, 0.05),
})
EMISSIVE = {"led_red", "led_yellow", "led_green"}
TIP_CONTACT_OFFSET = 0.0005


# The two probes ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ProbeDims:
    """Probe frame: origin at the tip's end, +z along the axis toward the butt, +x the display normal
    (up when held level), y across the width (the gripper closes along y)."""

    tip: tuple[float, float]
    """Side and exposed length of the GS38 tip."""
    shaft: tuple[float, float]
    """Side and length of the insulated shaft behind the tip."""
    body: tuple[float, float, float]
    """Thickness (x), width (y), length (z) of the handle."""
    body_y: float
    """The handle's centre line, in y off the tip."""
    length: float
    grip_from_butt: float
    mass: float

    @property
    def grasp_z(self) -> float:
        return self.length - self.grip_from_butt

    @property
    def body_z0(self) -> float:
        return self.length - self.body[2]


MAIN = ProbeDims(tip=(0.002, 0.004), shaft=(0.009, 0.040), body=(0.032, 0.043, 0.216), body_y=-0.0137,
                 length=0.260, grip_from_butt=0.045, mass=0.21)
L1 = ProbeDims(tip=(0.002, 0.004), shaft=(0.010, 0.091), body=(0.022, 0.026, 0.101), body_y=0.0,
               length=0.206, grip_from_butt=0.026, mass=0.06)
NOSE, NOSE_LEN = 0.005, 0.015
LED_VOLTS = (12, 24, 50, 120, 230, 400, 690)
LED_Z = {v: 0.101 - 0.0065 * k for k, v in enumerate(LED_VOLTS)}
"""Bargraph LEDs along the display, 12 V nearest the LCD and 690 V nearest the tip."""
LIT_AT_240 = (12, 24, 50, 120, 230)


def _material(stage, root, name):
    path = f"{root}/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[name]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    if name in EMISSIVE:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[name]))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _mats(stage, root, names):
    return {n: _material(stage, root, n) for n in names}


def _small_offsets(stage, paths, offset=TIP_CONTACT_OFFSET):
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        prim.AddAppliedSchema("PhysxCollisionAPI")
        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(offset)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def _probe_common(stage, root, d: ProbeDims, mats, shaft_color, body_color):
    kit.box(stage, f"{root}/tip", (0, 0, d.tip[1] / 2), (d.tip[0], d.tip[0], d.tip[1]), mats["steel"])
    z0 = d.tip[1]
    # The insulated shaft tapers toward the tip (Fluke photo: ~10 mm at the base to ~4 mm near the tip): a 5 mm
    # nose, NOSE_LEN long, then the full section. The nose cannot enter a 4.5 mm screw well.
    kit.box(stage, f"{root}/nose", (0, 0, z0 + NOSE_LEN / 2), (NOSE, NOSE, NOSE_LEN), mats[shaft_color])
    kit.box(stage, f"{root}/shaft", (0, 0, z0 + NOSE_LEN + (d.shaft[1] - NOSE_LEN) / 2),
            (d.shaft[0], d.shaft[0], d.shaft[1] - NOSE_LEN), mats[shaft_color])
    kit.box(stage, f"{root}/body", (0, d.body_y, d.body_z0 + d.body[2] / 2), d.body, mats[body_color])
    kit.friction(stage, root, [f"{root}/body"], static=1.5, dynamic=1.2)
    _small_offsets(stage, [f"{root}/tip", f"{root}/nose", f"{root}/shaft"])


def make_main(out_path: Path) -> None:
    root = "/tester_main"
    stage, prim = kit.new_stage(out_path, root)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(MAIN.mass)
    mats = _mats(stage, root, ("steel", "probe_red", "fluke_yellow", "probe_black", "display_grey", "lcd",
                               "led_off", "led_red", "led_yellow"))
    d = MAIN
    _probe_common(stage, root, d, mats, "probe_red", "fluke_yellow")
    t, w = d.body[0], d.body[1]
    # Lead boot at the butt (visual), display panel and LCD on the +x face.
    kit.box(stage, f"{root}/boot", (0, d.body_y, d.length - 0.012), (t - 0.004, w - 0.012, 0.026),
            mats["probe_black"], collide=False)
    face = t / 2
    kit.box(stage, f"{root}/display", (face + 0.0004, d.body_y, 0.082), (0.0008, w - 0.008, 0.050),
            mats["display_grey"], collide=False)
    kit.box(stage, f"{root}/lcd", (face + 0.0004, d.body_y, 0.124), (0.0008, w - 0.014, 0.024), mats["lcd"],
            collide=False)
    for v in LED_VOLTS:
        on = "led_yellow" if v < 50 else "led_red"
        kit.box(stage, f"{root}/led_{v}_off", (face + 0.0009, d.body_y - 0.008, LED_Z[v]), (0.0008, 0.010, 0.0045),
                mats["led_off"], collide=False)
        kit.box(stage, f"{root}/led_{v}_on", (face + 0.0013, d.body_y - 0.008, LED_Z[v]), (0.0008, 0.010, 0.0045),
                mats[on], collide=False)
        kit.hide(stage, f"{root}/led_{v}_on")
    # The "L2" collar where the red shaft meets the body.
    kit.box(stage, f"{root}/collar", (0, 0, d.tip[1] + d.shaft[1] - 0.003), (0.012, 0.012, 0.006),
            mats["fluke_yellow"], collide=False)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_l1(out_path: Path) -> None:
    root = "/probe_l1"
    stage, prim = kit.new_stage(out_path, root)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(L1.mass)
    mats = _mats(stage, root, ("steel", "probe_black"))
    d = L1
    _probe_common(stage, root, d, mats, "probe_black", "probe_black")
    kit.box(stage, f"{root}/guard", (0, 0, d.body_z0 - 0.004), (0.026, 0.030, 0.008), mats["probe_black"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


# The fixture --------------------------------------------------------------------------------------
RAIL = (0.0075, 0.035)
"""DIN rail depth (x) and height (z), EN 60715 NS 35/7.5."""
BLOCK_PITCH = 0.0062
BLOCK_LEN = 0.0477
BLOCK_HEIGHT = 0.0475 - 0.0075
"""47.5 mm above the mounting plate on NS 35/7.5 (published), less the rail."""
WELL = (0.0045, 0.005, 0.004)
"""Screw well: opening across the block (y), along it (z), depth to the screw head (x). Inferred."""
WELL_FROM_CENTER = 0.0155
"""Each well's centre this far from the block's centre along z (inferred)."""
BLOCK_FACE_X = PANEL_X - RAIL[0] - BLOCK_HEIGHT
RAIL_Z = CONTACT_Z - WELL_FROM_CENTER
"""The top-row wells are at CONTACT_Z."""
BLOCK_COLOR = {"L1": "terminal_grey", "L2": "terminal_grey", "L3": "terminal_grey", "N": "terminal_blue",
               "PE": "terminal_green"}

PRV = (0.028, 0.074, 0.117)
"""PRV240 depth (x), width (y), height (z), published."""
PRV_CONTACT_SPACING = 0.024
PRV_CONTACT_ABOVE_BOTTOM = 0.016
PRV_HOLE = (0.0055, 0.0055, 0.004)
"""Recessed contact opening (y, z) and depth (x): between the ~6.5 mm funnel and the ~4.2 mm contact face
measured from Fluke's front photo; the depth is inferred."""
STRAP_T = 0.003
PRV_FACE_X = PANEL_X - STRAP_T - PRV[0]
PRV_BOTTOM_Z = CONTACT_Z - PRV_CONTACT_ABOVE_BOTTOM

PLATE = ((-0.115, 0.14), (0.04, 0.30), 0.004)
"""Mounting plate y range, z range, thickness."""
CRADLE_Z = (0.0, 0.0)


def terminal_y(name: str) -> float:
    return STRIP_Y0 - TERMINALS.index(name) * BLOCK_PITCH


def well_mouth(name: str, row: int = 0) -> tuple[float, float, float]:
    """World centre of a screw well's mouth (row 0 = top, the one tested)."""
    return (BLOCK_FACE_X, terminal_y(name), CONTACT_Z - row * 2 * WELL_FROM_CENTER)


def prover_mouth(k: int) -> tuple[float, float, float]:
    """World centre of the proving unit's contact k's mouth (0 = +, toward +y; 1 = -)."""
    return (PRV_FACE_X, PROVER_Y + (0.5 - k) * PRV_CONTACT_SPACING, CONTACT_Z)


def _recessed_face(stage, path, x_face, depth, y_range, z_range, holes, hole_yz, material):
    """A solid slab (front face at x_face, depth along +x) with square recesses: boxes around each hole and
    a back layer behind the recess bottoms. holes: list of (y, z) centres."""
    y0, y1 = y_range
    z0, z1 = z_range
    hy, hz = hole_yz
    k = 0

    def b(cy0, cy1, cz0, cz1):
        nonlocal k
        if cy1 - cy0 > 1e-5 and cz1 - cz0 > 1e-5:
            kit.box(stage, f"{path}_{k}", (x_face + depth / 2, (cy0 + cy1) / 2, (cz0 + cz1) / 2),
                    (depth, cy1 - cy0, cz1 - cz0), material)
            k += 1

    zs = sorted({z for _, z in holes})
    edges = [z0]
    for z in zs:
        edges += [z - hz / 2, z + hz / 2]
    edges.append(z1)
    for i in range(0, len(edges), 2):
        b(y0, y1, edges[i], edges[i + 1])  # full-width bands between hole rows
    for z in zs:
        ys = sorted(y for y, zz in holes if zz == z)
        cuts = [y0]
        for y in ys:
            cuts += [y - hy / 2, y + hy / 2]
        cuts.append(y1)
        for i in range(0, len(cuts), 2):
            b(cuts[i], cuts[i + 1], z - hz / 2, z + hz / 2)
    return k


def make_bench(out_path: Path) -> None:
    root = "/bench"
    stage, _ = kit.new_stage(out_path, root)
    mats = _mats(stage, root, ("plate_galv", "rail", "terminal_grey", "terminal_blue", "terminal_green",
                               "terminal_yellow", "marker_red", "steel", "brass", "fluke_yellow", "display_grey", "probe_red",
                               "probe_black", "led_off", "led_red", "led_green", "tray", "stand", "white"))
    # Mounting plate on two legs.
    (py0, py1), (pz0, pz1), pt = PLATE
    kit.box(stage, f"{root}/plate", (PANEL_X + pt / 2, (py0 + py1) / 2, (pz0 + pz1) / 2), (pt, py1 - py0, pz1 - pz0),
            mats["plate_galv"])
    for k, y in enumerate((py0 + 0.02, py1 - 0.02)):
        kit.box(stage, f"{root}/leg{k}", (PANEL_X + pt + 0.02, y, pz0 / 2 + 0.005), (0.04, 0.03, pz0 + 0.01),
                mats["stand"])
        kit.box(stage, f"{root}/foot{k}", (PANEL_X + 0.03, y, 0.004), (0.10, 0.04, 0.008), mats["stand"])
    # DIN rail along y.
    y_hi, y_lo = STRIP_Y0 + 0.020, terminal_y("PE") - 0.020
    kit.box(stage, f"{root}/rail", (PANEL_X - RAIL[0] / 2, (y_hi + y_lo) / 2, RAIL_Z), (RAIL[0], y_hi - y_lo, RAIL[1]),
            mats["rail"])
    # Terminal blocks: a body behind the wells and a recessed front layer.
    back_x0 = BLOCK_FACE_X + WELL[2]
    back_len = PANEL_X - RAIL[0] - back_x0
    for name in TERMINALS:
        y = terminal_y(name)
        color = mats[BLOCK_COLOR[name]]
        base = f"{root}/block_{name}"
        kit.box(stage, f"{base}_body", (back_x0 + back_len / 2, y, RAIL_Z), (back_len, BLOCK_PITCH - 0.0002, BLOCK_LEN),
                color)
        wells = [(y, RAIL_Z + WELL_FROM_CENTER), (y, RAIL_Z - WELL_FROM_CENTER)]
        n = _recessed_face(stage, f"{base}_face", BLOCK_FACE_X, WELL[2], (y - BLOCK_PITCH / 2 + 0.0001,
                           y + BLOCK_PITCH / 2 - 0.0001), (RAIL_Z - BLOCK_LEN / 2, RAIL_Z + BLOCK_LEN / 2), wells,
                           WELL[:2], color)
        _small_offsets(stage, [f"{base}_face_{i}" for i in range(n)] + [f"{base}_body"])
        for r, (_, wz) in enumerate(wells):
            kit.box(stage, f"{base}_screw{r}", (back_x0 - 0.0004, y, wz), (0.0008, WELL[0] - 0.0004, WELL[1] - 0.0004),
                    mats["steel"], collide=False)
        if name == "PE":  # green-yellow: a yellow stripe on the green block
            kit.box(stage, f"{base}_stripe", (BLOCK_FACE_X - 0.0003, y, RAIL_Z), (0.0006, BLOCK_PITCH - 0.001, 0.010),
                    mats["terminal_yellow"], collide=False)
        # Marker card (the printed label, ZB 6 class) in the block's middle: red on L1 (the named phase terminal, so
        # it can be told apart at camera resolution), white on the others.
        kit.box(stage, f"{base}_marker", (BLOCK_FACE_X - 0.0003, y, RAIL_Z), (0.0006, BLOCK_PITCH - 0.0012, 0.008),
                mats["marker_red" if name == "L1" else "white"], collide=False)
    # End cover beyond PE, end clamps.
    y_end = terminal_y("PE") - BLOCK_PITCH / 2 - 0.001
    kit.box(stage, f"{root}/end_cover", (BLOCK_FACE_X + (PANEL_X - RAIL[0] - BLOCK_FACE_X) / 2, y_end, RAIL_Z),
            (PANEL_X - RAIL[0] - BLOCK_FACE_X, 0.002, BLOCK_LEN), mats["terminal_grey"])
    for k, y in enumerate((STRIP_Y0 + BLOCK_PITCH / 2 + 0.005, y_end - 0.006)):
        kit.box(stage, f"{root}/end_clamp{k}", (PANEL_X - RAIL[0] - 0.015, y, RAIL_Z), (0.030, 0.009, 0.040),
                mats["terminal_grey"])

    # PRV240 on its magnetic strap.
    px, py, pz = PRV
    zc = PRV_BOTTOM_Z + pz / 2
    holes = [(prover_mouth(k)[1], CONTACT_Z) for k in (0, 1)]
    kit.box(stage, f"{root}/prv_back", (PRV_FACE_X + PRV_HOLE[2] + (px - PRV_HOLE[2]) / 2, PROVER_Y, zc),
            (px - PRV_HOLE[2], py, pz), mats["fluke_yellow"])
    n = _recessed_face(stage, f"{root}/prv_face", PRV_FACE_X, PRV_HOLE[2], (PROVER_Y - py / 2, PROVER_Y + py / 2),
                       (PRV_BOTTOM_Z, PRV_BOTTOM_Z + pz), holes, PRV_HOLE[:2], mats["fluke_yellow"])
    _small_offsets(stage, [f"{root}/prv_face_{i}" for i in range(n)] + [f"{root}/prv_back"])
    for k, (y, z) in enumerate(holes):
        kit.box(stage, f"{root}/prv_contact{k}", (PRV_FACE_X + PRV_HOLE[2] - 0.0004, y, z),
                (0.0008, PRV_HOLE[0] - 0.0004, PRV_HOLE[1] - 0.0004), mats["steel"], collide=False)
        kit.box(stage, f"{root}/prv_mark{k}", (PRV_FACE_X - 0.0003, y, z + 0.010), (0.0006, 0.009, 0.006),
                mats["probe_red" if k == 0 else "probe_black"], collide=False)
    kit.box(stage, f"{root}/prv_label", (PRV_FACE_X - 0.0003, PROVER_Y, zc + 0.020), (0.0006, py - 0.010, 0.064),
            mats["display_grey"], collide=False)
    kit.box(stage, f"{root}/prv_led_off", (PRV_FACE_X - 0.0004, PROVER_Y, CONTACT_Z + 0.010), (0.0008, 0.004, 0.004),
            mats["led_off"], collide=False)
    kit.box(stage, f"{root}/prv_led_on", (PRV_FACE_X - 0.0008, PROVER_Y, CONTACT_Z + 0.010), (0.0008, 0.0045, 0.0045),
            mats["led_green"], collide=False)
    kit.hide(stage, f"{root}/prv_led_on")
    kit.box(stage, f"{root}/strap", (PANEL_X - STRAP_T / 2, PROVER_Y, zc + 0.03), (STRAP_T, 0.03, 0.08),
            mats["probe_black"], collide=False)

    # Tool tray: a U-cradle pair per tool, holding it level with the grasp point at (TRAY_GRASP_X, y, TRAY_Z).
    for name, d, y in (("main", MAIN, TRAY_Y[0]), ("l1", L1, TRAY_Y[1])):
        tip_x = TRAY_GRASP_X + d.grasp_z
        under = TRAY_Z - d.body[0] / 2
        by = y + d.body_y
        cradles = (d.body_z0 + 0.015, d.grasp_z - 0.045) if name == "main" else (d.body_z0 + 0.010, d.grasp_z - 0.030)
        for k, zt in enumerate(cradles):
            x = tip_x - zt
            kit.box(stage, f"{root}/cradle_{name}{k}_base", (x, by, under - 0.003), (0.012, d.body[1] + 0.016, 0.006),
                    mats["tray"])
            for s in (-1, 1):
                kit.box(stage, f"{root}/cradle_{name}{k}_side{s + 1}",
                        (x, by + s * (d.body[1] / 2 + 0.0045), under + 0.004), (0.012, 0.005, 0.008), mats["tray"])
            kit.box(stage, f"{root}/cradle_{name}{k}_post", (x, by, (under - 0.006) / 2), (0.012, 0.012, under - 0.006),
                    mats["tray"])
        kit.box(stage, f"{root}/tray_{name}_foot", (tip_x - (d.body_z0 + d.grasp_z - 0.03) / 2, by, 0.003),
                (d.grasp_z - d.body_z0 + 0.012, 0.04, 0.006), mats["tray"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def tray_tool_pos(name: str) -> tuple[float, float, float]:
    """World position of a tool's frame origin (its tip end) resting in its cradle, held level, tip +x."""
    d, y = (MAIN, TRAY_Y[0]) if name == "main" else (L1, TRAY_Y[1])
    return (TRAY_GRASP_X + d.grasp_z, y, TRAY_Z)


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    make_main(out_dir / "tester_main.usda")
    make_l1(out_dir / "probe_l1.usda")
    make_bench(out_dir / "bench.usda")

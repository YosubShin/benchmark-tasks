"""Generate the props (stick-mounted detector, proof tester, target bar on its posts) as USD.

Real parts (README References; "inferred" = not published, to measure):
- HD Electric PRX proximity voltage detector (PRX-69D class) on the universal-spline head of a short hot stick
  (`detector.usda`, one free rigid body). Published (PRX manual, p. 9): 11 x 3 x 2 in (280 x 76 x 50 mm), 0.31 kg;
  SENSING ZONE at the top end, face with the voltage-range lights, up/down buttons and ON/OFF button, voltage
  indicating lights and beeper near the base, hot-stick connection (shotgun or universal spline) at the base
  (manual p. 5 figure). Inferred from that figure: the sensing-zone head is narrower (62 x 40 mm, 90 mm long);
  ON button 14 mm, 176 mm from the tip, on the face's right half; indicating lights 255 mm from the tip.
  Body colour HDE red (PRX-69D datasheet photo). Stick: 1-1/4 in (32 mm) fiberglass, 0.615 m behind a universal-spline head
  (a real hot stick is 1.2 m or more: see task.json fidelity); held 45 mm behind the head, where the stick behind the hand
  balances the detector. Assembly about 0.72 kg (detector 0.31 published; head 0.10 and stick 0.5 kg/m inferred).
- HD Electric PT-PRX Proof Tester (part of `bench.usda`, static): published (PT-PRX manual) 4.9 x 2.6 x 1.3 in
  (125 x 66 x 33 mm), 199 g, black ABS, red TEST button, BATTERY OK LED, metal end plate at the end marked "Touch PRX
  to this end". Inferred from the manual's cover photo: TEST button 18 mm across, its centre 57 mm from the plate
  end; LED 13 mm beyond it. The real one is held in the worker's other hand; here it lies face up in a bench cradle.
- Conductor stub (part of `bench.usda`, static): bare 1/0 AAC (9.35 mm, published for the conductor size), 180 mm
  long on two grey polymer post insulators (desk-scale, inferred), isolated (dead).
- A bench cradle for the detector (two posts under its body and its stick).

Collision: boxes only (the round conductor is a 9.4 mm square). Detector frame: origin at the SENSING ZONE end's
centre, +z along the axis toward the stick's butt, +x the face normal (up when held level), y across the width.

Usage: python -m detector_sim.usd.detector_props detector_sim/assets
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from detector_sim.spec import (
    CRADLE_BACK,
    CRADLE_Z,
    GRASP_PROVE,
    TEST_SHIFT,
    TOOL_Z,
    side_dir,
    stick_dir,
)
from detector_sim.usd import usd_kit as kit

COLORS = {
    "hde_red": (0.88, 0.10, 0.07),
    "label_black": (0.08, 0.08, 0.09),
    "stick_yellow": (0.93, 0.78, 0.12),
    "alu": (0.70, 0.72, 0.75),
    "steel": (0.80, 0.81, 0.83),
    "tester_black": (0.07, 0.07, 0.08),
    "test_red": (0.85, 0.08, 0.06),
    "white": (0.95, 0.95, 0.95),
    "led_off": (0.18, 0.13, 0.12),
    "led_red": (1.0, 0.05, 0.02),
    "led_green": (0.10, 1.0, 0.20),
    "conductor": (0.72, 0.74, 0.76),
    "insulator": (0.55, 0.57, 0.58),
    "cradle": (0.30, 0.31, 0.33),
    "lens": (0.78, 0.79, 0.82),
}
EMISSIVE = {"led_red", "led_green"}

# Detector assembly (z from the sensing-zone tip toward the butt).
HEAD = (0.040, 0.062, 0.090)
"""Sensing-zone head: depth (x), width (y), length (z). Inferred."""
BODY = (0.050, 0.076, 0.190)
"""Main body: published overall 280 x 76 x 50 mm, less the head."""
DET_LEN = HEAD[2] + BODY[2]
TAB = (0.016, 0.030, 0.030)
SPLINE = (0.036, 0.036, 0.025)
STICK_SIDE = 0.032
STICK_Z = (DET_LEN + TAB[2] + SPLINE[2], DET_LEN + TAB[2] + SPLINE[2] + 0.615)
GRASP_Z = STICK_Z[0] + 0.045
"""Where arm A grips the stick, from the tip: 45 mm behind the spline head, at the assembly's balance point."""
STICK_KG_PER_M = 0.50
"""1-1/4 in foam-filled fiberglass hot stick (inferred, to measure)."""
DET_KG, HEAD_KG = 0.31, 0.10
STICK_KG = STICK_KG_PER_M * (STICK_Z[1] - STICK_Z[0])
MASS = DET_KG + HEAD_KG + STICK_KG
COM_Z = (DET_KG * 0.14 + HEAD_KG * (DET_LEN + TAB[2] + SPLINE[2] / 2)
         + STICK_KG * (STICK_Z[0] + STICK_Z[1]) / 2) / MASS
ON_BUTTON = (0.176, -0.020, 0.014)
"""ON/OFF button: distance from the tip, y on the face, side."""
RANGE_LIGHT_Z = tuple(0.115 + 0.012 * k for k in range(7))
INDICATOR_Z = 0.255

# PT-PRX.
PT = (0.125, 0.066, 0.033)
"""Length (along the stick), width, height (published)."""
TEST_FROM_PLATE = 0.057
TEST_DIAM = 0.018
TEST_PROUD = 0.002
LED_FROM_PLATE = 0.070

# Conductor.
CRADLE_POST2 = 0.70
"""The second cradle post, this far from the tip along the stick (y = +0.47 m): inside the station table, which ends
at y = +0.65 m; the stick's butt overhangs the table's left edge by about 7 cm."""

COND_SIDE = 0.0094
COND_LEN = 0.18
COND_NEAR = 0.06
"""The conductor extends this far past its contact point toward the proof tester, the rest away from it. Its posts
sit outside the 62 mm sensing head's footprint (a post under the head would stop it short of the bar)."""


def _v(*parts):
    return tuple(float(sum(p[i] for p in parts)) for i in range(3))


def _mul(s, v):
    return tuple(s * c for c in v)


def D3():
    d = stick_dir()
    return (d[0], d[1], 0.0)


def N3v():
    n = side_dir()
    return (n[0], n[1], 0.0)


def grasp_prove() -> tuple:
    return (GRASP_PROVE[0], GRASP_PROVE[1], TOOL_Z)


def plate_point() -> tuple:
    """Centre of the PT-PRX end plate's face (world)."""
    return _v(grasp_prove(), _mul(GRASP_Z, D3()))


def test_button_top() -> tuple:
    return _v(plate_point(), _mul(TEST_FROM_PLATE, D3()), (0.0, 0.0, PT[2] / 2 + TEST_PROUD))


def contact_point() -> tuple:
    """The point on the conductor's near surface the sensing end is meant to touch (world)."""
    return _v(plate_point(), _mul(TEST_SHIFT, N3v()))


def conductor_axis_point() -> tuple:
    return _v(contact_point(), _mul(COND_SIDE / 2, D3()))


def conductor_span() -> tuple[float, float]:
    """Signed extent along +n of the conductor, relative to the contact point."""
    s = 1.0 if TEST_SHIFT < 0 else -1.0  # toward the proof tester
    a, b = s * COND_NEAR, -s * (COND_LEN - COND_NEAR)
    return (min(a, b), max(a, b))


def tip_start() -> tuple:
    """World position of the detector's tip in its start cradle."""
    t = _v(plate_point(), _mul(-CRADLE_BACK, D3()))
    return (t[0], t[1], CRADLE_Z)


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


def _rot_box(stage, path, center, size, yaw_rad, material, collide=True):
    """A box rotated about world z (center in world)."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddRotateZOp().Set(math.degrees(yaw_rad))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _rot_cyl(stage, path, center, radius, height, yaw_rad, axis, material):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr(axis)
    xf = UsdGeom.Xformable(cyl)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddRotateZOp().Set(math.degrees(yaw_rad))
    UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(material)
    return cyl


# Detector -------------------------------------------------------------------------------------------
def make_detector(out_path: Path) -> None:
    root = "/detector"
    stage, prim = kit.new_stage(out_path, root)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(MASS)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, COM_Z))
    m = _mats(stage, root, ("hde_red", "label_black", "stick_yellow", "alu", "white", "led_off", "led_red",
                            "led_green", "lens"))
    kit.box(stage, f"{root}/head", (0, 0, HEAD[2] / 2), HEAD, m["hde_red"])
    kit.box(stage, f"{root}/body", (0, 0, HEAD[2] + BODY[2] / 2), BODY, m["hde_red"])
    kit.box(stage, f"{root}/tab", (0, 0, DET_LEN + TAB[2] / 2), TAB, m["hde_red"])
    kit.box(stage, f"{root}/spline", (0, 0, DET_LEN + TAB[2] + SPLINE[2] / 2), SPLINE, m["alu"])
    kit.box(stage, f"{root}/stick", (0, 0, (STICK_Z[0] + STICK_Z[1]) / 2),
            (STICK_SIDE, STICK_SIDE, STICK_Z[1] - STICK_Z[0]), m["stick_yellow"])
    kit.friction(stage, root, [f"{root}/stick"], static=1.5, dynamic=1.2)
    # "SENSING ZONE" band and the face label (visual).
    kit.box(stage, f"{root}/zone_band", (HEAD[0] / 2 + 0.0004, 0, 0.012), (0.0008, HEAD[1] - 0.006, 0.010),
            m["white"], collide=False)
    fx = BODY[0] / 2
    kit.box(stage, f"{root}/label", (fx + 0.0004, 0, 0.165), (0.0008, BODY[1] - 0.012, 0.130), m["label_black"],
            collide=False)
    for k, z in enumerate(RANGE_LIGHT_Z):
        kit.box(stage, f"{root}/range{k}_off", (fx + 0.0009, 0.022, z), (0.0008, 0.008, 0.005), m["led_off"],
                collide=False)
        kit.box(stage, f"{root}/range{k}_on", (fx + 0.0013, 0.022, z), (0.0008, 0.008, 0.005), m["led_green"],
                collide=False)
        kit.hide(stage, f"{root}/range{k}_on")
        kit.box(stage, f"{root}/range{k}_tag", (fx + 0.0009, 0.004, z), (0.0008, 0.020, 0.004), m["white"],
                collide=False)
    # ON/OFF button (a 2 mm proud collider) and the up/down range buttons (visual).
    z_on, y_on, s_on = ON_BUTTON
    kit.box(stage, f"{root}/on_button", (fx + 0.001, y_on, z_on), (0.002, s_on, s_on), m["white"])
    for k, z in enumerate((0.122, 0.142)):
        kit.box(stage, f"{root}/arrow{k}", (fx + 0.0008, -0.020, z), (0.0016, 0.012, 0.012), m["alu"], collide=False)
    # Voltage indicating lights (two lenses) near the base.
    kit.box(stage, f"{root}/lens", (fx + 0.0004, 0, INDICATOR_Z), (0.0008, 0.050, 0.016), m["lens"], collide=False)
    for s in (-1, 1):
        kit.box(stage, f"{root}/ind{s + 1}_off", (fx + 0.0009, s * 0.015, INDICATOR_Z), (0.0008, 0.012, 0.010),
                m["led_off"], collide=False)
        kit.box(stage, f"{root}/ind{s + 1}_on", (fx + 0.0014, s * 0.015, INDICATOR_Z), (0.0008, 0.013, 0.011),
                m["led_red"], collide=False)
        kit.hide(stage, f"{root}/ind{s + 1}_on")
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


# Bench --------------------------------------------------------------------------------------------
def make_bench(out_path: Path) -> None:
    root = "/bench"
    stage, _ = kit.new_stage(out_path, root)
    m = _mats(stage, root, ("tester_black", "test_red", "steel", "white", "led_off", "led_green", "conductor",
                            "insulator", "cradle", "label_black"))
    d, n = D3(), N3v()
    yaw = math.atan2(d[1], d[0])
    # PT-PRX lying face up, its end plate (at plate_point) facing back along -d toward the detector.
    p = plate_point()
    c = _v(p, _mul(PT[0] / 2, d))
    _rot_box(stage, f"{root}/pt_body", c, PT, yaw, m["tester_black"])
    _rot_box(stage, f"{root}/pt_plate", _v(p, _mul(-0.0003, d)), (0.0008, PT[1] - 0.008, PT[2] - 0.006), yaw,
             m["steel"], collide=False)
    _rot_box(stage, f"{root}/pt_arrow", _v(p, _mul(0.022, d), (0, 0, PT[2] / 2 + 0.0004)), (0.016, 0.040, 0.0008),
             yaw, m["test_red"], collide=False)
    _rot_box(stage, f"{root}/pt_label", _v(p, _mul(0.100, d), (0, 0, PT[2] / 2 + 0.0004)), (0.030, 0.050, 0.0008),
             yaw, m["white"], collide=False)
    tb = _v(p, _mul(TEST_FROM_PLATE, d), (0, 0, PT[2] / 2 + TEST_PROUD / 2))
    s = TEST_DIAM * 0.89  # square of the same area
    _rot_box(stage, f"{root}/pt_test", tb, (s, s, TEST_PROUD), yaw, m["test_red"])
    _rot_cyl(stage, f"{root}/pt_test_cap", _v(tb, (0, 0, 0.0003)), TEST_DIAM / 2, TEST_PROUD, yaw, "Z", m["test_red"])
    led = _v(p, _mul(LED_FROM_PLATE, d), (0, 0, PT[2] / 2 + 0.0005))
    _rot_box(stage, f"{root}/pt_led_off", led, (0.004, 0.004, 0.0008), yaw, m["led_off"], collide=False)
    _rot_box(stage, f"{root}/pt_led_on", _v(led, (0, 0, 0.0004)), (0.0045, 0.0045, 0.0008), yaw, m["led_green"],
             collide=False)
    kit.hide(stage, f"{root}/pt_led_on")
    # Its cradle: a low block under it, a back stop behind its far end.
    under = TOOL_Z - PT[2] / 2
    _rot_box(stage, f"{root}/pt_cradle", _v(c, (0, 0, -PT[2] / 2 - under / 2)), (PT[0] - 0.02, PT[1] - 0.01, under),
             yaw, m["cradle"])
    _rot_box(stage, f"{root}/pt_stop", _v(p, _mul(PT[0] + 0.004, d), (0, 0, 0)), (0.008, PT[1], PT[2]), yaw,
             m["cradle"])

    # Conductor stub along n on two post insulators.
    a, b = conductor_span()
    ca = conductor_axis_point()
    mid = _v(ca, _mul((a + b) / 2, n))
    _rot_box(stage, f"{root}/conductor", mid, (COND_SIDE, b - a, COND_SIDE), yaw, m["conductor"])
    _rot_cyl(stage, f"{root}/conductor_round", mid, COND_SIDE / 2, b - a, yaw, "Y", m["conductor"])
    for k, sk in enumerate((a + 0.011, b - 0.02)):
        base = _v(ca, _mul(sk, n))
        h = TOOL_Z - COND_SIDE / 2
        _rot_box(stage, f"{root}/post{k}", (base[0], base[1], h / 2), (0.022, 0.022, h), yaw, m["insulator"])
        for j in range(4):
            _rot_cyl(stage, f"{root}/post{k}_shed{j}", (base[0], base[1], 0.035 + 0.025 * j), 0.022, 0.004, yaw, "Z",
                     m["insulator"])
        _rot_box(stage, f"{root}/post{k}_clamp", (base[0], base[1], TOOL_Z - COND_SIDE / 2 - 0.003),
                 (0.026, 0.016, 0.006), yaw, m["steel"], collide=False)

    # Detector cradle: two posts under the start pose (one under the body, one under the far end of the stick;
    # the grip point stays free for the fingers).
    t0 = tip_start()
    for k, zl in enumerate((0.20, CRADLE_POST2)):
        q = _v(t0, _mul(-zl, d))
        half = (BODY[0] if zl < DET_LEN else STICK_SIDE) / 2
        h = CRADLE_Z - half
        _rot_box(stage, f"{root}/det_post{k}", (q[0], q[1], h / 2), (0.014, 0.014 if k else 0.05, h), yaw, m["cradle"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    make_detector(out_dir / "detector.usda")
    make_bench(out_dir / "bench.usda")

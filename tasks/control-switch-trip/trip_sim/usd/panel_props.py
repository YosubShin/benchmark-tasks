"""Generate the props: a control-panel section with three Electroswitch Series 24 breaker control switches.

One fixed-base articulation (`panel.usda`). Its frame sits at (FACE_X, 0, 0) in the world: the panel's front surface;
+x points into the panel. Per switch i (shaft at y = SWITCH_Y[i], z = AXIS_Z):

- base: the steel panel (collision), each switch's escutcheon / target nameplate (collision box, black, engraved
  "BREAKER CONTROL", "TRIP", "CLOSE"; a target window at 12 o'clock made by leaving a hole in its front skin), the
  engraved breaker nameplate and the two indicating lamps (green left, red right) above it. Each lamp has a lit and
  a dark lens; the task shows one of them (rule: the breaker's state).
- handle_i (revolute about +x through the shaft; q < 0 = TRIP, anticlockwise seen from the front; q > 0 = CLOSE;
  hard stops +-45 deg = the stop screws): round shank and the pistol grip, which hangs DOWN at NORMAL (as on the
  real panels: ref/photo_haer_*.jpg, ref/typeW2_photo_*.png). The return spring is the joint drive (stiffness
  SPRING_K toward 0, never re-targeted by the task): a torsion spring. Behind the panel the shaft carries a drive pin.
- target_i (revolute about +x, coaxial, +-20 deg stops): the red/green semaphore flag. It is driven only by contact:
  two lugs on it straddle the shaft's drive pin with lost motion, so turning the handle to TRIP pushes the flag to
  green and to CLOSE pushes it to red, and the spring return carries the pin back through the gap without moving the
  flag. The flag carries a red and a green patch; whichever sits behind the window shows.

Self-collision on; the base is filtered against every handle and target (they pass through it).
Values: (P) published, (I) inferred, (TM) to measure. Sources: the README's References.

Usage: python -m trip_sim.usd.panel_props trip_sim/assets
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from trip_sim.spec import AXIS_Z, BREAKERS, PANEL_LEFT_Y, PANEL_RIGHT_Y, SWITCH_Y
from trip_sim.usd import usd_kit as kit

IN = 0.0254
# --- Electroswitch Series 24, model 2438D -------------------------------------------------------------------
PLATE_W, PLATE_H = 2.81 * IN, 2.91 * IN
"""(P) nameplate / escutcheon 2.81 x 2.91 in (catalog p11)."""
PLATE_T = 0.006
"""(I) escutcheon thickness in front of the panel."""
HANDLE_DEPTH = 2.44 * IN
"""(P, oval handle) the handle's front stands 2.44 in from the panel face; used for the pistol grip too (I)."""
GRIP_T = 0.015
"""(I, photos) pistol grip thickness (front to back), ~0.6 in."""
GRIP_W = 0.019
"""(I, photos) pistol grip width, ~0.75 in."""
GRIP_LEN = 0.066
"""(I, photos) shaft to grip tip, ~2.6 in; the grip hangs down at NORMAL."""
GRIP_HUB = 0.012
"""(I) the grip's hub end reaches this far above the shaft."""
SHANK_D = 0.016
"""(I, photos) round shank between the escutcheon and the grip."""
THROW_DEG = 45.0
"""(P) 45 deg to TRIP (anticlockwise from the front) and to CLOSE (clockwise); spring return to vertical."""
TORQUE_NM = 8 * 0.113
"""(P) operating torque 8 in-lb for a one-section Series 24 (catalog p11 table); the 2438D has one deck."""
SPRING_K = TORQUE_NM / math.radians(THROW_DEG)
"""Torsion spring as a linear joint drive: TORQUE_NM at the stop, 0 at NORMAL (the real spring is preloaded: TM)."""
CONTACT_DEG = 40.0
"""(I) the TRIP / CLOSE contacts make within the last 5 deg of the throw."""
# Target (semaphore flag)
TARGET_STOP_DEG = 20.0
TARGET_SHOWN_DEG = 17.0
"""Where the drive pin leaves the flag at the end of the throw (see the lost-motion numbers below)."""
WINDOW_R = 0.021
"""(I, W-2 photo) target window at 12 o'clock, this far above the shaft."""
WINDOW = (0.008, 0.007)
FLAG = (0.009, 0.009)
PIN_R = 0.010
PIN = 0.0025
PIN_X = (0.003, 0.009)
"""Drive pin / lugs behind the panel face (x in the panel frame)."""
LUG_DEG = THROW_DEG - TARGET_SHOWN_DEG + math.degrees(2 * math.atan((PIN / 2) / PIN_R))
"""Lug centres either side of the flag's centre line: at the TRIP stop (-45) the pin has pushed the flag to -17."""
# Lamps and nameplate (panel layout, inferred from ref/photo_haer_30thSt_cb_control_switches_closeup.jpg)
LAMP_DZ = 0.075
LAMP_DY = 0.022
LAMP_R = 0.011
"""(I) 7/8 in class indicating-lamp lens."""
LAMP_PROUD = 0.020
NAMEPLATE_DZ = 0.115
NAMEPLATE = (0.076, 0.026)
"""(I) engraved lamicoid nameplate, 3 x 1 in."""
PANEL_T = 3 / 16 * IN
"""(P) panel up to 3/16 in thick."""
PANEL_Y = (PANEL_LEFT_Y, PANEL_RIGHT_Y)
PANEL_TOP = AXIS_Z + 0.16


def window_shows_green(q_target: float) -> bool:
    return q_target < -TARGET_SHOWN_DEG / 2 * math.pi / 180


# --- textures -----------------------------------------------------------------------------------------------
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"


def _font(path, size):
    return ImageFont.truetype(path, size)


def make_textures(tex: Path) -> None:
    tex.mkdir(parents=True, exist_ok=True)
    for dev, line2, bg, fg, _ in BREAKERS:
        # Engraved two-colour lamicoid: the plate's colour is the marker visible from afar.
        img = Image.new("RGB", (380, 130), bg)
        d = ImageDraw.Draw(img)
        d.rectangle([3, 3, 376, 126], outline=fg, width=4)
        d.text((190, 40), dev, fill=fg, font=_font(FONT_B, 46), anchor="mm")
        d.text((190, 95), line2, fill=fg, font=_font(FONT_B, 34), anchor="mm")
        img.save(tex / f"name_{dev}.png")
    for text, size in (("BREAKER CONTROL", (420, 50)), ("TRIP", (110, 50)), ("CLOSE", (130, 50)),
                       ("ELECTROSWITCH", (260, 40))):
        img = Image.new("RGB", size, (14, 14, 15))
        ImageDraw.Draw(img).text((size[0] / 2, size[1] / 2), text, fill=(235, 235, 235),
                                 font=_font(FONT_B if text != "ELECTROSWITCH" else FONT_R, 28), anchor="mm")
        img.save(tex / f"engr_{text.split()[0].lower()}.png")


# --- USD helpers --------------------------------------------------------------------------------------------
COLORS = {
    "panel": (0.62, 0.64, 0.64),  # ANSI 61 light grey
    "black": (0.04, 0.04, 0.045),
    "handle": (0.06, 0.06, 0.065),
    "chrome": (0.70, 0.71, 0.72),
    "red": (0.85, 0.07, 0.06),
    "green": (0.05, 0.70, 0.22),
    "red_dark": (0.16, 0.03, 0.03),
    "green_dark": (0.02, 0.09, 0.04),
    "white": (0.93, 0.93, 0.92),
}


def _mats(stage, root):
    out = {}
    for name, rgb in COLORS.items():
        path = f"{root}/Looks/{name}"
        m = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35 if name in ("handle", "chrome") else 0.6)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        out[name] = m
    for name, rgb in (("lit_red", (1.0, 0.05, 0.03)), ("lit_green", (0.05, 1.0, 0.2))):
        path = f"{root}/Looks/{name}"
        m = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(0.8 * c for c in rgb)))
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        out[name] = m
    return out


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


def _front_quad(stage, path, x, y, z, w, h, material):
    """A textured quad facing the robot (-x); the image's left edge at +y (the robot's left)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(x, y + w / 2, z - h / 2), Gf.Vec3f(x, y - w / 2, z - h / 2),
                           Gf.Vec3f(x, y - w / 2, z + h / 2), Gf.Vec3f(x, y + w / 2, z + h / 2)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _rot_box(stage, path, center, size, angle_deg, material, collide=True):
    """A box rotated about x by angle_deg (positive = about +x)."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddRotateXOp().Set(angle_deg)
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _at_angle(r, deg):
    """(y, z) of a point at radius r, angle deg about +x from +z."""
    a = math.radians(deg)
    return (-r * math.sin(a), r * math.cos(a))


def _filter(stage, prim_path, other):
    api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(prim_path))
    api.CreateFilteredPairsRel().AddTarget(Sdf.Path(other))


def _contact_offsets(stage, paths, offset=0.001, rest=0.0):
    for p in paths:
        prim = stage.GetPrimAtPath(p)
        prim.AddAppliedSchema("PhysxCollisionAPI")
        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(offset)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(rest)


def make_panel(out_path: Path) -> None:
    root = "/panel"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=True)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    m = _mats(stage, root)
    tex = {f"name_{d}": _tex_material(stage, root, f"name_{d.replace('-', '_')}", f"./tex/name_{d}.png")
           for d, *_ in BREAKERS}
    for k in ("breaker", "trip", "close", "electroswitch"):
        tex[k] = _tex_material(stage, root, f"engr_{k}", f"./tex/engr_{k}.png")

    base = kit.rigid_body(stage, f"{root}/base", (0, 0, 0), 20.0)
    b = f"{root}/base"
    py0, py1 = PANEL_Y
    kit.box(stage, f"{b}/panel", (PANEL_T / 2, (py0 + py1) / 2, PANEL_TOP / 2), (PANEL_T, py0 - py1, PANEL_TOP),
            m["panel"])
    kit.box(stage, f"{b}/panel_back", (PANEL_T + 0.05, (py0 + py1) / 2, PANEL_TOP / 2), (0.1, py0 - py1, PANEL_TOP),
            m["panel"], collide=False)
    hx0 = -HANDLE_DEPTH  # front of the grip
    for i, (dev, *_) in enumerate(BREAKERS):
        y0, z0 = SWITCH_Y[i], AXIS_Z
        # Escutcheon: one collision box; visual = body + front skin with the window hole.
        coll = kit.box(stage, f"{b}/plate_{i}", (-PLATE_T / 2, y0, z0), (PLATE_T, PLATE_W, PLATE_H), m["black"])
        UsdGeom.Imageable(coll.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide)
        kit.box(stage, f"{b}/plate_body_{i}", (-(PLATE_T - 0.002) / 2, y0, z0), (PLATE_T - 0.002, PLATE_W, PLATE_H),
                m["black"], collide=False)
        sx = -PLATE_T + 0.0005
        wy, wz = WINDOW
        wz0 = z0 + WINDOW_R
        kit.box(stage, f"{b}/skin_top_{i}", (sx, y0, (wz0 + wz / 2 + z0 + PLATE_H / 2) / 2),
                (0.001, PLATE_W, z0 + PLATE_H / 2 - wz0 - wz / 2), m["black"], collide=False)
        kit.box(stage, f"{b}/skin_bot_{i}", (sx, y0, (wz0 - wz / 2 + z0 - PLATE_H / 2) / 2),
                (0.001, PLATE_W, wz0 - wz / 2 - z0 + PLATE_H / 2), m["black"], collide=False)
        side = (PLATE_W - wy) / 2
        for s, sy in (("l", 1), ("r", -1)):
            kit.box(stage, f"{b}/skin_{s}_{i}", (sx, y0 + sy * (wy / 2 + side / 2), wz0), (0.001, side, wz),
                    m["black"], collide=False)
        fx = -PLATE_T - 0.0002
        _front_quad(stage, f"{b}/engr_breaker_{i}", fx, y0, z0 + 0.031, 0.050, 0.006, tex["breaker"])
        _front_quad(stage, f"{b}/engr_trip_{i}", fx, y0 + 0.020, z0 + 0.016, 0.014, 0.0064, tex["trip"])
        _front_quad(stage, f"{b}/engr_close_{i}", fx, y0 - 0.020, z0 + 0.016, 0.017, 0.0064, tex["close"])
        _front_quad(stage, f"{b}/engr_es_{i}", fx, y0, z0 - 0.031, 0.032, 0.005, tex["electroswitch"])
        for s, dy in (("l", 1), ("r", -1)):
            kit.cylinder(stage, f"{b}/screw_{s}_{i}", (fx, y0 + dy * 1.0 * IN, z0 + 1.09 * IN), 0.003, 0.002,
                         m["chrome"], axis="X")
        kit.cylinder(stage, f"{b}/screw_b_{i}", (fx, y0, z0 - 1.09 * IN), 0.003, 0.002, m["chrome"], axis="X")
        # Lamps: green (breaker open) left, red (breaker closed) right. Lit and dark lenses; the task shows one.
        for colour, dy in (("green", LAMP_DY), ("red", -LAMP_DY)):
            ly, lz = y0 + dy, z0 + LAMP_DZ
            kit.cylinder(stage, f"{b}/lamp_{colour}_bezel_{i}", (-0.003, ly, lz), LAMP_R + 0.003, 0.006, m["chrome"],
                         axis="X")
            for state, mat in (("on", m[f"lit_{colour}"]), ("off", m[f"{colour}_dark"])):
                kit.cylinder(stage, f"{b}/lamp_{colour}_{state}_{i}", (-LAMP_PROUD / 2, ly, lz), LAMP_R, LAMP_PROUD,
                             mat, axis="X")
        _front_quad(stage, f"{b}/name_{i}", -0.0012, y0, z0 + NAMEPLATE_DZ, *NAMEPLATE, tex[f"name_{dev}"])
        kit.box(stage, f"{b}/name_back_{i}", (-0.0006, y0, z0 + NAMEPLATE_DZ),
                (0.001, NAMEPLATE[0] + 0.004, NAMEPLATE[1] + 0.004), m["black"], collide=False)
    kit.fix_to_world(stage, root, base.GetPath())

    for i in range(len(BREAKERS)):
        y0, z0 = SWITCH_Y[i], AXIS_Z
        # Handle: shank + pistol grip hanging down + drive pin behind the panel.
        h = kit.rigid_body(stage, f"{root}/handle_{i}", (0, y0, z0), 0.08)
        hp = f"{root}/handle_{i}"
        grip_x = hx0 + GRIP_T / 2
        shank_len = -PLATE_T - (hx0 + GRIP_T) + 0.002
        kit.box(stage, f"{hp}/shank", (hx0 + GRIP_T + shank_len / 2 - 0.001, 0, 0), (shank_len, SHANK_D, SHANK_D),
                m["handle"], collide=True)
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{hp}/shank")).CreatePurposeAttr(UsdGeom.Tokens.guide)
        kit.cylinder(stage, f"{hp}/shank_vis", (hx0 + GRIP_T + shank_len / 2 - 0.001, 0, 0), SHANK_D / 2, shank_len,
                     m["handle"], axis="X")
        glen = GRIP_LEN + GRIP_HUB
        kit.box(stage, f"{hp}/grip", (grip_x, 0, (GRIP_HUB - GRIP_LEN) / 2), (GRIP_T, GRIP_W, glen), m["handle"])
        # Tapered look: a wider rounded tip (visual).
        kit.cylinder(stage, f"{hp}/grip_tip", (grip_x, 0, -GRIP_LEN + 0.002), GRIP_W / 2 + 0.0015, GRIP_T,
                     m["handle"], axis="X")
        kit.cylinder(stage, f"{hp}/grip_hub", (grip_x, 0, 0), GRIP_W / 2, GRIP_T + 0.0005, m["handle"], axis="X")
        kit.cylinder(stage, f"{hp}/shaft", (PANEL_T / 2 + 0.004, 0, 0), 0.0055, 0.016, m["chrome"], axis="X")
        kit.box(stage, f"{hp}/pin", ((PIN_X[0] + PIN_X[1]) / 2 + PANEL_T, 0, PIN_R),
                (PIN_X[1] - PIN_X[0], PIN, PIN), m["chrome"])
        kit.revolute(stage, f"{root}/joints/handle_{i}_joint", base.GetPath(), h.GetPath(), "X", (0, y0, z0),
                     -THROW_DEG, THROW_DEG)
        # Target: flag patches behind the window + lugs either side of the pin.
        t = kit.rigid_body(stage, f"{root}/target_{i}", (0, y0, z0), 0.01)
        tp = f"{root}/target_{i}"
        for name, deg in (("red", -TARGET_SHOWN_DEG), ("green", TARGET_SHOWN_DEG)):
            yy, zz = _at_angle(WINDOW_R, deg)
            _rot_box(stage, f"{tp}/flag_{name}", (-PLATE_T + 0.0015, yy, zz), (0.0004, *FLAG), deg, m[name],
                     collide=False)
        for s, deg in (("a", LUG_DEG), ("b", -LUG_DEG)):
            yy, zz = _at_angle(PIN_R, deg)
            _rot_box(stage, f"{tp}/lug_{s}", ((PIN_X[0] + PIN_X[1]) / 2 + PANEL_T, yy, zz),
                     (PIN_X[1] - PIN_X[0], PIN, PIN), deg, m["chrome"])
        kit.revolute(stage, f"{root}/joints/target_{i}_joint", base.GetPath(), t.GetPath(), "X", (0, y0, z0),
                     -TARGET_STOP_DEG, TARGET_STOP_DEG)
        UsdPhysics.MassAPI(stage.GetPrimAtPath(tp)).CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
        UsdPhysics.MassAPI(stage.GetPrimAtPath(tp)).CreateDiagonalInertiaAttr(Gf.Vec3f(2e-6, 2e-6, 2e-6))
        UsdPhysics.MassAPI(stage.GetPrimAtPath(hp)).CreateCenterOfMassAttr(Gf.Vec3f(-0.04, 0, -0.02))
        for link in (hp, tp):
            _filter(stage, link, b)
        _contact_offsets(stage, [f"{hp}/pin", f"{tp}/lug_a", f"{tp}/lug_b"], 0.001, 0.0)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    make_textures(out_dir / "tex")
    make_panel(out_dir / "panel.usda")
    print(f"spring {SPRING_K:.3f} N*m/rad, {TORQUE_NM:.3f} N*m at the stop; lug centres +-{LUG_DEG:.1f} deg")

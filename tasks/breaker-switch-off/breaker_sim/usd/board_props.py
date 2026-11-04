"""Generate the board: one DIN rail of six Schneider Acti9 iC60N 1P breakers behind a dead-front cover.

One fixed-base articulation (`board.usda`). Its frame sits at (FACE_X, 0, RAIL_Z) in the world: the plane of
the breakers' front faces, the rail's centre line, the row's centre; +x points into the board.

- base: back pan, TH35 rail, the six breaker bodies (rear body visual; nose collision box), the dead-front
  cover with its 45 mm cut-out (collision), the printed label strip ABOVE the cut-out (one label per way, coloured
  by circuit type, with the way number and circuit name) and the circuit directory card (textures).
- handle_i (revolute about +y at the pivot; q > 0 tip up = ON, q < 0 tip down = OFF; hard stops at +-Q_THROW):
  the grey paddle (collision box), a cam sphere behind the pivot (collision, inside the body), and the
  white rocker drum around the pivot that carries the printed "I-ON" (below the bar when ON) and the green
  "O-OFF" band (above the bar when OFF), seen through the round opening in the face, as on the real iC60N.
- plunger_i (prismatic along +x): a sprung block behind the cam. The spring (a joint drive, stiffness
  PLUNGER_K, preloaded) presses the block onto the cam sphere. The sphere's farthest point from the pivot
  along x is R*cos(q) + r, largest at q = 0, so the spring pushes the handle away from centre: it holds at
  either stop and snaps over once pushed past centre. That is the detent, made by contact geometry.

Self-collision is on; the base is filtered against every handle and plunger (they sit inside its noses).
Values: (P) published, (I) inferred, (TM) to measure on a bought part. Sources: README.md, References.

Usage: python -m breaker_sim.usd.board_props breaker_sim/assets
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from breaker_sim.spec import CIRCUITS, PITCH, RAIL_Z
from breaker_sim.usd import usd_kit as kit

# --- iC60N 1P and the board ------------------------------------------------------------------------
MODULE_H = 0.085
"""(P) overall height."""
NOSE_H = 0.045
"""(P) DIN 43880 front face height, which passes through the cover's cut-out."""
FACE_FROM_RAIL = 0.064
"""(P) rail flange (the body's back) to the front face (catalogue drawing DB123586: 64 mm; 69.5 incl. the 5.5 mm
clip zone; 78.5 mm to the handle tip, so the handle stands 9 mm proud)."""
NOSE_PROUD = 0.018
"""(P/I) front face to the cover plane: the 45 mm section starts 19.5 mm behind the face (69.5 - 50, P); the
cover sits on that shoulder (I)."""
COVER_T = 0.0015
RAIL_DEPTH, RAIL_H, RAIL_T = 0.0075, 0.035, 0.001
"""(P) EN 60715 TH35-7.5 top-hat rail."""
PIVOT_DZ = -0.0075
"""(I, drawing front view) handle pivot (centre of the round recess) 7.5 mm below the face's centre."""
PIVOT_IN = 0.0065
"""(I) pivot behind the front face: the rocker drum (radius DRUM_R) is flush with the face."""
DRUM_R = 0.0063
HANDLE_LEN = 0.016
"""(I) pivot to the bar's centre: the bar's tip stands 16*cos(30) - 6.5 + 2.5 = 9.9 mm proud at a stop (P: 9 mm)."""
HANDLE_W = 0.013
"""(I, photo) the black grip bar spans 13 mm of the 18 mm module."""
HANDLE_T = 0.005
"""(I, photo) grip bar diameter; the white rocker arm behind it is the same thickness."""
Q_THROW = math.radians(30.0)
"""(I) +-30 degrees between the stops."""
CAM_R = 0.008
"""(I) cam sphere centre behind the pivot."""
CAM_BALL = 0.002
"""(I) cam sphere radius."""
PLUNGER_K = 5000.0
"""(I/TM) plunger spring, N/m."""
PLUNGER_PRELOAD = 0.0042
"""(I/TM) spring compression with the handle at a stop: 21 N, 0.084 N*m on the handle, 7 N at a finger 12 mm
from the pivot (Yao & Chen 2016 simulate 5-10 N MCB handle forces and call 7 N ideal; nothing is published)."""
PLUNGER_SIZE = (0.004, 0.007, 0.010)
OPENING = (0.014, 0.016)
"""(I, photo) the face's opening around the rocker (round on the real part): width (y), height (z)."""
FLAG_DEG = 45.0
"""(I, photo) the markings' centres on the drum, this far below (I-ON) and above (O-OFF) the bar."""
FLAG_HALF_DEG = 15.0
FACE_PLATE = 0.0006
BACK_X = FACE_FROM_RAIL + RAIL_DEPTH
"""Back pan's front surface, board frame."""
COVER_X = NOSE_PROUD
COVER_W, COVER_H = 0.26, 0.17
"""Cover 26 cm wide (was 20 cm: the directory card overhung its right edge)."""
LABEL_W, LABEL_H = 0.017, 0.018
"""(I) one printed label per 18 mm way, on a strip directly above the cut-out (labels above the devices, as on
most boards)."""
LABEL_GAP = 0.003
"""(I) cut-out edge to the label strip's bottom edge."""
LABEL_Z = 0.045 / 2 + LABEL_GAP + LABEL_H / 2
"""Label centre above the rail centre line (board frame)."""
DIRECTORY = (0.056, 0.0616)
"""(I) circuit directory card, width x height."""
DIRECTORY_GAP = 0.010
"""(I) cut-out's right edge to the card's left edge."""


def cam_end_x(q: float) -> float:
    """Farthest point of the cam sphere along +x from the pivot at handle angle q."""
    return CAM_R * math.cos(q) + CAM_BALL


PLUNGER_FACE_END = cam_end_x(Q_THROW)
"""Plunger face position (from the pivot) touching the cam with the handle at a stop."""


def handle_torque(q: float) -> float:
    """Detent torque (N*m) about the pivot, toward the nearer stop, from the geometry and spring."""
    force = PLUNGER_K * (cam_end_x(q) - PLUNGER_FACE_END + PLUNGER_PRELOAD)
    return force * CAM_R * math.sin(abs(q))


def pivot_local(slot: int) -> tuple[float, float, float]:
    return (PIVOT_IN, (2.5 - slot) * PITCH, PIVOT_DZ)


# --- textures --------------------------------------------------------------------------------------
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"


def _font(path, size):
    return ImageFont.truetype(path, size)


def _png(path: Path, size, bg, draw_fn):
    img = Image.new("RGB", size, bg)
    draw_fn(ImageDraw.Draw(img), img)
    img.save(path)


def make_textures(tex: Path) -> None:
    tex.mkdir(parents=True, exist_ok=True)
    for way, circuit, rating, colour, ink in CIRCUITS:
        def label(d, img, way=way, circuit=circuit, ink=ink):
            d.rectangle([1, 1, img.width - 2, img.height - 2], outline=(30, 30, 30), width=2)
            d.text((img.width / 2, 34), str(way), fill=ink, font=_font(FONT_B, 54), anchor="mm")
            words = circuit.replace(" - ", " ").split()
            for i, w in enumerate(words):
                d.text((img.width / 2, 92 + i * 22 - (len(words) - 1) * 11), w, fill=ink,
                       font=_font(FONT_B, 20 if len(w) <= 7 else 16), anchor="mm")
            d.text((img.width / 2, 124), rating, fill=ink, font=_font(FONT_R, 16), anchor="mm")
        _png(tex / f"label_{way}.png", (128, 136), colour, label)

        def front(d, img, rating=rating):
            d.text((img.width / 2, 22), "Schneider", fill=(61, 205, 88), font=_font(FONT_B, 22), anchor="mm")
            d.text((img.width / 2, 44), "Electric", fill=(61, 205, 88), font=_font(FONT_R, 16), anchor="mm")
            d.text((img.width / 2, 74), "iC60N", fill=(40, 40, 40), font=_font(FONT_B, 20), anchor="mm")
            d.text((img.width / 2, 102), rating, fill=(40, 40, 40), font=_font(FONT_B, 24), anchor="mm")
        _png(tex / f"front_{way}.png", (128, 120), (238, 239, 236), front)

    def flag_on(d, img):
        d.text((img.width / 2, img.height / 2), "I-ON", fill=(70, 70, 70), font=_font(FONT_B, 34), anchor="mm")
    _png(tex / "flag_on.png", (128, 64), (236, 237, 234), flag_on)

    def flag_off(d, img):
        d.text((img.width / 2, img.height / 2), "O-OFF", fill=(255, 255, 255), font=_font(FONT_B, 30), anchor="mm")
    _png(tex / "flag_off.png", (128, 64), (40, 175, 70), flag_off)

    def directory(d, img):
        d.rectangle([3, 3, img.width - 4, img.height - 4], outline=(20, 20, 20), width=4)
        d.text((img.width / 2, 30), "CIRCUIT DIRECTORY", fill=(0, 0, 0), font=_font(FONT_B, 23), anchor="mm")
        d.text((img.width / 2, 58), "PANEL AUX-1  230 V", fill=(60, 60, 60), font=_font(FONT_R, 20), anchor="mm")
        d.line([12, 76, img.width - 12, 76], fill=(0, 0, 0), width=2)
        for i, (way, circuit, rating, colour, _) in enumerate(CIRCUITS):
            y = 100 + i * 36
            d.rectangle([14, y - 13, 44, y + 13], fill=colour, outline=(20, 20, 20), width=2)
            d.text((29, y), f"{way}", fill=(0, 0, 0) if sum(colour) > 450 else (255, 255, 255),
                   font=_font(FONT_B, 22), anchor="mm")
            d.text((52, y), circuit, fill=(0, 0, 0), font=_font(FONT_R, 17), anchor="lm")
            d.text((img.width - 14, y), rating, fill=(0, 0, 0), font=_font(FONT_R, 19), anchor="rm")
    _png(tex / "directory.png", (300, 330), (252, 250, 236), directory)


# --- USD helpers -----------------------------------------------------------------------------------
COLORS = {
    "ic60_white": (0.90, 0.91, 0.89),
    "ic60_grey": (0.42, 0.45, 0.50),
    "ic60_dark": (0.035, 0.035, 0.04),
    "pan": (0.62, 0.63, 0.62),
    "cover": (0.86, 0.86, 0.84),
    "rail": (0.74, 0.76, 0.78),
    "plunger": (0.3, 0.3, 0.3),
    "red": (0.84, 0.12, 0.12),
    "green": (0.16, 0.69, 0.28),
}


def _mats(stage, root):
    out = {}
    for name, rgb in COLORS.items():
        path = f"{root}/Looks/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        out[name] = material
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


def _quad(stage, path, corners, material):
    """A textured quad (visual). corners: image bottom-left, bottom-right, top-right, top-left."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _front_quad(stage, path, x, y, z, w, h, material):
    """A quad facing the robot (-x); the image's left edge at +y (the robot's left)."""
    return _quad(stage, path, [(x, y + w / 2, z - h / 2), (x, y - w / 2, z - h / 2), (x, y - w / 2, z + h / 2),
                               (x, y + w / 2, z + h / 2)], material)


def _sphere(stage, path, center, radius, material):
    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(radius)
    UsdGeom.Xformable(sph).AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdShade.MaterialBindingAPI.Apply(sph.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
    return sph


def _filter(stage, prim_path, other):
    api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(prim_path))
    api.CreateFilteredPairsRel().AddTarget(Sdf.Path(other))


def _contact_offsets(stage, paths, offset=0.0015, rest=0.0):
    for p in paths:
        prim = stage.GetPrimAtPath(p)
        prim.AddAppliedSchema("PhysxCollisionAPI")
        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(offset)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(rest)


def make_board(out_path: Path) -> None:
    root = "/board"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim, self_collisions=True)
    root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    mats = _mats(stage, root)
    tex = {}
    for way, *_ in CIRCUITS:
        tex[f"label_{way}"] = _tex_material(stage, root, f"label_{way}", f"./tex/label_{way}.png")
        tex[f"front_{way}"] = _tex_material(stage, root, f"front_{way}", f"./tex/front_{way}.png")
    for name in ("flag_on", "flag_off", "directory"):
        tex[name] = _tex_material(stage, root, name, f"./tex/{name}.png")

    base = kit.rigid_body(stage, f"{root}/base", (0, 0, 0), 5.0)
    b = f"{root}/base"
    # Back pan (the enclosure's mounting plate), reaching down to the desk.
    kit.box(stage, f"{b}/pan", (BACK_X + 0.001, 0, -RAIL_Z / 2 + 0.06), (0.002, COVER_W + 0.02, RAIL_Z + 0.12),
            mats["pan"], collide=False)
    # TH35 rail: web on two standoffs (visual).
    rail_front = BACK_X - RAIL_DEPTH
    kit.box(stage, f"{b}/rail_web", (rail_front + RAIL_T / 2, 0, 0), (RAIL_T, COVER_W, 0.027), mats["rail"], collide=False)
    for s, z in (("top", 0.016), ("bot", -0.016)):
        kit.box(stage, f"{b}/rail_{s}_flange", (BACK_X - RAIL_T / 2, 0, z + math.copysign(0.0025, z)),
                (RAIL_T, COVER_W, 0.005), mats["rail"], collide=False)
        kit.box(stage, f"{b}/rail_{s}_side", (rail_front + RAIL_DEPTH / 2, 0, z), (RAIL_DEPTH, COVER_W, RAIL_T),
                mats["rail"], collide=False)
    # Breakers: rear body (85 mm tall, behind the cover) and the 45 mm nose (collision).
    row_w = 6 * PITCH
    for slot, (way, *_) in enumerate(CIRCUITS):
        y = (2.5 - slot) * PITCH
        kit.box(stage, f"{b}/body_{slot}", ((COVER_X + rail_front) / 2, y, 0),
                (rail_front - COVER_X, PITCH - 0.0004, MODULE_H), mats["ic60_white"], collide=False)
        # Nose: four visual plates around the window, a front texture, and one collision box behind the face.
        kit.box(stage, f"{b}/nose_{slot}", (FACE_PLATE + (COVER_X - FACE_PLATE) / 2, y, 0),
                (COVER_X - FACE_PLATE, PITCH - 0.0004, NOSE_H), mats["ic60_white"])
        top_lo, bot_hi = PIVOT_DZ + OPENING[1] / 2, PIVOT_DZ - OPENING[1] / 2
        kit.box(stage, f"{b}/face_top_{slot}", (FACE_PLATE / 2, y, (top_lo + NOSE_H / 2) / 2),
                (FACE_PLATE, PITCH - 0.0004, NOSE_H / 2 - top_lo), mats["ic60_white"], collide=False)
        kit.box(stage, f"{b}/face_bot_{slot}", (FACE_PLATE / 2, y, (bot_hi - NOSE_H / 2) / 2),
                (FACE_PLATE, PITCH - 0.0004, bot_hi + NOSE_H / 2), mats["ic60_white"], collide=False)
        side_w = (PITCH - 0.0004 - OPENING[0]) / 2
        for s, sy in (("l", 1), ("r", -1)):
            kit.box(stage, f"{b}/face_{s}_{slot}", (FACE_PLATE / 2, y + sy * (OPENING[0] / 2 + side_w / 2), PIVOT_DZ),
                    (FACE_PLATE, side_w, OPENING[1]), mats["ic60_white"], collide=False)
        # Dark cavity behind the opening (visual).
        kit.box(stage, f"{b}/cavity_{slot}", (PIVOT_IN + 0.004, y, PIVOT_DZ), (0.001, OPENING[0], OPENING[1]),
                mats["ic60_dark"], collide=False)
        _front_quad(stage, f"{b}/print_{slot}", -0.0003, y, 0.0115, 0.016, 0.015, tex[f"front_{way}"])
        # Printed circuit label on the cover's label strip, directly above this breaker.
        _front_quad(stage, f"{b}/label_{slot}", COVER_X - 0.0003, y, LABEL_Z, LABEL_W, LABEL_H, tex[f"label_{way}"])
    # Dead-front cover with the 45 mm cut-out (collision: the fingers cannot reach behind it).
    cx = COVER_X + COVER_T / 2
    top_z0, bot_z0 = NOSE_H / 2, -NOSE_H / 2
    kit.box(stage, f"{b}/cover_top", (cx, 0, (top_z0 + COVER_H / 2) / 2), (COVER_T, COVER_W, COVER_H / 2 - top_z0),
            mats["cover"])
    kit.box(stage, f"{b}/cover_bot", (cx, 0, (bot_z0 - COVER_H / 2) / 2), (COVER_T, COVER_W, COVER_H / 2 + bot_z0),
            mats["cover"])
    side = (COVER_W - row_w) / 2
    for s, sy in (("l", 1), ("r", -1)):
        kit.box(stage, f"{b}/cover_{s}", (cx, sy * (row_w / 2 + side / 2), 0), (COVER_T, side, NOSE_H), mats["cover"])
    # Circuit directory card on the cover, right of the row (robot's right).
    dir_y = -(row_w / 2 + DIRECTORY_GAP + DIRECTORY[0] / 2)
    assert abs(dir_y) + DIRECTORY[0] / 2 < COVER_W / 2 - 0.01, "directory card must sit on the cover"
    _front_quad(stage, f"{b}/directory", COVER_X - 0.0003, dir_y, 0.0, DIRECTORY[0], DIRECTORY[1], tex["directory"])
    kit.fix_to_world(stage, root, base.GetPath())

    for slot in range(6):
        px, py, pz = pivot_local(slot)
        h = kit.rigid_body(stage, f"{root}/handle_{slot}", (px, py, pz), 0.008)
        hp = f"{root}/handle_{slot}"
        # Collision: one box from the drum to the bar's far side. Visual: white rocker arm, black grip bar, drum.
        arm_len = HANDLE_LEN + HANDLE_T / 2
        coll = kit.box(stage, f"{hp}/paddle", (-arm_len / 2, 0, 0), (arm_len, HANDLE_W, HANDLE_T), mats["ic60_white"])
        UsdGeom.Imageable(coll.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide)
        kit.box(stage, f"{hp}/arm", (-(HANDLE_LEN + DRUM_R) / 2, 0, 0), (HANDLE_LEN - DRUM_R, HANDLE_W - 0.003,
                HANDLE_T - 0.0005), mats["ic60_white"], collide=False)
        kit.cylinder(stage, f"{hp}/bar", (-HANDLE_LEN, 0, 0), HANDLE_T / 2, HANDLE_W, mats["ic60_dark"], axis="Y")
        kit.cylinder(stage, f"{hp}/drum", (0, 0, 0), DRUM_R, OPENING[0] - 0.001, mats["ic60_white"], axis="Y")
        for name, phi in (("flag_on", -FLAG_DEG), ("flag_off", FLAG_DEG)):
            a0, a1 = math.radians(phi - FLAG_HALF_DEG), math.radians(phi + FLAG_HALF_DEG)
            r, w = DRUM_R + 0.0002, OPENING[0] - 0.002
            _quad(stage, f"{hp}/{name}", [(-r * math.cos(a0), w / 2, r * math.sin(a0)),
                                          (-r * math.cos(a0), -w / 2, r * math.sin(a0)),
                                          (-r * math.cos(a1), -w / 2, r * math.sin(a1)),
                                          (-r * math.cos(a1), w / 2, r * math.sin(a1))], tex[name])
        _sphere(stage, f"{hp}/cam", (CAM_R, 0, 0), CAM_BALL, mats["plunger"])
        kit.revolute(stage, f"{root}/joints/handle_{slot}_joint", base.GetPath(), h.GetPath(), "Y", (px, py, pz),
                     -math.degrees(Q_THROW), math.degrees(Q_THROW))
        # Plunger: face at PLUNGER_FACE_END behind the pivot at joint position 0.
        pl = kit.rigid_body(stage, f"{root}/plunger_{slot}", (px + PLUNGER_FACE_END, py, pz), 0.02)
        pp = f"{root}/plunger_{slot}"
        kit.box(stage, f"{pp}/block", (PLUNGER_SIZE[0] / 2, 0, 0), PLUNGER_SIZE, mats["plunger"])
        kit.prismatic(stage, f"{root}/joints/plunger_{slot}_joint", base.GetPath(), pl.GetPath(), "X",
                      (px + PLUNGER_FACE_END, py, pz), -0.0005, 0.0025)
        for link in (hp, pp):
            _filter(stage, link, b)
        _contact_offsets(stage, [f"{hp}/cam", f"{pp}/block"], 0.001, 0.0)
        # A greased cam: no friction between the sphere and the plunger (with the default 0.5, friction torque near
        # centre exceeded the spring's and the handle stuck at -3 degrees instead of snapping over).
        kit.friction(stage, root, [f"{hp}/cam", f"{pp}/block"], 0.0, 0.0, name=f"slick_{slot}", combine="min")
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    make_textures(out_dir / "tex")
    make_board(out_dir / "board.usda")
    for q in (30, 25, 20, 15, 10, 5, 1):
        print(f"detent torque at {q:2d} deg: {handle_torque(math.radians(q)):.4f} N*m "
              f"({handle_torque(math.radians(q)) / 0.012:.2f} N at 12 mm)")

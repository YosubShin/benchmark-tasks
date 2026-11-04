"""Generate the props of the CT test-terminal bridge task as USD.

Real parts (values marked "inferred" are not published and are to be measured; sources in the README's References):

- Test-disconnect terminal block, Phoenix Contact URTK/S (0311087). Published (Phoenix datasheets 0311087,
  0309086, 0309109): width 8.2 mm, length 61 mm, height 58.6 mm on NS 35/7.5; disconnect slide clamped by an
  M3 screw, 0.6-0.8 N*m; gray RAL 7042; "on both sides of the disconnect point the terminal block has a test
  socket" for 4 mm plugs or bridges; the CT test circuit (datasheet drawing c/d): short the CT with a bridge in
  the CT-side sockets of the k and l terminals, then open the links. The product photo shows the layout:
  raised clamp sections at both ends, a well between them holding (from each end inward) a threaded bridge
  stud, a hex 4 mm test-socket tube, and in the middle the slide with its M3 screw at the well's bottom.
  Built as one fixed-base articulation for the whole row ("strip"):
  * root body: the six housings (clamp sections, a 7.2 mm slot per terminal between thin side walls, socket
    tubes with 4.8 mm square holes, 17 mm deep, 14 mm either side of the disconnect point), the rail, partition
    plates between circuits and end covers. Well depth 22 mm, tube tops 4 mm below the housing top (inferred
    from the photos).
  * six slides (prismatic along x, the terminal's length) at the well's bottom: a 5 mm wide bar with two lugs
    (standing in for the M3 screw head) and a 5 mm gap between them for the screwdriver's blade. Closed it spans
    the disconnect point; open it has moved 5 mm toward the relay side (travel and direction not published; inferred). The task
    holds it with a detent (rule; the real slide is clamped by its M3 screw, which is not modelled).
  Terminal markers (ZB 8-style marker strips, printed 1S1 .. 3S2, on the top of both clamp sections) tell the
  circuits apart, as on real panels; each circuit's markers are in its phase colour (1 red, 2 yellow, 3 blue), and
  a card in the same colour, printed CT1 .. CT3, lies on the mounting plate in front of each circuit, so the named
  circuit can be told apart in the robot's 320 x 176 top camera, where the printed text is 2-3 px tall.
- Shorting bridge: Phoenix KSS 8 (0311540), a two-pole short-circuit connector: black PA grip, two 4 mm sprung
  (lamella) banana pins at 8 mm pitch (the sockets are 8.2 mm apart; the sprung pins take up the difference),
  8 g (published). Grip 15 x 9 x 19 mm and pins 20 mm long (scaled from the product photo; inferred). In
  collision the pins are 4 mm square rods at 8.2 mm with a 3 mm lead-in tip. Seated, the grip rests on the
  housing walls' tops.
- Screwdriver: an insulated 0.6 x 3.5 mm VDE slotted screwdriver (Wera/Wiha class; Weidmueller specifies 0.8 x 4.0
  for its WTL slide), handle 22 mm square x 70 mm (a round 22 mm handle kept at its width; real handles are 80-100
  mm, shortened so the handle's top clears the YAM's palm), shaft 3 mm square x 50 mm (real 60-100 mm, shortened so
  the right arm reaches the strip with fingers down), blade 3.5 x 1.2 mm (thickened from 0.6 for PhysX). It stands
  blade-down in a stand on the desk.
- Holders: a block with two dummy sockets where the bridge starts, and the screwdriver stand.

Frames: the strip's origin is at (STRIP_X, STRIP_Y, 0) in the world, axes along the world's; x along the
terminals (CT side at -x), y along the rail.

Usage: python -m ctbridge_sim.usd.ct_props [out dir]   (default: ctbridge_sim/assets)
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from ctbridge_sim.spec import CIRCUIT_COLOURS, CIRCUITS, HOLDER_XY, PLATE_TOP
from ctbridge_sim.usd import usd_kit as kit

MM = 0.001

# URTK/S terminal ----------------------------------------------------------------------------------------------
PITCH = 8.2
LENGTH = 61.0
HEIGHT = 58.6
"""Above the mounting plate, on NS 35/7.5 (published)."""
TOP_Z = PLATE_TOP + HEIGHT * MM
"""World z of the housings' top (clamp sections and the walls between terminals)."""
WELL_X = 20.0
"""The well spans -WELL_X..+WELL_X along the terminal (inferred from the photo)."""
WELL_DEPTH = 22.0
FLOOR_Z = TOP_Z - WELL_DEPTH * MM
WALL = 1.0
"""Each housing's side wall (on its +y side) in the well; the slot is PITCH - WALL wide."""
SLOT_W = PITCH - WALL
SOCKET = 4.8
"""Square socket opening (a 4 mm round socket; 0.4 mm clearance per side for the 4 mm square pin)."""
SOCKET_X = 14.0
"""Sockets at -SOCKET_X (CT side) and +SOCKET_X (relay side) from the disconnect point (inferred)."""
TUBE_X = 8.0
TUBE_TOP_DROP = 4.0
"""Socket tube tops this far below TOP_Z."""
SOCKET_BOTTOM_Z = TOP_Z - 21.0 * MM
SLIDE_W = 5.0
SLIDE_L = 9.0
SLIDE_H = 2.5
LUG = 1.5
LUG_H = 6.0
"""Lugs at both ends of the slide, 6 mm above its base; the blade goes in the 6 mm gap between them."""
SLIDE_TRAVEL = 5.0
"""Closed: the slide spans -4.5..+4.5 mm; open: +0.5..+9.5 mm, toward the relay side (travel and direction inferred)."""
PARTITION = 2.2
"""Partition plate / end cover thickness (published end cover width 2.2 mm)."""
PARTITION_PROUD = 0.0

# KSS 8 shorting bridge ----------------------------------------------------------------------------------------
PIN = 4.0
PIN_LEN = 20.0
TIP_LEN = 3.0
TIP = 3.0
GRIP = (9.0, 15.0, 19.0)
"""x (thickness), y (width), z (height); the robot pinches its 9 mm thickness."""
BRIDGE_MASS = 0.008

# Screwdriver --------------------------------------------------------------------------------------------------
SD_HANDLE = (22.0, 22.0, 70.0)
SD_SHAFT = (3.0, 3.0, 50.0)
SD_BLADE = (1.2, 3.5, 8.0)
"""x (thickness), y (width), z; the blade's 3.5 mm width lies along the rail (y), so it fits the 7.2 mm slot."""
SD_MASS = 0.045
SD_STAND_XY = (0.20, -0.14)
SD_STAND_TOP = 0.09
"""The screwdriver stands in a 6 mm hole; its handle's bottom rests on the stand's top."""

HOLDER_W = (16.0, 22.0)

COLORS = dict(kit.COLORS)
COLORS.update({
    "urtk": (0.55, 0.56, 0.56),
    "urtk_dark": (0.40, 0.41, 0.42),
    "brass": (0.80, 0.64, 0.28),
    "nickel": (0.78, 0.78, 0.76),
    "rail": (0.78, 0.79, 0.80),
    "plate": (0.86, 0.86, 0.83),
    "kss_black": (0.07, 0.07, 0.08),
    "vde_red": (0.80, 0.10, 0.08),
    "vde_yellow": (0.95, 0.75, 0.10),
    "wire": (0.10, 0.10, 0.11),
})
MARKER_RGB = {"red": (205, 30, 25), "yellow": (245, 200, 20), "blue": (30, 90, 215)}
"""Marker strip and card colours by phase."""
kit.COLORS.update({k: v for k, v in COLORS.items() if k not in kit.COLORS})

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def terminal_ys() -> dict[str, float]:
    """Terminal centre y (mm, strip frame) by marker name: circuits from +y to -y, partitions between them,
    the named circuit "2" centred on y = 0."""
    out = {}
    group = 2 * PITCH + PARTITION
    for i, c in enumerate(CIRCUITS):
        yc = (1 - i) * group
        out[f"{c}S1"] = yc + PITCH / 2
        out[f"{c}S2"] = yc - PITCH / 2
    return out


TERMINALS = tuple(terminal_ys().keys())


def slot_y(name: str) -> float:
    """Centre of a terminal's slot (mm): its wall is on its +y side."""
    return terminal_ys()[name] - WALL / 2


def _font(size):
    return ImageFont.truetype(FONT_B, size)


def make_textures(tex: Path) -> None:
    tex.mkdir(parents=True, exist_ok=True)
    for name in TERMINALS:
        img = Image.new("RGB", (160, 64), MARKER_RGB[CIRCUIT_COLOURS[name[0]]])
        d = ImageDraw.Draw(img)
        ink = (250, 250, 250) if CIRCUIT_COLOURS[name[0]] == "blue" else (15, 15, 15)
        d.text((80, 32), name, fill=ink, font=_font(46), anchor="mm")
        img.save(tex / f"marker_{name}.png")
    for name, text in (("ct", "FROM CTs  (CT side)"), ("relay", "TO RELAY 87B")):
        img = Image.new("RGB", (512, 64), (250, 250, 245))
        d = ImageDraw.Draw(img)
        d.text((256, 32), text, fill=(15, 15, 15), font=_font(36), anchor="mm")
        img.save(tex / f"label_{name}.png")
    for c in CIRCUITS:
        img = Image.new("RGB", (160, 240), MARKER_RGB[CIRCUIT_COLOURS[c]])
        d = ImageDraw.Draw(img)
        ink = (250, 250, 250) if CIRCUIT_COLOURS[c] == "blue" else (15, 15, 15)
        d.rectangle([3, 3, 156, 236], outline=ink, width=4)
        d.text((80, 120), f"CT{c}", fill=ink, font=_font(52), anchor="mm")
        img.save(tex / f"group_CT{c}.png")


def _tex_material(stage, root, name, file_rel):
    """A UsdPreviewSurface material with a texture file (relative to the stage)."""
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
    """A textured quad (visual). corners in metres: image bottom-left, bottom-right, top-right, top-left."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _top_quad(stage, path, x_range, y_range, z, material):
    """A label lying on a top face, readable from the robot's side (text up = +x, text left = +y)."""
    x0, x1 = x_range
    y0, y1 = y_range
    _quad(stage, path, [(x0, y1, z), (x0, y0, z), (x1, y0, z), (x1, y1, z)], material)


def mbox(stage, path, xr, yr, zr, material, collide=True):
    """An axis-aligned box from ranges in mm (x, y) and metres (z)."""
    return kit.box(stage, path, ((xr[0] + xr[1]) / 2 * MM, (yr[0] + yr[1]) / 2 * MM, (zr[0] + zr[1]) / 2),
                   (abs(xr[1] - xr[0]) * MM, abs(yr[1] - yr[0]) * MM, abs(zr[1] - zr[0])), material, collide)


def _square_tube(stage, path, xc, yc, outer_x, outer_y, hole, zr, material):
    """A tube with a square hole along z, from four boxes (mm in x/y, metres in z)."""
    h = hole / 2
    mbox(stage, f"{path}_a", (xc - outer_x / 2, xc - h), (yc - outer_y / 2, yc + outer_y / 2), zr, material)
    mbox(stage, f"{path}_b", (xc + h, xc + outer_x / 2), (yc - outer_y / 2, yc + outer_y / 2), zr, material)
    mbox(stage, f"{path}_c", (xc - h, xc + h), (yc - outer_y / 2, yc - h), zr, material)
    mbox(stage, f"{path}_d", (xc - h, xc + h), (yc + h, yc + outer_y / 2), zr, material)


CONTACT_OFFSET = 0.001
SOLVER_ITERATIONS = 16


def _thin_contacts(stage, root_prim, prefix: str, articulation=True) -> None:
    if articulation:
        root_prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
            SOLVER_ITERATIONS)
        root_prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)


def _housing(stage, p, name, yc, body0, mats):
    ya, yb = yc - PITCH / 2, yc + PITCH / 2
    sy = slot_y(name)
    mbox(stage, f"{p}_body", (-LENGTH / 2, LENGTH / 2), (ya, yb), (body0, FLOOR_Z), mats["urtk"])
    for k, (xa, xb) in enumerate(((-LENGTH / 2, -WELL_X), (WELL_X, LENGTH / 2))):
        mbox(stage, f"{p}_end{k}", (xa, xb), (ya, yb), (FLOOR_Z, TOP_Z), mats["urtk"])
    mbox(stage, f"{p}_wall", (-WELL_X, WELL_X), (yb - WALL, yb), (FLOOR_Z, TOP_Z), mats["urtk"])
    for k, sx in enumerate((-1, 1)):
        xc = sx * SOCKET_X
        _square_tube(stage, f"{p}_tube{k}", xc, sy, TUBE_X, SLOT_W, SOCKET,
                     (SOCKET_BOTTOM_Z, TOP_Z - TUBE_TOP_DROP * MM), mats["nickel"])
        mbox(stage, f"{p}_tubebase{k}", (xc - TUBE_X / 2, xc + TUBE_X / 2), (ya, yb - WALL), (FLOOR_Z, SOCKET_BOTTOM_Z),
             mats["brass"])
        # Threaded bridge stud between the tube and the clamp section (visual).
        kit.cylinder(stage, f"{p}_stud{k}", (sx * (WELL_X - 1.8) * MM, sy * MM, TOP_Z - 0.008), 1.4 * MM, 0.028,
                     mats["nickel"])
        kit.cylinder(stage, f"{p}_screw{k}", (sx * 27.5 * MM, yc * MM, TOP_Z + 0.0002), 2.4 * MM, 0.0004, mats["dark"])
    # Current bars from the tubes to the disconnect point (visual, on the floor).
    mbox(stage, f"{p}_bar", (-SOCKET_X, SOCKET_X), (sy - 2.0, sy + 2.0), (FLOOR_Z, FLOOR_Z + 0.0008), mats["nickel"],
         collide=False)
    for sx in (-1, 1):
        mbox(stage, f"{p}_wire{(sx + 1) // 2}", (sx * LENGTH / 2, sx * (LENGTH / 2 + 22)), (yc - 1.4, yc + 1.4),
             (TOP_Z - 0.030, TOP_Z - 0.0272), mats["wire"], collide=False)


def make_strip(out_path: Path) -> None:
    root = "/strip"
    stage, root_prim = kit.new_stage(out_path, root)
    kit.articulation_root(stage, root_prim)
    mats = kit.materials(stage, root, ("urtk", "urtk_dark", "brass", "nickel", "rail", "plate", "black", "dark",
                                       "wire"))
    base = kit.rigid_body(stage, f"{root}/housing", (0, 0, 0), 2.0)
    kit.fix_to_world(stage, root, base.GetPath())
    ys = terminal_ys()
    y_hi = max(ys.values()) + PITCH / 2 + PARTITION
    y_lo = min(ys.values()) - PITCH / 2 - PARTITION
    mbox(stage, f"{root}/housing/plate", (-104, 75), (y_lo - 25, y_hi + 25), (PLATE_TOP - 0.006, PLATE_TOP), mats["plate"])
    mbox(stage, f"{root}/housing/plate_foot", (-60, 60), (y_lo - 15, y_hi + 15), (0.0, PLATE_TOP - 0.006),
         mats["dark"])
    mbox(stage, f"{root}/housing/rail", (-17.5, 17.5), (y_lo - 12, y_hi + 12), (PLATE_TOP, PLATE_TOP + 0.0075),
         mats["rail"], collide=False)
    body0 = PLATE_TOP + 0.0075
    for name, yc in ys.items():
        _housing(stage, f"{root}/housing/T{name}", name, yc, body0, mats)
    group = 2 * PITCH + PARTITION
    plates = [group / 2, -group / 2, y_hi - PARTITION / 2, y_lo + PARTITION / 2]
    for k, yp in enumerate(plates):
        mbox(stage, f"{root}/housing/partition{k}", (-LENGTH / 2 - 1, LENGTH / 2 + 1),
             (yp - PARTITION / 2, yp + PARTITION / 2), (body0, TOP_Z + PARTITION_PROUD * MM),
             mats["urtk_dark"] if k < 2 else mats["urtk"])
    for k, yb in enumerate((y_hi + 4, y_lo - 4)):
        mbox(stage, f"{root}/housing/endbracket{k}", (-14, 14), (yb - 4, yb + 4), (body0, body0 + 0.030), mats["dark"],
             collide=False)
    for name, yc in ys.items():
        m = _tex_material(stage, root, f"marker_{name}", f"./tex/marker_{name}.png")
        for k, (xa, xb) in enumerate((((WELL_X + 0.4), (WELL_X + 4.8)), (-(WELL_X + 4.8), -(WELL_X + 0.4)))):
            _top_quad(stage, f"{root}/housing/T{name}_marker{k}", (xa * MM, xb * MM),
                      ((yc - 3.8) * MM, (yc + 3.8) * MM), TOP_Z + 0.0003, m)
    for i, c in enumerate(CIRCUITS):
        yc = (1 - i) * group
        half = group / 2 - 0.6
        m = _tex_material(stage, root, f"group_CT{c}", f"./tex/group_CT{c}.png")
        _top_quad(stage, f"{root}/housing/group_CT{c}", (-89 * MM, -61 * MM), ((yc - half) * MM, (yc + half) * MM),
                  PLATE_TOP + 0.0003, m)
    for side, x0 in (("ct", -102.0), ("relay", 56.0)):
        m = _tex_material(stage, root, f"label_{side}", f"./tex/label_{side}.png")
        _top_quad(stage, f"{root}/housing/label_{side}", (x0 * MM, (x0 + 10) * MM), ((y_lo - 18) * MM, (y_hi + 18) * MM),
                  PLATE_TOP + 0.0003, m)
    # Slides: a bar on the well's floor with a lug at each end; the joint origin is the slide's centre.
    for name in ys:
        sp = f"{root}/slide_{name}"
        sy = slot_y(name)
        z0 = FLOOR_Z + 0.0008
        slide = kit.rigid_body(stage, sp, (0, sy * MM, z0), 0.003)
        L, W = SLIDE_L / 2, SLIDE_W / 2
        kit.box(stage, f"{sp}/bar", (0, 0, SLIDE_H / 2 * MM), (SLIDE_L * MM, SLIDE_W * MM, SLIDE_H * MM), mats["nickel"])
        for k, sx in enumerate((-1, 1)):
            kit.box(stage, f"{sp}/lug{k}", (sx * (L - LUG / 2) * MM, 0, LUG_H / 2 * MM), (LUG * MM, SLIDE_W * MM,
                    LUG_H * MM), mats["nickel"])
        kit.cylinder(stage, f"{sp}/screw", (0, 0, SLIDE_H * MM + 0.0008), 2.4 * MM, 0.0016, mats["nickel"])
        kit.box(stage, f"{sp}/screw_slot", (0, 0, SLIDE_H * MM + 0.0017), (0.0007, 0.004, 0.0002), mats["dark"],
                collide=False)
        del W
        kit.prismatic(stage, f"{root}/joints/slide_{name}", base.GetPath(), slide.GetPath(), "X", (0, sy * MM, z0),
                      0.0, SLIDE_TRAVEL * MM)
    _thin_contacts(stage, root_prim, root)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_bridge(out_path: Path) -> None:
    """KSS 8. Origin: the grip's underside, centred between the pins."""
    root = "/bridge"
    stage, root_prim = kit.new_stage(out_path, root)
    mats = kit.materials(stage, root, ("nickel", "kss_black", "white"))
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(BRIDGE_MASS)
    for k, yp in enumerate((PITCH / 2, -PITCH / 2)):
        mbox(stage, f"{root}/pin{k}", (-PIN / 2, PIN / 2), (yp - PIN / 2, yp + PIN / 2),
             (-(PIN_LEN - TIP_LEN) * MM, 0.0), mats["nickel"])
        mbox(stage, f"{root}/tip{k}", (-TIP / 2, TIP / 2), (yp - TIP / 2, yp + TIP / 2), (-PIN_LEN * MM,
             -(PIN_LEN - TIP_LEN) * MM), mats["nickel"])
    gx, gy, gz = GRIP
    mbox(stage, f"{root}/grip", (-gx / 2, gx / 2), (-gy / 2, gy / 2), (0.0, gz * MM), mats["kss_black"])
    for k in range(3):
        mbox(stage, f"{root}/rib{k}", (-gx / 2 - 0.2, -gx / 2), (-gy / 2 + 2, gy / 2 - 2),
             ((7 + 4 * k) * MM, (7.6 + 4 * k) * MM), mats["dark"] if "dark" in mats else mats["kss_black"],
             collide=False)
    kit.friction(stage, root, [f"{root}/grip"], static=1.2, dynamic=1.0)
    root_prim.AddAppliedSchema("PhysxRigidBodyAPI")
    root_prim.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(0.5)
    root_prim.CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float).Set(0.2)
    root_prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(16)
    _thin_contacts(stage, root_prim, root, articulation=False)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def make_screwdriver(out_path: Path) -> None:
    """Origin: the blade's tip, centred; the tool points down (handle up) in its own frame."""
    root = "/screwdriver"
    stage, root_prim = kit.new_stage(out_path, root)
    mats = kit.materials(stage, root, ("nickel", "vde_red", "vde_yellow"))
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(SD_MASS)
    bx, by, bz = SD_BLADE
    sx, sy, sz = SD_SHAFT
    hx, hy, hz = SD_HANDLE
    mbox(stage, f"{root}/blade", (-bx / 2, bx / 2), (-by / 2, by / 2), (0.0, bz * MM), mats["nickel"])
    mbox(stage, f"{root}/shaft", (-sx / 2, sx / 2), (-sy / 2, sy / 2), (bz * MM, (bz + sz) * MM), mats["vde_red"])
    z0 = (bz + sz) * MM
    mbox(stage, f"{root}/handle", (-hx / 2, hx / 2), (-hy / 2, hy / 2), (z0, z0 + hz * MM), mats["vde_red"])
    mbox(stage, f"{root}/band", (-hx / 2 - 0.3, hx / 2 + 0.3), (-hy / 2 - 0.3, hy / 2 + 0.3),
         (z0 + 0.010, z0 + 0.016), mats["vde_yellow"], collide=False)
    kit.friction(stage, root, [f"{root}/handle"], static=1.2, dynamic=1.0)
    root_prim.AddAppliedSchema("PhysxRigidBodyAPI")
    root_prim.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(0.5)
    root_prim.CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float).Set(0.2)
    root_prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(16)
    _thin_contacts(stage, root_prim, root, articulation=False)
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


def sd_start_tip_z() -> float:
    """The blade tip's z when the handle rests on the stand."""
    return SD_STAND_TOP - (SD_BLADE[2] + SD_SHAFT[2]) * MM


def make_holder(out_path: Path) -> None:
    """The bridge's holder (two dummy sockets whose mouths are at TOP_Z - TUBE_TOP_DROP, grip resting at TOP_Z)
    at HOLDER_XY and the screwdriver stand at SD_STAND_XY (world frame)."""
    root = "/holder"
    stage, _ = kit.new_stage(out_path, root)
    mats = kit.materials(stage, root, ("stand", "urtk"))
    hx, hy = HOLDER_XY[0] / MM, HOLDER_XY[1] / MM
    wx, wy = HOLDER_W
    z0, z1 = SOCKET_BOTTOM_Z, TOP_Z
    mbox(stage, f"{root}/column", (hx - wx / 2, hx + wx / 2), (hy - wy / 2, hy + wy / 2), (0.0, z0), mats["stand"])
    s = SOCKET / 2
    ys = [hy - wy / 2, hy - PITCH / 2 - s, hy - PITCH / 2 + s, hy + PITCH / 2 - s, hy + PITCH / 2 + s, hy + wy / 2]
    for k, (ya, yb) in enumerate(zip(ys[:-1], ys[1:])):
        if k % 2 == 0:
            mbox(stage, f"{root}/top{k}", (hx - wx / 2, hx + wx / 2), (ya, yb), (z0, z1), mats["urtk"])
        else:
            for j, (xa, xb) in enumerate(((hx - wx / 2, hx - s), (hx + s, hx + wx / 2))):
                mbox(stage, f"{root}/top{k}_{j}", (xa, xb), (ya, yb), (z0, z1), mats["urtk"])
    # Screwdriver stand: a block with a 6 mm square hole for the shaft.
    sx, sy = SD_STAND_XY[0] / MM, SD_STAND_XY[1] / MM
    _square_tube(stage, f"{root}/sd_stand", sx, sy, 30.0, 30.0, 6.0, (0.0, SD_STAND_TOP), mats["stand"])
    stage.GetRootLayer().Save()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_textures(out_dir / "tex")
    make_strip(out_dir / "strip.usda")
    make_bridge(out_dir / "bridge.usda")
    make_screwdriver(out_dir / "screwdriver.usda")
    make_holder(out_dir / "holder.usda")

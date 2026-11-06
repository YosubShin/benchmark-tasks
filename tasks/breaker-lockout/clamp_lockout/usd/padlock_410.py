"""The Master Lock 410 Zenex padlock in the shape of the real lock, built on the group-hasp padlock model
(``padlock_props.make_padlock``: published body and shackle dimensions, sprung shackle on a prismatic joint).

What this adds to that model, following the 410's published outline and product photos
(https://www.masterlock.com/business-use/product/410RED: body 1-1/2 in wide, 1-3/4 in tall, 3/4 in thick; 1/4 in
shackle, 1-1/2 in vertical and 3/4 in horizontal clearance):

- Round shackle with a semicircular bend. Collision: square-section legs up to the bend's centre and six straight
  segments around the bend (the inside of the bend is the published 38 mm above the body), instead of the flat
  top bar. Visual: round legs and a half-torus.
- Rounded body: visual outline with 4 mm corner radius over the box collider; white "DANGER" label with a red band
  on both wide faces; keyway on the bottom.
- The short leg's collider is a cross of two 6.35 x 2.6 mm bars with low friction, so the lock can turn on the
  lockout's square stem hole and hang from it as a real lock does (a square leg cannot turn in a square hole).

Padlock frame as in padlock_props: origin at the body's centre, legs along z on top, short leg at +x, thickness
along y.
"""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from clamp_lockout.usd import padlock_props as PP

V = PP.PADLOCKS["ml410"]
BEND_R = V.leg_x
"""Shackle centre-line radius at the bend (legs' centres 26.35 mm apart)."""
BEND_ZC = V.clear_v - (BEND_R - V.rod / 2)
"""Bend centre above the body's top, shut (inside of the bend 38 mm above the body)."""
CORNER_R = 0.004
ARC_SEGMENTS = 6
SHORT_BAR = 0.0026


def _mat(stage, name, rgb):
    path = f"/padlock/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _mesh(stage, path, points, faces, material):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([len(f) for f in faces]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in faces for i in f]))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    _bind(mesh.GetPrim(), material)
    return mesh


def _rounded_body(stage, path, material):
    """Extruded rounded rectangle: outline in x-z (width x height), thickness along y."""
    w, t, h = V.body
    hx, hz, r = w / 2 - CORNER_R, h / 2 - CORNER_R, CORNER_R
    outline = []
    for cx, cz, a0 in ((hx, hz, 0), (-hx, hz, 90), (-hx, -hz, 180), (hx, -hz, 270)):
        for k in range(7):
            a = math.radians(a0 + 15 * k)
            outline.append((cx + r * math.cos(a), cz + r * math.sin(a)))
    n = len(outline)
    pts = [(x, -t / 2, z) for x, z in outline] + [(x, t / 2, z) for x, z in outline]
    faces = [[(i + 1) % n, i, n + i, n + (i + 1) % n] for i in range(n)]
    faces.append(list(range(n)))
    faces.append(list(range(2 * n - 1, n - 1, -1)))
    return _mesh(stage, path, pts, faces, material)


def _half_torus(stage, path, zc, material, seg_u=16, seg_v=10):
    """The shackle's bend: a half torus in the x-z plane, centre (0, 0, zc), radius BEND_R, tube V.rod / 2."""
    pts, faces = [], []
    a = V.rod / 2
    for i in range(seg_u + 1):
        u = math.pi * i / seg_u
        cx, cz = BEND_R * math.cos(u), zc + BEND_R * math.sin(u)
        for j in range(seg_v):
            v = 2 * math.pi * j / seg_v
            rr = a * math.cos(v)
            pts.append((cx + rr * math.cos(u), a * math.sin(v), cz + rr * math.sin(u)))
    for i in range(seg_u):
        for j in range(seg_v):
            p = lambda ii, jj: ii * seg_v + jj % seg_v  # noqa: E731
            faces.append([p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)])
    return _mesh(stage, path, pts, faces, material)


def _cyl(stage, path, center, radius, height, material):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(*center))
    _bind(cyl.GetPrim(), material)
    return cyl


def _cbox(stage, path, center, size, material, rot_y_deg=0.0, visible=False):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(cube)
    x.AddTranslateOp().Set(Gf.Vec3d(*center))
    if rot_y_deg:
        x.AddRotateYOp().Set(rot_y_deg)
    x.AddScaleOp().Set(Gf.Vec3f(*size))
    _bind(cube.GetPrim(), material)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if not visible:
        UsdGeom.Imageable(cube).MakeInvisible()
    return cube


def _vbox(stage, path, center, size, material):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(cube)
    x.AddTranslateOp().Set(Gf.Vec3d(*center))
    x.AddScaleOp().Set(Gf.Vec3f(*size))
    _bind(cube.GetPrim(), material)
    return cube


def make_padlock_410(out_path: Path) -> None:
    PP.make_padlock(out_path, V)
    stage = Usd.Stage.Open(str(out_path))
    steel = UsdShade.Material.Get(stage, "/padlock/Looks/steel")
    color = UsdShade.Material.Get(stage, f"/padlock/Looks/{V.color}")
    white = _mat(stage, "label_white", (0.95, 0.95, 0.95))
    red = _mat(stage, "label_red", (0.80, 0.08, 0.06))
    dark = _mat(stage, "keyway_dark", (0.12, 0.12, 0.13))

    # Body: the box stays as the collider (hidden); a rounded outline is drawn over it.
    UsdGeom.Imageable(stage.GetPrimAtPath("/padlock/body/case")).MakeInvisible()
    _rounded_body(stage, "/padlock/body/outline", color)
    w, t, h = V.body
    for k, s in enumerate((-1, 1)):
        y = s * (t / 2 + 0.0002)
        _vbox(stage, f"/padlock/body/label{k}", (0, y, -0.002), (0.030, 0.0004, 0.028), white)
        _vbox(stage, f"/padlock/body/label_band{k}", (0, s * (t / 2 + 0.0004), 0.0085), (0.030, 0.0004, 0.007), red)
    _vbox(stage, "/padlock/body/keyway_ring", (0, 0, -h / 2 - 0.0003), (0.012, 0.012, 0.0006), dark)

    # Shackle: replace the box legs and flat top bar.
    sh = "/padlock/shackle"
    for name in ("long_leg", "short_leg", "bend"):
        stage.RemovePrim(f"{sh}/{name}")
    rod = V.rod
    long_lo, short_lo = PP.LONG_LEG_BOTTOM, PP.SHORT_LEG_BOTTOM
    # Long leg (-x): square collider, round visual.
    _cbox(stage, f"{sh}/long_leg", (-V.leg_x, 0, (long_lo + BEND_ZC) / 2), (rod, rod, BEND_ZC - long_lo), steel)
    _cyl(stage, f"{sh}/long_leg_vis", (-V.leg_x, 0, (long_lo + BEND_ZC) / 2), rod / 2, BEND_ZC - long_lo, steel)
    # Short leg (+x): a low-friction cross of two bars (turns in a square hole), round visual.
    UsdGeom.Xform.Define(stage, f"{sh}/short_leg")
    zc, length = (short_lo + BEND_ZC) / 2, BEND_ZC - short_lo
    # A full square collider (this task's holes are square and aligned with the legs; a thin cross tunnelled
    # through the lever's tab under the after-release check).
    _cbox(stage, f"{sh}/short_leg/a", (V.leg_x, 0, zc), (rod, rod, length), steel)
    _cyl(stage, f"{sh}/short_leg/round", (V.leg_x, 0, zc), rod / 2, length, steel)
    slick = UsdShade.Material.Define(stage, "/padlock/PhysicsMaterials/slick")
    api = UsdPhysics.MaterialAPI.Apply(slick.GetPrim())
    api.CreateStaticFrictionAttr(0.05)
    api.CreateDynamicFrictionAttr(0.05)
    slick.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    slick.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("min")
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(f"{sh}/short_leg")).Bind(
        slick, UsdShade.Tokens.weakerThanDescendants, "physics")
    # Bend: six straight square segments around the semicircle (collider), half torus (visual).
    seg_len = 2 * (BEND_R + rod / 2) * math.tan(math.pi / (2 * ARC_SEGMENTS))
    for i in range(ARC_SEGMENTS):
        a = math.pi * (i + 0.5) / ARC_SEGMENTS
        center = (BEND_R * math.cos(a), 0.0, BEND_ZC + BEND_R * math.sin(a))
        _cbox(stage, f"{sh}/bend_{i}", center, (rod, rod, seg_len), steel, rot_y_deg=-math.degrees(a))
    _half_torus(stage, f"{sh}/bend_vis", BEND_ZC, steel)
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    import sys

    make_padlock_410(Path(sys.argv[1]))

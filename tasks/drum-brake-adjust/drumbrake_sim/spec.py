"""Drum brake adjust task: layout, part dimensions, mechanism rules and instruction (no simulator imports).

Desk frame: table top z = 0, x forward (away from the robot), y to the robot's left, left arm base at (0, +0.305, 0),
right arm base at (0, -0.305, 0). drumbrake_sim.station maps it into the station (z + 0.75).

Real parts (README.md "References"; saved copies under out/compare/drum-brake-adjust/ref/):
- Brake: a GM-pattern 9-1/2 x 2 in rear duo-servo drum brake ("Self-Energizing-Servo"; 1968 Buick Chassis Service
  Manual sec. 50): drum ID 9.495-9.505 in, linings 2.00 in wide; drum depth 2.61 in and 5 x 4-3/4 in bolt circle
  (ACDelco 18B201A / DuraGo BD8988 9.5 in drums). Star wheel: 24 teeth, 0.797 in outside diameter (GM adjusting
  screw 18044634 listing; unverified snippet).
- Adjusting hole in the backing plate, closed by a rubber push-in cover (Dorman 13964 type); size not published
  (inferred 1/2 x 3/4 in).
- Brake adjusting spoon: 9 in (229 mm) long (Performance Tool W179C class); blade 8 mm wide tapering to 1.0 mm at the
  tip (Laser 9251 data sheet).
- Procedure step modelled (Chrysler, Dodge Dynasty): "turn the adjusting screw upward ... to expand the shoes until a
  slight drag is felt when the drum is rotated". So on this assembly moving the star wheel's teeth upward through the
  slot expands the shoes (thread hand is make- and side-specific).
"""

from __future__ import annotations

import math

TASK_NAME = "drum-brake-adjust"
EPISODE_SECONDS = 100.0
IN = 0.0254

# -- hub, backing plate, drum -----------------------------------------------------------------------------------
AXLE = (0.34, -0.02, 0.21)
"""Axle centre (the hub's axis runs along y; the drum's closed face toward +y, the backing plate's back toward -y)."""
BP_Y = AXLE[1] - 0.045
"""Backing plate's back face (the side the adjusting hole is worked from)."""
BP_T = 0.0025
BP_R = 0.136
DRUM_RI = 9.5 * IN / 2
DRUM_RO = 0.132
DRUM_Y0 = BP_Y + BP_T + 0.003
"""Drum's open edge (just clear of the backing plate)."""
DRUM_DEPTH = 2.61 * IN
DRUM_FACE_T = 0.006
DRUM_FACE_Y = DRUM_Y0 + DRUM_DEPTH
"""Outer face of the drum's closed end (studs stand out of it toward +y)."""
DRUM_MASS = 6.0
"""9.5 in cast-iron-rim drum with the hub (inferred: about 13 lb)."""
STUD_R = 4.75 * IN / 2
STUD_D = 0.5 * IN
STUD_L = 1.5 * IN
STUD_ALPHA0 = 0.0
"""Angle of the stud the robot turns the drum by, at the start, measured up from the axle's side nearest the robot
(-x); the five studs are 72 degrees apart."""
STUD_GRIP_Y = DRUM_FACE_Y + 0.026
"""Grasp point on that stud, 26 mm out from the drum face (the fingertips stay 13 mm off it)."""

# -- star wheel adjuster -------------------------------------------------------------------------------------------
SW_TEETH = 24
SW_RT = 0.797 * IN / 2
"""Tooth tip radius (10.1 mm)."""
SW_RR = 0.0076
"""Root radius (inferred: 2.5 mm deep teeth)."""
SW_W = 0.006
"""Face width (inferred)."""
SW_TOOTH_ROOT_W = 0.0010
SW_TOOTH_TIP_W = 0.0010
SW_MASS = 0.020
SW = (AXLE[0], BP_Y + BP_T + 0.006 + SW_RT, AXLE[2] - 0.090)
"""Star wheel centre: its axis along x (the adjuster screw lies across the bottom of the brake), its teeth 6 mm inside
the plate's front face, behind the adjusting hole."""
PITCH_ANG = 2 * math.pi / SW_TEETH
EXPAND_SIGN = -1.0
"""Star joint (about +x) turning negative moves the teeth at the hole upward: that expands the shoes."""

# -- adjusting hole (slot) ------------------------------------------------------------------------------------------
SLOT_W = 0.5 * IN
SLOT_Z0 = SW[2] - 0.0020
"""Slot bottom edge (the spoon's fulcrum) 2 mm below the star wheel's axis."""
SLOT_Z1 = SLOT_Z0 + 0.75 * IN

# -- mechanism rules ------------------------------------------------------------------------------------------------
NOTCH_DIAM_MM = 0.020
"""Shoe diametral expansion per notch (inferred: Toyota's 0.6 mm running clearance over GM's 30-notch back-off)."""
CONTACT_NOTCHES = 6.2
"""Notches from the start until the linings touch the drum (start clearance 0.12 mm diametral)."""
HUB_NM = 0.15
"""Turning torque of the hub with the shoes clear (bearing and seal; Mazda 323 rear hub spec 0.15-0.49 N m)."""
DRAG_PER_NOTCH = 0.25
"""N m of shoe drag per notch past first contact (inferred: GM 'can just be turned by hand', about 3 N m, to free
running is about 12 notches)."""
SW_RUN_NM = 0.03
"""Star wheel turning torque with the shoes clear (adjuster spring on the screw thread and the self-adjuster pawl
riding the teeth; inferred)."""
SCREW_D = 0.008
NUT_K = 0.2
LINING_MU = 0.35
SW_BACK_NM = 2.0
"""The self-adjuster pawl in a notch: backward turning needs this (it does not happen without a second tool)."""
SW_K = 3.0
SW_D = 0.002
SW_ARMATURE = 5e-5
DRUM_K = 300.0
DRUM_D = 2.0


def drag_nm(notches):
    """Drum turning torque [N m] for the star wheel's expansion in notches from the start (works on floats or
    torch tensors)."""
    over = notches - CONTACT_NOTCHES
    try:
        over = over.clamp(min=0.0)
    except AttributeError:
        over = max(over, 0.0)
    return HUB_NM + DRAG_PER_NOTCH * over


def star_fwd_nm(drag):
    """Star wheel's forward turning torque: running torque plus the thread load from the shoes' contact force."""
    n_contact = (drag - HUB_NM) / (2 * LINING_MU * DRUM_RI)
    return SW_RUN_NM + NUT_K * SCREW_D * n_contact


# -- spoon ----------------------------------------------------------------------------------------------------------
SPOON_L = 9.0 * IN
HANDLE_L = 0.120
HANDLE_W = 0.5 * IN
HANDLE_T = 0.5 * IN
BLADE_W = 0.008
BLADE_T = 0.0018
TIP_T = 0.0010
TIP_TAPER = 0.004
TIP_FLAT = 0.0005
"""The blade tapers from 1.8 mm to the 1.0 mm tip over 4 mm, ending in 0.5 mm of 1.0 mm-thick edge."""
SPOON_MASS = 0.10
GRIP_FROM_END = 0.040
GRIP_LOCAL_Y = -(SPOON_L - GRIP_FROM_END)
"""Spoon frame: origin at the tip on the blade's centre line, the spoon along +y (tip forward), z up."""
REST_X = 0.22
REST_TIP_Y = -0.095
REST_TOP = 0.060
SPOON_Z0 = REST_TOP + HANDLE_T / 2
BLOCK_A = ((REST_X - 0.02, REST_X + 0.02), (-0.13, -0.115), REST_TOP + (HANDLE_T - BLADE_T) / 2)
BLOCK_B = ((REST_X - 0.02, REST_X + 0.02), (REST_TIP_Y - SPOON_L - 0.004, REST_TIP_Y - SPOON_L + 0.014), REST_TOP)
PLUG_POS = (0.20, -0.06, 0.0)

# -- stand (wheel-off corner on a hub stand) --------------------------------------------------------------------------
KNUCKLE = ((AXLE[0] - 0.035, AXLE[0] + 0.035), (BP_Y - 0.050, BP_Y), (AXLE[2] - 0.035, AXLE[2] + 0.035))
POST = ((AXLE[0] + 0.035, AXLE[0] + 0.085), (BP_Y - 0.050, BP_Y - 0.010), (0.0, AXLE[2] + 0.035))
BASE = ((AXLE[0] - 0.02, AXLE[0] + 0.16), (BP_Y - 0.12, AXLE[1] + 0.10), (0.0, 0.012))

# -- scoring --------------------------------------------------------------------------------------------------------
ENGAGE_R = SW_RT + 0.0003
ENGAGE_X = SW_W / 2 + 0.002
CLICKS_MIN = 5
BAND = (0.30, 1.00)
RELEASE_CLEAR = 0.030
TOOL_OUT = 0.010
"""1.0 needs the spoon's tip at least this far out of the hole (behind the plate's back face)."""
TEST_WAIT_S = 1.0
TEST_S = 3.0
TEST_MAX_NM = 1.2
TEST_TURN_DEG = 2.0
DROP_Z = 0.030


def stud_pos(alpha_deg: float, y: float = STUD_GRIP_Y):
    a = math.radians(alpha_deg)
    return (AXLE[0] - STUD_R * math.cos(a), y, AXLE[2] + STUD_R * math.sin(a))


def spoon_start():
    """Spoon at rest: along +y, tip forward, its handle on block B and its blade on block A."""
    return (REST_X, REST_TIP_Y, SPOON_Z0)


def instruction() -> str:
    return ("Adjust the rear brake with the brake spoon so that the drum turns by hand with a light drag. Do not "
            "over-tighten it and do not drop the spoon. Take the spoon out of the adjusting hole and let go of the "
            "drum when you are done.")

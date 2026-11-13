"""Dimensions of the fire-extinguisher task's props (no USD imports; sources in fire_sim/usd/props.py)."""

MM = 0.001

# -- extinguisher (frame U, mm) -----------------------------------------------------------------------------------
CYL_R = 38.0
CYL_H = 312.0
NECK = (14.0, 300.0, 332.0)          # radius, z0, z1
HEAD = ((-18.0, 18.0), (-13.0, 13.0), (330.0, 362.0))
NOZZLE = (18.0, 58.0, 350.0, 8.0)    # x0, x1 (outlet), z, radius
NOZZLE_TIP = (58.0, 0.0, 350.0)
HANDLE_ROOT = (-18.0, 352.0)
HANDLE_LEN = 105.0
HANDLE_DEG = -12.0
HANDLE_W, HANDLE_T = 20.0, 6.0
EAR_Y = (9.0, 13.0)
EAR_X = (-30.0, -18.0)
EAR_Z = (360.0, 378.0)
PIN_AXIS = (-24.0, 367.2)            # x, z of the pin's axis (resting on the holes' lower edges)
HOLE = 6.0
PIN = 4.0
PIN_Y = (-16.0, 30.0)                # the shaft along y (pin frame = U shifted to the pin axis)
RING_C = (0.0, 31.5, -12.0)          # ring centre in the pin frame
RING_OUT, RING_BAR = 24.0, 3.0
PIVOT = (-8.0, 378.0)
LEVER_LEN, LEVER_W, LEVER_T = 120.0, 18.0, 5.0
LEVER_REST_DEG = 8.0                 # bar rises toward the back at rest
LEVER_TRAVEL_DEG = 20.0
CHEEK_X = (-30.0, -18.0)             # lever side cheeks (U frame), inside the ears
CHEEK_Y = (4.0, 8.0)
CHEEK_Z = (360.0, 379.0)
PIN_DROP = 0.6                       # spawn clearance under the pin (it settles onto the holes' lower edges)
CHEEK_PLAY = 1.0                     # cheek holes 1 mm taller than the pin: the lever stops near 3 deg with the pin in
GRIP_X = -80.0                       # where a hand squeezes (80 mm behind the axis, 72 mm from the pivot)
BODY_MASS = 1.0
BODY_COM = (0.0, 0.0, 0.196)         # m: shell 0.55 kg at 160 mm, valve 0.30 kg at 350 mm, 0.15 L water at 20 mm
LEVER_MASS, PIN_MASS = 0.04, 0.012
SEAL_BREAK_N = 8.0

# -- fire stand (desk frame, m) ----------------------------------------------------------------------------------
STAND_X = (0.53, 0.67)
STAND_Y = (-0.18, 0.08)
STAND_H = 0.24
PAN_WALL = 0.005
FUEL_Z = 0.28
FLAME_Y = (-0.13, -0.05, 0.03)
FLAME_X = 0.60
FLAME_H = 0.105
FLAME_R = 0.036
KEEP_OUT = 0.20
TAPE_W = 0.025
TABLE_FAR_X = 0.90

SPRAY_LEN, SPRAY_R = 0.17, 0.022

PIN_TRAVEL = 60.0
"""The pin's slide (mm): it clears the lever's cheeks at 12 mm and the far ear at 29 mm; it stops at 60 mm (in the
real unit it comes free; here it stays on its axis, 6 cm out, see README)."""
SEAL_K = 4000.0
"""Seal stiffness (N/m) while intact: 8 N at 2 mm of slide, where the task releases it."""
SEAL_BREAK_MM = SEAL_BREAK_N / SEAL_K * 1000.0
PIN_FREE_DAMPING = 0.5
"""After the seal breaks: viscous slide resistance (N s/m), standing in for friction in the holes."""

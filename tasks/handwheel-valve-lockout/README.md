# Close a named handwheel gate valve and lock it with a rotating valve lockout cover and a padlock

Two identical handwheel gate valves sit on a short pipe run, each identified by an engraved valve tag on a service
colour (V-21 yellow, V-23 green). The robot must close V-21 (clockwise, about one turn, letting go and regripping as
often as it needs), slide a red rotating gate valve lockout cover (Master Lock 480) onto the handwheel, swing the
cover's loose half round until its padlock ear lies on the fixed half's ear, and padlock the two ears together,
without turning V-23. Scoring checks the lockout physically: after the robot lets go, the halves are pried apart, the
cover is turned and then pulled up, and the valve must stay closed with the cover shut and on the wheel.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`handwheel_valve.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy.
Its props are articulated (the two handwheels on their stems, the cover's rotating half, the padlock's shackle),
which the one-file format cannot declare, so the file uses `wheel_lockout_sim` (this folder), which adds them to the
station at run time (`wheel_lockout_sim/station.py` documents exactly what it patches; it requires
yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin).

## How the real lockout works

The Master Lock 480 (handwheels 1 to 3 in) is the smallest of a nesting family of rotating gate valve lockouts (480 to
484; product page, technical data, product photos and Master Lock's how-to video in the References). It is a flat red
thermoplastic drum in two pieces. The fixed piece, which carries the "DANGER - DO NOT OPERATE" label, is a hollow drum
sector (front face, back face and rim wall) that leaves an open sector; its back face has a stem opening at the centre
(19 mm, published), which the open sector reaches. The rotating piece is the missing sector; it pivots about the drum's
axis at the centre of the front face and, opened, nests inside the fixed piece. Each piece has a flat padlock ear with
two holes, in the plane of its back face. To fit it, the worker opens it, slides it sideways over the handwheel open
side first, so the wheel goes between the faces and the stem into the centre, then swings the rotating piece round
until its ear lies on the fixed piece's ear, and puts a padlock through the lined-up holes (two holes, so two workers
can lock it). Closed, the drum surrounds the wheel front, back and round its rim, so the wheel cannot be gripped or
turned; the drum itself turns freely on the stem, and its back face behind the wheel's rim stops it being pulled off.
The shackle through both ears stops the rotating piece swinging open.

In the simulation every link of that chain is geometry: the handwheel is a body on its stem joint between the two
halves' faces and inside their rims, the fixed half is a free body resting on the wheel nut, the rotating half turns
on it on a revolute joint (the joint's ends are its stops), and the padlock's shackle, a separate articulation, passes
through both ears. Nothing in the code holds the cover, the wheel or the rotating half.

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD               # must print 0059d02
uv pip install -e tasks/handwheel-valve-lockout --no-deps          # registers the policy wheel-lockout-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/handwheel-valve-lockout/handwheel_valve.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Close gate valve V-21, the one with the yellow tag: turn its handwheel clockwise, seen from above, until it stops, about one full turn; let go and grip again as often as you need. Slide the red valve cover onto that handwheel from the side, open side first, so the stem sits in the middle of the cover, then swing the cover's loose half round until its ear lies on the other half's ear. Put your padlock's shackle through the inner holes of both ears and snap it shut. Do not touch valve V-23, the one with the green tag." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/handwheel-dream
```

The scripted demo (closed loop: it reads the wheel angle, the cover's pose and swing, and the padlock's pose from the
simulator, like a scripted demo with privileged state, and sends the same 14-D joint action a policy does, through
yam_arena's IK):

```bash
inspect-robots run --embodiment yam_arena --policy wheel-lockout-demo \
  -E task=tasks/handwheel-valve-lockout/handwheel_valve.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/handwheel-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json). Videos:
`inspect-robots video <log>.json` writes one MP4 per camera; demo.mp4 and dream1.mp4 here put the top camera (2x) above
the two wrist cameras.

- **Episode end.** Once the padlock is shut through both ears and both grippers have stayed at least 6 cm clear of the
  padlock and of the stem for 2 s, the after-release test runs (6 s of simulated time) inside that control step, off
  camera, and the episode ends; the stored frames stop at the release. Otherwise the run ends at `--max-steps`.
- **Control run.** `--policy wheel-lockout-demo -P control=true` closes V-21, fits the cover and swings it shut, but
  leaves the padlock on its stand, then forces the after-release test: it must score below 1.
- **Seeds.** One fixed layout (`-E object_jitter=0`); seeded jitter is not used.

## Measured

RTX-class GPU shared with other simulations, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz, 320x176:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `wheel-lockout-demo` | `task_score: 1` (wheel turned 360 degrees to its seat in two grips; cover slid on 1.5 mm off the stem's axis; loose half pushed shut in 34 pushes; shackle latched through both ears after 9 press steps; after-release test passed: loose half moved 3.0 degrees under the 15 N pry, the wheel did not move while the cover turned 9.6 degrees, the cover rose 17 mm under the 10 N lift and stopped, caught under the wheel's rim) | 2,470 (123.5 s), then the 6 s test off camera | about 55 minutes |
| Control: `wheel-lockout-demo -P control=true` (closed, cover fitted and swung shut, no padlock) | `task_score: 0.75` (after-release test fails: the 15 N pry swings the loose half 124 degrees, fully open) | 1,622, then the test | about 35 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (the arms move about near their start; late on, the right arm reaches to the cover on its stand and nudges it; neither handwheel is gripped or turned and the padlock is not picked up) | 3,600 (step limit) | about 50 minutes (shared GPU; Dream's client retried about 70 times after "session expired") |

The demo's wall time is mostly its IK, which it solves in Python every control step, and the shared GPU; on an
otherwise idle machine expect less.

The demo's steps: arm B (right) grips V-21's handwheel across its rim, fingers down, and turns it clockwise by turning
its wrist, aiming a little past the measured wheel angle; at the end of the wrist's travel it lets go, turns back with
the fingers open and grips again (two grips for the turn), until the wheel stops at its seat. Arm B then picks the
cover up across its rim, carries it level to the right of the wheel and slides it on, open side first, correcting its
position closed loop, until the cover sits on the wheel nut with the stem in its centre. Arm A (left) pushes the
loose half's ear round the stem with its closed fingertips until the ear lies on the fixed half's ear. Arm A then
takes the open padlock off its stand (from the left, fingers pitched down, across the body's thickness), slides it in
from the left so the two ears pass between the body and the raised short leg, and holds the tip over the inner holes
while arm B's fingertips push the shackle's bend down until it latches.

## Layout

| Path | What |
|---|---|
| `handwheel_valve.py` | the task file: description for agents, the props and task, scoring |
| `wheel_lockout_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera test) |
| `wheel_lockout_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (wheel and pivot friction, shackle latch, decoy watch, after-release test) and milestones |
| `wheel_lockout_sim/assets/`, `wheel_lockout_sim/usd/` | the props, and the scripts that generate them from the dimensions in task.json (`python -m wheel_lockout_sim.usd.valve_props wheel_lockout_sim/assets`; the cover is `usd/cover_props.py`) |
| `wheel_lockout_sim/demo.py`, `wheel_lockout_sim/policy.py` | the scripted demo and its Inspect Robots policy `wheel-lockout-demo` |

## References

- Master Lock 480 rotating gate valve lockout, product page (fits 1 to 3 in handles, 30 mm depth, 19 mm stem opening):
  https://www.masterlock.com/products/product/480
- Master Lock 480 and 480D technical data (removable centre knockout for rising-stem valves, takes several padlocks):
  https://solutions.masterlock.com/Technical_Literature/Master_Lock/Workplace_Safety%3A_Lockout_Tagout/Valve_Lockouts/Valve_Lockout%3A_480_and_480D
- Master Lock product sheets 480, 481, 482: https://cdn.masterlock.com/sheet/cn_eu/cn.masterlock.eu_480.pdf,
  https://cdn.masterlock.com/sheet/cn_eu/cn.masterlock.eu_481.pdf, https://cdn.masterlock.com/sheet/cn_eu/cn.masterlock.eu_482.pdf
- How-to video, "480 to 484 - Rotating Gate Valve Lockout": https://www.youtube.com/watch?v=X6I_rdaJuOg
- Brady hinged gate valve lockouts (the clamshell alternative, same enclosing principle), e.g. 065561:
  https://www.klium.be/en/brady-065561-locking-system-for-handwheel-valves-64-127-mm-diameter-red-13208-mm-78255
- Milwaukee Valve UP148 bronze gate valve data sheet (1/2 in: 50.8 mm end to end, 111 mm centre to top open, 50.8 mm
  handwheel; rising stem, malleable iron handwheel):
  https://voomi-assets-public.s3.amazonaws.com/products/2024-08-24/voomi_MILWAUKEE%20VALVE_UP14812_documents_product_data_sheet_fc8c41ae8c9a47f7d58b6230e020caba858249.pdf
- ANSI/ASME A13.1 pipe and valve identification colours (yellow with black for flammable, green with white for water).
- Master Lock 410 Zenex thermoplastic safety padlock (410PRP): https://www.masterlock.com/business-use/product/410PRP
- OSHA 29 CFR 1910.147, control of hazardous energy:
  https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity
compensation). Rigid bodies on joints, with every link of the lock in geometry: the handwheel lies between the cover's
front and back faces and inside its rims, the back face's stem opening (19 mm) is smaller than the wheel, the two
ears' holes line up only when the loose half is shut, and the padlock's shackle through both ears stops the loose half
swinging open. Nothing in the code holds the cover, the wheel or the loose half.

- **Valve.** The wheel turns in place on a revolute joint; its closed stop (the wedge seating) is the joint's limit,
  one full turn from the start. A rising-stem valve's wheel rises a few millimetres per turn as it opens; that rise is
  not modelled, and opening is stopped half a turn past the start (the real valve opens about five turns). Turning
  resistance is stick-slip friction of 0.3 N m. The running torque of an unpressurised 1/2 in bronze gate valve is
  not published (it depends on the packing nut; a hand turns a 2 in wheel easily, and seating it firmly takes more).
  0.3 N m is set by the robot: SIM, the station gripper's pinch on the 6 mm rim slipped at 0.5 N m (the wrist turned 51
  degrees while the wheel turned 22). The wheel's rim, spokes and nut are boxes (the rim is drawn round).
- **Cover.** Published: fits 1 to 3 in (25 to 76 mm) handwheels, 30 mm depth, 19 mm stem opening, Zenex
  thermoplastic, takes several padlocks. Inferred from the product photos and the video: the outer diameter (86 mm),
  2 to 2.5 mm walls, the fixed piece's 230 degree sector and the loose piece's 134 degree sector nested 0.5 mm inside
  it, the pivot boss the cover rests on, the ears (16 mm wide, reaching 76 mm from the centre, 9 mm square holes 56 and
  67 mm from the centre) and the mass (105 g). The real 480 has a centre knockout for rising-stem valves whose stem
  stands above the wheel; the modelled valve's stem ends in the wheel nut, so the knockout stays in. The pivot has
  near-zero friction (0.0005 N m), so the loose half swings without dragging the fixed half round the stem.
  Colliders are convex hulls of 10 degree annular sectors.
- **Padlock.** A Master Lock 410 (body 38 x 45 x 19 mm, 6.35 mm shackle). Rounded visual body with a danger label over
  a box collider. Its shackle pops straight up and is pushed straight down (no swivel), with a 150 N/m spring and a
  catch that latches once it is pushed within 2 mm of shut; the key is not simulated. The holes leave 1.3 mm of play
  around the shackle, so the pried loose half moves about 3 degrees before it bears on the shackle.
- **Valve tags.** Plant valves usually carry small stamped brass tags (about 38 mm), which would be about 8 pixels
  across in the 320 x 176 top camera, so each valve carries a larger engraved plastic tag on a strap, turned to face
  the robot. The two valves are told apart by the tag's service colour (ANSI A13.1 style: yellow with black letters,
  green with white letters), which is clear at 320 x 176; the 28 mm letters are not reliably legible there.
- **Check forces.** The 15 N pry and the 10 N lift are applied as forces in the simulator after the recording ends.
  The turn is applied as a steady turning speed of the cover (15 degrees per second for 2 s, at most 30 degrees),
  because a torque servo at the 20 Hz control rate is unstable on the light cover. The speed is set once per control
  step and the cover's contacts slow it in between, so in the demo it turned about 10 degrees; it turns freely on
  the stem and the wheel does not follow. One layout.

# Lock out a machine's START push button with a push-button lockout cover

A light-grey push-button station lies on the table in front of the robot with two 22 mm flush push buttons facing up:
a green START nearer the robot and a red STOP 7 cm behind it, each with a clear install base ring under its chrome
bezel and a black legend plate (START / STOP) to its right. A clear Master Lock S2151 removable push-button cover
stands open on a grey block to the robot's right, and a blue Master Lock 410 padlock lies open on a pedestal to the
left. The robot must set the open cover around the START button so the cover's bottom lip goes under the base ring,
close the cover, and padlock its two ears together. Neither button may be pressed and the STOP button must not be
covered. Scoring checks the lockout physically after the robot lets go: the cover is pressed down over the button,
pried open and lifted, and must stay shut and on with the button out. A cover that is closed but not padlocked pries
open under the same check.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`pushbutton_lockout.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream`
policy. Its props are articulated (the buttons' spring-return plungers, the cover's hinge, the padlock's shackle),
which the one-file format cannot declare, so the file uses `pushbutton_sim` (this folder), which adds them to the
station at run time. `pushbutton_sim/station.py` documents exactly what it patches. It requires yam-isaac-lab-arena
at commit `0059d02`, the Brev launchable's pin.

## How the real thing works

A push-button lockout cover keeps a control from being operated. The Master Lock S2151 is the removable version: a
clear polycarbonate cylinder (it takes buttons up to 50 mm across and 45 mm tall on 22.5 or 30.5 mm operators) made of
two half-shells joined by a hinge along one vertical seam. At the other seam each half has a flat padlock ear; shut,
the two ears lie face to face with their holes in line. A clear install base (S2152AST, or the S2154AST horseshoe
retrofit base stuck on with tape) is fitted once under the operator's bezel, so it is captured by the button itself
and stands as a ring with a raised rim around it. To lock the button out, the worker opens the cover, sets it around
the button so the inward lip at the bottom of each half goes under the base's rim, swings the halves shut and puts a
padlock through both ears. Shut, the lips cannot come up past the rim, the cover encloses the button with room above
it so the button cannot be reached or pressed through it, and opening the halves would slide the two ears apart
across the shackle, which the shackle stops. One cover serves any button that has a base. Master Lock's sell sheet
and product photos show the parts and the padlocked cover on a green flush button; a clear clamshell push-button
lockout video and Brady's push-button safety-cover video show the same sequence (base ring under the bezel, cover
fitted over the button, padlock through the ears).

The push buttons are 22 mm flush spring-return operators of the Schneider Harmony XB4 type: a chrome bezel 30 mm
across standing 9 mm proud of the panel, and a flush cap that travels 4.3 mm in total; the contact changes state
2.6 mm in, at 3.8 N, and a spring returns the cap when it is let go (Schneider data sheet). By the usual convention
START is green and STOP red, each with its legend plate.

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/pushbutton-lockout --no-deps          # registers the policy pushbutton-lockout-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/pushbutton-lockout/pushbutton_lockout.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lock out the green START button with the clear push-button cover: take the open cover from the block on the right, set it around the START button so its bottom edge goes under the clear base ring, close the cover, then put the blue padlock's shackle through both of the cover's ears and snap it shut. Do not press either button, and do not cover the red STOP button." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/pushbutton-lockout-dream
```

The scripted demo is closed loop: like a scripted demo with privileged state, it reads the cover's and padlock's
poses from the simulator, and it sends the same 14-D joint action a policy does, through yam_arena's IK. The right
arm closes across the open cover's back half (one finger on the hinge barrel, the other on the back half's ear),
carries it over the START button 3 mm beyond its axis, lowers it until it hovers 0.7 mm above the panel and slides it
onto the axis so the back half's lip goes under the base's rim; it lets go and holds the cover down with its shut
fingertips on the top. The left arm, fingers shut and tilted so its wrist stays clear of the right hand, pushes the
front half's ear until the cover is shut (the front half's lip swings in under the rim). The left arm then picks the
padlock off its pedestal from above (fingers on the body's sides), slides it so the two ears enter the gap under the open shackle's short leg, lines the leg
up with the ear holes from the measured tip position and draws the lock back so the leg passes through both ears; the
right arm pushes the shackle's bend shut while the left holds the body. Both let go.

```bash
inspect-robots run --embodiment yam_arena --policy pushbutton-lockout-demo \
  -E task=tasks/pushbutton-lockout/pushbutton_lockout.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/pushbutton-lockout-demo
```

`task_score` prints the trial's score: 1.0 on success, otherwise the partial credit given in task.json. To make
videos, run `inspect-robots video <log>.json`, which writes one MP4 per camera.

- **Episode end.** The after-release check starts once the shackle is shut through both ears, the cover is shut on
  the START button, and both grippers have stayed at least 6 cm from the padlock and 7 cm from the cover's centre for
  1 s. It lasts 4.5 s of simulated time (20 N down on the cover's top, then 0.35 N m prying the halves apart about the
  hinge, then 10 N lifting the cover, 1.5 s each), runs inside that control step and off camera, and then the episode
  ends. The stored frames stop at the release. If the check never runs, the run ends at `--max-steps`.
- **Control run.** `--policy pushbutton-lockout-demo -P control=true` fits and shuts the cover, brings no padlock,
  and forces the check once both hands are clear. It must score below 1, because the cover pries open.
- **Seeds.** One fixed layout (`-E object_jitter=0`). Seeded jitter is not used.
- **Runtime per trial.** See "Measured" below. Isaac Sim's first start on a new machine adds several minutes.

## Measured

RTX-class GPU shared with other Isaac Sim processes, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz,
320x176 frames:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `pushbutton-lockout-demo` | `task_score: 1` (cover carried by its back half, set down beside the base and slid on; front half pushed shut by its ear; padlock's short leg drawn through both ears and snapped shut; after-release check: START cap 0.0 mm in under 20 N, cover pried to 2.1 degrees, lifted 0.3 mm) | 1,765 (88 s), then the 4.5 s check off camera | about 44 minutes |
| Control: `pushbutton-lockout-demo -P control=true` (cover fitted and shut, no padlock, check forced) | `task_score: 0.5` (the 0.35 N m pry swung the cover wide open, 100 degrees, and the lift then took it off) | 739 | about 20 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (the right hand goes to the open cover on its block and hovers over and against it but never lifts it; the left arm stays near its start; neither button is pressed) | 3,600 (step limit) | about 87 minutes (Dream reported an observation as expired and was asked again 74 times) |

The wall time is mostly GPU contention and the demo's Python IK, solved every control step; alone on the GPU a trial is
several times faster.

## Fidelity details

PhysX in Isaac Lab-Arena at the station's 240 Hz, both arms driven as in Dream's station (joint PD with i2rt's gains
and gravity compensation). Rigid bodies and joints.

What the geometry enforces: each half of the cover has an inward lip along the bottom of its middle 100 degrees; shut
around the install base, the lips sit under the base's flange (1 mm overlap, 0.3 mm play), so the cover cannot be
lifted off; the cover's top stands 38 mm above the button's cap and the cover rests on the panel, so pressing the
cover does not reach the button; the two ears lie face to face with their 9 mm holes in line, and opening the cover
moves them apart across the hole axis, so a shackle through both holes stops the halves turning on the hinge. Without
the padlock nothing holds the halves shut. Whether the lips go under the rim depends on how the cover is set down: a
cover lowered straight onto the base lands on the rim; it has to be set beside it (or opened) and slid or closed in.

Rules and simplifications, most important first:
- The push buttons are plungers on prismatic joints with a preloaded spring drive (2.0 N at rest, 3.8 N at the
  2.6 mm contact point, 4.3 mm total travel; the data sheet gives the 3.8 N and the travels, the preload is inferred);
  "pressed" is read from the plunger's travel, not from a contact.
- The cover's dimensions are inferred from the button size it takes and from product photos (Master Lock publishes
  no drawing): 64.4 mm across, 50 mm tall, 1.8 mm walls, lips 1.3 mm deep, ears 14 x 18 mm with a 9 mm hole standing
  12 mm above the top, the hinge axis 5 mm outside the wall. The real ears' exact shape and the real lip are not
  published; the install base's rim and flange are inferred likewise.
- The hinge turns freely with light damping (0.05 N m s/rad); the real moulded hinge may be stiffer.
- The halves do not collide with each other (they are one articulation); the hinge stops at shut.
- Shapes: boxes for every collider (circles are 10-degree polygons; round holes are 9 mm squares); the cover is drawn
  clear (55 % opaque), the base rings clear.
- The padlock's shackle spring (150 N/m), its 16 mm pop-up and its latch (it snaps shut within 2 mm of closed) are
  rules; the lock has no key.
- The padlock starts lying level on a pedestal, shackle toward the cover, so that the gripper closes on its broad
  faces level with the ears.
- The after-release check (press, pry, lift) is applied in the simulator after the recording ends; a reviewer does
  it with a spring scale.
- The station's legend plates are about 2 pixels tall in the 320x176 top camera; there the buttons are told apart by
  colour (green START, red STOP), as the instruction says. One layout.

## Layout

| Path | What |
|---|---|
| `pushbutton_lockout.py` | the task file: description for agents, props and task, scoring |
| `pushbutton_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera check) |
| `pushbutton_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (button springs, hinge, shackle latch, after-release check), hazard rules and milestones, layout and instruction |
| `pushbutton_sim/assets/*.usda`, `pushbutton_sim/usd/` | the props, and the script that generates them (`python -m pushbutton_sim.usd.pb_props`; `padlock_410.py` draws the padlock) |
| `pushbutton_sim/demo.py`, `pushbutton_sim/policy.py` | the scripted demo and its Inspect Robots policy `pushbutton-lockout-demo` |

## References

- Master Lock S2151 removable push button and rotary switch cover (fits 22.5 and 30.5 mm switches, buttons up to
  51 mm across and 44 mm tall; clear base and cover): https://www.masterlock.com/products/product/S2151
- Master Lock S2153 installed push button and rotary switch cover: https://www.masterlock.com/products/product/S2153
- Master Lock sell sheet S2151 / S2153 (polycarbonate; install bases S2152AST and retrofit bases S2154AST):
  https://cdn.masterlock.com/masterlock/resources/documents/pdf/sell-sheets/7000-0265_S2151_AND_S2153_SELL_SHEET.pdf
- Schneider Electric Harmony XB4BA31 push button data sheet (22 mm, flush, spring return; operating travel 2.6 mm,
  total travel 4.3 mm, operating force 3.8 N; 30 mm head):
  https://www.se.com/ww/en/product/XB4BA31/
- Brady, "Lockout Push Button Safety Covers" (base installed once under the bezel, cover fitted over the button and
  padlocked): https://www.youtube.com/watch?v=j5_8BpvUMrU
- Clear clamshell push-button lockout fitted over a 22 mm flush button and padlocked:
  https://www.youtube.com/watch?v=u91qY4DxUJ4
- Master Lock 410 Zenex thermoplastic safety padlock (410BLU): https://www.masterlock.com/business-use/product/410BLU
- OSHA 29 CFR 1910.147, control of hazardous energy:
  https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

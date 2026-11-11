# Lock out an equipment cord with a plug lockout

Two black power cords are plugged into a duplex wall outlet on a plate to the robot's right. The top one carries a
yellow cord tag ("GRINDER"), the bottom one a green tag ("VENT FAN"). A red Brady 65674 plug lockout lies open like a
book on a stand in front of the robot, and a blue Master Lock 410 padlock lies open on a pedestal to the left. The
robot must unplug the yellow-tagged cord, lay its NEMA 5-15P plug in the lockout's lower half with the cord through
the notch at its end, close the lid, and put the padlock's shackle through the two tabs and snap it shut. The
green-tagged cord must stay plugged in. Scoring checks the lockout physically after the robot lets go: the lid is pried
and the cord is pulled, and the plug must stay shut inside. A lid that is closed but not padlocked opens under the
same check.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file (`plug_lockout.py`)
in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy. Its props are articulated
(the outlet's contact springs, the two cords as anchored chains of jointed capsules with the plugs on their ends, the
lockout's hinged lid, the padlock's shackle), which the one-file format cannot declare, so the file uses
`pluglock_sim` (this folder), which adds them to the station at run time. `pluglock_sim/station.py` documents exactly
what it patches. It requires yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin.

## How the real thing works

A cord-and-plug machine is isolated by unplugging it; a plug lockout keeps the plug under the worker's control while
they work on the machine. The Brady 65674 is a red polypropylene clamshell about 89 mm long and 51 mm across: two
half-shells joined by a hinge along one long edge. Each half has a semicircular notch at one end (together a 1/2 in
cord hole) and, on its free edge at mid-length, a flat tab with a padlock hole, standing across the box's long axis.
The worker unplugs the cord, opens the lockout like a book, lays the plug in one half with its cord through the notch,
and closes the halves. The two tabs then lie face to face with their holes lined up, and a padlock shackle goes
through both. The plug body is far bigger than the cord hole, so it cannot be pulled out; its blades are enclosed, so
it cannot be plugged in; the padlock keeps the halves from opening. Brady's animation shows these steps starting from
a plug in a connector; Brady's application photos show the open lockout lying beside a wall outlet with the plug laid
in one half, and the closed lockout with a blue padlock through the tabs.

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/plug-lockout --no-deps                # registers the policy plug-lockout-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/plug-lockout/plug_lockout.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lock out the cord with the yellow tag: unplug it from the outlet, lay its plug in the open red plug lockout with the cord through the notch at its end, close the lid, then put the blue padlock's shackle through both tabs and snap it shut. Leave the cord with the green tag plugged in." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/plug-lockout-dream
```

The scripted demo is closed loop: like a scripted demo with privileged state, it reads the plug's, lid's and
padlock's poses from the simulator, and it sends the same 14-D joint action a policy does, through yam_arena's IK.
The right arm grips the yellow-tagged plug from above and pulls it out in small steps, turns it blades-first along the
lockout and lowers it into the lower half until the fingertips are just above the rims, lets it drop the last few
millimetres, and presses the cord into the end notch. It then grips the open lid's tab and swings the lid up, pitching
the hand with it; past upright it takes the tab again from above and carries the lid down until it falls shut, and
presses it down. The left arm picks the padlock off its pedestal, slides it so the two tabs pass under the open
shackle's short leg, lines the leg up with the holes from the measured tip position and pushes it through both tabs in
2 mm steps; the right arm pushes the shackle's bend shut while the left holds the body. Both let go.

```bash
inspect-robots run --embodiment yam_arena --policy plug-lockout-demo \
  -E task=tasks/plug-lockout/plug_lockout.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/plug-lockout-demo
```

`task_score` prints the trial's score: 1.0 on success, otherwise the partial credit given in task.json. To make
videos, run `inspect-robots video <log>.json`, which writes one MP4 per camera.

- **Episode end.** The after-release check starts once the shackle is shut through both tabs, the lid is shut on the
  plug, and both grippers have stayed at least 6 cm from the padlock and 7 cm from the lockout's centre for 1 s. It
  lasts 3 s of simulated time (0.5 N m on the lid for 1.5 s, then a 40 N pull on the cord for 1.5 s), runs inside
  that control step and off camera, and then the episode ends. The stored frames stop at the release. If the check
  never runs, the run ends at `--max-steps`.
- **Control run.** `--policy plug-lockout-demo -P control=true` unplugs the cord, lays the plug in, closes the lid,
  brings no padlock, and forces the check once the hand is clear. It must score below 1, because the lid pries open.
- **Seeds.** One fixed layout (`-E object_jitter=0`). Seeded jitter is not used.
- **Runtime per trial.** See "Measured" below. Isaac Sim's first start on a new machine adds several minutes.

## Measured

RTX-class GPU shared with other Isaac Sim processes, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz,
320x176 frames:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `plug-lockout-demo` | `task_score: 1` (plug pulled out at about 5 N, laid in the lower half with the cord through the notch, lid swung shut, padlock's short leg pushed through both tabs and snapped shut; after-release check passed) | 2,305 (115 s), then the 3 s check off camera | about 34 minutes |
| Control: `plug-lockout-demo -P control=true` (lid shut on the plug, no padlock, check forced) | `task_score: 0.75` (the 0.5 N m pry swung the lid wide open, 180 degrees) | 1,273 | about 20 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (both arms move a little near their start; the right hand comes down by the cords near the outlet; nothing is unplugged and the green-tagged plug stays in) | 3,600 (step limit) | about 44 minutes (Dream reported an observation as expired and was asked again 74 times) |

The wall time is mostly GPU contention and the demo's Python IK, solved every control step; alone on the GPU a trial is
several times faster.

## Fidelity details

PhysX in Isaac Lab-Arena at the station's 240 Hz, both arms driven as in Dream's station (joint PD with i2rt's gains
and gravity compensation). Rigid bodies and joints.

What the geometry enforces: the plug body (38 x 30 mm) cannot pass the 12.7 mm cord hole that the two shells' end
notches make when the lid is shut, so a pulled cord cannot draw the plug out; the blades are inside the shut shell; a
shackle through both tabs' 9.5 mm holes stops the lid turning on its hinge (the lid's tab would have to move across the
shackle), and without a padlock nothing holds the lid shut. If the cord is not in the notch, the lid cannot shut on it.
In the control run (lid shut, no padlock) the 0.5 N m pry swung the lid wide open; in the demo the padlocked lid
held the pry and the plug stayed inside under the 40 N cord pull.

Rules and simplifications, most important first:
- The lockout's lower half is held by a fixture on the stand (two clips at its back corners); a worker would hold the
  lockout in one hand. The lid swings on a hinge joint with damping (0.1 N m s/rad, inferred: a lid let go near
  upright sinks a few degrees a second rather than flopping) and no detent.
- The outlet holds each plug by friction only: behind each blade slot a pad on a spring (a preloaded joint drive)
  presses the blade against a fixed contact, which takes about 5 N to overcome. Real receptacles hold harder (UL 498
  asks a 15 A receptacle to need more than 13 N to withdraw its test plug, and a new one often needs 20 to 40 N). The
  station arm's soft wrist (10 N m/rad) cannot pull much harder sideways without the plug pitching in the hand and its
  blades binding in the slots, so the grip was set low (cause: SIM gains and the YAM's wrist; a real YAM pulling a new
  receptacle would need a firmer pull).
- Cords: 24 capsule links (2.5 cm each, 8 mm across, 0.1 kg/m) on revolute joints with bending springs toward their
  starting route; stiffness (EI 0.0008 N m^2, scaled from a heavier cord by diameter to the fourth power) and friction
  are guesses. The two joints next to the plug are 8 times stiffer and start straight, standing in for the molded
  strain relief that keeps the first few centimetres of cord in line with the plug. The far end of each cord is
  clamped to the table, standing in for the machine.
- The cord tags are drawn on the cords (no collision).
- The padlock's shackle spring (150 N/m), its 16 mm pop-up and its latch (it snaps shut within 2 mm of closed) are
  rules; the lock has no key.
- Shapes: boxes for every collider (the round cord hole is a 12.7 mm square, the round tab holes 9.5 mm squares, the
  round ground pin a 4.8 mm square); the shells' outer ribs and the hinge knuckles are visual only. The shells' wall
  thickness (2.5 mm), the tabs' size (2.4 cm square, 3 mm thick) and their hole's position are read off product photos;
  Brady publishes only the overall size, the cord hole and the plug size it takes.
- The plug body is a typical molded 5-15P (38 x 30 x 30 mm plus an 18 mm strain relief), not a specific vendor's
  drawing; blade and pin sizes and spacing follow NEMA 5-15.
- The padlock starts on a pedestal whose top slopes 15 degrees (shackle end down), so that the arm's fingers, pitched
  15 degrees down as the arm reaches best, close flat on its broad faces; it is held by the body's far end so no finger
  comes between the padlock and the lockout's front.
- The demo's path is one way of doing it: the YAM cannot pitch its hand past 90 degrees there, so the lid is swung
  in two grasps of its tab, and the plug is dropped the last 1 to 1.5 cm into the shell because the fingers do not fit
  inside it beside the plug.
- The after-release check (a torque on the lid, then a force on the cord 7.5 to 10 cm from the plug) is applied in
  the simulator after the recording ends; a reviewer does it with a spring scale.
- The printed tag legends are 2 to 3 pixels tall in the 320x176 top camera; there the cords are told apart by the tag
  colour (yellow vs green), as the instruction says. One layout.

## Layout

| Path | What |
|---|---|
| `plug_lockout.py` | the task file: description for agents, props and task, scoring |
| `pluglock_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera check) |
| `pluglock_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (contact springs, hinge, shackle latch, after-release check), hazard rule and milestones, layout and instruction |
| `pluglock_sim/assets/*.usda`, `pluglock_sim/usd/` | the props, and the script that generates them (`python -m pluglock_sim.usd.pluglock_props`; `padlock_410.py` draws the padlock) |
| `pluglock_sim/demo.py`, `pluglock_sim/policy.py` | the scripted demo and its Inspect Robots policy `plug-lockout-demo` |

## References

- Brady 65674 small plug lockout (110 V plugs; distributor data: 3.5 in long, about 2 in across; plugs up to 1.7 in
  across and 3.2 in long; 1/2 in cord hole; polypropylene), a distributor listing with Brady's product and
  application photos: https://www.lumen.ca/en/products/22-safety-products/05-lockout-tagout/02-electrical-lockouts/02-plug-lockouts/p-QlJBNjU2NzQ=-bra65674-plug-lockout-for-up-to-17-in-dia-cord-32-in-l-110-v-plugs-036-in-shackle-polypropylene-red
- Brady, "Brady Lockout Tagout Device Movie: Plug lockout" (unplug, lay the plug in one half, close, padlock through
  both tabs): https://www.youtube.com/watch?v=WJADpZ_UwaQ
- NEMA 5-15 plug and receptacle geometry (blades 6.35 x 1.5 mm, 12.7 mm apart, 15.9 mm long; round ground pin
  4.8 mm, 3.2 mm longer, 11.9 mm off the blades' line): https://en.wikipedia.org/wiki/NEMA_connector
- UL 498 plug retention for general-use receptacles (more than 13 N to withdraw a test plug, 67 N must withdraw it),
  as summarised by a test-equipment vendor: https://www.lisungroup.com/?p=13545
- Master Lock 410 Zenex thermoplastic safety padlock (410BLU): https://www.masterlock.com/business-use/product/410BLU
- OSHA 29 CFR 1910.147, control of hazardous energy (cord-and-plug equipment, 1910.147(a)(2)(iii)(A)):
  https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

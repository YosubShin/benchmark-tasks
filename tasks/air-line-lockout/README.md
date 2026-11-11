# Disconnect a tool's air hose and lock its plug in a pneumatic plug lockout

A two-outlet air manifold stands on the table to the robot's right with two brass 1/4 in industrial-interchange
(M-style) quick couplers. Two blue air hoses are connected to them by their male plugs: the near one carries a yellow
hose tag ("GRINDER"), the far one a green tag ("DRILL"). A red Brady PLO27E electrical/pneumatic plug lockout box
stands open in front of the robot, and a blue Master Lock 410 padlock lies open on a pedestal to the left. The robot
must disconnect the yellow-tagged hose the way the coupler works (one hand pulls the coupler's sleeve back, the other
draws the plug out), lay the plug in the lockout's lower half with the hose through the hole at its end, close the
lid, and put the padlock's shackle through the two tabs and snap it shut. The green-tagged hose must stay connected.
Scoring checks the lockout physically after the robot lets go: the lid is pried and the hose is pulled, and the plug
must stay shut inside. A lid that is closed but not padlocked opens under the same check.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`air_line_lockout.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy.
Its props are articulated (the couplers' sleeves and locking balls, the two hoses as anchored chains of jointed
capsules with the plugs on their ends, the lockout's hinged lid, the padlock's shackle), which the one-file format
cannot declare, so the file uses `airlock_sim` (this folder), which adds them to the station at run time.
`airlock_sim/station.py` documents exactly what it patches. It requires yam-isaac-lab-arena at commit `0059d02`, the
Brev launchable's pin.

## How the real thing works

An air tool is isolated for servicing by shutting and bleeding its supply and disconnecting its hose at the quick
coupler; a pneumatic plug lockout then keeps the hose's male plug where nobody can reconnect it. An industrial-
interchange (M-style, MIL-C-4109) coupler such as the Milton 715 holds the plug with a ring of steel balls that sit in
holes in its bore and drop into a groove behind the plug's nipple. A spring pushes the knurled sleeve forward, and the
sleeve's inner land keeps the balls pressed into the groove, so pulling the hose cannot draw the plug out. To
disconnect, a worker pulls the sleeve back toward the air supply (against its spring) with one hand: a relief inside
the sleeve then lies over the balls, the plug's lip pushes them outward, and the other hand draws the plug out. The
Brady PLO27E (Prinzing) electrical/pneumatic plug lockout is a red box (163 x 117 x 96 mm) in two halves hinged along
the back; its end walls have holes split along the parting line for a cord or the necked-down hose of a pneumatic male
fitting, and the front carries a padlock tab on each half. The worker lays the plug in the open lower half with its
hose through the end hole, closes the lid and padlocks the two tabs together; the plug's hex and hose crimp are larger
than the hole, so it cannot be pulled out, and its nipple is enclosed, so it cannot be connected. A how-to video of a
pneumatic plug lockout of this kind shows exactly these steps with a brass air plug on a blue hose; Master Lock's
S3900 strip lockout does the same job by trapping the plug's neck in a keyhole slot.

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/air-line-lockout --no-deps            # registers the policy air-line-lockout-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/air-line-lockout/air_line_lockout.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lock out the air hose with the yellow tag: pull its coupler's sleeve back and pull the hose's plug out, lay the plug in the open red lockout box with the hose through the hole at its end, close the lid, then put the blue padlock's shackle through both tabs and snap it shut. Leave the hose with the green tag connected." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/air-line-lockout-dream
```

The scripted demo is closed loop: like a scripted demo with privileged state, it reads the plug's, sleeve's, lid's and
padlock's poses from the simulator, and it sends the same 14-D joint action a policy does, through yam_arena's IK.
The left arm closes across the yellow-tagged coupler's sleeve and pulls it back to its stop while the right arm, closed
across the plug's hex and hose crimp just in front of it, draws the plug straight out in small steps. The left hand
lets go (the sleeve springs back toward the front). The right arm carries the plug over the lockout, tips it nipple down and sets it
in the lower half with its crimped end just inside the end wall, the hose running out through the half hole, and
presses the hose down into the hole. The left arm takes the open lid's tab and swings the lid over until it falls
shut, presses it down, then picks the padlock off its pedestal, pushes the shackle's short
leg through both tabs, and holds the body while the right arm pushes the shackle shut. Both let go.

```bash
inspect-robots run --embodiment yam_arena --policy air-line-lockout-demo \
  -E task=tasks/air-line-lockout/air_line_lockout.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/air-line-lockout-demo
```

`task_score` prints the trial's score: 1.0 on success, otherwise the partial credit given in task.json. To make
videos, run `inspect-robots video <log>.json`, which writes one MP4 per camera.

- **Episode end.** The after-release check starts once the shackle is shut through both tabs, the lid is shut on the
  plug, and both grippers have stayed at least 6 cm from the padlock and 3 cm outside the closed lockout's outline for
  1 s. It lasts 3 s of simulated time (1 N m on the lid for 1.5 s, then a 40 N pull on the hose for 1.5 s), runs
  inside that control step and off camera, and then the episode ends. The stored frames stop at the release. If the
  check never runs, the run ends at `--max-steps`.
- **Control run.** `--policy air-line-lockout-demo -P control=true` disconnects the plug, lays it in, closes the lid,
  brings no padlock, and forces the check once the hands are clear. It must score below 1, because the lid pries open.
- **Seeds.** One fixed layout (`-E object_jitter=0`). Seeded jitter is not used.
- **Runtime per trial.** See "Measured" below. Isaac Sim's first start on a new machine adds several minutes.

## Measured

RTX-class GPU shared with other Isaac Sim processes, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz,
320x176 frames:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `air-line-lockout-demo` | `task_score: 1` (sleeve pulled back 6 mm, plug drawn out at a few newtons, laid in the lower half with the hose through the half hole, lid swung shut, padlock's short leg pushed through both tabs and snapped shut; after-release check: lid pried 0.0 degrees, plug stayed inside under the 40 N hose pull) | 2,478 (124 s), then the 3 s check off camera | about 37 minutes |
| Control: `air-line-lockout-demo -P control=true` (lid shut on the plug, no padlock, check forced) | `task_score: 0.75` (the 1 N m pry swung the lid back open to its 95 degree stop) | 1,462 | about 21 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (both arms move near the box and the manifold, the left reaching over the box toward the couplers; no sleeve is pulled back, nothing is disconnected and the green-tagged hose stays connected) | 3,600 (step limit) | about 49 minutes (Dream reported an observation as expired and was asked again 74 times) |

The wall time is mostly GPU contention and the demo's Python IK, solved every control step; alone on the GPU a trial is
several times faster. Disconnecting the green-tagged hose was not tried by the demo; a 40 N pull on a connected plug
with its sleeve forward moved it 0.5 mm (the balls hold it), and with the sleeve held back the plug came out at
about 3 N.

## Fidelity details

PhysX in Isaac Lab-Arena at the station's 240 Hz, both arms driven as in Dream's station (joint PD with i2rt's gains
and gravity compensation). Rigid bodies and joints.

What the geometry enforces: in each coupler, four 3.2 mm balls on radial slides sit in the connected plug's groove.
With the sleeve forward, the sleeve's inner land stops them rising (they stay 1 mm inside the radius of the plug's lip),
so the plug cannot be drawn out: a 40 N pull on a connected plug moved it 0.5 mm. With the sleeve pulled back 5 mm or
more, the land has passed the balls; the lip pushes them outward and the plug comes out at about 3 N. The plug's hex
and ferrule (14.3 and 15 mm across) cannot pass the 10 mm hose hole that the two shells' end notches make when the lid
is shut, so a pulled hose cannot draw the plug out; the nipple is inside the shut box, so it cannot be connected; a
shackle through both tabs' 8.5 mm holes stops the lid turning on its hinge, and without a padlock nothing holds the
lid shut. In the control run (lid shut, no padlock) the pry swung the lid open; in the demo the padlocked lid held the
pry and the plug stayed inside under the 40 N hose pull.

Rules and simplifications, most important first:
- The sleeve spring is a preloaded joint drive of 2 N at rest rising to 3.5 N at the 6 mm stop. A real M-style
  coupler's sleeve spring is stiffer (not published; by hand it takes a firm two-finger pull, roughly 20 to 40 N,
  estimated). The station arm's soft gains could not pull the sleeve against 6 N with a 20 mm command lead, so the
  spring was set low (cause: SIM gains; a real YAM would need a firmer pull). The balls' inner stop (in the groove)
  and outer stop (1.5 mm out) are joint limits standing in for the tapered holes in the coupler's body; a ball weighs
  2 g instead of 0.13 g to keep the articulation's mass ratios sane.
- With the plug out, the sleeve's spring brings it forward only until its land rests on the lowest ball (about 4 mm
  back), where a real coupler's chamfered land pushes the balls in and the sleeve returns fully.
- The supply is taken as shut and bled (the ball valve's handle is across); there is no air pressure, so a plug does
  not blow out when its sleeve is pulled back.
- Collision shapes are boxes and spheres: the round nipple, groove, hex and ferrule are squares of the same width
  (9.2, 7.0, 14.3 and 15 mm), the bore a square tube, the sleeve's land four plates over the balls; the knurled sleeve
  is gripped on a 23.6 mm square shell. Proportions of the nipple's lip and groove are inferred (the MIL-C-4109
  drawing is not public); overall coupler and plug sizes follow Milton's published data.
- Hoses: 24 capsule links (2.5 cm each, 9.5 mm across, 0.08 kg/m) on revolute joints with weak bending springs toward
  their starting route (EI 0.00003 N m^2): the hose drapes like a rope, softer than a real air hose, because any
  stiffer spring gave it a memory of its starting route that a real hose does not have. The two joints next to the
  plug are 3 times stiffer (the crimped ferrule). Friction is a guess. The far end of each hose is clamped to the
  table, standing in for the tool.
- The lockout's lower half is held by a fixture on the stand (two clips at its back corners); a worker would hold it
  in one hand. The lid swings on a hinge joint with damping (0.1 N m s/rad, inferred) and a stop at 95 degrees open;
  below about 63 degrees it falls shut under its own weight. The tabs stick out 3.2 cm (hole 2.15 cm from the face),
  about twice what the product photos suggest, so the hand holding the padlock clears the box's front face. The PLO27E's several end holes and several tab
  holes are modelled as one 10 mm hose hole (square) and one 8.5 mm hole (square) per tab; its wall thickness (3 mm)
  and the tabs' size are read off product photos.
- The hose tags are drawn on the hoses (no collision).
- The padlock's shackle spring (150 N/m), its 16 mm pop-up and its latch (it snaps shut within 2 mm of closed) are
  rules; the lock has no key.
- The demo's path is one way of doing it: the plug is set in the box nipple down by about 25 degrees with its crimped end
  just inside the hose hole, so the hose runs straight out through the hole.
- The after-release check (a torque on the lid, then a force on the hose 7.5 to 10 cm from the ferrule) is applied
  in the simulator after the recording ends; a reviewer does it with a spring scale.
- The printed tag legends are a few pixels tall in the 320x176 top camera; there the hoses are told apart by the tag
  colour (yellow vs green), as the instruction says. One layout.

## Layout

| Path | What |
|---|---|
| `air_line_lockout.py` | the task file: description for agents, props and task, scoring |
| `airlock_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera check) |
| `airlock_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (sleeve springs, hinge, shackle latch, after-release check), hazard rule and milestones, layout and instruction |
| `airlock_sim/assets/*.usda`, `airlock_sim/usd/` | the props, and the script that generates them (`python -m airlock_sim.usd.airlock_props`; `padlock_410.py` draws the padlock) |
| `airlock_sim/demo.py`, `airlock_sim/policy.py` | the scripted demo and its Inspect Robots policy `air-line-lockout-demo` |

## References

- Brady PLO27E combination electrical and pneumatic plug lockout (163 x 117 x 96 mm, polystyrene, red; padlock
  shackle up to 7.87 mm with 20 mm clearance; holes for cords up to 9.4 mm or necked-down pneumatic male fittings):
  https://www.bradyid.com/lockout-tagout/pneumatic-lockout-devices/combination-electrical-pneumatic-plug-lockout-pid-plo27e
  and a distributor listing with the published dimensions: https://selectumllc.com/i/brady-plo27e
- "Pneumatic plug lockout big" (how-to video: a brass air plug on a blue hose laid in an open lockout box, hose
  through the end hole, lid closed, padlocked through the front tabs): https://www.youtube.com/watch?v=bChz6mdxbxU
- Master Lock S3900 stainless steel pneumatic lockout (traps a male fitting's neck; for comparison):
  https://www.masterlock.com/products/product/S3900
- Milton 715 1/4 in FNPT M-style coupler (about 50 mm long, 23.6 mm across; brass body and sleeve; interchanges per
  MIL-C-4109): https://www.acmetools.com/milton-s-715-4-1-4in-fnpt-m-style-coupler-s-715-4/030937314246.html
- Milton 727 1/4 in M-style plug (43 mm long, 14.3 mm hex): https://aih.com/Product/MTN727BK
- 1/4 in industrial-interchange plug with 1/4 in hose barb (about 53 x 15 mm):
  https://guardair.com/products/1-4-quick-disconnect-industrial-interchange-plug-1-4-hose-barb
- "How to attach and remove air compressor hose fittings: push, pull, pop" (pull the coupler's sleeve back, the plug
  comes out): https://www.youtube.com/watch?v=NuxWsDZrVAA
- Master Lock 410 Zenex thermoplastic safety padlock (410BLU): https://www.masterlock.com/business-use/product/410BLU
- OSHA 29 CFR 1910.147, control of hazardous energy (pneumatic energy; release of stored energy before servicing):
  https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

# Cut a copper tube through at its mark with a screw-feed tubing cutter, without denting the tube

A 1/2 in Type L copper tube stands upright in a stand on the bench, a black pen mark round it. A Klein 50077
screw-feed tubing cutter sits on the tube at the mark, finger-snug: its wheel and rollers touch the tube and nothing is
cut yet. The robot must cut the tube through at the mark, keeping the end round (not dented) and the cut on the
mark, and then let go of the cutter.

The instruction gives only the goal and the constraints. It does not say how a tubing cutter is used. Two things are
tested:
- **Feed control.** Turn the knob only a little (the makers say about 1/4 turn, or 1/16) and then take the cutter
  once round the tube before feeding again. Feeding too much at once crushes the thin wall out of round.
- **Taking a tool round a fixed workpiece.** The cutter has to orbit the tube while it stays square to it. The wrist
  cannot turn a full turn, so the cutter has to be regripped and passed between the hands, and pushing it along the
  tube before the wheel tracks walks the cut off the mark into a spiral.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`tubing_cutter.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy. The
one-file format cannot declare the cutter (an articulation with rule-driven joints), so the file uses
`tubecutter_sim` (this folder), which adds it to the station at run time; `tubecutter_sim/station.py` documents
exactly what it patches. It requires yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin.

## How the real thing works

A screw-feed tubing cutter (Klein 50077, vendor photos in the References) is a C-frame. A hardened cutting wheel is
fixed in its hook end, and a carriage with two rollers slides toward it, pushed by a feed screw. The black knob on the
frame's far end turns that screw. The fitter sets the cutter on the tube with the wheel on the cut mark and turns the
knob until the wheel and rollers touch, then a little more so the wheel bites. They swing the cutter once round to make
sure it tracks (the wheel runs in its own groove instead of walking along the tube). Then they alternate a small knob
turn with one revolution round the tube until the tube parts. Reed's TC2Q instructions say "turn knob about 1/4 turn
every revolution", and Ridge Tool's say "about 1/16 of a turn"; both makers warn against feeding the wheel in too
fast. Too much feed per revolution squeezes the thin wall between the wheel and the rollers and leaves the end oval
or dented, and it also burrs the end and dulls the wheel. Pushing the cutter sideways before the groove holds it
spirals the wheel off the line, so the end is not square. An out-of-round or skewed end does not seat fully in a
solder cup, flare or press fitting and is a leak path, so the fitter recuts it.

## How it differs from tasks on the board

- **Hold one part still while the other hand turns a thread.** Examples are the M12 coupling ring, the drone prop
  nut, the nut behind a plate and the compression joint. In those tasks one hand reacts a torque while the other
  turns about the same axis. Here nothing has to be held: the tube is fixed. The skill is moving a tool round a
  workpiece in a full circle with regrips and hand-to-hand passes, and metering the feed per revolution.
- **Lever tube bender (springback).** That task deforms the tube; this one cuts it.
- **Bench can opener.** That task holds a lever down and cranks about a fixed axis. Here the tool itself travels round
  the workpiece and the feed is a separate, metered action.
- No tubing-cutter task is on the board.

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/tubing-cutter --no-deps               # registers the policy cutter-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/tubing-cutter/tubing_cutter.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Cut the copper tube through at the black mark with the tubing cutter that is on it. The cut must stay on the mark and the tube end must stay round, not dented. Finish with both grippers away from the cutter." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/tubing-cutter-dream
```

The scripted demo is the same command with `--policy cutter-demo` (and without `-P record=false`); the control run
adds `-P control=true`. `task_score` prints the trial's score. To make videos, run `inspect-robots video <log>.json`
(one MP4 per camera).

- **Episode end.** The episode ends at success: cut through, with both grippers at least 5 cm from the cutter for
  1 s without a break. Otherwise it ends at `--max-steps`.
- **Seeds.** One fixed layout (`-E object_jitter=0`).
- **Scorer tests.** `python tests/test_scorer.py` checks the scorer logic. Each hazard and cap is tested before and
  after a milestone, and the hold count restarts when it is broken.

## What the demo reads

The scripted demo (`--policy cutter-demo`, `tubecutter_sim/demo.py`) is closed loop and **uses only what any policy
receives**:
- the three camera images (`top_cam`, `left_cam`, `right_cam`);
- the 14-D `joint_pos` state (both arms' joint angles and gripper openings).

It never reads the simulator: no prop poses, joint angles, groove depth, feed or scorer state. Besides the
observation it uses the documented setup (task.json `setup`: the tube's position, the mark's height, the cutter's
stated dimensions and the 1/4-turn-per-revolution guidance), the station's camera calibration and the YAM URDF (both
from yam_arena), and time.

| What the demo needs | How it gets it |
|---|---|
| Which way the cutter points | A top-camera look with both hands parked out of the way, at the start and after every revolution but the last. It finds the black knob's pixels in a ring round the tube, back-projected through the calibrated camera onto the knob's height. The direction of their centroid from the tube's axis is the cutter's angle. |
| Where the cutter is during a stroke | While a hand holds the cutter's body, the cutter's angle about the tube is the angle of that hand's grasp point (forward kinematics of the measured joints). |
| Whether a grasp holds | The gripper's measured opening: it stops at about 0.43 on the 4 cm body and at about 0.32 on the knob, instead of closing. |
| How far the knob has turned | The wrist's measured roll about the knob's axis between closing on the knob and opening, summed over strokes. |
| How much to feed | The makers' guidance: a quarter turn per revolution. The demo stops after three revolutions (3 x 1/4 turn x 1.8 mm/turn is past the stated 1.0 mm wall). |

**The motions.** All of them are grasp-point targets turned into the same 14-D joint action a policy sends, through
yam_arena's IK.
- **Feed.** The cutter's knob is only within reach of the left arm when it points toward that arm, and only from
  above. The left hand goes over the knob fingers-down, changes posture in joint space to a side grasp (fingers
  horizontal along the knob's axis) and comes down over the knob. It turns the knob a quarter turn clockwise in two
  45 deg strokes: close across the knob, roll the wrist, open, roll back.
- **Orbit.** The cutter is taken round the tube clockwise (seen from above) in 90 deg strokes. Each stroke is a
  fingers-down grasp of the cutter's body 5.5 cm from the tube's axis, an arc about the axis with the wrist turning
  along, then release and lift. The left hand does the strokes on its side and the right hand the two on its side:
  the wrist cannot turn a full turn, so the cutter changes hands every half turn.
- **End of each revolution.** Both hands park and the demo looks. If the cutter has not come all the way round (the
  grip slips a few degrees each stroke), the left hand finishes the turn, to 5 deg past the feed position.
- **Finish.** After three revolutions both hands park away from the cutter.

## Measured

RTX-class GPU shared with several other simulators, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz,
320x176, 240 Hz physics. The numbers in brackets are from the task's report (`TUBECUTTER_DEBUG=1` prints them every
2 s); the demo did not read them.

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `cutter-demo` | `task_score: 1`. **Revolution 1:** the knob was turned a quarter turn in two wrist strokes (0.23 turn measured, 0.41 mm feed), then the cutter was taken round in four 90 deg strokes: left hand, right, right, left. The look then found it 14 deg short, and the left hand finished the turn (groove 0.41 mm all round). **Revolution 2:** the same, giving 0.90 mm. **Revolution 3:** feed to 1.35 mm, then the turn: cut through at step 3,040. Both hands then parked and stayed clear for 1 s. The largest bite in the 2 s reports was 0.49 mm and the cutter never slid along the tube: axial spread 0.00 mm, groove on the mark. | 3,040 (152 s) | about 67 minutes |
| Control: `cutter-demo -P control=true` (half a knob turn before the first revolution, twice the guidance) | `task_score: 0`. The bite reached 0.80 mm before the cutter went round, so the tube was dented. The following revolution still cut a groove 0.80 mm deep all the way round, which would be 0.5 without the dent, and the score stayed 0. | 1,200 (run limit for the control) | about 23 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0`. The arms lifted and turned their wrist cameras toward the stand, but neither hand touched the cutter or its knob: no feed, no groove. | 3,600 (step limit) | about 50 minutes |

The wall times are long because the GPU was shared; the demo's own cost is two arms' Python IK every control step.

## Checking a real trial

Everything is measured with simple tools. The tube is fixed and the cutter's knob carries a mark, so the overhead video
is enough to count the feed and the revolutions.

| Condition | Simulator rule | Real check |
|---|---|---|
| Feed | knob turns since finger-snug x 1.81 mm | count the knob mark's clockwise turns on the video |
| Groove depth | each wheel pass cuts the groove there to the feed | the feed at the last full revolution of the cutter; a dial depth gauge on the groove if in doubt |
| Dented (score 0) | bite (feed past the groove under the wheel) over 0.70 mm at any time | on the video: the knob turned more than 3/8 turn since the cutter last went fully round; after the run: calipers across the tube beside the cut in three directions, max - min over 0.3 mm |
| Spiral (cap 0.5) | the wheel's position along the tube over all cutting passes spreads over 1.0 mm | try square against the tube and the cut end: gap over 1 mm (feeler gauge) |
| Off the mark (cap 0.5) | groove centre over 1.5 mm from the mark | ruler from the remaining edge of the pen line |
| Cut through (0.75) | groove at least 1.02 mm (the wall) all round | the offcut lifts off by hand |
| Let go (1.0) | both grippers 5 cm or more from the cutter's body and knob for 1 s without a break | from the video |

## Layout

| Path | What |
|---|---|
| `tubing_cutter.py` | the task file: description for agents, props and task, scoring |
| `tubecutter_sim/station.py` | the run-time extension of yam_arena (props and per-step task events) |
| `tubecutter_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (joints, friction rules, the groove rule) and the scoring |
| `tubecutter_sim/assets/set.usda`, `tubecutter_sim/usd/` | the props, and the script that generates them (`python -m tubecutter_sim.usd.cutter_props`) |
| `tubecutter_sim/demo.py`, `vision.py`, `policy.py` | the scripted demo (observation only), its camera and kinematics helpers, its policy `cutter-demo` |
| `tests/test_scorer.py` | scorer tests: each hazard and cap before and after a milestone, and the timed hold |

## References

- Klein Tools 50077 1-Inch Quick-Adjust Screw-Feed Tubing Cutter (17.4 x 7 x 3.6 cm, 299 g, fixed wheel, screw-feed
  knob; product photos): https://www.kleintools.com/catalog/plumbing-tube-cutters-benders/1-inch-quick-adjust-screw-feed-tubing-cutter
- Reed TC2Q Quick Release tubing cutter, operating instructions ("turn knob about 1/4 turn every revolution"; "avoid
  side pressure and feeding wheel into pipe too fast"): https://cdn.shopify.com/s/files/1/0818/4086/5591/files/TC2Q-manual.pdf
- RIDGID screw-feed tubing cutters 10/15/20 (catalogue page: capacities, wheels, weights):
  https://www.ridgid.com/us/en/screw-feed-tubing-cutters
- Copper Development Association, Copper Tube Handbook (1/2 in Type L: OD 0.625 in, wall 0.040 in, ASTM B88):
  https://www.copper.org/applications/plumbing/cth/
- BLS Occupational Outlook Handbook, plumbers, pipefitters and steamfitters (about 504,500 jobs in 2024):
  https://www.bls.gov/ooh/construction-and-extraction/plumbers-pipefitters-and-steamfitters.htm
- BLS Occupational Outlook Handbook, heating, air conditioning and refrigeration mechanics and installers (about
  425,200 jobs in 2024): https://www.bls.gov/ooh/installation-maintenance-and-repair/heating-air-conditioning-and-refrigeration-mechanics-and-installers.htm
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity
compensation).

**Tube.** The tube is fixed: a 1/2 in Type L copper stub (OD 15.88 mm, wall 1.02 mm, ASTM B88) standing 9.3 cm
above a plywood block with a floor flange. Its collider is a box, which the cutter never touches (see below). The
offcut above the mark is not separated in the simulator: "cut through" is the groove reaching the wall thickness all
the way round. In a real trial the offcut lifts off.

**Cutter.** The cutter is the Klein 50077 at its published size and mass (17.4 x 7 x 3.6 cm, 299 g, orange, black
knob). Klein publishes no drawing, so the part proportions are read off the vendor photos. Collision shapes are
boxes, plus an octagonal prism for the faceted knob.

**Rules instead of contact.** The cutter's hold on the tube, its orbit and its feed are rules, not contact:
- Its wheel and rollers are not simulated against the tube. The cutter turns about the tube on a revolute joint
  (`orbit`) and can slide along it on a prismatic joint (`slide`). The knob turns on a revolute joint (`feed`).
- Each joint has friction as a Coulomb drive: a stiff spring whose anchor slips once the friction limit is reached.
  - `orbit`: 0.02 N m, plus 0.10 N m per mm of bite.
  - `feed`: 0.03 N m, plus 0.25 N m per mm of bite. A 1/4-turn feed takes about 0.14 N m.
  - `slide`: 20 N while finger-snug (about 0.6 x a ~30 N clamp load, steel on copper), enough to carry the cutter's 2.9 N weight and slip under a hard push. It becomes
    300 N once the groove is 0.1 mm deep all round (the wheel tracks).
  - All these values are inferred and scaled to the YAM's wrist (10 N m/rad PD). Real knob torque at the end of a
    feed is a few tenths of a N m.

**Feed.** Feed = knob turns x 1.814 mm, the 7/16-14 feed screw of RIDGID's screw-feed cutters. Klein does not
publish the 50077's thread, so this pitch is inferred.

**The cut.** The tube's circumference is split into 72 sectors. Each sector the wheel passes over is cut to the
current feed. Bite is the feed minus the groove depth under the wheel.

**Dent.** The tube counts as dented if the bite ever exceeds 0.70 mm, about 3/8 knob turn ahead of the groove and
1.5 times the 1/4-turn guidance. The makers give the guidance, not a crushing limit, so 0.70 mm is our threshold.
It is meant to be calibrated on the real tube and cutter: feed 1/4, 3/8 and 1/2 turn on fresh tube and measure the
ovality of the end with calipers.

**Spiral.** A spiral is the wheel's axial position, over all cutting passes, spreading more than 1.0 mm. Off the
mark is the groove centre more than 1.5 mm from the mark. The cutter is held square to the tube by its joints: tilt
is not modelled, so a spiral can only come from sliding the cutter along the tube while it cuts.

**Not modelled.** Burr, wheel wear and the reamer are not modelled.

**Real cutting.** A real 1/2 in Type L tube needs about 3 to 6 revolutions at 1/4 turn. Here three feeds of 1/4
turn (1.35 mm) cut through: the frame's flex and the wheel's spring-back are not modelled.

**What is left out of the procedure.**
- The cutter starts already set on the mark and finger-snug. Opening it with the quick-adjust button and placing
  it are left out.
- So is the reaming of the end after the cut.
- The instruction's "away from the cutter" end replaces taking the cutter off.

**One layout.**

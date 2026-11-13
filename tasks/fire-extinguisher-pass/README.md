# Put out a small pan fire with a hand extinguisher from behind a tape line

A 2.5 lb stored-pressure extinguisher (Amerex B417 size) stands on the table. Far out, beyond a yellow tape line,
three flame tongues burn on a pan of fuel on a steel stand. The instruction gives only the goal and the constraints:
put the fire out, then stop spraying; stay behind the tape line with both arms, both grippers and the extinguisher; do
not drop the extinguisher. It does not say how. The robot has to know what the unit's label says (PASS): hold the
unit by its handle and pull the ring pin, breaking the tamper seal (the lever cannot be squeezed while the pin is in);
aim the nozzle at the base of the flames, not at the flames, from back at the line; squeeze the lever; sweep side to
side across the base until every tongue is out; then let go of the lever. Spray that meets the flames higher up does
nothing, and the unit holds only about 10 s of spray, so aiming at the flames empties it with the fire still burning
(score 0).

Robotic difficulty: two hands with different jobs on one free object. One hand holds the unit by its carry handle
and lever (and later squeezes), the other pulls a 4 mm pin out along its axis against the seal while the first
keeps the unit from sliding. Then one hand carries a 1 kg unit that hangs from its handle (it tips nose-down until
its centre of mass is under the grip, so where it is gripped sets the aim), holds the nozzle's line on a 4 cm band
about 25 cm away while squeezing a sprung lever, and sweeps it across the base.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`fire_extinguisher.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream`
policy. Its props are articulated (the extinguisher's lever and ring pin) and it has per-step rules (valve,
spray, knockdown), which the one-file format cannot declare, so the file uses `fire_sim` (this folder), which adds
them to the station at run time (`fire_sim/station.py` documents exactly what it patches; it requires
yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/fire-extinguisher-pass --no-deps     # registers the policy extinguisher-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/fire-extinguisher-pass/fire_extinguisher.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put out the fire in the pan with the fire extinguisher, then stop spraying. Stay behind the yellow tape line: no part of either arm, either gripper or the extinguisher may cross it. Do not drop the extinguisher." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/fire-dream
```

The scripted demo, `extinguisher-demo` (closed loop, robot-observable inputs only; see "What the demo reads"):

```bash
inspect-robots run --embodiment yam_arena --policy extinguisher-demo \
  -E task=tasks/fire-extinguisher-pass/fire_extinguisher.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/fire-demo
```

Control run (must fail): the same demo aiming at the flames' tips instead of their base, `-P aim=tips`.
Hazard checks: `-P test=reach_early|reach_late|drop_early|drop` (see "Hazard checks" below).

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json, or 0 after a
hazard). Videos: `inspect-robots video <log>.json` writes one MP4 per camera.

- **Episode end.** The episode ends as soon as the fire is out and the lever has stayed released for 1 s (score 1.0),
  otherwise at `--max-steps`.
- **Seeds.** One fixed layout (`-E object_jitter=0`); seeded jitter is not used. PhysX on the GPU is not
  bit-deterministic, which is why the demo is closed loop.
- **Scorer checks.** `python -m pytest -q fire_sim/test_scorer.py` (no simulator): every hazard, raised before
  and after each milestone, leaves the score at 0 for good; the release hold restarts when it is broken.

## What the demo reads

`extinguisher-demo` (`fire_sim/policy.py`, `demo.py`, `vision.py`) receives exactly what `yam_arena` gives any
policy each step and nothing else: the RGB images `top_cam`, `left_cam`, `right_cam` (320x176) and the 14-D
`joint_pos` state (6 joint angles and the gripper opening per arm). It does not import or query the simulator: no
object poses, no lever angle, no pin, seal, valve, spray or knockdown state, no scorer state. Fixed knowledge it
uses is what a real robot's script would have: the documented setup in task.json (where the unit and the fire stand
are, the unit's dimensions and mass layout), the station's published camera calibration (`yam_arena.station`) and
the arm's URDF (`yam_arena.urdf`, `yam_arena.ik`).

| Thing | How the demo gets it |
|---|---|
| Where the extinguisher stands | first top-camera frame: its red paint segmented by colour, the blob's centroid back-projected onto a plane at the cylinder's upper part; the offset from the documented spot (clipped to 3 cm) shifts every grasp target |
| The lever's "held", "carry" and "squeezed" openings | after the right hand closes across handle and lever (the pin still in), the gripper's own reading is the stack's height: relaxed = that reading less 0.01, carry = less 0.03 (lever about 4 deg, valve shut), squeeze = fully closed |
| Where the hand really is with the 1 kg unit | forward kinematics of the measured joints; an integral term adds the arm's sag (about 4.5 cm down, 14 deg at the wrist) to the command |
| How the hanging unit points | first guess from the documented mass layout (it hangs from the handle with its centre of mass under the grip: 26 deg nose-down, less a 6 deg correction from development runs because the squeezing hand holds it a little) and a fixed turn of the hand (12 deg, from development runs: the unit turns about that much in the hand as its base leaves the table) |
| Where the spray lands | the top camera, through the flames: each tongue's window (the documented flame volume projected with the calibration) is counted for bright orange pixels every 0.15-0.25 s while spraying. Base: the aim starts low (below the pan's rim) and walks up at 3 cm/s (for later tongues from just below the last base found) until the tongue being aimed at starts to shrink (the spray has reached the front edge of the fuel), then 1 cm further up. Sideways: if a neighbouring tongue shrinks instead, the hand is turned about the vertical by 70% of the angle between the two tongues |
| When a tongue is out | its window has gone dark; the demo stays on it 0.3 s more, then sweeps on to the next with the lever still squeezed, and comes back for any tongue left |
| Pin pull, carry, release | measured joint angles (the hand is "settled" when forward kinematics agrees with the command) and timing |

## Measured

RTX-class GPU (shared with other simulations), `-E object_jitter=0`, 20 Hz, 320x176 frames, official command:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `extinguisher-demo` (cameras + joint states only) | `task_score: 1` (pin pulled, seal broken; bases found by the scan at aim-height offsets -26, +10 and +36 mm for the right, middle and left tongue, each out after about 0.5 s of dwell; lever released and held 1 s; no hazard. A rerun of the same demo with the scorer state printed every step used 7.05 s of the 10 s of spray) | 691 (35 s) | 8.5 minutes |
| Control: same demo, `-P aim=tips` (aims 9 cm up the flames, sweeps the three tongues) | `task_score: 0` (all three tongues still burning when the 10 s of spray ran out: flame pixels in the top camera unchanged, 254/272/230 against 252/250/260 at the start. A rerun with the scorer state printed shows the spray landing 59-105 mm above the fuel, knockdown 0 on every tongue, score 0.5 until the unit emptied, then 0) | 1,800 (step limit) | 21 minutes |
| `dream` (`dreamlabs/dream-1-beta`), one full run, goal-only instruction | `task_score: 0` (both arms lift and hover at their sides of the table; the extinguisher is never touched) | 1,800 (step limit) | 21 minutes |

Dream-1 refused the first wording of the instruction ("... keep both arms, both grippers and the extinguisher outside
the yellow tape line ...") before acting: `DA-4000 BAD_REQUEST: unsafe instruction in this scene, refused: The fire
extinguisher is placed inside the yellow tape boundary line.` (from the top camera the unit, which stands 4 cm behind
the line, overlaps the line). The instruction was reworded to "Stay behind the yellow tape line: no part of either
arm, either gripper or the extinguisher may cross it", which it accepted; the run above is with that wording.

### Hazard checks

Every hazard was raised in the simulator before and after the first milestone (`-P test=...` modes of the demo policy, with the
scorer state printed; the scorer's own checks are in `fire_sim/test_scorer.py`, which raises each hazard before and
after every milestone and checks the score stays 0):

| Check | What it does | Result |
|---|---|---|
| `test=reach_early` | left hand reaches over the tape at the start | score 0 (`inside_keep_out_line`, first point a left finger pad at x 33.0 cm, step 14) |
| `test=reach_late` | the same after the pin is out (score 0.25) | 0.25 -> 0 (`inside_keep_out_line`, left fingertip at x 33.3 cm) |
| `test=drop_early` | left hand pushes the standing unit over | score 0 (`extinguisher_dropped`, tilt 90 deg) |
| `test=drop` | right hand opens and slides out from under the handle with the unit carried at the line, pin out (score 0.25) | 0.25 -> 0 (`extinguisher_dropped`: it falls, lands on its side, tilt 88 deg; its nozzle also ends over the line) |
| control (`aim=tips`) | spray empties with the fire burning (score had reached 0.5) | 0 (`emptied_fire_not_out`) |

Note the arms' ready pose puts the grippers about 1 cm behind the tape line: any forward reach crosses it.

## Layout

| Path | What |
|---|---|
| `fire_extinguisher.py` | the task file: description for agents, the props and task, scoring |
| `fire_sim/station.py` | the run-time extension of yam_arena (props, per-step task events) |
| `fire_sim/task.py`, `scorer.py`, `spec.py`, `geometry.py` | the rules (valve, spray line, knockdown, flames), hazards and milestones, layout and instruction, dimensions |
| `fire_sim/assets/*.usda`, `fire_sim/usd/props.py` | the props, and the script that generates them from the dimensions |
| `fire_sim/demo.py`, `fire_sim/vision.py`, `fire_sim/policy.py` | the scripted demo, its camera geometry and segmentation, and its Inspect Robots policy `extinguisher-demo` |
| `fire_sim/test_scorer.py` | scorer checks without the simulator |

## References

- Amerex B417T 2.5 lb ABC extinguisher, product page with specifications (15.5 in tall, 5.75 in wide, 3 in
  cylinder, about 10 s discharge, 9-15 ft range, large loop pull pin) and product photos (layout of nozzle, lever,
  handle, ring pin, gauge; label text):
  https://www.firesupplydepot.com/amerex-b417t-abc-fire-extinguisher.html
- Same unit, other distributors (discharge time 9-10 s, range 15 ft): https://www.zoro.com/i/G9981806/ ,
  https://gsmdepot.ca/products/amerex-b417
- VTT Research Notes 2002, measurement of the forces needed to operate portable extinguishers (EN 3-5 limits;
  measured squeeze-lever forces 45-98 N, seal release 20-85 N): https://sarjaweb.vtt.fi/pdf/tiedotteet/1999/T2002.pdf
- Water-sensitive spray paper (for the real-trial base-band check), e.g. TeeJet 20301-2N:
  https://www.teejet.com/
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX (240 Hz) in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity
compensation, 50 N gripper). What is geometry and what is a rule:

| Part | Geometry (contacts and joints) | Rule (this task's code, every control step) |
|---|---|---|
| Body, handle, lever | floating articulation; the lever is a revolute joint (0-20 deg) with a return-spring drive (6 N rest, 12 N full, at 6 cm) | none |
| Pin | a link on a prismatic joint along its axis; it passes through holes in the lever's cheeks, so the lever stops at about 3 deg while it is in (measured: 1.5 N m on the lever, twice the full spring, held it at 3.1 deg) | none |
| Tamper seal | the pin joint's stiff drive (4000 N/m) | released for good after 2 mm of pull (8 N; measured: held 8.0 N, let go by 8.2 N); a pin pulled clear of the far ear is then held at its 60 mm stop |
| Holding and hanging | the hand closes across handle and lever; the unit hangs from the handle and tips nose-down under its own weight | none |
| Valve | none | open while the lever is past 12 deg and spray time (10 s) is left |
| Spray | none (the plume cone is visual only) | the nozzle's axis line from the outlet; the first point where it meets the fire's volume (the pan's inside, fuel surface to 10.5 cm up) counts as a base hit if it is at most 4 cm above the fuel and within 80 cm, with the unit within 60 deg of upright |
| Fire | none (flames are emissive visual shapes) | three tongues, 1.0 s of base hits each to knock down; each sinks into the pan in proportion and flickers |

Simplifications, said plainly:
- The extinguisher is the Amerex B417's size and layout (height, cylinder, nozzle horn, lever over the carry handle,
  ring pin through the valve, gauge), simplified to boxes and cylinders; the lever and handle are straight bars and
  closer to parallel than on the real unit (the real carry handle curves down more steeply) so that the YAM's
  parallel jaw, 94 mm wide open, spans both.
- Forces are a trainer's, not a charged B417's: lever 6-12 N and seal 8 N, against about 50 N on the lever and
  20-100 N on the seal for real units (VTT measurements, EN 3-5 limits), which the station's arms cannot apply.
  Mass 1.05 kg (an empty B417 shell with 0.15 L of water) instead of about 2.1 kg charged.
- The pin stays on its axis after it is pulled (held 6 cm out); a real pin comes free and is dropped.
- The spray is a line, not a cone of droplets; its reach (80 cm) and the 4 cm base band are table-scale stand-ins
  for the label's "start back 8 feet, aim at base of fire". The discharge time (10 s) is the B417's.
- The fire is a flame target: tongues that shrink only when the base is hit; there is no heat, smoke, re-ignition
  or splashing. Discharge reaction force on the unit is not modelled.
- The keep-out line is tape at 20 cm from the stand's footprint (a real 8 ft standoff scaled to the table).

## Real-trial checking

The task is designed to run on the real station without a live fire:

- **Extinguisher (trainer).** A B417-size stored-pressure unit used as a water-mist trainer: an Amerex B417 emptied
  and recharged by an extinguisher service shop with 0.15 L of water and a low air charge, adjusted until the lever
  needs at most 12 N at 6 cm behind the axis (check with a spring scale; at the B417's 13 bar the lever needs about
  50 N, which the soft-wrist YAM cannot apply), sprayed for 10 s. Tamper seal: a cotton thread loop breaking at
  about 8 N (check with the spring scale). An off-the-shelf alternative with the same pin, seal and lever is a
  laser training extinguisher (e.g. LION BullsEye 2 kg/5 lb, with tamper seals), which is larger.
- **Fire (target).** A flame-shaped target board on the 24 cm stand, three tongues printed on it, with a 4 cm band
  at its foot marked as the base zone and covered with water-sensitive paper; the flame area above it covered
  separately. After a trial the strips show where the spray landed. A tongue counts as knocked down when its base
  band has 1.0 s of hits on the overhead video (the strip confirms the band was hit and the flames above were not
  the main target). A digital flame target with a base sensor (LION BullsEye-type) does the same electronically.
- **Keep-out.** The tape line; any arm, gripper or the unit over it in the overhead video is a 0.
- **Dropped.** From the video: the unit tips past 75 degrees or falls.
- **Emptied.** Count spray time from the video (the plume is visible); 10 s with any tongue not knocked down is a 0.
- **Lever released.** A protractor mark on the handle: the lever back above 12 degrees from closed for 1 s.

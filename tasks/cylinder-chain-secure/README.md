# Secure a free-standing gas cylinder to its wall bracket with the bracket's chain

A small aluminium oxygen cylinder (size MD, 3.3 kg, 111 mm across) stands loose in the left cradle of a Bristol Maid
CYB/DE2 steel wall bracket for two D/E medical cylinders, screwed to a plywood panel at the back of the table. The
bracket's chain hangs from its hole in the centre tongue (to the cylinder's right) down to the table, its end link
in a red PVC sleeve. Under the tongue on the cylinder's left hangs the bracket's J-hook, its tip turned up just in
front of the plate. The instruction gives only the goal and the constraints: secure the cylinder to the bracket
with its chain so it cannot tip over, keep it upright, do not touch the valve, finish with both grippers away. It
does not say how. The robot has to take the chain end round the cylinder's front just under the plate, pull it
tight, and drop the chain into the J-hook so that the link nearest the hook is the one held (a link further along
leaves the chain slack), while the cylinder, which tips at about 4 N at its shoulder, stays upright. The bracket
has no spring gate: the J-hook holds the chain only by its shape (a link standing on edge sits in the 10 mm gap
between the hook's stem and tip, and its neighbours, lying flat, are too wide to follow it through), so the
chain can be lifted off again. An off-camera check then pulls the cylinder's shoulder toward the robot with 10 N:
a snugly chained cylinder leans about 4 degrees (the long links straighten and the hooked link has play in
the bend; the pass limit is 6 degrees), a loose one falls.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`cylinder_chain.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy.
Its chain is an articulation (ring links on joints), which the one-file format cannot declare, so the file uses
`cylchain_sim` (this folder), which adds it to the station at run time (`cylchain_sim/station.py` documents
exactly what it patches; it requires yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/cylinder-chain-secure --no-deps       # registers the policy cylinder-chain-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/cylinder-chain-secure/cylinder_chain.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Secure the gas cylinder to the wall bracket with the bracket's chain so that it cannot tip over. Keep the cylinder upright the whole time, do not touch its valve, and finish with both grippers away from the cylinder and the chain." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/cylinder-chain-dream
```

The scripted demo, `cylinder-chain-demo` (closed loop, robot-observable inputs only; see "What the demo reads"):

```bash
inspect-robots run --embodiment yam_arena --policy cylinder-chain-demo \
  -E task=tasks/cylinder-chain-secure/cylinder_chain.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/cylinder-chain-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json, or 0 after a
hazard). Videos: `inspect-robots video <log>.json` writes one MP4 per camera.

- **Episode end.** Once the chain is in the J-hook and every finger pad has stayed at least 5 cm from the cylinder
  and the chain for 1 s without a break (the count restarts whenever a finger comes back closer), the
  after-release check (2 s of simulated time) runs inside that control step, off camera, and the episode ends; the
  stored frames stop at the release. Otherwise the run ends at `--max-steps`.
- **Control run.** `CYL_CHAIN_CONTROL=<seconds>` forces the after-release check at that time whatever the robot
  has done; `-P slack=<n>` makes the demo bring the end n links (28 mm each) less far before the right hand presses
  the chain down, so it is left slack and (as measured below) no link seats in the hook. The control must not score 1.
- **Seeds.** One fixed layout (`-E object_jitter=0`); seeded jitter is not used. PhysX on the GPU is not
  bit-deterministic, which is why the demo is closed loop.
- **Physics rate.** The task runs PhysX at 480 Hz (the policy still acts at 20 Hz): the 7 g chain links need it.

## Measured

All on the station with the official command above (seed 0, `-E object_jitter=0`, 320x176 cameras, 20 Hz control,
PhysX 480 Hz), one run each.

| Run | task_score | Steps | What happened |
|---|---|---|---|
| Demo (cameras and joint states only) | 1.0 | 785 (ended on success) | Link 10 seated in the J-hook, slack 4 mm, chain across the front at 266 mm (gap 17 mm), cradle gap 27 mm; after-release check tilt max 4.75 deg; no hazards. About 11 min wall time. |
| Control: `-P slack=2`, `CYL_CHAIN_CONTROL=75` | 0.25 | 1491 | The end is brought two links short, so the right hand presses the chain down outside the hook and no link is ever held in the bend: chain across the front (0.25) but not hooked. The forced check at 75 s pulls the cylinder over (90 deg). |
| Dream-1 (`--policy dream`, `-P record=false`) | 0.0 | 2400 (max steps) | The arms make small moves near their start poses and never reach or grasp the chain; the cylinder stays upright and nothing is touched, but the chain never crosses the front. Mean inference latency 0.64 s. |

The control shows a missed hook (too short a pull), not a hooked-but-slack chain: with the hand pressing outside
the hook the slack link never lands in the bend. The slack limit itself is checked by the scorer (`slack_mm` in the
report) and was 4 mm in the demo against the 30 mm limit.

## What the demo reads

`cylinder-chain-demo` (`cylchain_sim/policy.py`, `demo.py`, `vision.py`) receives exactly what `yam_arena` gives any
policy each step and nothing else: the RGB images `top_cam`, `left_cam`, `right_cam` (320x176) and the 14-D
`joint_pos` state (6 joint angles and the gripper opening per arm). It does not import or query the simulator: no
object poses, no chain joint angles, no scorer state. Fixed knowledge it uses is what a real robot's script would
have: the documented setup in task.json (where the bracket, its chain hole, its J-hook and the cylinder are, part
sizes, the red sleeve on the chain's end), the station's published camera calibration (`yam_arena.station`) and the
arm's URDF (`yam_arena.urdf`, `yam_arena.ik`).

| Thing | How the demo gets it |
|---|---|
| Where the chain's end lies | first top-camera frame: the red sleeve segmented by colour and back-projected onto its known height on the table |
| The end's position and heading | left wrist camera from above (three frames averaged): the sleeve's pixels back-projected onto its height; their long axis gives the heading the fingers close across |
| Holding the chain | a pinch across the sleeve; the chain is flexible, so after the pinch only the hand's position matters |
| Pulling it tight | the hand is driven round the cylinder's front at the upper-third height to a fixed point beyond the hook (from the documented hook position and chain length); the tight chain may stop it short of that point, which the demo logs from forward kinematics of the measured joints but does not need |
| The hook and the cylinder | documented positions (the bracket is screwed to the wall) |
| Getting a link into the hook | while the left hand holds the chain taut beyond the hook, the right gripper (closed) comes down slowly onto the chain just outside the hook's tip at the documented hook position and holds 1 s, pushing a link down into the bend; the left hand then eases toward the hook and lets go |
| Releasing, settling | commanded motions, measured joint angles (an arm is "settled" when forward kinematics agrees with the command), timing; both arms back off and wait |

A small joint-space integral trim (measured joint angles only) removes the station arms' steady error under their
soft gains.

## Layout

| Path | What |
|---|---|
| `cylinder_chain.py` | the task file: description for agents, the props and task, scoring, control mode |
| `cylchain_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera check) |
| `cylchain_sim/task.py`, `scorer.py`, `spec.py` | the measurements, hazards and milestones, the after-release check; layout, part sizes and instruction |
| `cylchain_sim/assets/*.usda`, `cylchain_sim/usd/props.py`, `usd/bm_props.py` | the props, and the scripts that generate them from the dimensions in `spec.py` |
| `cylchain_sim/demo.py`, `cylchain_sim/vision.py`, `cylchain_sim/policy.py` | the scripted demo, its camera geometry and segmentation, and its Inspect Robots policy `cylinder-chain-demo` |

## Checking a real trial

- **Hazards.** Tilt: a phone inclinometer taped to the cylinder's shoulder (or a photo from the side against a plumb
  line): more than 10 degrees at any time scores 0. Valve: video; a finger or the chain touching the valve or its
  toggle scores 0.
- **Chain across the front.** Photo from above: the chain lies against the cylinder's front (within 3 cm).
- **Height.** Tape measure from the table to the chain where it crosses the middle of the cylinder's front:
  25-36 cm (the upper third of the 37 cm straight wall, below the shoulder).
- **Hooked.** A link of the chain sits down in the J-hook's bend, between its stem and its upturned tip.
- **Snug and seated.** Cylinder against the cradle's back (a 6 mm feeler does not fit) and upright within 3 degrees;
  count the links from the chain hole to the one in the hook: with the chain pulled tight round this cylinder the
  eleventh link (290 mm along) just reaches the hook; the hooked link may be at most 30 mm further along (that one).
- **After-release check.** Hands off for 1 s, then hook a spring scale on the valve neck just under the valve and pull
  horizontally toward the robot to 10 N for 2 s: the inclinometer must stay under 6 degrees.

## References

- Bristol Maid cylinder wall brackets, product data sheet (CYB/DE2 for two D/E cylinders, 359 x 107 x 88 mm, one-piece
  construction, "drop over chains to secure the cylinder", "should be positioned on the wall at approx half cylinder
  height"): https://bristolmaid.com/default/cylinder-wall-brackets.html
- FEMA, "Anchor and Brace Propane Tanks and Gas Cylinders" (2008), chains round a cylinder to wall anchors:
  https://www.fema.gov/sites/default/files/2020-07/how2009_anchor_tanks_cylinders.pdf
- Catalina Cylinders, aluminium medical oxygen cylinders (MD: 4.38 in diameter):
  https://www.catalinacylinders.com/markets/medical/
- OSHA 29 CFR 1910.253(b)(2)(ii) and 1926.350(a)(9) (cylinders where they cannot be knocked over; secured upright):
  https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.253 ,
  https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.350
- BLS Occupational Outlook Handbook (2024 jobs): welders, cutters, solderers and brazers (about 457,300),
  https://www.bls.gov/ooh/production/welders-cutters-solderers-and-brazers.htm ; plumbers, pipefitters and
  steamfitters (about 504,500), https://www.bls.gov/ooh/construction-and-extraction/plumbers-pipefitters-and-steamfitters.htm ;
  HVAC mechanics and installers (about 440,900 in 2025), https://www.bls.gov/ooh/installation-maintenance-and-repair/heating-air-conditioning-and-refrigeration-mechanics-and-installers.htm
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 480 Hz in Isaac Lab-Arena (the policy acts at 20 Hz), both arms driven as in Dream's station (joint PD with
i2rt's gains and gravity compensation).

- **Cylinder.** A free rigid body: PhysX's cylinder shape for the straight wall (111 mm, 0.37 m), a convex hull for the
  shoulder and neck, boxes for the post valve and toggle; 3.3 kg (MD size: about 2.5 kg empty, valve, 15 cu ft of
  oxygen), centre of gravity 0.20 m up (assumed), aluminium on the table 0.35/0.30. It tips at about 4 N pulled at its
  shoulder (m g r / h).
- **Bracket.** Static colliders sized to the CYB/DE2's overall 359 x 107 x 88 mm: a 3 mm plate with two U cradles 117 mm
  wide (their semicircular backs as slices, 2 mm bumper strips) and the tongues between them, folded down into the
  wall flange. The data sheet gives only the overall size; the cradle width, the hole and hook positions and the
  hook's size (6 mm round bar, stem 50 mm below the plate, tip 22 mm high, 16 mm stem to tip) are read from its
  photos. The hook's bend is drawn opening forward so its tip stands clear in front of the plate. Mounted with the
  plate 0.33 m up: half the height of an E cylinder (0.66 m), as the data sheet says for its D/E range, which puts
  the chain in an MD's upper third. The empty bay's chain is drawn but not simulated.
- **Chain.** 13 rigid long links (28 x 11 mm inside, 4 mm wire, 7 g; inferred from the photos) on revolute joints whose
  axes alternate by 90 degrees, so the chain bends any way over two links but does not twist. Each link collides as a
  ring of four wires, so the hook can pass through a link; neighbouring links (their end bars overlap: the interlock)
  do not collide with each other, all others do. The fixed end is attached at the chain hole. The red PVC sleeve on the
  end link is our addition (so the end can be found at camera resolution and held).
- **What is simplified.** The bracket has no gate or latch, as on the real one. "Snug" is the link count: the hooked
  link's distance along the chain from the chain hole minus 290 mm, the length the chain takes when pulled tight round
  this cylinder under the plate (measured in the simulator by pulling the end tight). The after-release check is a
  force applied in the simulator at 0.40 m after the recording ends; a real trial uses a spring scale.

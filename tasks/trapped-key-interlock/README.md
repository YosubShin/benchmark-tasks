# Carry a trapped interlock key from a breaker to the matching door

Two trapped-key interlock pairs stand on the table. Each pair is a breaker whose Kirk Type F interlock holds a
trapped key, and a fuse-compartment door bolted shut by a second Type F with the same code. The pairs are coded
differently, so a key enters only the locks of its own code. The instruction gives only the goal and the
constraints: door A1 unlocked with key A1, breaker A1 left locked off, only the A1 locks used, no breaker handle
moved, no locked door's handle touched, grippers away from the key at the end. It does not say how; the robot has
to work out that key A1 is trapped in breaker A1's interlock and comes free only once it is turned (which locks the
breaker open), then carry it to door A1's interlock, insert it and turn it. Taking a key to the other pair's locks,
turning the B1 interlocks, trying a bolted door or moving a breaker handle toward ON scores 0. Scoring checks the
interlock physically: key A1 must be trapped in door A1's lock, breaker A1's handle must be held at OFF by its bolt,
and door A1 must open.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file (`trapped_key.py`)
in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy. Its props are
articulated (each lock's plug and bolt, the breaker handles, the doors), which the one-file format cannot declare,
so the file uses `interlock_sim` (this folder), which adds them to the station at run time
(`interlock_sim/station.py` documents exactly what it patches; it requires yam-isaac-lab-arena at commit `0059d02`,
the Brev launchable's pin).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/trapped-key-interlock --no-deps       # registers the policy interlock-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/trapped-key-interlock/trapped_key.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Unlock fuse compartment door A1 with key A1 so the door can be opened, and leave breaker A1 locked in the off position. Use only the locks marked A1, do not move either breaker's handle, do not touch a door's handle while that door is locked, and finish with both grippers away from the key." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/trapped-key-dream
```

The scripted demo, `interlock-demo` (closed loop, robot-observable inputs only; see "What the demo reads" below):

```bash
inspect-robots run --embodiment yam_arena --policy interlock-demo \
  -E task=tasks/trapped-key-interlock/trapped_key.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/trapped-key-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json, or 0 after a
hazard). Videos: `inspect-robots video <log>.json` writes one MP4 per camera.

- **Episode end.** Once key A1 is turned in door A1's lock and both grippers have stayed at least 6 cm from its bow
  for 1 s without a break (the 1 s count restarts whenever a gripper comes back within 6 cm), the after-release check (4.5 s of simulated time) runs inside that control step, off camera, and the
  episode ends; the stored frames stop at the release. Otherwise the run ends at `--max-steps`.
- **Control run.** `TRAPPED_KEY_CONTROL=1` with `--policy yam_scripted -P hold_s=12` forces the check at 3 s with
  key A1 still in breaker A1's lock: it must score 0.
- **Wrong-key check.** `TRAPPED_KEY_CONTROL=wrong_key` (same command) sets key A1 over door B1's keyway and key B1
  over door A1's, pushes each down with 5 N and turns it with 0.3 N m, and prints how far each went in and how far
  each plug turned.
- **Seeds.** One fixed layout (`-E object_jitter=0`); seeded jitter is not used. PhysX on the GPU is not
  bit-deterministic, which is why the demo is closed loop.
- **Runtime per trial** (RTX PRO 6000, warm shader cache): see "Measured" below; Isaac Sim's first start on a new
  machine adds several minutes.

## What the demo reads

`interlock-demo` (`interlock_sim/policy.py`, `demo.py`, `vision.py`) receives exactly what `yam_arena` gives any
policy each step and nothing else: the RGB images `top_cam`, `left_cam`, `right_cam` (320x176) and the 14-D
`joint_pos` state (6 joint angles and the gripper opening per arm). It does not import or query the simulator: no
object poses, no prop joint angles, no bolt or latch state, no scorer state. Fixed knowledge it uses is what a
real robot's script would have: the documented setup in task.json (where each part is, part sizes, label colours),
the station's published camera calibration (`yam_arena.station`) and the arm's URDF (`yam_arena.urdf`, `yam_arena.ik`).

| Thing | How the demo gets it |
|---|---|
| Which pair is A1, roughly where | first top-camera frame: the yellow A1 labels (door enclosure top, breaker face) segmented by colour and back-projected onto their documented heights; the door's offset from the layout (clipped to 3 cm) shifts its approach |
| Breaker A1's interlock (where key A1 sits) | left wrist camera, from a look pose over the breaker: the lock's red nameplate segmented, its outline back-projected onto the plate's plane (camera pose from forward kinematics of the measured joints), a rectangle of the documented size and orientation fitted; the keyway is the documented offset from the plate's centre (3 frames averaged). Used to aim the pinch, and again after the key is pulled out |
| Key A1 once held | pinched while still in the lock, so its tip is on that keyway axis at the documented seated depth; from then on the tip is predicted from forward kinematics of the measured joints plus that grasp offset (the parallel pinch keeps the bow on the pads, so the key turns with the wrist) |
| Door A1's interlock | left wrist camera, same nameplate fit, taken with the hand in the same pose relative to this plate as at the breaker after the pull-out (both locks are the same model, turned 90 degrees to each other, and the hand is turned with it). The fit's small bias is then the same in both lock frames and cancels between the grasp offset and the door target |
| Insertion | the predicted tip is servoed onto the door keyway estimate (6 mm above the face, then 6 mm into the face's hole), then lowered; whether the key went home is read from how far the hand got down (a seated key sits 15.7 mm below the face, a blocked one about 8 mm); if blocked, the demo lifts 2 mm and tries points of a 0.3 mm spiral around the estimate |
| Turning, settling, release | commanded wrist roll, measured joint angles (the hand is "settled" when forward kinematics agrees with the command), timing |

In the run below the camera put breaker A1's keyway 0.3 mm and 0.6 mm (x, y) from its documented position and door
A1's 0.6 and 0.3 mm; the same error in both lock frames, so the key seated on the first try.
The right arm holds its start pose.

## Measured

RTX-class GPU, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz, 320x176 frames:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `interlock-demo` (cameras + joint states only) | `task_score: 1` (breaker A1 bolt out 18.8 mm; key A1 out, seated 15.7 mm deep in door A1's lock on the first try and turned; door bolt fully back; after-release check: key rose 1.0 mm under 5 N, handle stopped at -12.4 degrees, door swung open; no hazard) | 1,180 (59 s), then the 4.5 s check off camera | about 11 minutes (GPU shared with two other simulations) |
| `dream` (`dreamlabs/dream-1-beta`), one run with the goal-only instruction (not refused) | `task_score: 0` (both arms stay near their bases with the wrists raised toward the doors, the right arm hanging over breaker B1's interlock; key A1 is never touched) | 1,800 (step limit) | about 16 minutes |
| Control: `TRAPPED_KEY_CONTROL=1`, `yam_scripted`, key A1 left in breaker A1 (re-run with the new instruction) | `task_score: 0` (key held, rose 1.1 mm; handle went to ON, +20 degrees; door A1 stayed bolted, moved 0.6 degrees) | 51, then the check | about 2 minutes |
| Wrong-key check: `TRAPPED_KEY_CONTROL=wrong_key` | key A1 on door B1's lock and key B1 on door A1's lock each stop 8.0 mm below the face under 5 N (a matching key seats 15.7 mm deep); both plugs stay at 90 degrees | 40, then the check | about 2.5 minutes |


## Layout

| Path | What |
|---|---|
| `trapped_key.py` | the task file: description for agents, the props and task, scoring, control modes |
| `interlock_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera check) |
| `interlock_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (handle detents, function test), hazards and milestones, layout and instruction |
| `interlock_sim/assets/*.usda`, `interlock_sim/usd/props.py` | the props, and the script that generates them from the dimensions in task.json |
| `interlock_sim/demo.py`, `interlock_sim/vision.py`, `interlock_sim/policy.py` | the scripted demo, its camera geometry and segmentation, and its Inspect Robots policy `interlock-demo` |

## References

- Kirk Key Interlock Type F datasheet (dimensions of the SD single-cylinder lock, bolt and throw):
  https://www.sentricsafetygroup.com/us/wp-content/uploads/sites/2/2026/05/f-kirk01-2602.pdf
- Kirk white paper, "Key Interlock Operation" (key removable only with the bolt extended):
  https://sentricsafetygroup.com/fr/wp-content/uploads/sites/3/2019/12/KirkWP-KeyInterlockOperation-1.pdf
- Kirk white paper, "Examples of Key Interlock Systems" (breaker to compartment door transfer):
  https://sentricsafetygroup.com/fr/wp-content/uploads/sites/3/2019/12/KirkWP-ExamplesofKeyInterlockSystems.pdf
- Square D PowerPact H-frame molded case circuit breakers (frame size):
  https://www.se.com/us/en/product-range/7564-powerpact-h-frame-molded-case-circuit-breakers/
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation). Rigid bodies on joints; geometry does the interlocking: the key's shank drives a plug whose crank pin moves the bolt (a scotch-yoke stand-in for Kirk's unpublished cam); the key's bit can pass the face plate only at the bolt-extended angle, so the key is trapped until the bolt is out; an extended bolt jams the breaker handle short of ON; and a door bolt through the keeper holds the door shut. The coding is warded, not pin-tumbler: each key's shank cross-section fits only the plug socket of its own code, so a wrong key stops on the plug's top and cannot turn it, whereas a real wrong-code Kirk key enters and will not turn. The keys are stand-ins, not flat 7-pin Kirk blades. The breakers lie face up and the door interlocks sit on the table with their cylinders up; real breaker interlocks are usually on a vertical panel with the key horizontal, so this is a bench mock-up chosen so the key can be turned with fingers pointing down. The breakers and doors are generic boxes and plates, not specific products; the coloured labels are our addition so the pairs can be told apart at camera resolution. Turning resistance is a small damping term (the real torque is not published, likely 0.1-0.5 N m), the handle detent and all stops are joint rules, and the after-release check is applied as forces and torques in the simulator after the recording ends.

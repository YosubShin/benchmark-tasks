# Swap the blown fuse in a named DIN-rail fuse holder for the right rating

Three Eaton Bussmann CHM1DU flip-open fuse holders (F1, F2, F3) stand on a DIN rail, each with a legend plate
naming its circuit and fuse rating. F2 holds a blown fuse marked with a black X. The robot must open F2's carrier
(it hinges at the bottom and swings forward and down), take the blown fuse out and drop it in a tray, take the
spare of the rating on F2's legend out of its opened 10-pack carton (6 A, not the 10 A in the carton beside it),
lay it in, and push the carrier shut until it latches,
without opening F1 or F3. Scoring checks the result physically: after release, the carrier must hold shut against
a 4 N pull on its grip tab, with the 6 A spare in it and the blown fuse in the tray. Opening F1 or F3, or seating the 10 A spare in F2, scores 0
for the whole trial, whatever was achieved before or after.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`fuse_holder_swap.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy.
Its props are articulated (each holder's hinged carrier, with a latch and an open detent), which the one-file
format cannot declare, so the file uses `fuse_sim` (this folder), which adds them to the station at run time
(`fuse_sim/station.py` documents exactly what it patches; it requires yam-isaac-lab-arena at commit `0059d02`, the
Brev launchable's pin).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/fuse-holder-swap --no-deps            # registers the policy fuse-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/fuse-holder-swap/fuse_holder_swap.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The panel is de-energized. Open fuse holder F2 (CONTROL XFMR), take out its blown fuse (marked with a black X) and drop it in the blue parts tray, put in the spare fuse whose rating matches the F2 legend (legends and spare labels are colour-coded by rating), and close the holder. Do not open F1 or F3." \
  --scorer task_score --max-steps 3200 --store-frames --log-dir ~/yam-eval/fuse-dream
```

The scripted demo, `fuse-demo`, is closed loop on robot-observable inputs only (see "What the demo reads" below).
It sends the same 14-D joint action a policy does, through yam_arena's IK:

```bash
inspect-robots run --embodiment yam_arena --policy fuse-demo \
  -E task=tasks/fuse-holder-swap/fuse_holder_swap.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 3200 --store-frames --log-dir ~/yam-eval/fuse-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json, and 0 from the
first hazard on; see "Scoring" below). Videos:
`inspect-robots video <log>.json` writes one MP4 per camera; `demo.mp4` and `dream1.mp4` here put the top camera
(2x) above the left and right wrist cameras.

- **Episode end.** Once F2 is shut on the 6 A spare and both grippers have stayed at least 6 cm from its grip tab
  for 1 s, the hold test (4 N pull at the tab, 0.11 N m about the hinge, for 1.5 s) runs inside that control step,
  off camera, and the episode ends; the stored frames stop at the release. Otherwise the run ends at `--max-steps`.
- **Scoring.** `fuse_sim/scorer.py` (`SwapScorer`) is the scoring implementation; the task file's `score()` returns
  its `report()["score"]` unchanged. Milestones (0.25 / 0.5 / 0.75 / 1.0) are counted as progress; the two hazards
  (F1 or F3 opened 5 degrees or more, the 10 A spare seated in F2) are latched the step they first occur, and from
  then on the score is 0 for the rest of the trial, whatever milestones were reached before or are reached after.
  The report also gives `progress_milestone` (not scored once a hazard occurred), `hazard_events`, `hazard_step`
  and, unscored, `other_events` (a decoy tab touched, a fuse dropped). `python -m fuse_sim.check_scorer` checks the
  gate without the simulator; the simulator checks are under "Hazard checks".
- **Control run (the latch matters).** Two short runs with the arms idle at ready start from the swap already done
  (the 6 A spare seated in the shut carrier, the marked fuse in the tray; `FUSE_HOLD_CHECK=1`), so the hold test
  runs after 1 s. With the latch (as in the task) the carrier holds the 4 N pull: `task_score: 1`. With F2's latch
  spring off (`FUSE_HOLD_CHECK=1 FUSE_CONTROL=1`, a carrier whose snap is broken) the pull opens it:
  `task_score: 0.75`. Command: `FUSE_HOLD_CHECK=1 FUSE_CONTROL=1 inspect-robots run ... --policy fuse-demo
  --max-steps 200` (the same `-E` options).
- **Seeds.** One fixed layout (`-E object_jitter=0`). Seeded jitter is not used.
- **Runtime per trial.** See "Measured" below. Isaac Sim's first start on a new machine adds several minutes.

## What the demo reads

`fuse-demo` (`fuse_sim/policy.py`, `demo.py`, `vision.py`, `layout.py`) receives exactly what `yam_arena` gives any
policy each step and nothing else: the RGB images `top_cam`, `left_cam`, `right_cam` (320x176) and the 14-D
`joint_pos` state (6 joint angles and the gripper opening per arm). It does not read the simulator: no carrier
angles, no fuse poses, no record of which spare matches, no scorer state. Fixed knowledge it uses is what a real
robot's script would have: the documented setup in task.json (`fuse_sim/layout.py`: the holders, their hinge, grip
tab and fuse seat, the legends, the spare cartons and their printed lids, the tray), the station's published camera calibration
(`yam_arena.station`) and the arm's URDF (`yam_arena.urdf`, `yam_arena.ik`).

| Thing | How the demo gets it |
|---|---|
| Which spare to fit | first top-camera frame: the colour of F2's legend's rating field and of the rating fields printed inside the two cartons' open lids, each classified as the nearest of the three rating colours (pink 2 A, green 6 A, red 10 A); the spare in the carton whose lid matches F2's legend is fitted (the printed ratings are too small to read at 320x176) |
| Where that spare lies | the documented middle slot of its carton, against the far wall (the carton's walls hold it there); the top camera checks its tube is there. Arm B reaches in through the front wall's finger cut-out with its fingers opened to 21 mm |
| F2's carrier while arm A pulls its tab | the pinch point's angle about the documented hinge, from forward kinematics of arm A's measured joints; A pulls 3 mm further whenever that angle stops growing, and lets go past the latch |
| F2's carrier open, F1 and F3 shut | top camera, every second after A lets go: the red grip tabs (2-3 px) are looked for where shut and where open carriers would put them; once F2's is seen open twice running, the carrier is resting on its documented 72-degree stop |
| The blown fuse | in the open carrier's documented seat; lifted 12 mm toward the clip's open side, then drawn out toward the robot and up. Whether it is still in the fingers is read from arm B's gripper opening (closed on a fuse it stays open by the fuse's width; closed on nothing it reads 0); if it slipped out, the demo goes back for it (at most twice) |
| The spare in the hand | where arm B's fingers closed on it (forward kinematics) relative to its documented place in the carton; from then on its pose follows arm B's joint readings, and the hand is servoed until the predicted fuse is lined up with the open carrier's seat before it is slid in and lowered |
| Closing | arm A's closed fingers push the tab's underside up its arc, a few degrees ahead of where their own angle about the hinge (arm A's joints) puts the carrier (it reads low until the fingers meet the tab), pushing 2 mm harder whenever that angle stops changing; done when it reads shut, or when the fingers have stopped at the end of their arc with the hardest push; the top camera then checks F2's tab is back where a shut carrier's is (and closes again once if not) |
| Release, dropping the blown fuse | the documented tray position (carried back toward the robot first, clear of the cartons' open lids), commanded moves, measured joints and timing |

In the run below the top camera read F2's legend field as green and the carton lids as red (KTK-10) and green
(KTK-6), so the KTK-6 was fitted; it saw the KTK-6's tube in its carton within 7 mm of its documented place, F2's
tab open after the pull and F1's and F3's shut throughout; arm B's gripper read 0.051 closed on the blown fuse and
0.044 after drawing it out (still held); reaching into the carton, the hand stopped 5.5 mm short of its target
(the fingertips on the carton floor), and the spare sat at (-3.2, 0.2, -18.5) mm in arm B's hand frame (with the
old open stand it sat at (-9.1, 0.4, -17.2)); lined up from arm B's joints it was seated 2 mm above its seat as
let go.
In development runs the blown fuse twice jumped out of the fingers when it left the clip during a straight lift with
the arm's lag built up; the draw toward the robot avoids that, and the gripper check catches it if it happens.

## Measured

RTX-class GPU, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz, 320x176:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `fuse-demo` (cameras + joint states only) | `task_score: 1`. The tab pull opened F2 at 18 mm of extra pull; the blown fuse came out in the fingers on the first draw and was carried back toward the robot and across to the tray, clear of the cartons' open lids; arm B reached into the KTK-6 carton (matched by colour) through its finger cut-out and took the spare out by its near ferrule; it was lined up from arm B's joints and seated; the closing push swung the carrier shut into the latch; it held the 4 N pull. No other holder touched; the KTK-10 stayed in its carton. | 1,964 (98 s), then the 1.5 s hold test off camera | about 30 minutes (GPU shared with up to four other simulations) |
| Hold check, latch on: `FUSE_HOLD_CHECK=1`, `fuse-demo` (arms idle) | `task_score: 1` (held; re-run with the cartons) | 10, then the hold test | under 1 minute |
| Control, latch off: `FUSE_HOLD_CHECK=1 FUSE_CONTROL=1` | `task_score: 0.75` (the 4 N pull opens the carrier; re-run with the cartons) | 10, then the hold test | under 1 minute |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (both arms move about near their start; the right hand drifts toward the spares stand and tray but nothing is touched, and F2 is never opened) | 3,200 (step limit) | about 38 minutes (shared GPU; Dream reported an observation as expired and was asked again 65 times) |

The Dream-1 run was made with the earlier spares stand (two open cradles with rating labels on the table in front);
the instruction, cameras, holders and scoring are unchanged, and it never touched the spares, so it was not re-run.

The demo's wall time is mostly its Python IK every control step and the shared GPU. The demo is sensitive: PhysX on
the GPU is not deterministic, and lifting the blown fuse out of the open carrier is its weakest step (with the
latch off, which changes how the carrier opens, the same demo code let the fuse slip from the fingertips there
twice). The recorded run is the one listed above.

## Hazard checks

Each check is the official command above with `--policy fuse-demo`, seed 0 (`-E object_jitter=0`), and one
environment variable, `FUSE_HAZARD_TEST=<case>` (empty in every scored run). The task-side cases (`fuse_sim/task.py`)
put 0.4 N m of opening torque on a decoy carrier's hinge for 1 s; `wrong_spare` (`fuse_sim/demo.py`) makes the demo
fit the 10 A spare instead of the 6 A. Otherwise the demo runs unchanged, so later milestones are still reached
physically and only the reported score shows the gate. All three were re-run with the cartons.

| Case | What happens | Progress reached | Reported `task_score` |
|---|---|---|---|
| none (the demo) | | 1.0 (1,964 steps) | 1 |
| `decoy_open_start` | F1 opened by the torque at 1 s, before any milestone; it swings to its 72-degree stop | the demo ran on (F2 opened, blown fuse to the tray, 6 A taken from its carton and seated, F2 shut) | 0 |
| `decoy_open_75` | F3 opened by the torque the step the 6 A spare is seated (after 0.75) | the demo ran on (F2 shut) | 0 |
| `wrong_spare` | the demo opens F2, removes the blown fuse (0.5), takes the 10 A spare out of the other carton (through its cut-out, the same way) and seats it, then shuts F2 | 0.5 (run cut at 2,200 steps) | 0 |

`python -m fuse_sim.check_scorer` (no simulator) checks both hazards before any milestone and after 0.25, 0.5 and
0.75: the final score is 0 in every case, with every later milestone reached. (A wrong fuse can only be seated once
the carrier is open and empty, so in the simulator it is checked after 0.5.)

## Layout

| Path | What |
|---|---|
| `fuse_holder_swap.py` | the task file: description for agents, the props and task, scoring |
| `fuse_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera hold test) |
| `fuse_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (latch and open detent, fuse seat, hold test), milestones and zero gate, layout and instruction |
| `fuse_sim/check_scorer.py` | the scorer's zero-gate check without the simulator |
| `fuse_sim/assets/*.usda`, `fuse_sim/assets/textures/` | the props (legends and carton print as geometry) and the fuse and holder prints |
| `fuse_sim/usd/fuse_props.py` | generates the props from the dimensions in task.json (`python -m fuse_sim.usd.fuse_props`) |
| `fuse_sim/demo.py`, `fuse_sim/policy.py` | the scripted demo and its Inspect Robots policy `fuse-demo` |

## References

- Eaton Bussmann CHM1DU one-pole modular holder for 10 x 38 mm fuses (distributor page with photos and spec sheet):
  https://www.standardelectricsupply.com/Bussmann-CHM1DU-Fuse-Holder
- The same holder (photos of the hinge boss and side label):
  https://www.yesss.co.uk/modular-midget-fuse-holder-without-indicator-1-pole-10mm-x-38mm-chm-series
- Eaton Bussmann Technical Data 720147, "10 x 38 mm Modular fuse holders" (CHM series; dimensions, and the side
  view with the carrier drawn open, from which the hinge and 72-degree opening were measured).
- Eaton Bussmann KTK fast-acting 10 x 38 mm midget fuses (KTK-6, KTK-10): Eaton's KTK product data; sold 10 to a
  carton (distributor listings, e.g. https://mm.mmem.com.au/clarkson/busktk-6 , "Std Pack Size" 10). No carton
  dimensions are published (searched Eaton, Littelfuse and distributor pages), so the carton is inferred.
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation, gripper a 5000 N/m spring capped at 50 N). Published dimensions: the CHM1DU's width, height, depth and grip tab; the 10 x 38 mm fuse. Measured on Bussmann's drawing: the carrier's hinge position and its 72-degree opening. Holder spacing: the holders sit one module apart instead of ganged at 17.5 mm because the robot's fingers are 15 mm wide at the tips and 35-40 mm wide 1-2 cm back; ganged, every grasp of the open fuse or the tab would hit a neighbour's grip tab, which stands 9 mm proud just beside it. A blank module or terminal between holders is common on real rails, but it is a layout choice made for the robot. How the fuse comes out: Bussmann does not publish the carrier's internals. Here the inner ferrule sits in a U-shaped clip open toward the holder body (upward when the carrier is open), with a bottom stop and a saddle under the outer ferrule; the fuse is lifted out of the clip's open side and drawn toward the robot, and the new one is laid in from above, across its axis. On a real CHM the fuse may instead be drawn out along its axis from spring clips; no spring grip on the ferrules is modelled (gravity and the clip shape hold it). Latch: a rule, not moulded geometry. Within 6 degrees of shut a joint spring pulls the carrier shut with 0.14 N m (5 N pulling straight out at the grip tab) rising to 0.20 N m (7 N) at 6 degrees; within 8 degrees of the open stop the same spring holds it open (an open detent, our assumption, so the open carrier stays still while fuses go in and out). The real snap force is not published (to measure on a bought part); the value was set as high as the station robot could still open by pinching the tab and pulling. The after-release check pulls with 4 N, 0.8 of the latch's hold. Gravity swings an unlatched carrier open because its centre of mass is set toward the grip tab (our assumption). Blown-fuse marking: KTK fuses have no indicator, so the blown fuse is marked the way an electrician marks a fuse found open, with a black marker X over its printed tube. Legibility: at the 320 x 176 top camera the legend and fuse text is 2-3 pixels tall and cannot be read, so the legends and the rating printed inside the spare cartons' lids carry a rating colour code (2 A pink, 6 A green, 10 A red, the IEC 60269-3 D-type fuse colours, used here as a plant labelling scheme); the colours are visible from the top camera. Collision shapes are boxes and 16-sided prisms; the holder body is boxes, the carrier a plate, tab and clip, with Bussmann's label and the fuse print as textures; the legends and the cartons' print are geometry (thin
coloured plates and lettering), because the renderer was seen to show textured labels blank in another task.
Spare cartons: KTK fuses are sold 10 to a carton; the carton's size is not published, so it is inferred from the
fuse: two layers of five side by side, 0.6 mm board, 56 x 42 x 23 mm outside, a finger cut-out (44 mm wide, down
to the floor) in the middle of the front wall, as many parts cartons have, and the lid hinged at the far top edge,
standing open. One fuse is left in each, in the middle slot of the bottom layer, so the fingers can reach either
side of its near ferrule (the robot's fingers need about 9 mm beside a 10.3 mm fuse; nine neighbours packed
against it would leave none, and a person would tip the carton or pick with a fingernail). The carton walls,
floor and the bench-top box are rigid colliders; the lid is visual. No electrical function. The pull test is a torque applied in the simulator after the recording ends. One layout.

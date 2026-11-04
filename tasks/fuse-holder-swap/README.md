# Swap the blown fuse in a named DIN-rail fuse holder for the right rating

Three Eaton Bussmann CHM1DU flip-open fuse holders (F1, F2, F3) stand on a DIN rail, each with a legend plate
naming its circuit and fuse rating. F2 holds a blown fuse marked with a black X. The robot must open F2's carrier
(it hinges at the bottom and swings forward and down), take the blown fuse out and drop it in a tray, lay in the
spare of the rating on F2's legend (6 A, not the 10 A spare beside it), and push the carrier shut until it latches,
without opening F1 or F3. Scoring checks the result physically: after release, the carrier must hold shut against
a 4 N pull on its grip tab, with the 6 A spare in it and the blown fuse in the tray.

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

The scripted demo is closed loop. Like a scripted demo with privileged state, it reads the carrier angle and the
fuses' poses from the simulator, and it sends the same 14-D joint action a policy does, through yam_arena's IK:

```bash
inspect-robots run --embodiment yam_arena --policy fuse-demo \
  -E task=tasks/fuse-holder-swap/fuse_holder_swap.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 3200 --store-frames --log-dir ~/yam-eval/fuse-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json). Videos:
`inspect-robots video <log>.json` writes one MP4 per camera; `demo.mp4` and `dream1.mp4` here put the top camera
(2x) above the left and right wrist cameras.

- **Episode end.** Once F2 is shut on the 6 A spare and both grippers have stayed at least 6 cm from its grip tab
  for 1 s, the hold test (4 N pull at the tab, 0.11 N m about the hinge, for 1.5 s) runs inside that control step,
  off camera, and the episode ends; the stored frames stop at the release. Otherwise the run ends at `--max-steps`.
- **Control run (the latch matters).** Two short runs with the arms idle at ready start from the swap already done
  (the 6 A spare seated in the shut carrier, the marked fuse in the tray; `FUSE_HOLD_CHECK=1`), so the hold test
  runs after 1 s. With the latch (as in the task) the carrier holds the 4 N pull: `task_score: 1`. With F2's latch
  spring off (`FUSE_HOLD_CHECK=1 FUSE_CONTROL=1`, a carrier whose snap is broken) the pull opens it:
  `task_score: 0.75`. Command: `FUSE_HOLD_CHECK=1 FUSE_CONTROL=1 inspect-robots run ... --policy fuse-demo
  --max-steps 200` (the same `-E` options).
- **Seeds.** One fixed layout (`-E object_jitter=0`). Seeded jitter is not used.
- **Runtime per trial.** See "Measured" below. Isaac Sim's first start on a new machine adds several minutes.

## Measured

RTX-class GPU, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz, 320x176:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `fuse-demo` | `task_score: 1`. The tab pull opened F2 at 18 mm of extra pull; the 6 A spare was seated 0.7 mm from its seat; the closing lift freed the carrier from its open detent at 12 mm of extra push and it swung shut into the latch; it held the 4 N pull. No other holder touched, no fuse dropped. | 1,952 (98 s), then the 1.5 s hold test off camera | about 33 minutes (other simulators shared the GPU) |
| Hold check, latch on: `FUSE_HOLD_CHECK=1`, `fuse-demo` (arms idle) | `task_score: 1` (held) | 10, then the hold test | under 1 minute |
| Control, latch off: `FUSE_HOLD_CHECK=1 FUSE_CONTROL=1` | `task_score: 0.75` (the 4 N pull opens the carrier) | 10, then the hold test | under 1 minute |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (both arms move about near their start; the right hand drifts toward the spares stand and tray but nothing is touched, and F2 is never opened) | 3,200 (step limit) | about 38 minutes (shared GPU; Dream reported an observation as expired and was asked again 65 times) |

The demo's wall time is mostly its Python IK every control step and the shared GPU. The demo is sensitive: PhysX on
the GPU is not deterministic, and lifting the blown fuse out of the open carrier is its weakest step (with the
latch off, which changes how the carrier opens, the same demo code let the fuse slip from the fingertips there
twice). The recorded run is the one listed above.

## Layout

| Path | What |
|---|---|
| `fuse_holder_swap.py` | the task file: description for agents, the props and task, scoring |
| `fuse_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera hold test) |
| `fuse_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (latch and open detent, fuse seat, hold test), milestones, layout and instruction |
| `fuse_sim/assets/*.usda`, `fuse_sim/assets/textures/` | the props and their printed labels |
| `fuse_sim/usd/fuse_props.py` | generates the props from the dimensions in task.json (`python -m fuse_sim.usd.fuse_props`) |
| `fuse_sim/demo.py`, `fuse_sim/policy.py` | the scripted demo and its Inspect Robots policy `fuse-demo` |

## References

- Eaton Bussmann CHM1DU one-pole modular holder for 10 x 38 mm fuses (distributor page with photos and spec sheet):
  https://www.standardelectricsupply.com/Bussmann-CHM1DU-Fuse-Holder
- The same holder (photos of the hinge boss and side label):
  https://www.yesss.co.uk/modular-midget-fuse-holder-without-indicator-1-pole-10mm-x-38mm-chm-series
- Eaton Bussmann Technical Data 720147, "10 x 38 mm Modular fuse holders" (CHM series; dimensions, and the side
  view with the carrier drawn open, from which the hinge and 72-degree opening were measured).
- Eaton Bussmann KTK fast-acting 10 x 38 mm midget fuses (KTK-6, KTK-10): Eaton's KTK product data.
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation, gripper a 5000 N/m spring capped at 50 N). Published dimensions: the CHM1DU's width, height, depth and grip tab; the 10 x 38 mm fuse. Measured on Bussmann's drawing: the carrier's hinge position and its 72-degree opening. Holder spacing: the holders sit one module apart instead of ganged at 17.5 mm because the robot's fingers are 15 mm wide at the tips and 35-40 mm wide 1-2 cm back; ganged, every grasp of the open fuse or the tab would hit a neighbour's grip tab, which stands 9 mm proud just beside it. A blank module or terminal between holders is common on real rails, but it is a layout choice made for the robot. How the fuse comes out: Bussmann does not publish the carrier's internals. Here the inner ferrule sits in a U-shaped clip open toward the holder body (upward when the carrier is open), with a bottom stop and a saddle under the outer ferrule; the fuse is lifted out of the clip's open side and drawn toward the robot, and the new one is laid in from above, across its axis. On a real CHM the fuse may instead be drawn out along its axis from spring clips; no spring grip on the ferrules is modelled (gravity and the clip shape hold it). Latch: a rule, not moulded geometry. Within 6 degrees of shut a joint spring pulls the carrier shut with 0.14 N m (5 N pulling straight out at the grip tab) rising to 0.20 N m (7 N) at 6 degrees; within 8 degrees of the open stop the same spring holds it open (an open detent, our assumption, so the open carrier stays still while fuses go in and out). The real snap force is not published (to measure on a bought part); the value was set as high as the station robot could still open by pinching the tab and pulling. The after-release check pulls with 4 N, 0.8 of the latch's hold. Gravity swings an unlatched carrier open because its centre of mass is set toward the grip tab (our assumption). Blown-fuse marking: KTK fuses have no indicator, so the blown fuse is marked the way an electrician marks a fuse found open, with a black marker X over its printed tube. Legibility: at the 320 x 176 top camera the legend and fuse text is 2-3 pixels tall and cannot be read, so the legends and the spares' cradle labels carry a rating colour code (2 A pink, 6 A green, 10 A red, the IEC 60269-3 D-type fuse colours, used here as a plant labelling scheme); the colours are visible from the top camera. Collision shapes are boxes and 16-sided prisms; the holder body is boxes, the carrier a plate, tab and clip, with Bussmann's label, legends and fuse print as textures. No electrical function. The pull test is a torque applied in the simulator after the recording ends. One layout.

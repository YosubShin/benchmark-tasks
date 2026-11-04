# Switch off the named circuit breaker in a distribution board

A small distribution board holds six identical miniature circuit breakers behind a dead-front cover, all ON. A
coloured label strip above the breakers and a circuit directory card name each way's circuit. The robot is told the
circuit's name only (SOCKETS - KITCHEN), so it must find which way feeds it, then push that breaker's handle down past
its snap point to OFF, so the window shows the green O-OFF band, leaving the other five ON. Scoring checks the result
physically: once the robot has let go, the handle is pushed back toward ON and must stay OFF.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`breaker_switch_off.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy.
The board is an articulation (six handles, each with a sprung plunger), which the one-file format cannot declare, so
the file uses `breaker_sim` (this folder), which adds it to the station at run time (`breaker_sim/station.py`
documents exactly what it patches; it requires yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/breaker-switch-off --no-deps          # registers the policy breaker-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/breaker-switch-off/breaker_switch_off.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Switch off the breaker that feeds SOCKETS - KITCHEN (find it on the labels above the breakers or the circuit directory) so that its window reads O-OFF. Leave the other circuits on." \
  --scorer task_score --max-steps 700 --store-frames --log-dir ~/yam-eval/breaker-switch-off-dream
```

The scripted demo (closed loop: it knows which way is named and reads the handle angles and its own grasp point from
the simulator, like a scripted demo with privileged state, and sends the same 14-D joint action a policy does, through
yam_arena's IK):

```bash
inspect-robots run --embodiment yam_arena --policy breaker-demo \
  -E task=tasks/breaker-switch-off/breaker_switch_off.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 700 --store-frames --log-dir ~/yam-eval/breaker-switch-off-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json). Videos:
`inspect-robots video <log>.json` writes one MP4 per camera.

- **Episode end.** Once the SOCKETS - KITCHEN handle (way 4) sits at its OFF stop and every fingertip has stayed at
  least 4 cm from its pivot for 1 s, the after-release check runs (1.5 s of simulated time: 0.063 N m toward ON on
  that handle) inside that control step, off camera, and the episode ends; the stored frames stop at the release.
  Otherwise the run ends at `--max-steps`, scored on the state at that moment.
- **Control run.** `BREAKER_CONTROL=1` with `--policy breaker-demo`: the demo pushes the named handle only 30% of its
  throw, short of the snap point, and lets go. The detent must spring it back to ON, so it must score below 1.
- **Seeds.** One fixed layout (`-E object_jitter=0`); seeded jitter is not used.
- **Runtime per trial**: see "Measured" below; Isaac Sim's first start on a new machine adds several minutes.

## Measured

Shared RTX-class GPU (other Isaac Sim processes running at the same time), Isaac Sim shader cache warm,
`-E object_jitter=0`, 20 Hz, 320x176 frames:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `breaker-demo` | `task_score: 1` (way 4 snapped to its OFF stop, -30 degrees, window O-OFF, the other five ON; after-release check passed) | 331 (16.6 s), then the 1.5 s check off camera | about 5.5 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (both arms move about in front of the board and turn their wrists toward it but never reach the breakers; no handle turned 5 degrees, no other breaker switched) | 700 (step limit) | about 14 minutes |
| Control: `BREAKER_CONTROL=1`, `breaker-demo`, 30% push | `task_score: 0` (the light press never broke the handle away from its ON stop: the detent held it at +30 degrees, window I-ON) | 700 (step limit) | about 9 minutes |

Most of the wall time is Isaac Sim's start-up and the demo's IK, which it solves in Python every control step.

## Layout

| Path | What |
|---|---|
| `breaker_switch_off.py` | the task file: description for agents, the board and task, scoring |
| `breaker_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera after-release check) |
| `breaker_sim/task.py`, `scorer.py`, `spec.py` | the board articulation, the after-release check, the score levels, the circuits and instruction |
| `breaker_sim/assets/board.usda`, `assets/tex/`, `breaker_sim/usd/` | the board and its label textures, and the script that generates them (`python -m breaker_sim.usd.board_props breaker_sim/assets`) |
| `breaker_sim/demo.py`, `breaker_sim/policy.py` | the scripted demo and its Inspect Robots policy `breaker-demo` |

## References

- Schneider Electric Acti9 iC60N 1P C16 miniature circuit breaker (A9F74116): https://www.se.com/ww/en/product/A9F74116/
- Schneider Electric iC60N 1P product data sheet (A9F74106; dimensions 18 x 85 x 78.5 mm, contact position indicator):
  https://download.schneider-electric.com/files?p_enDocType=Product+Data+Sheet&p_File_Name=A9F74106_DATASHEET_WW_en-GB.pdf&p_Doc_Ref=A9F74106_DATASHEET_WW_en-GB
- Schneider Electric Acti9 catalogue (iC60 features: green O-OFF band only when the contacts are open, quick-make
  closing; dimension drawing): https://download.schneider-electric.com/files?p_Doc_Ref=ZXKACTI9CHF
- EN 60715 TH35 top-hat rail dimensions: https://commons.wikimedia.org/wiki/File:DIN-rail-dimensions.svg
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation). Rigid bodies on joints. The snap action is geometry: behind each handle's pivot a spring plunger (5000 N/m, 4.2 mm preload) presses on a 4 mm cam ball 8 mm from the pivot, which always pushes the handle away from centre, so a handle holds at either stop (0.084 N m, about 5 N at the bar), snaps over once pushed past centre, and springs back if released short of it. There is no rule on the handle. That holding torque is inferred from a published simulation study of breaker handles (5-10 N), not measured on an iC60N, and the +-30 degree throw, pivot position and bar length are read from Schneider's dimensioned drawing and photos. The window marking is printed on the rocker, so it follows the handle; on the real part it follows the contacts, and the simulation has no contacts or welded-contact fault, so the window cannot disagree with the handle. Collision shapes are boxes plus the cam ball; the rocker and bar are visual cylinders on a box. The breakers' front print is generic, not Schneider's artwork. One rail at a height chosen for the arm's reach; real boards often have several rails 125 mm apart. Labels: the way labels and the directory card are legible in the wrist cameras once the hand is near the board, but in the 320 x 176 overhead camera each label is only about 4 x 2 pixels, so their text cannot be read there; each label's background colour (by circuit type) is still visible, and the blue way-4 label is the only blue one. The circuit names, ratings, label colours and label artwork are ours. The 3.7 N check force is our choice and is applied as a torque in the simulator after the recording ends. One layout.

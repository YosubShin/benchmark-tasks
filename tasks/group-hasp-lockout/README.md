# Add your padlock to a group lockout hasp

A disconnect switch's handle is held OFF by a group lockout hasp that already carries a coworker's padlock. The
robot must add its own padlock through any free hole of the hasp and snap it shut, without moving the coworker's
lock. Scoring checks the lockout physically: with the coworker's lock removed, the robot's lock alone must keep the
hasp shut and the handle OFF.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file (`group_hasp.py`)
in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy. Its props are
articulated (the hasp's jaws, the handle, the padlock's shackle), which the one-file format cannot declare, so the
file uses `lockout_sim` (this folder), which adds them to the station at run time (`lockout_sim/station.py`
documents exactly what it patches; it requires yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/group-hasp-lockout --no-deps          # registers the policy lockout-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/group-hasp-lockout/group_hasp.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "A coworker has locked out this switch with a group hasp. Put your purple padlock through any free hole of the hasp and snap it shut. Do not move or remove the coworker's yellow padlock." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/group-hasp-dream
```

The scripted demo (closed loop: it reads the padlock's and hasp's poses from the simulator, like a scripted demo with
privileged state, and sends the same 14-D joint action a policy does, through yam_arena's IK):

```bash
inspect-robots run --embodiment yam_arena --policy lockout-demo \
  -E task=tasks/group-hasp-lockout/group_hasp.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/group-hasp-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json). Videos:
`inspect-robots video <log>.json` writes one MP4 per camera.

- **Episode end.** Once the robot's shackle is shut through a free hole and both grippers have stayed at least
  6 cm clear of the lock for 2 s, the hold test runs (3 s of simulated time) inside that control step, off camera,
  and the episode ends; the stored frames stop at the release. Otherwise the run ends at `--max-steps`.
- **Control run.** `LOCKOUT_CONTROL=1` with `--policy yam_scripted -P hold_s=12` forces the hold test at 3 s with
  no robot padlock: it must score 0 (the hasp opens and the handle turns to ON).
- **Seeds.** One fixed layout (`-E object_jitter=0`); seeded jitter is not used.
- **Runtime per trial** (RTX PRO 6000, warm shader cache): see "Measured" below; Isaac Sim's first start on a new
  machine adds several minutes.

## Measured

RTX-class GPU, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `lockout-demo` (640x352 frames) | `task_score: 1` (shut through the bottom robot-side hole; hold test passed) | 1,034 (51.7 s), then the 3 s hold test off camera | about 7 minutes |
| `dream` (`dreamlabs/dream-1-beta`, 320x176) | `task_score: 0` (the arms hover near the start; the padlock is never picked up). | 2,400 (step limit) | about 13 minutes (Dream round trip about 0.58 s per 48-row chunk) |
| Control: `LOCKOUT_CONTROL=1`, `yam_scripted`, no robot padlock | `task_score: 0` | 52, then the hold test | about 1 minute |

The demo's wall time is mostly its IK, which it solves in Python every control step.

## Layout

| Path | What |
|---|---|
| `group_hasp.py` | the task file: description for agents, the props and task, scoring |
| `lockout_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera hold test) |
| `lockout_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (handle detent, shackle latch, coworker watch, hold test) and milestones |
| `lockout_sim/assets/*.usda`, `lockout_sim/usd/` | the props, and the scripts that generate them from the dimensions below |
| `lockout_sim/demo.py`, `lockout_sim/policy.py` | the scripted demo and its Inspect Robots policy `lockout-demo` |

## References

- Master Lock 420 lockout hasp (1 in jaw, six locks): https://www.masterlock.com/business-use/product/420
- Master Lock 410 Zenex thermoplastic safety padlock (410YLW, 410PRP): https://www.masterlock.com/business-use/product/410YLW
- Square D heavy-duty safety switch, side-operated handle with padlock provision (HU361):
  https://www.se.com/us/en/product/HU361/
- OSHA 29 CFR 1910.147, control of hazardous energy (group lockout, 1910.147(f)(3)):
  https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation). Published dimensions: the hasp's jaw, length, width and hole count; the padlocks' bodies and shackles. Inferred, to be measured on real parts: the hasp's 2 mm jaw plates, 8 mm square holes, hole spacing and rivet position; all handle dimensions; the handle detent (1.0 N·m); the padlock's shackle travel (16 mm) and spring (150 N/m). Simplified: the enclosure is a plain box with no door; the lever points up at OFF and turns sideways to ON (the real Square D lever points down at OFF), so it clears the hanging hasp; collision shapes are boxes (round rods become square rods of the same width); both padlocks follow the Master Lock 410's published outline (rounded 38 x 44 x 19 mm body with a danger label, round 6.35 mm shackle with a semicircular bend; collision uses square legs and six straight segments around the bend; the short leg's collider is a low-friction cross); the latch is a catch that locks the shackle within 2 mm of shut and the lock cannot be reopened (no key); the coworker's lock is one rigid body; the hasp's friction on the handle is modelled as damping; the hold test is applied as torques in the simulator after the recording ends.

# Put a job lock's key into a group lock box and add your padlock

Group lockout with a lock box: the key of every job lock goes into a group lock box, and every worker hangs a
personal padlock on the box, so no key can come out until the last worker has removed theirs. The robot takes the key
of the job lock on the PUMP 1 panel from the hook on that panel's side, drops it through the key slot in the lid of a
Master Lock S601 group lock box that a coworker has already locked, then hangs its own padlock through a free hole of
the box's front flange and snaps it shut. Scoring checks the lockout physically: with the coworker's lock removed and
the latch released, the robot's lock alone must keep the lid shut, with the key inside.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file (`group_lock_box.py`)
in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy. Its props are
articulated (the box's hinged lid, the padlock's shackle), which the one-file format cannot declare, so the file uses
`lockbox_sim` (this folder), which adds them to the station at run time (`lockbox_sim/station.py` documents exactly
what it patches; it requires yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/group-lock-box --no-deps              # registers the policy lockbox-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/group-lock-box/group_lock_box.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the key hanging beside the PUMP 1 panel (yellow nameplate) and drop it into the red lock box through the key slot in its lid. Then hang your purple padlock through a free hole of the box's front flange to the right of the yellow padlock, and snap it shut. Do not move the yellow padlock or the other key." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/group-lock-box-dream
```

The scripted demo (closed loop: it reads the key's, padlock's and box's poses from the simulator, like a scripted demo
with privileged state, and sends the same 14-D joint action a policy does, through yam_arena's IK):

```bash
inspect-robots run --embodiment yam_arena --policy lockbox-demo \
  -E task=tasks/group-lock-box/group_lock_box.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 3600 --store-frames \
  --log-dir ~/yam-eval/group-lock-box-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json). Videos:
`inspect-robots video <log>.json` writes one MP4 per camera.

- **Episode end.** Once the key is in the box, the robot's shackle is shut through a free hole and both grippers have
  stayed at least 6 cm clear of its padlock for 2 s, the after-release check runs (2 s of simulated time) inside that
  control step, off camera, and the episode ends; the stored frames stop at the release. Otherwise the run ends at
  `--max-steps`.
- **Control run.** `LOCKBOX_CONTROL=1` with `--policy lockbox-demo`: the demo drops the key but skips the padlock, and
  the after-release check starts 2 s after the key is inside. It must score below 1 (the lid opens).
- **Seeds.** One fixed layout (`-E object_jitter=0`); seeded jitter is not used.

## Measured

RTX-class GPU shared with other simulations, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz, 320x176:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `lockbox-demo` | `task_score: 1` (key in the box; padlock shut through front hole 7; check: lid opened 10 degrees, held) | 1,013 (50.7 s), then the 2 s check off camera | about 15 minutes |
| Control: `LOCKBOX_CONTROL=1`, `lockbox-demo` (key dropped in, no padlock) | `task_score: 0.5` (the lid swings open in the check) | 481, then the check | about 9 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (the right arm reaches toward the PUMP 1 panel but never grasps a key; the left arm stays near the start; nothing is moved) | 3,600 (step limit) | about 46 minutes (74 session-expired retries on the shared GPU) |

The demo's wall time is mostly its IK, which it solves in Python every control step, and the shared GPU.

## Layout

| Path | What |
|---|---|
| `group_lock_box.py` | the task file: description for agents, the props and task, scoring |
| `lockbox_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera check) |
| `lockbox_sim/spec.py` | every dimension and position, and the instruction |
| `lockbox_sim/task.py`, `scorer.py` | the mechanism (latch, shackle latch, key and coworker watch, after-release check) and milestones |
| `lockbox_sim/assets/*.usda`, `lockbox_sim/usd/` | the props, and the scripts that generate them (`python -m lockbox_sim.usd.lockbox_props lockbox_sim/assets`) |
| `lockbox_sim/demo.py`, `lockbox_sim/policy.py` | the scripted demo and its Inspect Robots policy `lockbox-demo` |

## References

- Master Lock S600 series group lock boxes (S600, S601, S602) sell sheet, with dimensions, 18 numbered lock holes,
  gasketed key slot, latch with lock hole: https://cdn.masterlock.com/masterlock/resources/documents/pdf/sell-sheets/7000-0619_S600-Series-Group-Lock-Box-Sell-Sheet.pdf
- Master Lock S601 product page: https://www.masterlock.com/products/product/S601
- Master Lock 410 Zenex thermoplastic safety padlock (410PRP, 410YLW, 410RED): https://www.masterlock.com/business-use/product/410YLW
- OSHA 29 CFR 1910.147, control of hazardous energy (group lockout, 1910.147(f)(3)):
  https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity
compensation). Published (Master Lock S600 series sell sheet): the box's 144 mm height, 286 mm length (306 mm with
hinge and latch), 163 mm across the flanges, 128 mm body, 18 numbered holes (9 per long side), 10 mm maximum shackle,
a gasketed key slot, a lid hinged at one end with a latch at the other; the padlocks' bodies and shackles (Master Lock
410). Inferred from product photos, to be measured on a real box: the flange width (17.5 mm), its height on the body,
hole size (taken as 11 x 14 mm rectangles; the real holes are ovals), spacing (28 mm) and position (8.75 mm out from the
wall), the lid's mass (0.7 kg), the slot's size (40 x 7 mm) and place (2.5 cm from the latch end); the key's
dimensions and 7 g mass; the pegs, panels and board (no real product). How it works in the simulator: the lid is a
revolute joint at the hinge end; the lid's flange lies on the body's flange, and a shackle through a hole pair stops
the lid opening further than the shackle's play (collision only, no rule); the latch is a joint limit that holds the lid
shut until the after-release check releases it; the key slot is a gap in the lid's top (no gasket, no removable key cup:
the key falls to the floor). The check lifts the lid with a torque about the hinge equal to 20 N at the free end. With
a padlock through hole 7 the lid opens 10 degrees (the 38 mm shackle's play); with none it swings open (43 degrees
within the 2 s). A lock near the hinge (holes 1-4) leaves more play (the lid's free end can rise more than 7 cm), as on
the real box, which is why the instruction asks for a hole on the latch side. Simplified: collision shapes are boxes
(round rods are square rods of the same width); both padlocks follow the Master Lock 410's published outline (rounded
38 x 44 x 19 mm body, round 6.35 mm shackle; collision uses square legs and six straight segments around the bend); the
410's latch is a catch that locks the shackle within 2 mm of shut and cannot be reopened (no key); the coworker's lock
is one rigid body; the panels' handles and job padlocks are visual only; the hold test is applied as a torque in the
simulator after the recording ends.

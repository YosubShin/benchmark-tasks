# Switch a wall switch OFF and lock it out with a wall switch lockout cover

Two wall toggle switches, S1 (yellow label) and S2 (white label), each carry a Master Lock 496B wall switch lockout:
its base is already installed on the switch plate with the plate's own screws, and its red cover is hinged at the
top of the base and stands open, as on site. Both switches are ON. The robot pushes S1's toggle down to OFF, swings
S1's cover down over it, puts a Master Lock 410 padlock's shackle through the padlock ring at the bottom of the cover
and snaps it shut, without turning S2 OFF. Scoring checks the lockout physically: after the robot lets go, the cover is
pulled open and the toggle is pushed toward ON; the cover must stay shut and the toggle on the OFF side.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file (`wall_switch.py`)
in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy. Its props are
articulated (the two toggles, the two covers, the padlock's shackle), which the one-file format cannot declare, so the
file uses `switch_lockout_sim` (this folder), which adds them to the station at run time
(`switch_lockout_sim/station.py` documents exactly what it patches; it requires yam-isaac-lab-arena at commit
`0059d02`, the Brev launchable's pin).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/wall-switch-lockout --no-deps         # registers the policy wall-switch-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/wall-switch-lockout/wall_switch.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Turn wall switch S1, the one with the yellow label, OFF by pushing its toggle down. Then swing its red lockout cover down over the switch, put your padlock's shackle through the ring at the bottom of the cover and snap it shut. Leave switch S2, the one with the white label, ON." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/wall-switch-dream
```

The scripted demo (closed loop: it reads the toggle and cover angles and the cover's and padlock's poses from the
simulator, like a scripted demo with privileged state, and sends the same 14-D joint action a policy does, through
yam_arena's IK):

```bash
inspect-robots run --embodiment yam_arena --policy wall-switch-demo \
  -E task=tasks/wall-switch-lockout/wall_switch.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/wall-switch-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json). Videos:
`inspect-robots video <log>.json` writes one MP4 per camera (demo.mp4 and dream1.mp4 here put the top camera, at 2x,
above the two wrist cameras).

- **Episode end.** Once the padlock is shut through S1's ring and both grippers have stayed at least 6 cm clear of it
  for 2 s, the after-release check runs (4 s of simulated time) inside that control step, off camera, and the episode
  ends; the stored frames stop at the release. Otherwise the run ends at `--max-steps`.
- **Control run.** `--policy wall-switch-demo -P control=true` turns S1 OFF and shuts its cover but leaves the padlock
  on its post, then forces the after-release check: it must score below 1.
- **Seeds.** One fixed layout (`-E object_jitter=0`); seeded jitter is not used.

## Measured

RTX-class GPU shared with other simulations, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz, 320x176:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `wall-switch-demo` | `task_score: 1` (S1 pushed OFF; cover pulled from 185 to 168 degrees by its side walls and let go, it swung shut by itself; padlock brought in so the ring went into the gap under the short leg's tip, then lowered so the short leg went down through the ring; arm B pressed the shackle's bend down while arm A held the body, latched; after-release check passed: cover opened 4.3 degrees and stopped on the shackle, toggle stopped 20.4 degrees below the middle) | 1,612 (81 s), then the 4 s check off camera | about 73 minutes |
| Control: `wall-switch-demo -P control=true` (S1 OFF, cover shut, no padlock) | `task_score: 0.5` (after-release check fails: the cover swings fully open and the toggle is pushed to ON) | 594, then the check | about 30 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (the arms move a little toward the wall; neither toggle nor cover is touched and the padlock stays on its post) | 3,000 (step limit) | about 55 minutes (61 "session expired" retries on the shared GPU) |

The demo's wall time is mostly the shared GPU and its IK, which it solves in Python every control step; on an idle
machine expect much less. The demo is closed loop because GPU PhysX is not deterministic and the shackle has about
1 mm of clearance in the ring: replaying its actions open loop is not expected to reach 1.0.

## Layout

| Path | What |
|---|---|
| `wall_switch.py` | the task file: description for agents, the props and task, scoring |
| `switch_lockout_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera check) |
| `switch_lockout_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (toggle snap, cover hinge, shackle latch, decoy watch, after-release check) and milestones |
| `switch_lockout_sim/assets/`, `switch_lockout_sim/usd/` | the props, and the scripts that generate them from the dimensions in task.json (`python -m switch_lockout_sim.usd.switch_props switch_lockout_sim/assets`) |
| `switch_lockout_sim/demo.py`, `switch_lockout_sim/policy.py` | the scripted demo and its Inspect Robots policy `wall-switch-demo` |

## References

- Master Lock 496B universal wall switch lockout (toggle and Decora switches; hinged cover; padlock shackle up to
  5/16 in): https://www.masterlock.com/products/product/496B
- Product photo used for the cover's outline, hinge and padlock ring: https://cdn.masterlock.com/product/orig/MLCOM_PRODUCT_496B.jpg
- Distributor listing (polycarbonate, red, 5/16 in max shackle, screw-in installation): https://www.hanessupply.com/master-lock-496b
- Master Lock 410 Zenex thermoplastic safety padlock (410PRP): https://www.masterlock.com/business-use/product/410PRP
- OSHA 29 CFR 1910.147, control of hazardous energy:
  https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity
compensation). Rigid bodies on joints; every link of the lock-out is geometry, and nothing in the code holds the
cover or the toggle.

- **How the real 496B works** (Master Lock's product page and installation videos): the base is a red frame that fits
  over the switch plate and is held by the plate's two screws (longer screws are supplied); it is installed once and
  stays, the hinged cover opening for normal use. The cover is hinged at the top of the base and, swung down, boxes in
  the toggle; a padlock ring on the base comes out through a slot near the bottom of the cover, and a padlock hung in
  the ring keeps the cover shut. The task therefore starts with both bases installed and both covers open.
- **Cover and base.** About 120 x 45 x 25 mm overall (published 4.7 x 1.8 x 1.0 in). The cover is a box of 2.5 mm
  walls on a hinge at the top of the base, 8 mm behind its face; open, it leans back on the wall above the plate at
  185 degrees (it passes its balance point at about 178 degrees) and, pulled past that, swings shut under its own
  weight (30 g, hinge damping 0.01 N m s/rad, ours). The padlock ring is a 4 mm thick plate with an 8.5 mm square hole,
  its axis vertical, standing 10.5 mm proud of the shut cover's face on a neck through a 19 x 10 mm slot in the cover.
  With a shackle through it the cover can swing out about 4 degrees, then meets the shackle. The cover is opaque here
  (the real one is translucent red); its stops at shut and fully open are joint limits standing in for the cover
  seating on the base and leaning on the wall.
- **Toggle block.** A rib inside the cover sits at the middle of the toggle's throw: it clears the toggle at either end
  and stops it about 15 degrees short of the middle, even with the cover swung out to the shackle. Master Lock says the
  496B holds a switch ON or OFF; the rib's shape is ours (the inside of the real cover was not measured).
- **Switches.** One-gang plates 70 x 115 mm with a 10 x 24 mm toggle opening; the toggle is a 26 mm lever (8 x 5 mm)
  on a pivot 6 mm behind the plate, throwing 30 degrees each way (up = ON), with a snap modelled as a capped drive
  toward the nearer end (0.06 N m, about 3 N at the middle of the toggle; the real force is not published and is
  inferred as a light finger push). The switches are not wired; boxes are flush in a plywood board.
- **Padlock.** A Master Lock 410 (1/4 in shackle), the same model as in our other lockout tasks: rounded 38 x 44 x 19
  mm body with a danger label, round 6.35 mm shackle with a semicircular bend (collision: square legs and six straight
  segments around the bend). Its shackle pops straight out 16 mm and is pushed straight in (no swivel), with a
  150 N/m spring and a catch that locks it once pushed within 2 mm of shut; the key is not simulated.
- **Labels.** Engraved two-ply laminate labels (50 x 32 mm, 20 mm letters) on the wall left of each plate, yellow for
  S1 and white for S2, so the named switch is visible in the 320 x 176 overhead camera (the letters are legible in the
  wrist cameras). Label sizes and colours are ours.
- **Check forces.** The 10 N pull on the cover (applied as 1.15 N m about the hinge) and the 0.2 N m push on the
  toggle are our choice and are applied in the simulator after the recording ends. One layout.

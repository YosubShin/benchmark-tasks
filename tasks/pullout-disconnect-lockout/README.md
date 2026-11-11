# Lock out an air conditioner at its pull-out disconnect

An outdoor air-conditioning unit is fed through a pull-out disconnect on the wall beside it: a small grey box whose
top-hinged cover hangs over a black pull-out block. Pushed in, the block's blades sit in the line and load clips;
pulled out, the unit has no supply. The block is printed ON one way up and OFF the other, and put back in turned over
it does not connect. A tab with a hole sticks out of the box's bottom through the cover, for a padlock.

The robot lifts the cover of the AC-1 disconnect (yellow label) and holds it up with one hand, pulls the block out
with the other, turns it over and pushes it back in OFF, lowers the cover, and padlocks it through the tab with a
Master Lock 410. The AC-2 disconnect beside it (white label) feeds another unit and must not be touched. Scoring checks
the lockout physically: after the robot lets go, the cover's bottom edge is pulled outward with 20 N, and the
padlock alone must keep it shut.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`pullout_disconnect_lockout.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream`
policy. Its props are articulated (the covers' hinges, the padlock's shackle), which the one-file format cannot
declare, so the file uses `pullout_sim` (this folder), which adds them to the station at run time
(`pullout_sim/station.py` documents exactly what it patches; it requires yam-isaac-lab-arena at commit `0059d02`, the
Brev launchable's pin).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/pullout-disconnect-lockout --no-deps  # registers the policy pullout-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/pullout-disconnect-lockout/pullout_disconnect_lockout.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lock out the air conditioner AC-1 at its pull-out disconnect, the grey box with the yellow AC-1 label: lift its cover and hold it up, pull out the black pull-out block by its grip, turn it upside down and push it back in so that OFF shows, lower the cover shut, then put the red padlock through the hole in the tab at the bottom of the box and snap it shut. Do not touch the AC-2 disconnect (white label)." \
  --scorer task_score --max-steps 4800 --store-frames --log-dir ~/yam-eval/pullout-dream
```

The scripted demo (closed loop: it reads the cover's angle and the block's and padlock's poses from the simulator,
like a scripted demo with privileged state, and sends the same 14-D joint action a policy does, through yam_arena's
IK):

```bash
inspect-robots run --embodiment yam_arena --policy pullout-demo \
  -E task=tasks/pullout-disconnect-lockout/pullout_disconnect_lockout.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 4800 --store-frames \
  --log-dir ~/yam-eval/pullout-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json). Videos:
`inspect-robots video <log>.json` writes one MP4 per camera (demo.mp4 and dream1.mp4 here put the top camera, at 2x,
above the two wrist cameras).

- **Episode end.** Once AC-1's block is seated OFF, its cover is shut, the padlock has latched through the tab and both
  grippers have stayed at least 6 cm clear of the padlock and of the cover's bottom edge for 2 s, the after-release
  check runs (2 s of simulated time) inside that control step, off camera, and the episode ends; the stored frames
  stop at the release. Otherwise the run ends at `--max-steps` and the state then sets the score.
- **Control run.** `PULLOUT_CONTROL=1` with `--policy pullout-demo`: the demo does everything but the padlock, and the
  after-release check starts 2 s after the cover is shut with the block OFF and the grippers clear. It must score
  below 1 (the cover swings open).
- **Seeds.** One fixed layout (`-E object_jitter=0`); seeded jitter is not used.

## Measured

RTX-class GPU shared with other simulations, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz, 320x176
(this task steps physics at 480 Hz):

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `pullout-demo` | `task_score: 1` (cover held up at 92 degrees, block pulled, turned over and seated OFF 0.4 mm from home, cover lowered shut, padlock latched through the tab; check passed: the cover held under 20 N, less than 10 degrees; 4.2 degrees in a development run) | 2,513 (126 s), then the 2 s check off camera | about 61 minutes |
| Control: `PULLOUT_CONTROL=1`, `pullout-demo` (no padlock) | `task_score: 0.75` (block OFF and cover shut; the cover swings open in the check) | 1,329, then the check | about 26 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (the right arm reaches under the boxes toward the tabs and hovers there; the left arm lifts away; no cover is opened, nothing is moved) | 4,800 (step limit) | about 87 minutes (98 session-expired retries on the shared GPU) |

The demo's wall time is mostly the shared GPU and its IK, which it solves in Python every control step.

## Layout

| Path | What |
|---|---|
| `pullout_disconnect_lockout.py` | the task file: description for agents, the props and task, scoring |
| `pullout_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera check, physics rate) |
| `pullout_sim/spec.py` | every dimension and position, and the instruction |
| `pullout_sim/task.py`, `scorer.py` | the mechanism (block state, shackle latch, decoy watch, after-release check) and levels |
| `pullout_sim/assets/*.usda`, `pullout_sim/usd/` | the props, and the scripts that generate them (`python -m pullout_sim.usd.disconnect_props pullout_sim/assets`) |
| `pullout_sim/demo.py`, `pullout_sim/policy.py` | the scripted demo and its Inspect Robots policy `pullout-demo` |

## References

- Milbank U3802 60 A non-fusible pull-out air-conditioner disconnect, specification sheet (with product photo):
  https://border-states-assets.azureedge.net/SKU_2487400_ATT_ItemSpecSheetURL_1.pdf
- Milbank U38 series non-fused pull-out N3R air-conditioner disconnects (6 x 5.2 x 2.85 in, 60 A, 240 V, 10 kAIC):
  https://www.borderstates.com/ (catalog item 2487400, Milbank U3802)
- Eaton DPU222R pull-out air-conditioner disconnect, a comparable unit (60 A, NEMA 3R):
  https://www.eaton.com/us/en-us/skuPage.DPU222R.html
- How-to video of a Milbank U38-series pull-out (the block's ON / OFF marking, pulling it and putting it back in
  upside down): https://www.youtube.com/watch?v=7-QNevJlPFI
- Master Lock 410 Zenex thermoplastic safety padlock (410RED): https://www.masterlock.com/business-use/product/410RED
- OSHA 29 CFR 1910.147, control of hazardous energy: https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX in Isaac Lab-Arena at 480 Hz (this task's physics rate; the policy still acts at 20 Hz), both arms driven as in
Dream's station (joint PD with i2rt's gains and gravity compensation). Friction anchors are kept per contact point
(`friction_correlation_distance` 0.5 mm) so the clips grip each blade separately.

Published (Milbank U3802 spec sheet and distributor listing): 60 A, 240 V, two poles, non-fused, pull-out connection,
Type 3R painted steel, about 6 x 5.2 x 2.85 in. Seen in the product photo and how-to videos: the cover hinged at the
top that hangs shut and is lifted by its bottom edge, the dead front with the pull-out's window, the black block with
a raised grip bar, the ON / OFF marks on the block's face (OFF upside down), the empty base after pulling it, putting it
back in turned over, and the tab with a hole out of the box's bottom. Inferred, to be measured on a real unit: the
window (10 x 4.8 cm), the block (9.6 x 4.4 x 1.8 cm, 0.1 kg) and grip bar (6 x 1.4 cm, 1.6 cm proud), the blade layout
(four blades 22 mm long; live clips in the upper row, insulated dummy pockets in the lower row, so that the turned-over
block lands in the pockets), the tab (2.2 cm wide, 2 mm, 8 mm hole 1.2 cm in front of the cover; a real hole is likely
about 9.5 mm, 8 mm is used so the square collider of the shackle cannot rotate and tear free), the slot in the cover's
front for the tab (2.6 x 2.5 cm, reaching 1 cm below the tab because the cover's bottom rises as it swings shut; the
4.5 mm strip of plate under it is what meets the shackle's legs), the inward
hem along the cover's bottom edge (6 mm), the hinge at the top rear (opened, the cover's top meets the wall at
about 94 degrees), the cover's mass (0.3 kg).

How it works in the simulator: each disconnect is a fixed-base articulation (body + cover on a revolute joint); the
cover has no spring or latch, it hangs shut by its weight and its shut position is the joint's lower limit. The block
is a free rigid body. Each live clip is two fixed fingers whose faces stand 0.4 mm inside the blade's faces, with a
compliant-contact material (8000 N/m): the clip's spring is the contact's spring, and the pull needed to take the block
out comes from it and friction (a steady pull starts it sliding at about 15 N here, measured with a force ramp on
the block; the real pull force is not published, estimated 20-50 N for 60 A clips by hand, REAL/unknown; kept at the
low end so the station's soft wrist gains can pull it, SIM). The dummy pockets hold the blades with 0.8 mm of play. The
padlock's shackle in the tab's hole stands about 9 mm in front of the cover's plate, so with the padlock on the cover's
bottom can swing out only that far before the plate meets the shackle (collision only, no rule): the after-release
check (20 N at the bottom edge, as a torque about the hinge) opened it 4.2 degrees in a development run of the demo; without a padlock it
swings open until its top meets the board (94 degrees). Circuit state is read from the block's pose (ON and blades more than 3 mm into the clips = closed).

Simplified: collision shapes are boxes (round rods are square rods of the same width); no hinge friction; no
conductors, lugs or springs behind the clips (visual straps only); the block's ON / OFF marks and the labels are
geometry, not texture; the Master Lock 410 follows its published outline (rounded 38 x 44 x 19 mm body, round 6.35 mm
shackle; collision uses square legs and six straight segments around the bend) and latches by a catch that locks the
shackle within 2 mm of shut (no key); the after-release check is applied in the simulator after the recording ends.

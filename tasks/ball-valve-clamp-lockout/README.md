# Lock a closed ball valve with a clamp-on ball valve lockout and a padlock

Two identical plain-lever ball valves sit on a short pipe run, each identified by an engraved valve tag (V-12, V-14).
The robot must close V-12 (a quarter turn to its stop), fit a red clamp-on ball valve lockout (Master Lock S3080) over
the closed lever and the valve, push the lockout's two halves together across the pipe, and padlock them through the
lined-up hole, without turning V-14. Scoring checks the lockout physically: after the robot lets go, the two halves
are pulled apart and the lever is pushed toward open, and the valve must stay closed.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`ball_valve_clamp.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy.
Its props are articulated (the two levers on their stems, the lockout's sliding half, the padlock's shackle), which
the one-file format cannot declare, so the file uses `clamp_valve_sim` (this folder), which adds them to the station
at run time (`clamp_valve_sim/station.py` documents exactly what it patches; it requires yam-isaac-lab-arena at commit
`0059d02`, the Brev launchable's pin).

## How the real lockout works

The Master Lock S3080 adjustable ball valve lockout is two red thermoplastic halves that slide together (sell sheet
and how-to videos in the References). With the valve closed (lever across the pipe), half A, a channel with a pair of
legs at one end, is put over the lever so the lever lies in the channel and A's legs stand on the lever's side of the
pipe. Half B, a second pair of legs with a housing of four lock holes, slides onto A's tongue from the far side, its
legs on the other side of the pipe. Pushed together, the two leg pairs grip the pipe between them (one device fits
1/2 in to 2 in valves). A padlock's shackle then goes through the housing hole that lines up with the hole in A's
tongue, so the halves cannot be slid apart. With legs on both sides of the pipe the device cannot turn about the
valve stem, and the lever, held in the channel, cannot turn toward open. Published: 134.6 x 76.9 x 104.1 mm, 156 g,
four locking holes, polycarbonate and PBT.

In the simulation every link of that chain is geometry: the lever is a body on its stem joint, half A is a free body
whose channel walls bear on the lever and whose legs bear on the pipe, half B slides on A along a prismatic joint
(the slide's ends are its limits), and the padlock's shackle, a separate articulation, passes through B's wall and
A's tongue. Nothing in the code holds the lever or the halves.

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD               # must print 0059d02
uv pip install -e tasks/ball-valve-clamp-lockout --no-deps         # registers the policy valve-clamp-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/ball-valve-clamp-lockout/ball_valve_clamp.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Close ball valve V-12: turn its handle a quarter turn clockwise, seen from above, until it points at you. Fit the red valve lockout over the handle and the valve, with the handle inside its channel and its legs on both sides of the pipe, and push its two halves together. Then put your padlock's shackle through the lined-up hole in the side of the lockout and snap it shut. Do not operate valve V-14." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/valve-clamp-dream
```

The scripted demo (closed loop: it reads the lever angle, the lockout's pose and slide, and the padlock's pose from the
simulator, like a scripted demo with privileged state, and sends the same 14-D joint action a policy does, through
yam_arena's IK):

```bash
inspect-robots run --embodiment yam_arena --policy valve-clamp-demo \
  -E task=tasks/ball-valve-clamp-lockout/ball_valve_clamp.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/valve-clamp-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json). Videos:
`inspect-robots video <log>.json` writes one MP4 per camera; demo.mp4 and dream1.mp4 here put the top camera (2x) above
the two wrist cameras.

- **Episode end.** Once the padlock is shut through the lined-up holes and both grippers have stayed at least 6 cm
  clear of the padlock and of the valve stem for 2 s, the after-release test runs (4 s of simulated time) inside
  that control step, off camera, and the episode ends; the stored frames stop at the release. Otherwise the run ends
  at `--max-steps`.
- **Control run.** `--policy valve-clamp-demo -P control=true` closes V-12, fits the lockout and pushes its halves
  together, but leaves the padlock in its cradle, then forces the after-release test: it must score below 1.
- **Seeds.** One fixed layout (`-E object_jitter=0`); seeded jitter is not used.

## Measured

RTX-class GPU shared with other simulations, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz, 320x176:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `valve-clamp-demo` | `task_score: 1` (lever closed to 90.0 degrees; lockout seated, halves pushed to the first setting; shackle latched through the lined-up holes after 11 press steps; after-release test passed: lever moved 8.5 degrees under 3.0 N m, halves moved 2.9 mm apart under the 15 N pulls) | 2,309 (115 s), then the 4 s test off camera | about 29 minutes |
| Control: `valve-clamp-demo -P control=true` (closed, lockout fitted and pushed together, no padlock) | `task_score: 0.75` (after-release test fails: the halves slide 39 mm apart to the end of the slide and the lever swings 34.5 degrees toward open) | 1,596, then the test | about 18 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (the arms stay near their start for most of the run; late on, the left arm reaches toward the lockout; neither valve lever is turned) | 3,000 (step limit) | about 30 minutes (shared GPU; Dream's client retried about 60 times after "session expired") |

The demo's wall time is mostly its IK, which it solves in Python every control step, and the shared GPU; on an
otherwise idle machine expect less.

The scripted demo's steps: arm B (right) pinches V-12's grip and turns it to the closed stop; arm A (left) picks the
lockout up by the outside of A's two legs, carries it forward over the closed lever (the legs pass either side of
the lever, the channel above it), corrects its height and position closed loop, and lowers it so the lever lies in
the channel and the leg pairs straddle the pipe; arm A then lets go and squeezes the two halves together across the
pipe, as a hand does; arm B carries the open padlock on its side and lowers it so the housing's wall slides between
the body and the raised short leg; arm A's closed fingertips push the shackle shut.

## Layout

| Path | What |
|---|---|
| `ball_valve_clamp.py` | the task file: description for agents, the props and task, scoring |
| `clamp_valve_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera test) |
| `clamp_valve_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (turning and slide friction, shackle latch, decoy watch, after-release test) and milestones |
| `clamp_valve_sim/assets/`, `clamp_valve_sim/usd/` | the props, and the scripts that generate them from the dimensions in task.json (`python -m clamp_valve_sim.usd.valve_props clamp_valve_sim/assets`; the lockout is `usd/clamp_props.py`) |
| `clamp_valve_sim/demo.py`, `clamp_valve_sim/policy.py` | the scripted demo and its Inspect Robots policy `valve-clamp-demo` |

## References

- Master Lock S3080 adjustable ball valve lockout, product page: https://www.masterlock.com/business-use/product/S3080
- Master Lock S3080 sell sheet (dimensions, weight, valve range, four locking holes, removable insert):
  https://cdn.masterlock.com/masterlock/resources/documents/pdf/sell-sheets/7000-0206_S3080_Adjustable_Valve_Lockout_Sell_Sheet.pdf
- Master Lock S3079/S3080 sell sheet ("encompassing the ball valve handle", S806 cable through the legs):
  https://cdn.masterlock.com/masterlock/resources/documents/pdf/S3079-S3080_Ball_Valve_Lockout_SellSheet.pdf
- Master Lock S3081 (same design) sell sheet:
  https://cdn.masterlock.com/latinamerica/resources/documents/pdf/sell-sheets/7000-0214_S3068_SELL_SHEET.pdf
- How-to videos: "How to use an Adjustable Ball Valve Lockout - Master Lock No. S3080" (https://www.youtube.com/watch?v=zoxZA2biVk0),
  "Master Lock - Ball Valve Lockout (S3080)" (https://www.youtube.com/watch?v=kwMZiarQ514),
  "S3081 - Adjustable Ball Valve Lockout" (https://www.youtube.com/watch?v=U1MmlQAUoRo)
- Milwaukee Valve BA-100 bronze two-piece ball valve, submittal (dimensions):
  https://www.milwaukeevalve.com/wp-content/uploads/spec-sheets/BA100100S.pdf
- Milwaukee Valve two-piece threaded ball valve IOM (closes clockwise):
  https://www.milwaukeevalve.com/wp-content/uploads/ioms/TwoPieceThreadedBallValveIOM.pdf
- Milwaukee Valve actuation sizing charts (valve torque, used as a proxy):
  https://www.milwaukeevalve.com/wp-content/uploads/sizing-charts/ActuationSizingCharts.pdf
- Master Lock 410 Zenex thermoplastic safety padlock (410PRP): https://www.masterlock.com/business-use/product/410PRP
- OSHA 29 CFR 1910.147, control of hazardous energy:
  https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity
compensation). Rigid bodies on joints, with every link of the lock in geometry: the lever bears on the lockout's
channel walls, the lockout's legs bear on the pipe on both sides, and the padlock's shackle passes through B's wall
and A's tongue, so it stops the halves sliding apart. Nothing in the code holds the lever or the halves.

- **Valve.** The pipe run is low (lever 13.5 cm above the table) so the arms can carry the lockout over the pipe well
  inside their reach. The quarter-turn stops are joint limits.
  Turning resistance is stick-slip friction of 1.0 N m. That is under a quarter of Milwaukee's published 4.5 N m for a
  1/2 in two-piece valve at full rated pressure, lowered so one soft-wristed arm can turn the lever to its stop (at
  1.5 N m the scripted pinch stalled 3.6 degrees short); the real torque on an unpressurised valve is not measured. The lever is flat (the real one is formed), and collision shapes are boxes (the valve body is
  one box).
- **Lockout.** The outline (134.6 x 76.9 x 104.1 mm), mass (156 g), the four locking holes and the way the halves
  slide together are published; everything inside that outline is inferred from Master Lock's photos and how-to
  videos, not measured: 3 mm walls, a 26 mm wide channel for the lever's 19 mm grip (the real channel takes levers up to 2 in valves, and its
  removable insert narrows it for small ones; the insert is not modelled), 10 x 7 mm legs, 9 mm square holes at a 12 mm pitch, and a 3.5 mm tongue. The first setting is made to
  fit the 21.3 mm pipe with 0.85 mm a side; on a real valve the nearest hole can leave a few millimetres of play.
  The slide has 1 N of stick-slip friction (inferred). The lock holes go through a 3 mm wall and a 3.5 mm tongue
  (7 mm with the gap), so the 410's raised short leg (12 mm clear of its body) can be slid straight over them. The real housing
  looks deeper; a person turns or tilts the padlock to thread it.
- **Not checked.** The real S3080's legs are open at the bottom: pulled straight up off the valve it is held only
  by its grip on the pipe, and Master Lock suggests looping an S806 cable lockout through the leg holes for the most
  secure fit. The simulation has no clamp preload, so a lift-off pull is not part of the check, and the cable is not
  modelled.
- **Padlock.** A Master Lock 410 (body 38 x 45 x 19 mm, 6.35 mm shackle; Master Lock lockouts take shackles up to
  about 7 mm). Rounded visual body with a danger label over a box collider; the short leg's collider is a
  low-friction cross. Its shackle pops straight out and is pushed straight in (no swivel), with a 150 N/m spring and a
  catch that latches once it is pushed within 2 mm of shut; the key is not simulated.
- **Valve tags.** Plant valves usually carry small stamped brass tags (about 38 mm), which would be about 8 pixels
  across and unreadable in the 320 x 176 top camera, so each valve carries a larger engraved plastic tag on a strap,
  turned to face the robot; its size and placement are our choice.
- **Check forces.** The 15 N pulls and the 3.0 N m on the lever are our choice and are applied as forces in the
  simulator after the recording ends. One layout.

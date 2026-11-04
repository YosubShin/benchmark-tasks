# Close a named ball valve and padlock its locking handle

Two identical lockable ball valves sit on a short pipe run, each identified by an engraved valve tag (V-12, V-14).
The robot must close V-12 (a quarter turn to its stop), slide the handle's locking sleeve in so it latches, and
padlock the handle through its lock hole, without turning V-14. Scoring checks the lockout physically: after the
robot lets go, the sleeve is pulled outward and the handle is pushed toward open, and the valve must stay closed.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file (`ball_valve.py`)
in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy. Its props are
articulated (the two levers on their stems, the latch sleeves, the padlock's shackle), which the one-file format
cannot declare, so the file uses `valve_lockout_sim` (this folder), which adds them to the station at run time
(`valve_lockout_sim/station.py` documents exactly what it patches; it requires yam-isaac-lab-arena at commit
`0059d02`, the Brev launchable's pin).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/ball-valve-lockout --no-deps         # registers the policy valve-lockout-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/ball-valve-lockout/ball_valve.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Close ball valve V-12: turn its handle a quarter turn clockwise, seen from above, to its stop. Slide the handle's locking sleeve toward the valve so it latches, then put your padlock's shackle through the hole in the handle and snap it shut. Do not operate valve V-14." \
  --scorer task_score --max-steps 2500 --store-frames --log-dir ~/yam-eval/ball-valve-dream
```

The scripted demo (closed loop: it reads the lever angle, the sleeve position and the padlock's pose from the
simulator, like a scripted demo with privileged state, and sends the same 14-D joint action a policy does, through
yam_arena's IK):

```bash
inspect-robots run --embodiment yam_arena --policy valve-lockout-demo \
  -E task=tasks/ball-valve-lockout/ball_valve.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 2500 --store-frames --log-dir ~/yam-eval/ball-valve-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json). Videos:
`inspect-robots video <log>.json` writes one MP4 per camera (demo.mp4 and dream1.mp4 here put the top camera, at 2x, beside the two wrist cameras).

- **Episode end.** Once the padlock is shut through V-12's lever hole and both grippers have stayed at least 6 cm
  clear of it for 2 s, the after-release test runs (about 4 s of simulated time) inside that control step, off
  camera, and the episode ends; the stored frames stop at the release. Otherwise the run ends at `--max-steps`.
- **Control run.** `--policy valve-lockout-demo -P control=true` closes V-12 and latches its sleeve but leaves the
  padlock on its pedestal, then forces the after-release test: it must score below 1.
- **Seeds.** One fixed layout (`-E object_jitter=0`); seeded jitter is not used.

## Measured

RTX-class GPU shared with other simulations, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz, 320x176:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `valve-lockout-demo` | `task_score: 1` (lever closed to 89.3 degrees, sleeve fully in, shackle latched through the hole after 10 press steps; after-release test passed) | 1,554 (77.7 s), then the 4 s test off camera | about 8 minutes |
| Control: `valve-lockout-demo -P control=true` (closed and latched, no padlock) | `task_score: 0.5` (after-release test fails: the sleeve pulls out its full 10 mm and the lever swings back 90 degrees to open) | 698, then the test | about 8 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (the arms move about above the table but never touch a valve lever or the padlock). | 2,500 (step limit) | about 19 minutes (Dream round trip about 0.54 s per chunk; a few early "session expired" retries) |

The demo's wall time is mostly its IK, which it solves in Python every control step, and the shared GPU; on an
otherwise idle machine expect less.

## Layout

| Path | What |
|---|---|
| `ball_valve.py` | the task file: description for agents, the props and task, scoring |
| `valve_lockout_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera test) |
| `valve_lockout_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (turning friction, shackle latch, decoy watch, after-release test) and milestones |
| `valve_lockout_sim/assets/`, `valve_lockout_sim/usd/` | the props, and the scripts that generate them from the dimensions in task.json (`python -m valve_lockout_sim.usd.valve_props valve_lockout_sim/assets`) |
| `valve_lockout_sim/demo.py`, `valve_lockout_sim/policy.py` | the scripted demo and its Inspect Robots policy `valve-lockout-demo` |

## References

- Milwaukee Valve BA-100 bronze two-piece ball valve, submittal (dimensions):
  https://www.milwaukeevalve.com/wp-content/uploads/spec-sheets/BA100100S.pdf
- Milwaukee Valve BA-100 options, "LD" locking device (takes a standard 5/16 in padlock):
  https://www.milwaukeevalve.com/wp-content/uploads/spec-sheets/BA100100SP2OPT.pdf
- Milwaukee Valve two-piece threaded ball valve IOM (closes clockwise):
  https://www.milwaukeevalve.com/wp-content/uploads/ioms/TwoPieceThreadedBallValveIOM.pdf
- Milwaukee Valve actuation sizing charts (valve torque, used as a proxy):
  https://www.milwaukeevalve.com/wp-content/uploads/sizing-charts/ActuationSizingCharts.pdf
- Apollo 70-100 series submittal, option -27 latch-lock lever (same mechanism):
  https://image.stockpileexpress.com/ProductInfo/70-100_SS1023.pdf
- Master Lock 410 Zenex thermoplastic safety padlock (410PRP): https://www.masterlock.com/business-use/product/410PRP
- OSHA 29 CFR 1910.147, control of hazardous energy:
  https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation). Rigid bodies on joints, with every link of the locking chain in geometry: the sleeve's tongue can only drop into the stop plate's notch at a stop, a latched tongue stops the lever from turning, the shackle's leg in the lever hole stops the sleeve from sliding back out, and the sleeve covers the hole while it is out. The quarter-turn stops are joint limits, and turning resistance is a constant 1.5 N m friction-like torque. That is a third of Milwaukee's published 4.5 N m for a 1/2 in two-piece valve at full rated pressure, lowered so one arm can turn the lever; the real torque on an unpressurised valve is not measured. The lever is flat (the real LD lever rises on a sloped leg), and the hole, sleeve, notch and tongue sizes are inferred from Milwaukee's line drawing and photos, not measured. Collision shapes are boxes (the valve body is one box). The padlock is a Master Lock 410 (1/4 in shackle; Milwaukee specifies a 5/16 in shackle, so the 410 fits with room). It follows the 410's published outline: rounded 38 x 44 x 19 mm body with a danger label, round 6.35 mm shackle with a semicircular bend (collision: square legs and six straight segments around the bend), the short leg's collider a low-friction cross. Its shackle pops straight up and is pushed straight down (no swivel), with a 150 N/m spring and a catch that locks it once it is pushed within 2 mm of shut; the key is not simulated. Valve tags: plant valves usually carry small stamped brass tags (about 38 mm), which would be about 8 pixels across and unreadable in the 320 x 176 overhead camera, so each valve carries a larger engraved plastic tag on a strap, turned to face the robot; its size and placement are our choice. The 10 N and 32 N check forces are our choice and are applied as forces in the simulator after the recording ends. One layout.

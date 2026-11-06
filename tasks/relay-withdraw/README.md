# Withdraw the named drawout relay

Two protection relays (ABB Relion 615 series, REF615 class) are flush-mounted side by side in a steel panel in front
of the robot. Each relay's plug-in unit (2.1 kg, 177 mm square front) is latched in its case by a bail handle. The
robot must take out the plug-in unit of the relay labelled FEEDER 2 (yellow marker on its engraved label): lift the
handle to 90 degrees, which releases the latch and pushes the unit about 7 mm out of its case (parting the rear
connectors), pull the unit out by the handle with one hand while the other hand goes under the body to carry its
weight, and set it on the black mat in front of the panel. It must not touch the FEEDER 1 relay and must not reach
into the empty case. Scoring checks the result physically: after release, the withdrawn unit must rest upright on
the mat, and a 30 N pull on the FEEDER 1 unit must show it still latched.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`relay_withdraw.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy.
Its props are articulated (each plug-in unit is a free body with its handle on a hinge; the latch and the eject cam
are geometry), which the one-file format cannot declare, so the file uses `relay_sim` (this folder), which adds them
to the station at run time. `relay_sim/station.py` documents exactly what it patches. It requires
yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin.

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/relay-withdraw --no-deps              # registers the policy relay-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/relay-withdraw/relay_withdraw.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take out the plug-in unit of the relay labelled FEEDER 2 (the label with the yellow marker): lift its handle up to release the latch, pull the unit out of its case by the handle with one hand while the other hand goes under the body to carry its weight, and set it down on the black mat in front of it. Do not touch the other relay, and do not reach into the empty case." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/relay-dream
```

The scripted demo is closed loop. Like a scripted demo with privileged state, it reads the handle's angle and the
unit's pose from the simulator, and it sends the same 14-D joint action a policy does, through yam_arena's IK:

```bash
inspect-robots run --embodiment yam_arena --policy relay-demo \
  -E task=tasks/relay-withdraw/relay_withdraw.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/relay-demo
```

`task_score` prints the trial's score: 1.0 on success, otherwise the partial credit given in task.json. To make
videos, run `inspect-robots video <log>.json`, which writes one MP4 per camera.

- **Episode end.** The after-release check runs once the unit rests upright on the mat, wholly out of its case, and
  both grippers have stayed at least 1.5 cm clear of it for 2 s. It lasts 1.5 s of simulated time, runs inside that
  control step and off camera, and then the episode ends. The stored frames stop at the release. If the check never
  runs, the run ends at `--max-steps`.
- **Control run.** `--policy relay-demo -P control=true` keeps both arms at the ready pose. At 2 s the task pulls the
  FEEDER 2 unit by its body with 40 N with the handle down for 2 s, lets go for 2 s, lifts the handle with a
  1.5 N m torque for 2 s, then keeps the torque on while it pulls with 15 N until the unit is 10 cm out (2 s), and
  ends the episode. The latch must hold the first pull (the unit moves only its 5 mm of play), the lift must release
  it, and with no hand under it the unit falls off the case floor, so the run must score below 1.
- **Seeds.** One fixed layout (`-E object_jitter=0`). Seeded jitter is not used.
- **Runtime per trial.** See "Measured" below. Isaac Sim's first start on a new machine adds several minutes.

## Measured

RTX-class GPU shared with several other simulators, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz,
320x176, 240 Hz physics:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `relay-demo` | `task_score: 1`. Arm A lifted the handle to 89 deg (latch released), pulled the unit 12 cm out by the handle; arm B slid its closed fingers under the body and lifted it 1 mm; both withdrew it to 18 cm, held it out of the case for 3 s, set it on the mat and let go. Check: 30 N on the FEEDER 1 unit moved it 5.0 mm (its latch play) and it stayed latched. | 1,072 (53.6 s), then the 1.5 s check off camera | about 25 minutes |
| Control: `relay-demo -P control=true` | `task_score: 0`. 40 N on the FEEDER 2 body with the handle down: moved 5.0 mm (latch play) and held. 1.5 N m lift: handle to 92 deg, unit cammed 16 mm out. 15 N pull: unit 20 cm out, no hand under it, dropped (hazard: score 0). | 200 (10 s) | about 3 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0`. The arms move toward the panel; around 60 s the left gripper reaches up to the front of the FEEDER 1 relay (the one not named), then both arms drift back toward the table. Neither handle is lifted. 49 expired-observation retries; mean inference latency 0.50 s. | 2,400 (step limit) | about 28 minutes |

Joint torques while the demo held the unit out of the case (3 s, mean / peak, N m; i2rt's gains, includes the arms'
own weight): arm A (handle) shoulder joints 1.9 / 2.1, 5.8 / 6.1, 7.0 / 7.5, wrist joints 0.7-1.0 / at most 1.2;
arm B (under the body) shoulder joints 0.3 / 0.6, 11.5 / 11.7, 3.7 / 4.1, wrist joints 2.1 / 2.2 at most. The
wrists stay under the DM4310's roughly 3 N m continuous rating; arm B's shoulder joint 2 is above the DM4340's roughly
9 N m continuous rating (both ratings from reseller listings), so the real-arm feasibility of the 2.1 kg carry is
marginal at that joint. The demo adds a bounded integral term to its joint targets while carrying (arm B's wrist
term reached its bound, so the unit rode about 1 cm lower than commanded).

The wall times are long because the GPU was shared; the demo's own cost is two arms' Python IK every control step.

## Layout

| Path | What |
|---|---|
| `relay_withdraw.py` | the task file: description for agents, props and task, scoring |
| `relay_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera check) |
| `relay_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (connector retention rule, hazard watch, after-release check, control sequence), milestones, layout and instruction |
| `relay_sim/assets/*.usda`, `label_*.png`, `relay_sim/usd/` | the props and label textures, and the script that generates them (`python -m relay_sim.usd.relay_props`) |
| `relay_sim/demo.py`, `relay_sim/policy.py` | the scripted demo and its Inspect Robots policy `relay-demo` |

## References

- ABB 615 series Installation Manual 1MRS756375 (section 4.2.1 detaching and installing the plug-in unit, figs. 2-3;
  fig. 41 and table 5 dimensions and weights: plug-in unit 2.1 kg, frame 177 x 177 mm, 153 mm behind the panel):
  https://library.e.abb.com/public/71f77e79135977bbc1257b130056cac1/RE_615_inst_756375_ENg.pdf
- ABB REF615 ANSI Product Guide (draw-out unit, case dimensions, cover photo of a unit half out with its handle up):
  https://library.e.abb.com/public/d87d7419da3a2634c1257b130056a497/REF615_Product_Guide_ANSI.pdf
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation of the arms' own links). Published, from ABB's 615 series installation manual: the 177 x 177 mm frame, the case's 164 x 160 mm section and 153 mm depth behind the panel, the panel cut-out, the plug-in unit's 2.1 kg, and the procedure (lift the handle to 90 degrees, the unit is pushed about 7 mm out and its connectors separate, pull it out; nothing stops it once out). Inferred, to be measured on a real unit: the body inside the case (155 x 151 x 147 mm, 2 mm clearance a side), the 24 mm frame thickness, the handle's pivot (5 cm below the frame top, 1.2 cm in front of the panel), its 12.2 cm arms, 12 mm round cross-bar and 0.1 kg, and the friction of the unit on the case floor (0.30 static, 0.25 dynamic, painted steel). The latch and the eject cam are inside the real frame and not published; here they are geometry of our own, beside the frame's sides: a lug on each handle hub drops behind a fixed catch (5 mm of play when pulled, self-locking), and a tooth on each hub, just outside the cut-out, bears on the panel's front face from about 30 degrees of lift, pushing the unit out about 7 mm by 90 degrees. The rear connectors are a rule, not geometry: while the unit is within 6 mm of seated, a force pulls it back in (10 N rising to 30 N at 6 mm, inferred; the real retention is not published), and past 6 mm they count as parted. The handle has no spring: gravity holds it down. Collision shapes are boxes (the cross-bar is a convex cylinder). The HMI is coloured patches, and the relays have no power or wiring (the hazard of the live empty case is scored as a rule: no finger inside the case opening). The designation labels are textured plates; at the top camera's 320x176 their letters are about 3 px tall and not legible, while the 3.4 cm yellow and blue markers are clearly visible; the wrist cameras can read the letters close up. The mat is split into two strips so a supporting hand's fingers can come out from under the set-down unit; a real bench would need a slot or a block to the same effect. Carrying the real 2.1 kg: the scripted demo succeeds at full mass under the station's soft gains, with one arm on the handle and the other's fingers under the front half of the body (it cannot reach farther back), adding an integral term to its joint targets while it carries; measured joint torques while holding the unit out of the case for 3 s are given in the README (the wrists stay at or below about 2.2 N m, inside the DM4310's roughly 3 N m continuous rating; the supporting arm's shoulder joint 2 carries about 11.5 N m including the arm's own weight, above the DM4340's roughly 9 N m continuous rating, from reseller listings). In the simulator the robot's lift reaches only about 4.5 mm of the 7 mm cam stroke (the tooth sinks into the panel face under the arm's push; not resolved), so the connectors are parted by the following pull rather than by the cam alone; the cam does push the unit out when the handle is lifted by a torque alone (control run). The real cross-bar in rubber finger pads may slip more than the simulated one, which bears on a faceted collision cylinder. The 30 N check pull is our choice; it is applied as a force in the simulator after the recording ends. One layout.

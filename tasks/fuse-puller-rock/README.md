# Rock a named midget fuse out of its open-clip block with a tong puller

A control sub-panel's back pan lies on the bench with three Eaton Bussmann BM6031B open midget fuse blocks
(F1, F2, F3, colour-coded circuit labels). Each 10 x 38 mm fuse is held by two spring clips at its ferrules. The
robot must pick up a one-piece nylon tong fuse puller (Ideal 34-015 class), close its jaws round F2's body, and
take F2 out the way the puller makers describe: lever one end out of its clip first, with the other clip as the
pivot, then the other end, and drop it in a parts tray, without disturbing F1 or F3. Disturbing F1 or F3 scores 0 for the whole trial; taking
F2 out of its clips without the puller holding it caps the score at 0.5 for the rest of the trial.

The clips grip hard enough that a straight pull (both ferrules at once) needs about 70 N, more than the robot delivers
through the puller: gripping the puller over its jaws with the full 50 N and commanding the hand up to 16 cm straight
up, the fingers slide off the puller's smooth arms at about 27 N of pull and the fuse does not move. Levered out one end first, with the puller held by its handles as a lever about the far clip,
the near ferrule comes out with the hand lifting about 12-15 N (0.9-1.2 N m over the 78 mm from the far clip to
the hand); the far ferrule then comes out as the puller swings
up and the lift pulls on that one clip.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`fuse_puller_rock.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy.
Its props are articulated (the puller's two flexing arms), and the clips use PhysX compliant contacts, which the
one-file format cannot declare, so the file uses `fuse_puller_sim` (this folder), which adds them to the station at
run time (`fuse_puller_sim/station.py` documents exactly what it patches; it requires yam-isaac-lab-arena at commit
`0059d02`, the Brev launchable's pin). The task runs PhysX at 480 Hz (the policy still acts at 20 Hz).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/fuse-puller-rock --no-deps            # registers the policy fuse-puller-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/fuse-puller-rock/fuse_puller_rock.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The panel is de-energized. Pick up the yellow fuse puller, grip fuse F2 (CONTROL XFMR, the fuse with the orange label) in its jaws, lift the puller's handles to lever the near end of the fuse out of its clip first, then the far end, and drop the fuse in the blue parts tray. Do not disturb fuses F1 and F3." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/fuse-puller-dream
```

The scripted demo is closed loop. Like a scripted demo with privileged state, it reads the fuse's and the puller's
poses from the simulator, and it sends the same 14-D joint action a policy does, through yam_arena's IK:

```bash
inspect-robots run --embodiment yam_arena --policy fuse-puller-demo \
  -E task=tasks/fuse-puller-rock/fuse_puller_rock.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/fuse-puller-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json; at most 0.5 once
F2 has left its clips without the puller's jaws on it; 0 once F1 or F3 has moved; see "Scoring" below). Videos: `inspect-robots video <log>.json` writes one MP4 per camera; `demo.mp4` and `dream1.mp4` here
put the top camera (2x) above the left and right wrist cameras.

- **Episode end.** Once F2 lies in the tray (out of both clips, carried there in the jaws) the episode ends on
  success (after 0.5 s); otherwise at `--max-steps`.
- **Scoring.** `fuse_puller_sim/scorer.py` (`PullScorer`) is the scoring implementation; the task file's `score()`
  returns its `report()["score"]` unchanged. Milestones (0.25 / 0.5 / 0.75 / 1.0) are counted as progress. The cap
  (F2 out of both clips with the jaws not on it, before 0.75) is latched the step it occurs: from then on the score
  is at most 0.5, even if the puller grips F2 afterwards and it reaches the tray. The hazard (F1 or F3 moved 2 mm or
  tilted 4 degrees) is latched the same way: from then on the score is 0. The scored milestone is
  `0 if hazard else min(progress, cap)`; the report also gives `progress_milestone`, `cap`, `cap_step`,
  `hazard_step` and, unscored, `other_events`. `python -m fuse_puller_sim.check_scorer` checks the gate and the cap
  without the simulator; the simulator checks are under "Hazard checks". A trial with a hazard or the cap never
  reaches success, so it runs to `--max-steps`.
- **Control run (a straight pull fails).** `FPR_CONTROL=1` with `--policy fuse-puller-demo`: the demo puts the
  puller's jaws on F2 exactly as in the demo, then lets go of the handles, pinches the puller's arms directly
  over the jaws with the gripper's full 50 N (which also clamps the jaws on the fuse) and pulls straight up,
  commanding up to 16 cm above the start. It must score below 1.
- **Seeds.** One fixed layout (`-E object_jitter=0`). Seeded jitter is not used.
- **Runtime per trial.** See "Measured" below. Isaac Sim's first start on a new machine adds several minutes.

## Measured

RTX-class GPU shared with three other Isaac Sim processes, shader cache warm, `-E object_jitter=0`, 20 Hz, 320x176
frames, PhysX at 480 Hz:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `fuse-puller-demo` | `task_score: 1`. Puller picked up by its handles 85 mm from the loop (light pinch, jaws open), lowered over F2 and seated within 1 mm, squeezed; lifting the handles along an arc about the far clip levered the near ferrule out first (the fuse tilted 20 degrees when the demo checked, after lifting 10 mm above the arc); swinging on to 60 degrees and lifting freed the far ferrule; F2 carried in the jaws and dropped in the tray. F1 and F3 untouched | 828 (41.4 s) | about 16 minutes |
| Control: `FPR_CONTROL=1`, `fuse-puller-demo` (straight pull over the jaws) | `task_score: 0.5`. Jaws seated and squeezed on F2 as in the demo; pinched over the jaws, the straight pull built up to about 27 N (estimated from the arm's joint torques, 8 cm behind its target) and then the fingers slid off the smooth arms; F2 never moved (0.0 mm) | 1,300 (step limit) | about 25 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (the right hand reaches toward the puller's rest and the tray early on, then both arms stay near their start; the puller is never picked up and no fuse is touched) | 3,000 (step limit) | about 51 minutes (shared GPU; Dream reported an observation as expired and was asked again 61 times; mean inference latency 0.55 s) |

The Dream-1 run was made before the handle friction was split (the arms' friction to the gripper was 0.55 everywhere
then; scene, cameras, instruction and scoring unchanged); it never touched the puller.

The demo's wall time is mostly the doubled physics rate, its Python IK every control step and the shared GPU. The
demo is closed loop (it lifts further while the fuse does not turn) because the arm's soft gains leave the hand
20-30 mm behind its target while it levers.

## Hazard checks

Each check is the official command above with `--policy fuse-puller-demo`, seed 0 (`-E object_jitter=0`), and one
environment variable, `FPR_HAZARD_TEST=<case>`, read by `fuse_puller_sim/task.py` (empty in every scored run). The
test harness moves a fuse by writing its pose (a stand-in for a hand); the demo otherwise runs unchanged, so the
later milestones are still reached physically and only the reported score shows the gate. `--max-steps 1100` (a trial
with a hazard or the cap never ends on success).

| Case | What happens | Progress reached | Reported `task_score` |
|---|---|---|---|
| none (the demo) | | 1.0 | 1 |
| `decoy_move_start` | F1 slid 4 mm along its clips at 1 s, before any milestone | 1.0 (puller picked up, jaws on F2, F2 levered out and put in the tray) | 0 |
| `decoy_move_75` | F3 slid 4 mm the step F2 is out of its clips in the jaws (after 0.75) | 1.0 | 0 |
| `out_by_hand_25` | F2 lifted 25 mm out of both clips with no jaws on it the step the puller is picked up (after 0.25), put back in its clips on the next step; the demo then grips it with the puller, levers it out and puts it in the tray | 1.0 | 0.5 |

`python -m fuse_puller_sim.check_scorer` (no simulator) checks the same gate for a decoy moved before any milestone
and after 0.25, 0.5 and 0.75 (final score 0 each time, with every later milestone reached), and the cap for F2
taken out without the puller before any milestone, after 0.25 and after 0.5 (final score 0.5, with the puller then
reaching 0.75 and 1.0; 0 if a decoy is moved afterwards).

## Layout

| Path | What |
|---|---|
| `fuse_puller_rock.py` | the task file: description for agents, the props and task, scoring |
| `fuse_puller_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, physics rate) |
| `fuse_puller_sim/task.py`, `scorer.py`, `spec.py` | props, ferrule-out and decoy checks, milestones, zero gate and cap, layout and instruction |
| `fuse_puller_sim/check_scorer.py` | the scorer's zero-gate and cap check without the simulator |
| `fuse_puller_sim/assets/`, `fuse_puller_sim/usd/` | the props (`puller_props.py`: pan, BM6031B blocks with compliant clip fingers, fuses, puller, rest, tray), generated with `python -m fuse_puller_sim.usd.puller_props fuse_puller_sim/assets` |
| `fuse_puller_sim/demo.py`, `fuse_puller_sim/policy.py` | the scripted demo and its Inspect Robots policy `fuse-puller-demo` |

## References

- Eaton Bussmann BM series supplementary fuse blocks (BM6031B: 76.2 x 21.0 mm footprint, 30.7 mm high), data sheet
  1104: https://elliottelectric.com/Media/BM6031B-BUS
- Eaton Bussmann BMM603-1PQ modular midget block (successor of the BM6031), cutsheet ("clip with reinforced spring")
  and drawing: https://cdn.automationdirect.com/static/specs/cutsheet/BMM603-1PQ_cutsheet.pdf ,
  https://cdn.automationdirect.com/static/drawings/BMM603-1PQ.pdf ; STEP model (clip positions, finger gap, wall
  heights measured on it): https://ftp.automationdirect.com/support/drawings/3d/step/BMM603-1PQ.STEP
- Littelfuse POWR-GARD LF series midget fuse blocks (L60030M), dimensions:
  https://shop.lakelandengineering.com/products/L60030M-1PQ
- Eaton Bussmann KTK fuse, data sheet 1011 (10.3 x 38.1 mm): https://static.rapidonline.com/pdf/566819_v1.pdf
- Schurter ASO 10.3x38 midget fuse, dimension drawing: https://www.schurter.com/en/datasheet/typ_ASO_10.3x38.pdf
- Weidmueller 10x38 fuse 1224910000 (9 g): https://de.misumi-ec.com/pdf/ds/WML1/1224910000_FUSE_1KV_DC_10A_10_38_de.pdf
- Ideal 34-015 fuse puller (5 in, one-piece high-impact nylon, for 1/4-1/2 in fuses):
  https://www.testequipmentdepot.com/ideal-electrical-34-015-5-in-fuse-puller.html ,
  https://www.steinerelectric.com/Product/Ideal-34-015-Small-Fuse-Puller1-4-1-2-in-Dia-Fuse5-in-OALNotched-HandleHigh-Impact-Nylon-23320
- US patent 4541311 (fuse puller, assigned to the fuse maker): hook under one end of the fuse, lift that end out
  of its clip, then remove the other end "with a lever type motion": https://patents.google.com/patent/US4541311
- US patent 4288138 (fuse puller): removing a cartridge fuse "requires some force usually between 20 and 40 pounds
  depending on the fuse and clip size": https://patents.google.com/patent/US4288138
- OSHA 29 CFR 1910.335(a)(2)(i)(A), fuse handling equipment:
  https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.335
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 480 Hz in Isaac Lab-Arena (the station's 240 Hz doubled, per task, with friction anchors per contact point:
`friction_correlation_distance` 0.5 mm, `friction_offset_threshold` 2 mm; at PhysX's default 25 mm one friction patch
covers a whole clip or jaw and caps its grip), both arms driven as in Dream's station (joint PD with i2rt's gains and
gravity compensation). Rigid bodies; collision shapes are boxes and convex prisms.

**What holds the fuse is geometry plus a contact spring.** Each clip is two fixed finger faces standing 0.7 mm inside
the 10.3 mm ferrule (the BMM603-1PQ STEP model: 4.45 mm finger half-gap against a 5.15 mm ferrule radius), from just
above the ferrule's equator to the clip top. The fingers use PhysX compliant contacts (a spring of 60,000 N/m and
30 N s/m on the penetration), which stands in for the finger's spring deflection; finger-to-ferrule friction is 0.4
(tinned copper on nickel). Nothing else holds the fuse: no joint, no scripted force. Measured with ramped forces on
the fuses, robot idle (`probe_clips.py`, at 480 Hz: about 70 N pulling the fuse's middle straight out (69-71 N on the three fuses), 32-35 N pulling at one ferrule, and 0.9-1.2 N m turning it about its middle). A sprung finger on a revolute joint (the first model) did not
work: PhysX let the link sink into the ferrule and the squeeze never passed about 3.6 N. The real pull-out force of
a BM/BMM clip is not published; puller patents quote 90-180 N for cartridge fuses in general, so the simulated clip
is weaker than many real ones (chosen so that the station arm can lever it out through the puller and cannot pull it
straight).

**The puller.** An Ideal 34-015 class one-piece tong: a hinge body (the loop) and two arms on revolute joints with a
spring drive (5 N m/rad, 8 degrees free opening) standing in for the nylon's flex. Each arm carries a curved jaw shell
(six flat facets) 13-27 mm from the loop that wraps the fuse body from 81 degrees above to 35 degrees below its
equator; the arms run above the block's housing, parallel to the fuse, as in the reference video. Squeezed by the
gripper's 50 N 85 mm from the loop, the jaws clamp the body with about 175 N. Inferred: jaw shape, length and
position, flex and opening (Ideal publishes no drawing); 34 g against about 20 g real (lighter articulations broke
under the 50 N pinch in an earlier build); arms 12 mm tall strips. The jaw-to-body friction is 1.0 (combined as the
larger), standing in for a firm grip of the moulded jaws on the fibre tube: at 0.6 the jaws turned on the body when
levering. The handles are notched from 50 mm behind the loop to their ends (the 34-015 is sold with a "notched
handle"): the gripper's fingers grip them with friction 0.8; on the smooth moulded arms between the loop and the
notches they grip with 0.25 (inferred). The puller is held by friction alone; nothing attaches it to the hand or the
fuse.

**The rock.** Lifting the handles (the gripper's pinch acts as a pin) swings the puller and the fuse about the far
ferrule, so the near ferrule leaves its clip first with a lever of about 78 mm from the far clip to the hand; the
scripted demo then keeps swinging the puller up and lifting, and with the near end free the lift ends up pulling
almost straight up on the far clip alone. A straight pull on both clips at once (the control) is beyond what the
robot can transmit: gripping the puller over its jaws with the full 50 N and commanding the hand up to 16 cm up, the
fingers slid off the smooth arms at about 27 N with the fuse unmoved. (Before the handle friction was split, the same
pinch held to about 51 N and, commanded 9 cm up, pulled the fuse straight out; the smooth arms near the jaws are what
keep a straight pull through this puller below the clips' 70 N.) Held by its handles, the puller cannot be pulled
straight by the robot at all (the pinch is a pin, so any lift at the handles rocks the fuse), which is why the
control grips over the jaws.

**Layout and looks.** The back pan lies flat on a 25 mm board on the bench (a sub-panel serviced on the bench), so
fingers-down grasps reach it and the circuit labels face the top camera. The blocks stand 1.5 mm apart (BM6031B
blocks gang side by side). Label text is 2-3 px at 320x176, so each label carries a circuit colour (F1 white, F2
orange, F3 green), a plant labelling choice, not a standard. Block terminals, screws and the fingers' curved roots
and reinforcing springs are visual only. One layout.

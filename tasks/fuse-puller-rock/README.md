# Rock a named midget fuse out of its open-clip block with a tong puller

A small wall-mount control enclosure (Rittal AE 1036.500, 300 x 300 x 155 mm steel) lies on its back on the bench
with its door open. Inside, three Eaton Bussmann BM6031B open midget fuse blocks (F1, F2, F3) sit on a DIN rail on
the mounting plate between two wire ducts, each with a 14 AWG wire landed on its line and load box lugs and a
colour-coded circuit label on the duct cover above it. Each 10 x 38 mm fuse is held by two spring clips at its
ferrules. The robot must pick up a one-piece nylon tong fuse puller (Ideal 34-015 class) standing in its holder on
the bench beside the enclosure, lift it over the enclosure's wall, hold it as a tong puller is used, perpendicular to the fuse (pointing down into the block, its jaws closing across F2's body from
either side between the clips), and take F2 out the way the puller makers describe: lever one end out of its clip
first, with the other clip as the pivot, then the other end, carry it out of the enclosure and drop it in a parts
tray beside it, without disturbing F1 or F3. Disturbing F1 or F3 scores 0 for the whole trial; taking F2 out of its clips without the puller holding it caps
the score at 0.5 for the rest of the trial.

The clips grip hard enough that a straight pull (both ferrules at once) needs about 70 N. With the puller's jaws
closed across the body, tipping the handles away from the robot turns the puller and the fuse about the far clip:
the near ferrule comes out of its clip first, then a lift draws the far ferrule out of its clip alone (about 33 N).
Pulled straight up through the puller instead (the control), the fuse stays in: pinched on the smooth arms just
above the jaws with the gripper's full 50 N, the fingers slide up the arms at 20-30 N of pull.

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
  --instruction "The control panel is de-energized and its door is open. Pick up the yellow fuse puller from its holder beside the enclosure, reach into the enclosure and grip fuse F2 (CONTROL XFMR, the fuse with the orange label) across its body in the puller's jaws, holding the puller perpendicular to the fuse, tip the puller's handles away from you to lever the near end of the fuse out of its clip first, then lift out the far end, and drop the fuse in the blue parts tray beside the enclosure. Do not disturb fuses F1 and F3 or the wiring." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/fuse-puller-dream
```

The scripted demo, `fuse-puller-demo`, is closed loop on robot-observable inputs only (see "What the demo reads"
below). It sends the same 14-D joint action a policy does, through yam_arena's IK:

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
  puller's jaws across F2 exactly as in the demo, then lets go of the handles, pinches the puller's smooth arms
  just above the jaws with the gripper's full 50 N (fingers 50 degrees down, to keep the gripper's housing clear of
  the enclosure's near wall) and pulls straight up, without tipping, commanding up to 16 cm
  above the start. It must score below 1.
- **Seeds.** One fixed layout (`-E object_jitter=0`). Seeded jitter is not used.
- **Runtime per trial.** See "Measured" below. Isaac Sim's first start on a new machine adds several minutes.

## What the demo reads

`fuse-puller-demo` (`fuse_puller_sim/policy.py`, `demo.py`, `vision.py`, `layout.py`) receives exactly what
`yam_arena` gives any policy each step and nothing else: the RGB images `top_cam`, `left_cam`, `right_cam` (320x176)
and the 14-D `joint_pos` state (6 joint angles and the gripper opening per arm). It does not read the simulator: no
fuse or puller pose, no record of which fuse is named, no joint targets, no scorer state. Fixed knowledge it uses is
what a real robot's script would have: the documented setup in task.json (`fuse_puller_sim/layout.py`: the enclosure's walls, the
blocks, the fuses' seats and clips, the labels, the puller's holder and its geometry, the tray), the station's published camera
calibration (`yam_arena.station`) and the arm's URDF (`yam_arena.urdf`, `yam_arena.ik`).

| Thing | How the demo gets it |
|---|---|
| Which fuse is F2 | first top-camera frame: the colour of the three circuit labels on the far wire duct's cover above the blocks (white, orange, green); the block with the orange label is F2's |
| The puller | standing in its documented holder beside the enclosure; the same frame checks the yellow puller is there |
| The enclosure | its documented walls (rim 15.5 cm above the table): the puller is lifted until its jaw tips are 25 mm above the rim before it is carried over the right wall to F2, and the fuse in the jaws is lifted the same way before it is carried out to the tray |
| The puller in the hand | where arm B's fingers closed on its handles (forward kinematics of arm B's measured joints) relative to its documented pose in the holder; from then on its pose follows arm B's joints |
| Jaws across F2 | F2 in its documented clips; the puller pointing straight down over it; lowering, arm B's joint readings show when the jaws' hood stops on the fuse body, and it pushes a little further before squeezing |
| How far the fuse has been levered | the puller and the fuse turn together about F2's far ferrule (documented), so the tip is the angle of the pinch point (arm B's joints) about that ferrule. The hand is moved along that arc, turning with the puller, a few degrees ahead of the measured angle, further while it stalls; past 16 degrees the near ferrule (28 mm from the pivot) is out of its clip, and the demo tips on to 40 degrees while lifting 30 mm |
| Far end out, over the tray, dropping | turning the puller upright again while moving it up and a third of the way toward the documented tray draws the far ferrule out; the fuse is then lifted clear of the rim, carried over the right wall, and the jaws are placed over the tray from the puller's geometry; the squeeze is relaxed and the jaws open |
| Control run's pulling force | arm B's joint readings and the joint targets the demo itself sent (as before, through the arm's Jacobian) |

In the run below the top camera read the circuit labels as white, orange, white (the green one reads as white at
this resolution; only the orange one has to be found) and saw the puller in its holder; the puller's hinge sat at
(-34.4, -0.1, 21.3) mm in arm B's hand frame, and arm B's joints put it 113.6 mm above F2's centre when the jaws
closed (114 mm wanted). The tip read from arm B's joints climbed slowly to about 15 degrees while the near ferrule
held, then jumped to 18.2 degrees as it came out of its clip.

## Measured

RTX-class GPU shared with up to five other Isaac Sim processes (this task's own runs), shader cache warm, `-E object_jitter=0`, 20 Hz,
320x176 frames, PhysX at 480 Hz:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `fuse-puller-demo` (cameras + joint states only) | `task_score: 1`. F2 found by its orange label; puller lifted out of its holder beside the enclosure by its notched handles 40 mm from the loop (light pinch, jaws open), lifted until its jaw tips cleared the rim, carried over the right wall pointing straight down to above F2, jaws lowered across F2's body between the clips and squeezed; tipping the handles away from the robot about the far clip levered the near ferrule out first (the tip, read from arm B's joints, jumped to 18.2 degrees as it came out); tipped on to 40 degrees and lifted, then turned upright while moving up and toward the tray, which drew the far ferrule out; F2 lifted in the jaws over the right wall, carried over the tray and dropped in it. F1 and F3 untouched | 1,388 (69.4 s) | about 49 minutes (five other simulations running) |
| Control: `FPR_CONTROL=1`, `fuse-puller-demo` (straight pull, smooth arms) | `task_score: 0.5`. Jaws seated and squeezed across F2 as in the demo; pinched on the smooth arms just above the jaws (fingers 50 degrees down: at 30 degrees the gripper's housing ran into the enclosure's near wall and never reached the arms, a first attempt we discarded), the straight pull slid the fingers up the arms (the hand rose 23 mm while commanded 13 cm, rising to about 30 N estimated from the arm's joint torques); F2 stayed in its clips (moved 0.4 mm, tilted about 1 degree) | 1,500 (step limit) | about 29 minutes (two other simulations running) |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`, the instruction above) | `task_score: 0` (both arms rise and hover over the enclosure, the right hand near the puller's holder and the panel; the puller is never picked up and no fuse is touched) | 3,000 (step limit) | about 82 minutes (GPU shared with up to five other simulations; Dream reported an observation as expired and was asked again 61 times; mean inference latency 0.70 s) |

The demo's wall time is mostly the doubled physics rate, its Python IK every control step and the shared GPU. The
demo is closed loop (it tips further ahead while the fuse does not turn) because the arm's soft gains leave the hand
15-20 mm behind its target while it levers.

## Hazard checks

Each check is the official command above with `--policy fuse-puller-demo`, seed 0 (`-E object_jitter=0`), and one
environment variable, `FPR_HAZARD_TEST=<case>`, read by `fuse_puller_sim/task.py` (empty in every scored run). The
test harness moves a fuse by writing its pose (a stand-in for a hand); the demo otherwise runs unchanged, so the
later milestones are still reached physically and only the reported score shows the gate. `--max-steps 1600` (a trial
with a hazard or the cap never ends on success).

| Case | What happens | Progress reached | Reported `task_score` |
|---|---|---|---|
| none (the demo) | | 1.0 | 1 |
| `decoy_move_start` | F1 slid 4 mm along its clips at 1 s, before any milestone | 1.0 (puller picked up, jaws on F2, F2 levered out and put in the tray) | 0 |
| `decoy_move_75` | F3 slid 4 mm the step F2 is out of its clips in the jaws (after 0.75); the demo then drops F2 in the tray | 1.0 | 0 |
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
| `fuse_puller_sim/assets/`, `fuse_puller_sim/usd/` | the props (`puller_props.py`: enclosure with door, mounting plate, DIN rail, wire ducts and wires, BM6031B blocks with compliant clip fingers, fuses, puller, holder, tray), generated with `python -m fuse_puller_sim.usd.puller_props fuse_puller_sim/assets` |
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
- Rittal AE 1036.500 compact enclosure, 300 x 300 x 155 mm, data sheet (material thicknesses, mounting plate):
  https://www.rittal.com/pdf-creator/variant/us-en/1036500
- Eaton Bussmann BM6031B (DIN-rail mounting, box lugs for 14-6 AWG):
  https://www.crescentelectric.com/product/88694/eaton-bussmann-series-fuse-block-01-30a-600-vac-600-vdc-din-rail-mounting-box-lug-connection-10-kaic-interrupt-rating-14-to-6-awg-copper-wire-size-used-with-ktk-fnq-fnm-baf-pv-and-agu-fuse-thermoplastic-material
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

**The puller and how it is held.** An Ideal 34-015 class one-piece tong: a hinge body (the loop) and two flat arms
on revolute joints with a spring drive (8 N m/rad, 4 degrees free opening) standing in for the nylon's flex. A tong
fuse puller is used perpendicular to the fuse: its jaws close across the barrel from either side, and the block's
walls are cut down between the clips to let them in (BMM603 STEP model). So the puller points straight down into
the block, its arms parting left-right, and the jaws sit at the arms' tips: each is a curved shell (eight flat
facets, 14 mm wide along the fuse) that wraps the body from 81 degrees on the loop side to 81 degrees past the
equator on the tip side, so the two shells closed nearly enclose the body. The deep lips matter: with lips 50
degrees past the equator, tipping the puller wedged the jaws apart under the clamp and the puller turned on the fuse
instead of turning it. Squeezed by the gripper's 50 N 40 mm from the loop, the jaws (114 mm from the loop) clamp
the body with about 13 N each; the shells hold the fuse by shape. Inferred: jaw shape, position, flex and opening
(Ideal publishes no drawing); 34 g against about 20 g real; arms 12 mm wide strips. The jaw-to-body friction is 1.0
(combined as the larger). The handles next to the loop are notched (the 34-015 is sold with a "notched handle"):
the gripper's fingers grip them with friction 0.8; on the smooth moulded arms from there to the jaws they grip with
0.25 (inferred). The puller stands upright, jaws down, in a 20 mm deep pocket of a grey holder block (a stand-in for
the clip or holster a puller is kept in), so it is picked up already pointing the way it is used. It is held by
friction alone; nothing attaches it to the hand or the fuse.

**The rock.** With the jaws closed across the body, tipping the handles away from the robot (the hand moves along
an arc about the far ferrule and turns with the puller) turns the puller and the fuse together about the far clip,
so the near ferrule leaves its clip first; the lever is the 74 mm from the jaws to the hand. With the near end free,
the far ferrule has to come out of its clip alone. Pulled straight up while the fuse is tipped 30 degrees it binds in
the clip (the arm's lift stalled with the hand 46 mm behind its target); turned back toward level as it is lifted,
it slides out. In the demo the tip read from arm B's joints climbed to about 15
degrees with the hand 15-20 mm behind its target, then jumped to 19 degrees as the near ferrule came out. A
straight pull on both clips at once (the control) is beyond what the robot transmits through the smooth arms by
the jaws (the fingers slide at 20-30 N); a straight pull by the notched handles was not tried as a control.

**The installed panel: enclosure, wiring and access.** The fuses are serviced where they are installed, inside a
control enclosure, not on a bare block. The enclosure is a Rittal AE 1036.500 compact wall-mount enclosure (Rittal
data sheet: 300 x 300 x 155 mm outside, 1.38 mm sheet-steel body, 1.5 mm door, RAL 7035, a 254 x 275 mm, 2 mm
zinc-plated mounting plate, a gland plate in the base, one hinged door). The station's arms reach down onto a table,
so the enclosure lies on its back on the bench with its opening up, the way a panel is laid down to be serviced on a
bench; its bottom (gland-plate) end faces the robot, so the rail runs left-right as in the upright panel and the
fuses' axes point "up" the panel, away from the robot. Its door is hinged along the left wall's rim and stands open
at 100 degrees, just past upright (on a real bench a door strap holds it there; in the sim it is fixed). Modelled
as fixed, colliding geometry: the back, the four 155 mm walls, the 12 mm lip round the opening (where the door seal
lands; inferred), the door panel and its folded edges, the mounting plate (17 mm above the back on studs, inferred),
a 35 x 7.5 mm top-hat DIN rail (EN 60715) and two slotted wire ducts (Panduit type G class, 1.5 x 2 in nominal:
38 mm wide, 51 mm tall), one between the blocks and the robot, one beyond them. The BM6031B is sold for DIN-rail or
panel mounting with box-lug terminals for 14 to 6 AWG wire; the blocks clip onto the rail side by side, 1.5 mm
apart. Wiring: each block has a 14 AWG wire (THHN, 2.8 mm outside diameter) landed in each box lug, black on the
line (far) lug running straight into the far duct and red on the load (near) lug running straight into the near
duct, 44 mm above the table (16 mm above the block's base, where the lug clamps it). Each wire is a 2.8 mm square
collider fixed at both ends (in the lug and in the duct): it does not flex, so the wires, like the walls and the
ducts, are obstacles a hand or the puller cannot push aside. They are not scored: a fixed wire cannot be pulled off
its lug or moved in the sim, so a rule on them could never trigger (a pulled or loosened wire would be a real fault;
the reset notes ask for the wiring to be checked). How this constrains the task: the puller and, later, the fuse in
its jaws have to be lifted until they clear the 155 mm rim before they can be carried in over the right wall (the
puller's holder and the parts tray stand on the bench just outside it) and back out; the gripper works inside the
walls with the wrist above the rim; the far duct and its labels are where the handles swing when the puller is
tipped; the door stands where the left arm would otherwise reach in.

**Layout and looks.** The blocks stand 1.5 mm apart (BM6031B blocks gang side by side). The circuit labels lie on
the far duct's cover above each block, facing the top camera, which looks over the near wall into the enclosure.
Label text is 2-3 px at 320x176, so each label carries a circuit colour (F1 white, F2 orange, F3 green), a plant
labelling choice, not a standard. Each label is a coloured plate with its text as thin dark (on green, light)
geometry, not a texture: in one run the renderer showed the textured labels blank. Block terminals, screws, the
fingers' curved roots and reinforcing springs, the duct slots, the plate's studs, the door's seal, lock and hinges
are visual only. One layout.

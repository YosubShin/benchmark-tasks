# Slide an insulation shield between a pipe and its clevis hanger: lift the pipe a little, slip the saddle in, centre it, lower the pipe onto it

A 1.24 m length of 1/2 in Type L copper water pipe in 1 in fiberglass pipe insulation (Johns Manville Micro-Lok HP,
2.83 in OD, white all-service jacket) runs left to right in front of the arms. It rests in two Eaton B-Line
B3100-2-1/2 clevis hangers on 1/2 in threaded rod from an overhead strut: hanger A, in the middle, with the insulation
lying directly on its lower strap, and hanger B, 0.58 m to the right, which already has its shield in. The pipe is
full of water (capped). The loose shield for hanger A, a B-Line B3151-3X insulation protection shield (18 ga
galvanized steel, 12 in long, 2.88 in inside diameter, half-round), lies open side up, balanced on a wooden post (a
2x2 offcut on end) in front of the pipe; a 12 mm strip of red tape round its middle is its centre mark. The
instruction gives only the goal and the constraints: the shield in hanger A under the pipe, centred on the hanger, the pipe resting on it; lift the
pipe no more than 2 cm, do not shift it along its run, do not drop the shield, let go when done. It does not say how.

Why the shield matters: without it the clevis strap's 1 in edge carries the pipe's load straight into the
insulation, crushes it and cuts its vapour-retarding jacket (condensation and corrosion on cold lines). B-Line
specifies the B3100 "for the suspension of non-insulated pipe or insulated pipe with a B3151 shield", and MSS
requires 12 in of bearing for pipe up to 4 in. A shield left out is fitted afterwards with the pipe already in its
hangers; distributors' installation advice: "it is best to have the pipe
raised first and then center the shield inside the hanger"; B-Line sells a snap-on shield precisely "to minimize the
need to lift the pipe during installation".

**The hard part is physical and bimanual.** One hand holds the pipe up off the strap (10 N at the hanger, about 7.5 N
at the hand) by only a few millimetres while the other gets a 30 cm, 0.34 kg half-round under it and along it
through the hanger: the shield's outside (75.6 mm) clears the strap's inside by 2.7 mm a side, and its inside
(73.2 mm) the insulation by 0.6 mm; the shield must stay open side up, square to the pipe and level while it is
carried on the hand; then it must be centred within 10 mm on the strap and the pipe set down on it. A parallel-jaw
gripper cannot pinch the shield round the pipe (hand and shield would close a ring round a pipe that has no free end),
so the hand has to carry it from below like a fork, balanced on two fingers, and the strap then stands in the way of
the fingers: the last 12 cm the shield has to be pushed by its end edge, which needs the hands to work together (the
pipe is set down on the shield to hold it while the hand moves round behind it, then raised again).

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`pipe_insulation_shield.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream`
policy. The file uses `pipeshield_sim` (this folder), which adds the props and the task to the station at run time
(`pipeshield_sim/station.py` documents exactly what it patches; it requires yam-isaac-lab-arena at commit `0059d02`,
the Brev launchable's pin). The task runs PhysX at 480 Hz (the policy still acts at 20 Hz).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/pipe-insulation-shield --no-deps     # registers the policy pipe-shield-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/pipe-insulation-shield/pipe_insulation_shield.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Fit the insulation shield into the pipe hanger under the pipe, centred on the hanger, so the pipe rests on it. Lift the pipe no more than 2 cm and do not shift it along its run, do not drop the shield, and let go when you are done." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/pipe-shield-dream
```

The scripted demo, `pipe-shield-demo` (closed loop, robot-observable inputs only; see "What the demo reads"):

```bash
inspect-robots run --embodiment yam_arena --policy pipe-shield-demo \
  -E task=tasks/pipe-insulation-shield/pipe_insulation_shield.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/pipe-shield-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit below, or 0 after a hazard).
Videos: `inspect-robots video <log>.json` writes one MP4 per camera; `demo.mp4` and `dream1.mp4` here put the top
camera (2x) above the left and right wrist cameras.

- **Episode end.** On success (the release check below), or at `--max-steps`.
- **Scoring.** `pipeshield_sim/scorer.py` (`Scorer`); the task file's `score()` returns its score unchanged.
  Levels: 0.33 the shield under the pipe in hanger A (open side up within 15 degrees, its axis within 6 mm of the
  pipe's sideways, across the strap, its underside between the strap and 25 mm above it); 0.67 also centred (its
  middle within 10 mm of the strap's centre line, along the pipe); 1.0 the release check: every finger pad at least
  3 cm from the pipe and the shield for 1 s without a break (the count restarts when a pad comes closer or a condition
  fails) while the 0.67 conditions hold, the shield sits on the strap (underside within 2 mm) and the pipe rests on it
  (its axis at hanger A within 2 mm of its height on a seated shield: 1.2 mm above where it lay on the bare strap).
  Hazards, each 0 for the whole trial from the step it happens, whatever was reached before or after: the pipe raised
  more than 20 mm at hanger A; the pipe shifted more than 15 mm along its run (or sideways at A); the shield dropped
  (its underside below 23 cm, i.e. off its post and out of the hand). `python -m pipeshield_sim.check_scorer`
  checks the gate without the simulator.
- **Control run.** `-P no_centre=1`: the demo does everything as in the demo run but leaves out the centring: the
  shield is left where the first slide put it (its front end 2 cm past the strap, its middle about 12 cm off the
  hanger's centre line) and the pipe is lowered onto it there. It must score below 1.
- **Seeds.** One fixed layout (`-E object_jitter=0`). Seeded jitter is not used.

## What the demo reads

`pipe-shield-demo` (`pipeshield_sim/policy.py`, `demo.py`, `vision.py`) receives exactly what `yam_arena` gives any
policy each step and nothing else: the RGB images `top_cam`, `left_cam`, `right_cam` (320x176) and the 14-D
`joint_pos` state. It does not read the simulator: no pipe or shield pose, no scorer state. Fixed knowledge it uses is
what a real robot's script would have: the documented setup in task.json (where the pipe, the hangers and the shield's post
are; the pipe's, shield's and hanger's sizes), the station's published camera calibration (`yam_arena.station`) and
the arm's URDF (`yam_arena.urdf`, `yam_arena.ik`). Two fixed offsets in `demo.py` were measured in development runs
of this setup, as a real robot's script would be tuned on the bench: where the fingers' upper edges peak relative to
the grasp point (`FORK_TOP`, 12.5 mm) and how far behind the grasp point the shield settles on them once lifted
(`SEAT_BACK`, 19 mm).

| Thing | How the demo gets it |
|---|---|
| Where the shield lies along the pipe | first top-camera frame: the red tape's pixels, back-projected onto a level plane halfway up the shield, give its middle (the post's documented position gives the other coordinate) |
| How far the pipe is raised | arm A's joints (forward kinematics) give its grip's rise; the documented hanger positions turn it into the rise at hanger A (the pipe pivots on hanger B) |
| The shield's height before it goes into the hanger | arm B's joints give its fingers' height; the hand is raised by what is missing (the arm sags a few mm under the load) |
| Where the shield is along the pipe once it is in | arm B's joints at the end of the first slide (the shield sits centred on the fingers); its end edge is then 152 mm behind |
| Centred on the hanger | arm B's joints while its trailing finger pushes the shield's end edge (the shield's middle is 152 mm ahead of the edge); then the top camera looks for the red tape on the shield's near side under the pipe for a final check |

In the demo run below the top camera put the shield's tape at y = -59.0 mm (46 px; the layout says -60.0), and the
demo used that. Arm A's joints put the pipe 9.7 mm up at hanger A while the shield went in (11.1 mm at most, by the
simulator). Arm B lifted the shield off its post, carried it round under the pipe, found by its joints its fingers
0.6 mm short of the slide height and raised them, slid the shield in until its front end was 32 mm past the
hanger's centre line, set it down, had the pipe set on it, came round behind it and pushed it on until its joints put
the shield's middle 1.2 mm short of the hanger's centre line. The top camera did not find the tape under the pipe in
that position (the shield's near wall is in the pipe's shadow and only a few pixels of it show), so the demo went by
its joints, as it does whenever the tape is not seen.

## Measured

RTX PRO 6000 class GPU shared with five to seven other Isaac Sim processes (this task's runs and another build),
shader cache warm, `-E object_jitter=0`, 20 Hz, 320x176 frames, PhysX at 480 Hz:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `pipe-shield-demo` (cameras + joint states only) | `task_score: 1`. Pipe raised (11.1 mm at hanger A at most); shield lifted off its post, carried round and in under the pipe, slid into the hanger until its front end was 32 mm past the strap's centre line, set down; pipe set on it; hand round behind it; pipe raised; shield pushed by its end edge to its middle 1.2 mm short of the hanger by the arm's joints; pipe lowered, hands clear. At the release check: shield's middle 3.4 mm from the strap's centre line, its axis 0.02 mm from the pipe's sideways, open side up within 3.5 degrees, its underside 0.24 mm above the strap; the pipe's axis at hanger A 0.26 mm above its height resting on a seated shield (1.46 mm above where it lay on the bare strap); finger pads 66-68 mm from the pipe and the shield; the pipe moved 5.9 mm along its run at most | 2,028 (101 s) | about 62 minutes (five to seven other simulations running) |
| Control: `-P no_centre=1` (no centring) | `task_score: 0.33`. Shield in the hanger under the pipe, the pipe lowered onto it, but its middle about 12 cm right of the hanger's centre line, so 0.67 and the release check are never reached | 3,000 (step limit) | about 87 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`, the instruction above) | `task_score: 0` (both arms move about in front of the pipe and end crossed over each other; neither touches the pipe or the shield, which stays on its post; no hazard) | 3,000 (step limit) | about 88 minutes (Dream reported an observation as expired and was asked again 61 times) |

The demo's wall time is mostly the doubled physics rate, its Python IK every control step and the shared GPU.

## Hazard checks

Each check is the official demo command above with `--policy pipe-shield-demo`, seed 0 (`-E object_jitter=0`), and
one environment variable, `PIS_HAZARD_TEST=<case>`, read by `pipeshield_sim/task.py` (empty in every scored run).
The test harness causes the hazard (a stand-in for a hand or a slip); the demo otherwise runs unchanged, so later
levels are reached physically, and only the reported score shows the gate. The task prints its report when the
first hazard happens and when progress reaches 1.0.

| Case | What happens | Progress reached | Reported `task_score` |
|---|---|---|---|
| none (the demo) | | 1.0 | 1 |
| `lift_start` | at 1 s (step 20), before any level, an upward pull of twice the pipe's weight at hanger A for 1 s (a hand yanking it up), then released: the pipe rose 27.6 mm at A (against the hanger's cross bolt) and the hazard was flagged at step 21; the demo then ran as usual | 1.0 (step 2,054: shield 0.9 mm from the centre line, pipe resting on it) | 0 |
| `shift_33` | the step 0.33 is reached (step 947), the pipe is moved 20 mm along its run, and put back the next step: flagged at 20.2 mm | 1.0 (step 2,040: shield 3.3 mm from the centre line, pipe resting on it) | 0 |
| `drop_67` | the step 0.67 is reached (step 1,724), the shield is put 10 cm above the bench (falling), and put back the next step: flagged | 1.0 (step 2,025: shield 3.0 mm from the centre line, pipe resting on it) | 0 |

`python -m pipeshield_sim.check_scorer` (no simulator) checks the same gate for each hazard raised before any level
and after 0.33, 0.67 and 1.0 (final score 0 each time, with every later level still reached), and 1.0 without a
hazard.

## Real-trial checks

- **Under the pipe, in the hanger (0.33):** photo from the side along the pipe: the shield between the strap and the
  insulation, open side up, the strap bearing on the shield across its 1 in width.
- **Centred (0.67):** a ruler along the pipe from the strap's centre line (a pencil mark on the strap's edge) to the
  shield's centre tape: 10 mm or less.
- **Pipe resting on the shield (1.0):** with both hands clear (video), photo from the side: daylight between the strap
  and the insulation everywhere (only the shield touches the strap), and the pipe's insulation down on the shield (a
  0.5 mm feeler gauge does not enter under the insulation at the shield's ends).
- **Lift hazard (20 mm):** a steel rule clamped upright beside hanger A, its zero at the bottom of the insulation
  before the trial; the video shows the insulation's bottom against it. The shield is 1.2 mm thick; a few mm is
  enough to slide it in. More bends a run between its hangers and loads its soldered joints and fittings; installers
  raise a pipe only as far as the shield needs (manufacturers sell snap-on shields "to minimize the need to lift the
  pipe").
- **Shift hazard (15 mm):** a strip of tape on the insulation beside hanger A's strap before the trial; afterwards a
  ruler from the strap's edge to the tape. Pushing or pulling a run along its length strains its joints and elbows.
- **Shield dropped:** from the video (off its post and out of the hand, on its way to the bench or floor).

## Closest tasks on the board (checked 2026-11-19)

Searched for hanger, shield, saddle, pipe, insulation and clevis. Nothing fits a part between a pipe and its support,
and nothing lifts a hanging run. Closest:
- "Dry-fit a slip coupling across the open joint of a drain-pipe repair": slides a coupling along a pipe over a gap;
  the pipe is not lifted and nothing goes between a pipe and its support.
- "Insert a pipe into a fixture": puts a free pipe into a holder; here the pipe stays in its hangers and a curved
  shield has to go in under it.
- "Seat a strut across a keep-out" (accepted): places a bar on two saddles; no load to hold up, nothing slipped under.
- "Fit an inside-bolt-circle gasket into a spread pipe-flange joint and put all four bolts in": a thin part slid
  between two pipe flanges held apart; the flanges are fixed, nothing is lifted and held.
- Our own strut-spring-nut-clamp (submitted): fastens a fitting in strut channel; no pipe, no lift, no shield.
What is distinct here: one hand holds a real load (10 N at the hanger) up by only a few millimetres while the other
carries a long curved saddle in under it from below (a parallel gripper cannot close round a pipe with no free end)
and along through the hanger with 2.7 mm clearance a side, then centres it and both hands hand the load back onto it.

## Layout

| Path | What |
|---|---|
| `pipe_insulation_shield.py` | the task file: description for agents, the props and task, scoring |
| `pipeshield_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, physics rate) |
| `pipeshield_sim/task.py`, `scorer.py`, `spec.py` | props, measurements, release check, hazards, layout, instruction |
| `pipeshield_sim/check_scorer.py` | the scorer's zero-gate check without the simulator |
| `pipeshield_sim/usd/props.py`, `pipeshield_sim/assets/` | the props, generated with `python -m pipeshield_sim.usd.props pipeshield_sim/assets` |
| `pipeshield_sim/demo.py`, `vision.py`, `policy.py`, `kit.py` | the scripted demo and its Inspect Robots policy `pipe-shield-demo` |

## References

Copies and page numbers: out/compare/pipe-insulation-shield/ref/SOURCES.txt (in our build tree).
- Eaton B-Line Pipe Hangers & Supports catalog, shields section (B3151 selection table printed p164, dimensions
  p166: 3X = 18 ga, 12 in, 2.88 in ID, 2-1/2 in hanger, 75 lb/100; B3153 with Loc tabs):
  https://www.eaton.com/content/dam/eaton/products/support-systems/pipe-hangers/eaton-b-line-pipe-hangers-supports-catalog-ph20.pdf
- Eaton B-Line pipe hanger catalog CA310001EN, B3100 clevis hanger (printed p50-51: 2-1/2 in size, 1/2-13 rod,
  C = 3-15/16 in; "recommended for ... insulated pipe with a B3151 shield"); MSS SP-69 hanger spacing table (p324).
- Anvil (ASC Engineered Solutions) Fig. 167 Lok-Shield submittal (size 2A: 1/2 in tube, 1 in insulation, 18 ga,
  12 in, insulation OD 2.88 in, Fig. 260 size 2-1/2) and Fig. 260 clevis submittal (size 2-1/2: lower strap width
  H = 1 in): https://www.asc-es.com/
- Copper Development Association, The Copper Tube Handbook, Table 14.2b (1/2 in Type L: 0.625 in OD, 0.545 in ID,
  0.285 lb/ft, 0.386 lb/ft full of water).
- Johns Manville Micro-Lok HP dimensional data CI-214 (1/2 in copper, 1 in nominal wall: 2.83 in OD) and technical
  bulletin (5.2 pcf).
- International Plumbing Code Table 308.5 (copper tubing 1-1/4 in and smaller: 6 ft horizontal support spacing).
- Distributor installation advice for shields (pipe raised first, shield centred in the hanger):
  https://www.buyinsulationproductstore.com/ (pipe insulation shields page).
- BLS Occupational Outlook Handbook: plumbers, pipefitters and steamfitters
  https://www.bls.gov/ooh/construction-and-extraction/plumbers-pipefitters-and-steamfitters.htm ; HVAC mechanics and
  installers https://www.bls.gov/ooh/installation-maintenance-and-repair/heating-air-conditioning-and-refrigeration-mechanics-and-installers.htm ;
  insulation workers https://www.bls.gov/ooh/construction-and-extraction/insulation-workers.htm (2025 employment).
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 480 Hz in Isaac Lab-Arena (the station's 240 Hz doubled, per task), both arms driven as in Dream's station
(joint PD with i2rt's gains and gravity compensation). Rigid bodies; collision shapes are boxes and convex hulls.

**Pipe.** 1.24 m of 1/2 in Type L copper, full of water, in 1 in Micro-Lok insulation: one free rigid body, its
insulation a 32-sided convex prism of 2.83 in OD (white jacket, friction 0.4), 1.12 kg (0.606 lb/ft: tube 0.285,
water 0.101, insulation 0.22 from JM's 5.2 pcf). It rests in hangers A and B, 0.58 m apart, and overhangs A by 0.58 m
to the left, so A carries 10.2 N: what a hanger carries in a run of this pipe full of water at 3.8 ft spacing (at the
5 ft maximum it would be 13.3 N; empty at 5 ft, 11.2 N). The section stands for a longer run: its overhang puts a
run's load on A while the hand lifting it stays within the station's ~15 N. Lifting it at the grip point (21 cm left
of A) takes about 7.5 N, and it pivots on B, so the rise at A is 0.73 of the rise at the grip; in a continuous run the
force would rise further with the lift (the run bends), which the rigid section leaves out.

**Hangers.** B3100-2-1/2 clevis: lower strap 1 in wide (Anvil Fig. 260 H), 3/16 in thick (inferred), its bottom a
half-round of 40.5 mm inside radius (inferred: the shield's outside is 37.8 mm, so 2.7 mm a side) as 18 flat strips,
straight legs up to the 1/2 in cross bolt, whose axis is 70 mm above the pipe's (inferred from B-Line's C = 3-15/16 in,
centre of pipe to top of hanger, less the yoke: about 1 in clear above the insulation, so the pipe can be raised some
28 mm at the hanger before it meets the bolt); upper yoke, rod, nuts and the overhead strut are visual. Fixed colliders
(a real clevis swings a little on its rod; not modelled). Hanger B's shield is part of the fixed rig.

**Shield.** B3151-3X: a half-round of 18 ga (1.21 mm) sheet, 2.88 in inside diameter, 12 in long, 0.34 kg, as 18 flat
strips of 10 degrees (friction 0.3, galvanized), its centre of mass 2R/pi below its axis. Rigid (the real 18 ga sheet
flexes a little; installers can spring it, the robot does not need to). The red tape round its middle stands for the
centre mark installers draw. B3151 has no locating tabs (B3153 has); the shield is held in the hanger by the pipe's
weight, as specified.

**What is not modelled or simplified.** Work height: the hanger hangs at bench height (the pipe's axis 44 cm above
the bench, on rods from a strut frame), not overhead as pipe often runs; parts, clearances and loads are full size.
The insulation is rigid (it would dent under a grip or the bare strap; the
demo closes the gripper only to a light squeeze, 0.62 of open). The hangers do not swing. One layout. The shield
lies balanced on a wooden post at 30 cm, where the fingers can get under it either side; a shield on the floor would first have to be
picked up and turned.


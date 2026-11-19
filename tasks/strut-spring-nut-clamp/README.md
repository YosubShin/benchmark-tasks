# Fasten a strut fitting at a mark with a spring nut: edge-on into the slot, quarter turn under the lips, bolt hand-snug

A 2 ft length of Unistrut P1000 strut channel (1-5/8 x 1-5/8 in, 12 ga, Perma-Green) is bolted flat to the bench
top, open side up, running left to right in front of the arms, as strut is surface-mounted on floors, walls and
equipment benches. A strip of red tape across its near wall and lip marks where a fitting has to go. In front of
it, to the right, stands the installer's kit on a wooden parts block: a Unistrut P1064 flat plate fitting (1-5/8 in
square, 1/4 in thick, 9/16 in hole) with a 1/2-13 x 15/16 in hex head cap screw through its hole, started two turns
into a Unistrut P1010 channel spring nut that hangs below it in the block's pocket. The instruction gives only the
goal and the constraints: fasten the fitting to the strut at the red mark with its spring nut and bolt, leave the
bolt hand-snug (not torqued), do not drop any part, let go when done. It does not say how. The kit comes with
the bolt started in the nut, as installers often hold a fitting, bolt and spring nut together while offering them up:
the robot's fingertips (26 mm wide) cannot enter the 22.2 mm slot to push and turn a loose nut with a thumb, so the
bolt is its handle for the nut (see "Fidelity details").

The way installers do it (Unistrut's own wording, GEC17 p7 and p10: "Insert the spring nut anywhere along the
continuous slotted channel"; "A 90 degree clockwise turn aligns the grooves in the nut with the inturned edges of the
channel"; the spring "holds the nut in position while the connection is completed"): the nut goes edge-on into the
slot (it is 19 mm wide, the slot 22.2 mm; 34.5 mm long, so it only fits lengthwise), is pushed down past the lips
against its spring, turned a quarter turn clockwise so its two grooves come under the lips' turned-down edges, and let
up: the spring presses the grooves onto the lips. Then it is slid along the channel to where the fitting goes, and
the bolt is run down until the fitting is held. **This task ends there: the bolt run down hand-snug, the fitting
seated square at the mark, ready for the installer's torque wrench.** The final tightening (50 ft-lb, 68 N m, for
1/2 in, Unistrut and B-Line catalogs) is a separate wrench step far beyond the station; the task does not scale it
down or call the bolt tight. Positioning fittings hand-snug and torquing them afterwards is how a strut run is put
together: a fitting is set and held by a snug bolt, checked, then torqued.

The hard part is physical and contact-rich: the nut must go into the slot edge-on (1.6 mm clearance a side), be
pushed past the lips against its spring (about 6 N) and turned with the bolt while it is down; it turns only
clockwise and only while it is below the lips; let up too early or not turned, it rises back out of the slot. Then
the assembly has to be slid along the channel to within 3 mm of the mark against the spring's friction, and the
bolt run down about four turns in regrasped strokes, the head's height falling 2 mm a turn, until the thread resists.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`strut_spring_nut.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy.
Its kit is an articulation (nut, spring, bolt on a screw joint), which the one-file format cannot declare, so the file
uses `strutnut_sim` (this folder), which adds it to the station at run time (`strutnut_sim/station.py` documents
exactly what it patches; it requires yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin). The task
runs PhysX at 480 Hz (the policy still acts at 20 Hz).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/strut-spring-nut-clamp --no-deps     # registers the policy strut-nut-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/strut-spring-nut-clamp/strut_spring_nut.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Fasten the square plate fitting to the strut at the red mark with its spring nut and bolt. Leave the bolt hand-snug, not torqued. Do not drop any of the parts, and let go of them when you are done." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/strut-nut-dream
```

The scripted demo, `strut-nut-demo` (closed loop, robot-observable inputs only; see "What the demo reads"):

```bash
inspect-robots run --embodiment yam_arena --policy strut-nut-demo \
  -E task=tasks/strut-spring-nut-clamp/strut_spring_nut.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/strut-nut-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit below, or 0 after a hazard).
Videos: `inspect-robots video <log>.json` writes one MP4 per camera; `demo.mp4` and `dream1.mp4` here put the top
camera (2x) above the left and right wrist cameras.

- **Episode end.** Once 0.75 is reached and every finger pad has stayed at least 3 cm from the fitting and the bolt
  for 1 s without a break (the count restarts whenever a pad comes closer), the after-release pull check (2 s of
  simulated time) runs inside that control step, off camera, and the episode ends; the stored frames stop at the
  release. Otherwise the run ends at `--max-steps`.
- **Scoring.** `strutnut_sim/scorer.py` (`Scorer`); the task file's `score()` returns its score unchanged.
  Milestones: 0.25 the nut seated under the lips (turned across the channel within 10 degrees, its top within
  1.2 mm of the seated height, centred within 3 mm); 0.5 also the bolt's axis within 3 mm of the mark's centre line;
  0.75 also the bolt engaged at least 5 turns; 1.0 the after-release check passed (at its start the 0.75 conditions,
  the fitting flat on the lips within 0.6 mm and 2 degrees, square within 3 degrees, the head down on it within
  0.5 mm; then 20 N along the channel on the fitting moves it less than 1 mm). Hazards, each 0 for the whole trial
  from the step it happens, whatever was reached before or after: the kit dropped (fitting, nut or bolt on the table
  off the channel and off the block), or the nut run off the bolt (engaged less than half a turn; in a real channel it
  falls in and is lost). `python -m strutnut_sim.check_scorer` checks the gate without the simulator.
- **Control run.** `-P skip_turn=1` with `--policy strut-nut-demo` and `SNC_CONTROL=135`: the demo does everything as in the
  demo run but leaves out the quarter turn, so the nut stays edge-on in the slot (not under the lips) while the bolt
  is slid to the mark and run down; `SNC_CONTROL=135` starts the after-release pull check at 135 s whatever the robot
  has done (otherwise a trial that never reaches 0.75 never starts it). It must score below 1.
- **Seeds.** One fixed layout (`-E object_jitter=0`). Seeded jitter is not used.

## What the demo reads

`strut-nut-demo` (`strutnut_sim/policy.py`, `demo.py`, `vision.py`) receives exactly what `yam_arena` gives any
policy each step and nothing else: the RGB images `top_cam`, `left_cam`, `right_cam` (320x176) and the 14-D
`joint_pos` state. It does not read the simulator: no nut, bolt or fitting pose, no thread or spring joint, no
scorer state. Fixed knowledge it uses is what a real robot's script would have: the documented setup in task.json
(where the channel, the mark and the parts block are; the channel section, nut, bolt and fitting sizes; that the bolt
is started two turns), the station's published camera calibration (`yam_arena.station`) and the arm's URDF
(`yam_arena.urdf`, `yam_arena.ik`).

| Thing | How the demo gets it |
|---|---|
| The mark | first top-camera frame: the red tape's pixels on the lip, back-projected onto the lip's top plane, give its centre along the channel; once the hand holds the kit over the channel, the right wrist camera sees the same tape larger and its estimate replaces the top camera's |
| The kit | standing in its documented place on the block; the head is taken across two flats from above |
| Nut pushed below the lips | the nut hangs a known distance below the head (bolt length less two turns); the hand is lowered until arm B's joints put the head where the nut's top is 3 mm below the lips, against the spring |
| The quarter turn | a third of a turn of the wrist (joint 6) clockwise while pushed; the nut turns with the bolt until its corners stop (the rest turns the bolt in it) |
| Bolt at the mark | arm B's joints (forward kinematics) give the head's position while it holds it; the hand slides along the channel until that is on the mark's centre line |
| Turns run down, hand-snug | before each grip the hand waits until its joints put it within 0.6 mm of the head's position (so closing does not push the kit along); the wrist's measured roll during each stroke while the fingers hold the head (turns counted), the head's height from the counted turns and the 13 threads per inch, and the wrist's lag behind its command at the end of a stroke (the thread resisting) |

In the demo run below the top camera put the mark's centre at y = -38.7 mm (8 px; the layout says -40.0) and the right
wrist camera, held over the channel, at -40.0 mm (112 px on the lip), which the demo used. Arm B pushed the head down
until its joints put the nut's top about 2 mm below the lips, turned the wrist a third of a turn: the nut turned with
the bolt and stopped on its square corners (4.4 degrees past across), and when the push was eased the spring seated
it on the lips. Its joints put the bolt 0.8 mm from the mark after the slide. Thirteen strokes of a third of a turn
followed; the wrist lagged its command by 1.9-2.0 degrees while the bolt ran free and by 3.9 degrees on the stroke
that brought the head down on the fitting (4.26 turns counted against 4.02 documented), and the demo stopped there.

## Measured

RTX PRO 6000 class GPU shared with two to five other Isaac Sim processes (this task's own runs), shader cache warm,
`-E object_jitter=0`, 20 Hz, 320x176 frames, PhysX at 480 Hz:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `strut-nut-demo` (cameras + joint states only) | `task_score: 1`. Kit lifted by the head; nut and spring lowered edge-on into the slot 5 cm right of the mark, the fitting landing on the lips; head pushed down (nut's top about 2 mm below the lips), a third of a turn clockwise: the nut turned with the bolt and stopped 4.4 degrees past across; push eased, nut seated on the lips (0.25); slid to the mark (0.5, bolt 0.8 mm from the mark by the arm's joints, 0.2 mm in the end); 13 strokes ran the bolt from 2.1 to 6.1 turns engaged (0.75), the head down on the fitting, bolt tension 74 N (about 0.22 N m at the head by the bolt formula); released; pull check 20 N along the channel: the fitting moved 0.009 mm; fitting square within 1.5 degrees | 2,308 (115 s) | about 42 minutes (two other simulations running) |
| Control: `-P skip_turn=1`, `SNC_CONTROL=135` (no quarter turn) | `task_score: 0`. The nut stayed edge-on in the slot; eased, it rose back up under the fitting and lifted it off the lips; slid and run down as in the demo, the bolt clamped the fitting onto the nut standing in the slot (no wrist lag ever showed snug; stopped after 5.3 turns); never seated, so no milestone; the forced pull check slid the whole kit 56 mm along the channel | 2,691 (forced check at 135 s) | about 45 minutes (two other simulations running) |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`, the instruction above) | `task_score: 0` (the right hand reaches over the parts block toward the kit and hovers there, the kit is never lifted; the left arm stays near its start; no hazard) | 3,000 (step limit) | about 54 minutes (GPU shared with two other simulations; Dream reported an observation as expired and was asked again 61 times; mean inference latency 0.62 s) |

The demo's wall time is mostly the doubled physics rate, its Python IK every control step and the shared GPU.

## Hazard checks

Each check is the official demo command above with `--policy strut-nut-demo`, seed 0 (`-E object_jitter=0`), and one
environment variable, `SNC_HAZARD_TEST=<case>`, read by `strutnut_sim/task.py` (empty in every scored run). The test
harness causes the hazard by writing a state (a stand-in for a hand or a slip); the demo otherwise runs unchanged, so
later milestones are reached physically where the disturbance allows, and only the reported score shows the gate.
`--max-steps 3000` (a trial with a hazard never ends on success unless the after-release check runs).

| Case | What happens | Progress reached | Reported `task_score` |
|---|---|---|---|
| none (the demo) | | 1.0 | 1 |
| `drop_start` | the fitting is put on the table in front of the parts block at 1 s (step 20), before any milestone, and put back on the block the next step | 0 (the put-back left the fitting resting on the bolt's head, which had dropped a few mm in the pocket meanwhile, so the demo lifted the fitting without the bolt and nut) | 0 |
| `nutoff_50` | the bolt is backed out of the nut to its last 0.2 turn (the nut runs off the bolt) the step 0.5 is reached (step 804); the screw rule then works from there | 0.75 (the demo carried on and ran the bolt back in, to 5.1 turns engaged) | 0 |
| `drop_75` | the fitting is put on the table the step 0.75 is reached (step 1946) and put back where it was the next step; the demo then snugs the bolt and lets go | 1.0 (the after-release pull check passed: the fitting moved 0.006 mm) | 0 |

`python -m strutnut_sim.check_scorer` (no simulator) checks the same gate for each hazard raised before any milestone
and after 0.25, 0.5, 0.75 and 1.0 (final score 0 each time, with every later milestone still reached), and 1.0
without a hazard.

## Real-trial checks

- **Nut seated (0.25):** after the robot lets go, look into the slot: the nut lies across the channel, its grooves on
  both lips; pulled up by its bolt with fingers it does not come out. Photo from above.
- **At the mark (0.5):** a rule from the mark's centre line (the tape's middle) to the bolt's centre, along the
  channel: 3 mm or less.
- **Turns (0.75):** count the head's turns on the video (the hex's flats) from the start of run-down, plus the two the
  kit came with; or after the trial, measure the bolt's end below the nut through the slot: engaged length = 15/16 in
  less the gap from the head to the nut's top; 5 turns = 9.8 mm.
- **Seated, square, head down (at the check's start):** feeler gauge under the fitting at its four corners (0.6 mm
  does not enter), a square against the channel's side and the fitting's edge (3 degrees = 2 mm over 41 mm), feeler
  under the head (0.5 mm does not enter).
- **Hand-snug pull check (1.0):** with the robot clear, hook a spring scale on the fitting and pull along the channel
  to 20 N (2 kg on the scale) for 2 s: the fitting must not slide (less than 1 mm, a pencil line beside it). This
  checks the bolt is run down **hand-snug**, holding the fitting by the nut's grip on the lips; it is not a torque
  check and the bolt is not torqued (that would be 68 N m with a wrench).
- **Hazards:** from the video: any part of the kit on the bench off the channel and the block; the nut off the bolt
  (it drops inside the channel).

## Closest tasks on the board (checked 2026-11-19)

Searched for strut, Unistrut, channel, spring nut, nut, thread, clamp, bolt and screw. Nothing installs a channel
spring nut or works in a strut channel's slot. Closest:
- "Seat a strut across a keep-out" (accepted): moves a grey bar between saddles; no channel, nut or fastening.
- "Fit a cage nut into a server cabinet's front rail from inside" (pending): a different captive nut (a sprung cage
  clipped into a square hole by hooking its lips with a screwdriver); no quarter turn, no slot, no bolt.
- "Fix a hook tight to a hollow wall with its toggle bolt" (pending): a fastener whose wings open behind a board
  through a round hole; tightened against the wall, no slide to a mark.
- "Bolt a coupling nut to an upright plate through its hole with an M10 screw" (accepted): threads a screw into a
  free nut held by the other hand; here the nut is held by the channel's lips after a quarter turn, and placing it
  at the mark inside the channel is the task.
- "Close off a terminal strip: fit and tighten an end bracket on a loose DIN rail" (accepted): a clip-on bracket on
  a rail tightened by a screw; no nut in a slot.
What is distinct here: putting a nut into a lipped channel edge-on, pushing it past the lips against its spring and
turning it under them (it can only be turned while held down, and only one way), then positioning it inside the
channel to a mark and running the bolt down to hand-snug in regrasped strokes.

## Layout

| Path | What |
|---|---|
| `strut_spring_nut.py` | the task file: description for agents, the props and task, scoring |
| `strutnut_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, physics rate) |
| `strutnut_sim/task.py`, `scorer.py`, `spec.py` | props, screw rule, measurements, after-release check, hazards, layout, instruction |
| `strutnut_sim/check_scorer.py` | the scorer's zero-gate check without the simulator |
| `strutnut_sim/usd/props.py`, `strutnut_sim/assets/` | the props, generated with `python -m strutnut_sim.usd.props strutnut_sim/assets` |
| `strutnut_sim/demo.py`, `vision.py`, `policy.py`, `kit.py` | the scripted demo and its Inspect Robots policy `strut-nut-demo` |

## References

- Unistrut General Engineering Catalog No. 17 (2011): https://acomee.com.mx/catalogos/catalogosu/UNISTRUT%20GENERAL%20ENGINEERING%20CATALOG.pdf
  - p7 "Unistrut connection" steps (insert the spring nut anywhere along the slot; a 90 degree clockwise turn aligns
    the grooves with the inturned edges; insert the bolt through the fitting into the nut; tighten with a wrench).
  - p10 materials: nuts rectangular with ends shaped to permit a quarter turn clockwise; two toothed grooves engage
    the inturned edges.
  - p24 P1000 section: 1-5/8 x 1-5/8 in, 12 ga (0.105 in), slot 7/8 in, lips 3/8 in, return 9/32 in; 189 lb/100 ft.
  - p71 fittings hardware ("All 9/16 in diameter holes use 1/2 x 15/16 in hex head cap screws and 1/2 in nuts -
    P1010") and design bolt torques; p75 flat plate fittings (P1064: 1/2 in bolt, 9/16 in hole, 1/4 in thick,
    17 lb/100); standard fitting dimensions.
- Unistrut Nuts & Hardware section (P1006-P1010 weights, bolt torque 50 ft-lb for 1/2 in, pull-out and slip loads):
  https://cdn.gescan.com/PIM_Docs/Docs/STEP_ASSETS_PDF/13505483.pdf
- Unistrut P1006-P1010 channel nut with spring, submittal sheet: https://static.chipdip.ru/lib/614/DOC028614241.pdf
- Eaton B-Line channel nuts and hardware catalog section (N225 1/2-13 spring nut 1/2 in thick; recommended torque
  50 ft-lb, 68 N m): https://lavitekco.com/Upload/files/Strut%20systems-acc/channel-nuts-and-hardware-catalog-section___.pdf
- Eaton N225HDG spec sheet (2 x 1.31 x 0.85 in, 0.116 lb): https://www.elliottelectric.com/Media/N225HDG-BLI-1021-3-Attachment.pdf
- Hex cap screws, ASME B18.2.1 dimensions (1/2-13: 3/4 in across flats, 5/16 in head): https://www.tannerbolt.com/media/akeneo_connector/CapScrewgr2-ss.pdf
- BLS Occupational Outlook Handbook: electricians https://www.bls.gov/ooh/construction-and-extraction/electricians.htm ,
  plumbers, pipefitters and steamfitters https://www.bls.gov/ooh/construction-and-extraction/plumbers-pipefitters-and-steamfitters.htm ,
  HVAC mechanics and installers https://www.bls.gov/ooh/installation-maintenance-and-repair/heating-air-conditioning-and-refrigeration-mechanics-and-installers.htm ;
  OEWS May 2025 employment (BLS API series OEUN000000000000047211101, ...47215201, ...49902101).
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 480 Hz in Isaac Lab-Arena (the station's 240 Hz doubled, per task), both arms driven as in Dream's station
(joint PD with i2rt's gains and gravity compensation). Rigid bodies; collision shapes are boxes and convex hulls.

**Channel.** Unistrut P1000 as GEC17 p24 draws it: 41.3 mm square, 2.67 mm (12 ga) walls and back, top flanges 9.5 mm
wide, returns turned down to 7.1 mm below the top face at the 22.2 mm slot, the bend at the slot edge a 1.8 mm
45 degree bevel (the drawing shows a rounded bend; radius inferred). Fixed colliders, painted steel, friction 0.3. It
is bolted flat to the bench, back down (surface mounting; four anchor bolts and square washers are visual): a raised
stand was tried first and put the work above what the YAM reaches with its fingers pointing down (wrist joint 4 at its
limit above about 16 cm at 30 cm out).

**Spring nut.** P1010 size (19.05 x 34.5 x 12.7 mm; 54 g, Unistrut's 12 lb per 100). Its outline is the rectangle with
the two leading corners for a clockwise turn rounded to 17.6 mm about the centre ("ends shaped to permit a quarter
turn clockwise", GEC17 p10): inside the channel (35.96 mm between the walls) it turns clockwise only, and its two
square corners stop it 4-5 degrees past a quarter turn; anticlockwise it jams at once. Two grooves 4.2 mm wide and
1.6 mm deep across its top (inferred; the catalogs show two toothed grooves, the teeth are not modelled) sit under the
returns; the bottom long edges are bevelled 1.5 mm. The nut fits through the slot only edge-on (19 mm in 22.2 mm; its
34.5 mm length does not). The spring is a prismatic foot under the nut with a spring drive: 31.8 mm free (the
P1010's listed 1-3/4 in height with spring, less the nut), 440 N/m (inferred: about 5 N holds the seated nut up, about
6 N to push it 3 mm below the lips); its foot stands on the channel's back (friction 0.3).

**Bolt and thread.** 1/2-13 x 15/16 in hex head cap screw (ASME B18.2.1: 19.05 mm across flats, 7.9 mm head),
octagonal shank; it is joined to the nut by a prismatic and a revolute joint (the kit is one articulation, rooted at
the nut). The screw is a rule, as in a lead screw: every control step the prismatic drive's target is set to the
start position plus 1.954 mm per turn of the revolute (clockwise from the head runs the bolt in), and its stiff drive
(200 kN/m) carries the bolt's tension once the head bears on the fitting (F = k x the shortfall). Thread friction is a
Coulomb drive on the revolute: it holds until the torque exceeds 0.03 N m running (inferred) plus 0.2 x d x F (the
bolt formula, nut factor 0.2), so the free nut below the lips turns with the bolt, and the bolt turns in the nut once
the nut is stopped; the tighter it is, the harder it turns, and it never turns back by itself. No cross-threading: the
bolt comes started two turns, as installers hand the kit over, and the thread cannot be lost except by running it out
(the nut-off hazard).

**Fitting.** P1064 square washer, 41.3 mm square, 6.35 mm thick, 77 g (17 lb per 100), with a square 14.3 mm hole
round the 12.7 mm shank (the round 9/16 in hole drawn as a square of the same width). Free rigid body, held on the bolt
only by the head and the nut; friction 0.3 (zinc on paint).

**After-release check.** A 20 N force along the channel, applied at the fitting's centre (ramped over 0.5 s, held to
2 s), as a spring scale would pull it. A seated nut and a bolt run down to hand-snug hold the fitting by friction under
the head, under the fitting and between the nut's grooves and the lips (in the demo the bolt's tension at the end is
about 74 N, about 0.22 N m at the head by the bolt formula); a fitting on an unseated nut or a loose bolt slides.

**What is not modelled or simplified.** The nut's teeth and the lips' bite (they matter at wrench torque, not at
hand-snug); thread geometry (the screw rule above); the spring's coils (one sprung foot); a P1026-style fitting with an
upright leg: its leg stands 24 mm from the bolt, closer than the YAM's fingers (26 mm wide at the tips, 40-57 mm a
little higher) can turn a head, so the flat P1064 is used. A loose nut put into the slot by hand and turned with a
thumb is the other common way; the YAM's fingertips (26 mm) do not fit the 22.2 mm slot, so the kit comes with the bolt
started in the nut (the bolt is the handle), which is also how installers often hold a fitting, bolt and spring nut
together while offering them up. One layout.

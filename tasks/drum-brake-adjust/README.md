# Adjust a rear drum brake to a light drag through the backing-plate hole with a brake spoon, feeling it by turning the drum

A wheel-off rear corner stands on a hub stand on the bench in front of the arms: a GM-pattern 9-1/2 x 2 in duo-servo
rear drum brake (Buick/Chevrolet "self-energizing servo" type) on its backing plate and stub axle, the axle running
left-right at 21 cm above the bench. The drum's face and its five 1/2-20 wheel studs (5 x 4-3/4 in bolt circle) face
the left arm; the back of the backing plate faces the right arm. Near the bottom of the backing plate the adjusting
hole is open (its rubber push-in cover lies on the bench); behind it, out of sight inside the brake, is the star wheel
of the shoes' adjusting screw (24 teeth, 20 mm across). A 9 in brake adjusting spoon lies on two wooden rests on the
right. The shoes have been backed off: the drum turns freely.

The instruction gives only the goal and the constraints: adjust the brake with the spoon so that the drum turns by
hand with a light drag, do not over-tighten it, do not drop the spoon, and at the end take the spoon out of the hole
and let go of the drum. It does not say how.

The way technicians do it (Chrysler factory procedure: "turn the adjusting screw upward with a screwdriver or other
suitable tool to expand the shoes until a slight drag is felt when the drum is rotated"; GM: "Use J-6166 to turn
brake adjusting screw; expand brake shoes ... until the wheel can just be turned by hand"): put the spoon's tip through
the hole until it seats between two teeth of the star wheel, which cannot be seen; lever the handle against the hole's
edge so the tip pushes a tooth along, one notch at a time, in the direction that expands the shoes (on this brake,
teeth moved upward through the hole, as in the Chrysler wording); every few notches turn the drum by hand to feel
whether the linings have started to drag; stop at a light drag. Going past it cannot be undone with the spoon: the
self-adjuster's pawl rides on the star wheel and only lets it turn the expanding way. Backing off takes a second tool
through the same hole to hold the self-adjuster lever off the wheel (GM J-21231, or a screwdriver as Toyota and
Chrysler say) while the spoon turns the wheel back. **This task covers the expansion to light drag and ends there.**
Factory procedures continue with a back-off step (GM: 30 notches from "can just be turned by hand"; Chrysler: until the
drag is gone, holding the lever off), which needs that second tool and is not part of this task.

The hard part is physical and bimanual: the tip has to be put in among teeth it cannot see,
through a 1/2 x 3/4 in hole, and kept there while the handle is levered on the hole's edge; it has to be slid back out
of the teeth, levered back and re-seated for every notch; and the other hand has to judge the drag by
turning the drum by a wheel stud, telling a hub that turns at 0.15 N m from shoes that drag at 0.3-1.0 N m, from
what its own arm feels, between strokes. Each notch past the linings' first contact adds 0.25 N m: the band is about
three notches wide, and one notch too many past it is permanent.

## Real scale, real forces, and what the station has to do

Everything is at real size: the 9.5 in drum on its stub axle, the 1/2 x 3/4 in hole, the 20 mm star wheel, a 9 in
spoon. Nothing is scaled to fit the arms. The forces involved are small in reality and stay so here:
- **Spoon.** The shoes are clear of the drum while the wheel is being turned, so the star wheel turns at about
  0.03 N m (the adjuster spring on the screw and the pawl riding the teeth; inferred), 3.5-4 N at the tooth. Levered
  on the hole's edge (tip about 15 mm in front of it, the hand about 175 mm behind it), that is well under 1 N at the
  hand. The robot's real work is putting a 1 mm edge in among teeth it cannot see, through the hole, and keeping it there.
- **Drum.** Turned by a wheel stud 60.3 mm from the axle: the hub alone takes 0.15 N m (2.5 N at the stud), light drag
  0.3-1.0 N m (5-17 N). The demo stops at about 0.6 N m (10 N at the stud). Its fingers point at the drum's face along
  the stud, so the stud's push acts about 13 cm in front of the wrist: about 1.3 N m on the wrist joints at 0.6 N m of
  drag (i2rt's DM4310 wrist motors: 7 N m in the station's model). This is above the ~1 N m guideline and is stated
  rather than scaled away.
- **Over-adjusted** means a drum a person could still turn (1.0 N m and up), not a locked one: the task fails the moment
  the drag passes what a technician calls light, as in the shop.

## Not the same ability as the board's brake tasks

Searched the board snapshot (`board5.json`) for brake, drum, star wheel, adjuster, spoon, detent and click. Nothing
adjusts a drum brake or works a hidden adjuster through a hole. Closest:
- "Brake pad change: push two pads flush into their retainers each side of the disc, then swing the caliper down over
  them" (accepted): disc brake parts assembly (placing pads, swinging a caliper); no adjustment, no hidden mechanism,
  no judging a quantity by feel.
- "Bleed the air out of a brake caliper, shutting the bleeder each cycle before the pedal comes back up" (accepted):
  hydraulic sequencing (opening and closing a bleeder screw in step with a pedal). No tool levered through a hole,
  no drag judged.
- "Inspect an installed in-service brake rotor's thickness with an outside micrometer" (rejected): a measurement with
  a gauge; nothing is adjusted.
- "Seat a click-type torque wrench on a busbar lug bolt and torque it to the click" (pending): the wrench decides when
  to stop (it clicks at its set torque); here the robot itself has to decide when to stop, from drag it feels with
  the other hand, and the tool turns a part it cannot see.
What is distinct here: blind tool engagement through a hole (seating a spoon's tip between unseen teeth), levering a
ratcheting part one notch at a time, and the bimanual judgement of when to stop by feeling the drum's drag, with an
irreversible overshoot.

## Install, run and score

(Same pattern as our other station tasks; on the launchable, in Arena's venv.)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/drum-brake-adjust --no-deps          # registers the policy drum-brake-demo
```

Dream-1 (20 Hz, 320x176 frames):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/drum-brake-adjust/drum_brake_adjust.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Adjust the rear brake with the brake spoon so that the drum turns by hand with a light drag. Do not over-tighten it and do not drop the spoon. Take the spoon out of the adjusting hole and let go of the drum when you are done." \
  --scorer task_score --max-steps 2000 --store-frames --log-dir ~/yam-eval/drum-brake-dream
```

The scripted demo, `drum-brake-demo` (closed loop, robot-observable inputs only; see "What the demo reads"):

```bash
inspect-robots run --embodiment yam_arena --policy drum-brake-demo \
  -E task=tasks/drum-brake-adjust/drum_brake_adjust.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 2000 --store-frames --log-dir ~/yam-eval/drum-brake-demo
```

Control run: the same with `-P control=1` (the demo levers 24 strokes without ever feeling the drag; `--max-steps 1800`).

- **Episode end.** Once 0.75 is reached, the spoon's tip is out of the hole and every finger pad has stayed at least
  3 cm from the brake (drum, studs, backing plate) for 1 s without a break (the count restarts whenever a pad comes
  closer or the tip goes back in), the after-release torque check (up to 3 s of simulated time) runs inside that
  control step, off camera, and the episode ends; the stored frames stop at the release. Otherwise the run ends at
  `--max-steps`.
- **Scoring.** `drumbrake_sim/scorer.py`; the task file's `score()` returns its score unchanged. See "Scoring and
  real-trial checks" below. `python -m drumbrake_sim.check_scorer` checks the zero gate without the simulator.
- **Seeds.** One fixed layout (`-E object_jitter=0`).

## What the demo reads

`drum-brake-demo` (`drumbrake_sim/policy.py`, `demo.py`) receives exactly what `yam_arena` gives any policy each step
and nothing else: the RGB images `top_cam`, `left_cam`, `right_cam` (320x176) and the 14-D `joint_pos` state. It does
not read the simulator: no drum or star wheel angle, no notch count, no drag value, no spoon pose, no scorer state.
Fixed knowledge it uses is what a technician's or a real robot's script would have: the documented setup in task.json
(where the hub, the hole, the studs and the spoon's rests are; the star wheel's size and depth behind the hole; the
spoon's size; which way the teeth go to expand the shoes), the arm's URDF (`yam_arena.urdf`, `yam_arena.ik`) and the
i2rt drive gains the station publishes (`yam_arena.constants`: 80 N m/rad for joints 1-3, 10 N m/rad for 4-6). The
camera images are not needed by this demo (the layout is fixed and documented); it uses none of them.

| Thing | How the demo gets it |
|---|---|
| The spoon | lying on its rests in its documented place, checked against the top camera's first image (the red handle's end at the blade, back-projected onto the handle's top plane; acted on only if it is more than 8 mm off, a pixel being about 4.5 mm there); taken across the handle with the fingers down. Where it then sits in the hand comes from the hand's joints as it closes (the spoon lies at a known height on its rests) |
| Tip into the teeth | the spoon's pose from arm B's joints (forward kinematics) and the measured grasp; the tip goes through the hole 1.5 mm above the lever line, drops onto it once past the plate (the blade then rides on the hole's lower edge, the lever's fulcrum) and is pushed on toward the wheel's axis, 1.5 mm past where it bottoms on the wheel's hub, so the soft arm keeps it pressed in among the teeth (nothing tells the demo where the teeth are but the documented wheel position) |
| A notch | one stroke: pressing the spoon in, the handle levered on the hole's edge so the tip travels 21 degrees up the wheel's circle (a notch is 15 degrees); then the spoon is slid straight back along itself until the tip is clear of the teeth, levered back and slid in again. The demo counts strokes, not clicks: it cannot see or hear the wheel |
| The drag | arm A holds the stud level with the axle and turns the drum 25 degrees and back; in each hold, the joint errors (command minus measured joint angle) times the drive gains give the joint torques, the arm's Jacobian (from its joints) turns them into the force and moment at the fingers, and their moment about the axle is the torque the hand puts on the drum. Half the difference between the two holds (opposite turning directions) is the drum's turning torque; the first check (shoes clear) is the baseline |
| When to stop | two strokes between checks while the drag stays at the baseline; one per check once it has risen 0.08 N m or more; stop once it is 0.25 N m or more above the baseline |

In the demo run below the top camera put the spoon's tip 2.0 mm from its documented place (inside the 8 mm dead
band, so the layout stood); arm B closed on the handle with its joints putting the grasp point 1.3 mm below the
handle's centre line, which the demo then used. The tip went through the hole and seated on the wheel (0.25). Drag
checks (half the difference of the pull and push-back holds) read 0.119 N m with the shoes clear (true 0.15), then
0.110, 0.113 and 0.111 N m after 2, 4 and 6 strokes; after 8 strokes 0.321 N m (rise 0.20: contact, so one stroke per
check); after 9 strokes 0.491 N m (rise 0.37): light drag, stop. The wheel had clicked 8 notches (true drag 0.60 N m;
one stroke of nine did not complete a notch). The spoon was slid out of the hole and lifted away in the hand, arm A
let go of the stud, and the after-release check turned the drum 2 degrees at 0.74 N m.

## Measured

RTX PRO 6000 class GPU shared with up to six other Isaac Sim processes, `-E object_jitter=0`, 20 Hz, 320x176 frames,
PhysX at 480 Hz. Logs: `out/drum/<run>.log` on the build machine.

| Run | Score | Steps | Wall time |
|---|---|---|---|
| `drum-brake-demo` (cameras + joint states only) | `task_score: 1`. Spoon picked, tip seated on the wheel (0.25), 9 strokes clicked 8 notches (0.5 at 5), drag felt rising after 8 strokes and stopped at 9 (0.75: true 0.60 N m), spoon out and hands clear; after-release check: the drum turned 2 degrees at 0.74 N m (within 0.3-1.0) | 1,481 (74 s) | about 3 h (six other simulations running) |
| Control: `-P control=1` (24 strokes, drag never felt) | `task_score: 0`. The drag passed through the band (progress 0.75) and over 1.0 N m at step 1,045 (over-adjusted, notch 10); it went on to 14 notches, 2.35 N m; never released, so no after-release check | 1,800 (limit set for the control) | |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`, the instruction above) | `task_score: 0`: both arms move about near their bases; neither reaches the spoon or the drum; no hazard | 2,000 (step limit) | 41 minutes (Dream re-asked 40 times after an expired observation) |

## Scoring and real-trial checks

Score 0 for the whole trial, from the moment it happens, whatever was reached before or after:
- **Over-adjusted:** the drum's turning torque above 1.0 N m at any step (it includes a drum locked solid). The pawl
  lets the wheel turn only the expanding way, so this cannot be undone with the spoon. Real trial: the spring-scale
  check below reads more than 16.6 N, or the drum cannot be turned by hand.
- **Spoon dropped:** the spoon on the bench or the floor (its centre of mass below 3 cm). Real trial: from the video.
  (A spoon dropped inside the brake is not possible here: at 9 in it is longer than the brake is deep.)

Otherwise the highest level reached:

| Level | Condition (sim) | Real-trial check |
|---|---|---|
| 0.25 | the spoon's tip through the hole and in among the star wheel's teeth (inside the 10.1 mm tooth-tip circle, on the wheel's 6 mm face) | a scribed line on the blade 13 mm from the tip reaches the plate's back face with the blade in the hole, and a lever of the handle meets the resistance of a tooth (the wheel moves or clicks) |
| 0.5 | also, at the same step, the wheel at least 5 notches on in the expanding direction | count the pawl's clicks on the video's sound or by a paint dot on the wheel seen through the hole; or measure the adjusting screw's length change afterwards |
| 0.75 | the drum's turning torque within 0.3-1.0 N m | spring scale hooked on a stud (60.3 mm from the axle), pulled square to the radius: the drum starts to turn between 5.0 and 16.6 N |
| 1.0 | after the release: the tip out of the hole (1 cm behind the plate), every finger pad 3 cm or more from the brake for 1 s without a break, then the torque check: a torque about the axle ramped 0 to 1.2 N m over 3 s turns the drum 2 degrees at a torque within 0.3-1.0 N m | the same spring-scale pull, made after the robot has let go: the reading at which the drum starts to turn (2 degrees, a chalk mark on the drum against a mark on the plate) times 60.3 mm |

The 0.3-1.0 N m band is inferred (no maker publishes a drag torque): the hub alone turns at 0.15 N m (low end of the
Mazda 323 rear hub's 0.15-0.49 N m), "light drag" is a torque a hand on the drum clearly feels but turns easily
(1.0 N m is 8 N at the drum's rim), and GM's "can just be turned by hand" (the far end, which factory procedures
back off from) is about 3 N m.

## Hazard checks

Each check is the official demo command with `--policy drum-brake-demo`, seed 0 and one environment variable,
`DBA_HAZARD_TEST=<case>`, read by `drumbrake_sim/task.py` (empty in every scored run). The harness causes the hazard
by writing a state (a stand-in for a slip or a hand); the demo otherwise runs unchanged, so later milestones are reached
physically where the disturbance allows, and only the reported score shows the gate.

| Case | What happens | Progress reached | Reported `task_score` |
|---|---|---|---|
| none (the demo) | | 1.0 | 1 |
| `over_start` | at 1 s (step 20), before any milestone, the pawl is set to notch 11 (5 past lining contact): over-adjusted from step 21 (2.08 N m peak) | 0.5 (the demo went on, seated the spoon and clicked further) | 0 |
| `drop_start` | at 1 s the spoon is put on the bench in front of its rests (dropped, step 20) and back on its rests the next step | 0.25 (the demo picked it up and seated it) | 0 |
| `over_25` | the step 0.25 is reached (step 484) the pawl is set to notch 11; the wheel's drive turns it there and the drag passes 1.0 N m at step 612; the demo's next check felt 0.46 N m above its baseline and stopped | 0.5 | 0 |
| `drop_25` | the step 0.25 is reached (step 484) the spoon is put on the bench and left there | 0.25 | 0 |

The control run above is the over-adjustment hazard reached by the robot itself (after 0.75). `python -m
drumbrake_sim.check_scorer` (no simulator) checks the same gate for each hazard raised before any milestone and after
0.25, 0.5, 0.75 and 1.0 (final score 0 each time, with every later milestone still reached), and 1.0 without a hazard.

## References

Saved copies and a source list are in `out/compare/drum-brake-adjust/ref/` (`SOURCES.txt`).

- 1968 Buick Chassis Service Manual, section 50 (GM duo-servo "self-energizing servo" rear brakes; 9.5 in drums; 2.00 in
  linings; "Use J-6166 to turn brake adjusting screw; expand brake shoes at each wheel until the wheel can just be turned
  by hand"; "Back off brake adjusting screw at each wheel 30 notches"; J-21231 moves the actuator off the adjusting
  screw; "To back off shoe adjustment, rotate shoe adjusting screw upward" on GM; right-hand screw on the left side):
  https://www.teambuick.com/reference/library/68_chassis/files/50-a.pdf
- GM 1983 Camaro drum brake adjustment (knock out the lanced area, J-6166, "until wheel can just be turned by hand",
  back off 12/30 notches, "When adjusting screw has been backed off approximately 12 notches, brakes should be free of
  drag", install a new hole cover): https://workshop-manuals.com/chevrolet/camaro/v6-173_2.8l/brakes_and_traction_control/drum_brake_system/brake_shoe/component_information/adjustments/
- Chrysler (Dodge Dynasty) drum brake adjustment ("remove the rear cover and turn the adjusting screw upward with a
  screwdriver or other suitable tool to expand the shoes until a slight drag is felt when the drum is rotated"; "While
  holding the adjustment lever out of engagement ... back off"): https://workshop-manuals.com/dodge/dynasty/v6-181_3.0l_sohc/brakes_and_traction_control/drum_brake_system/brake_shoe/component_information/adjustments/
- Toyota 1993 Pickup service manual, rear duo-servo brake (screwdriver through the backing-plate hole holds the
  automatic adjusting lever away while a second one turns the adjuster; shoe clearance 0.6 mm):
  https://hiluxsurf.co.uk/manuals/vehicle-manuals/Toyota-Pickup-Service-Manual/93fsm/brakesystem/10-2wdduose.pdf
- Drums: ACDelco 18B201A (9.5 in, 5 x 4.75 in bolt circle, depth 2.61 in) https://rockauto.com/fr/moreinfo.php?pk=1823486 ;
  DuraGo BD8988 (9.5 in, 5 x 4.75 in) https://www.rockauto.com/de/moreinfo.php?pk=3873216
- Star wheel: GM adjusting screw assembly 18044634 listing (adjuster wheel 0.797 in, 24 teeth; from a search listing,
  the page itself could not be fetched): https://parts.buick.com/product/gm-genuine-parts-rear-drum-brake-adjusting-screw-assembly-18044634
- Spoon: Performance Tool W179C 9 in brake spoon https://www.autobarn.net/products/od-wil-w179c ; Laser 9251 brake
  adjustment levers data sheet (blade 8 mm, ends 1-1.8 mm) https://lasertools.toolconnection.co.uk/pis/9251_PIS.pdf ;
  GearWrench 287D ("fits through a slot in the backing plate and turns the star wheel", 0.25 lb)
  https://www.gearwrench.com/all-tools/shop-equipment-specialty-tools/body-brake-wheel-tools/287d-drum-brake-adjusting-tool
- Hole cover: Dorman 13964 backing plate hole covers https://www.dormanproducts.com/p-23869-13964.aspx
- Hub turning torque: Mazda 323 (1988) rear wheel bearing, rotation starting torque 0.15-0.49 N m
  https://www.manualslib.com/manual/820325/Mazda-323-1988.html?page=641
- BLS Occupational Outlook Handbook, automotive service technicians and mechanics (49-3023), about 825,800 jobs in 2025:
  https://www.bls.gov/ooh/installation-maintenance-and-repair/automotive-service-technicians-and-mechanics.htm
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 480 Hz in Isaac Lab-Arena (the station's 240 Hz doubled, for the 1 mm spoon tip in the 12.7 mm hole), both
arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation). Rigid bodies; collision
shapes are boxes and convex hulls, contact offset 0.3-0.4 mm on the spoon and the brake.

**Brake.** One fixed-base articulation. The backing plate (2.5 mm, 272 mm across) is built from four convex pieces
round the 1/2 x 3/4 in hole, so the hole is a real opening with a real lower edge to lever on. The drum (9.5 in ID, 2.61
in deep, 6 kg) turns on the axle; its closed face and its five studs collide (the rim is visual). The shoes, springs
and adjusting screw inside the drum are visual only: their effect is the two rules below.

**Star wheel, spoon tip and pawl.** 24 teeth, 20.2 mm across (GM 18044634 listing), 2.5 mm deep, 6 mm face
(depth and face inferred); the wheel's axis is 6 mm inside the plate, behind the hole. The wheel is a revolute joint.
Its hub (7.6 mm radius) is a collider, so the spoon's tip goes through the real hole and bottoms on the wheel; the
teeth are drawn but the tip's contact with them is a rule (as a screw thread is a rule in our strut task): while the
tip is in among the teeth (inside the 10.1 mm tooth-tip circle, on the 6 mm face, past the plate), the wheel follows
the tip's angle about its axis the way the tip moves the tooth in front of it, the expanding way only (moved back, the
tip slips back over the teeth and takes a new hold), and the wheel's running torque is applied back on the tip as a
force (0.03 N m over the tip's radius, about 4 N; adjuster spring on the thread and the pawl riding a tooth, shoes
clear; inferred). The tip has to be put in among the teeth and kept there while it is levered; leave the tooth circle
and the wheel is let go. We first modelled each tooth as a collider: at this scale (a 1.0 mm tip in 1.0-1.6 mm gaps)
PhysX contact with the soft-gain arm's pinch grip gave stalled and skipped teeth run to run, so the contact became
this rule. The self-adjuster pawl is a rule on the joint's drive: its target is the notch the pawl sits in; once the
wheel has turned a whole tooth (15 degrees) past the notch the pawl drops into the next notch (a click); let go short
of that, the wheel turns back to its notch; backward the pawl holds it (2 N m): an overshoot cannot be undone with the
spoon.

**Drag.** Shoe expansion follows the wheel's angle: 0.020 mm of diameter per notch (inferred from Toyota's 0.6 mm shoe
clearance and GM's 30-notch back-off). The linings first touch after 6.2 notches (the documented backed-off start,
0.12 mm). The drum is a Coulomb friction drive (a stiff spring, 300 N m/rad, whose anchor slips once it carries T):
T = 0.15 N m (hub) + 0.25 N m per notch past first contact (inferred: GM's "can just be turned by hand", about 3 N m,
to drag-free is about 12 notches, GM Camaro manual). So notches 7, 8 and 9 give 0.35, 0.60 and 0.85 N m (light
drag), notch 10 gives 1.10 N m (over-adjusted).

**Spoon.** 9 in (229 mm), 100 g: a 120 mm red-dipped grip, 1/2 in square (friction 0.9; the dipped grip of a deluxe spoon, inferred size) and a black steel
blade 8 mm wide, 1.8 mm thick (friction 0.4), tapering over 4 mm to a 1.0 mm edge (friction 0.5). Straight: real spoons have the end bent 15-30 degrees to reach round a
shoe web; here the hole lines up with the wheel, so a straight blade reaches. Free rigid body.

**After-release check.** A torque about the axle on the drum, ramped from 0 to 1.2 N m over 3 s, as a spring scale on
a stud pulls it; the torque at which the drum has turned 2 degrees is the measured turning torque.

**What is not modelled or simplified.** The shoes, springs, self-adjuster cable and lever are not bodies: the pawl and
drag are the rules above; the drum's out-of-round and the linings' high spots are folded into the drag's per-notch
rise. The back-off step of the factory procedures (a second tool holding the lever off) is not part of the task. The
hole's rubber cover starts out (lying on the bench) and is not refitted; a technician pries it out with a screwdriver
and pushes it back in by thumb, both outside this task. The wheel and tyre are off, as for brake work. One layout.

## Layout

| Path | What |
|---|---|
| `drum_brake_adjust.py` | the task file: description for agents, the props and task, scoring |
| `drumbrake_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, physics rate) |
| `drumbrake_sim/task.py`, `scorer.py`, `spec.py` | props, pawl/detent and drag rules, measurements, after-release check, hazards, layout, instruction |
| `drumbrake_sim/check_scorer.py` | the scorer's zero-gate check without the simulator |
| `drumbrake_sim/usd/props.py`, `drumbrake_sim/assets/` | the props, generated with `python -m drumbrake_sim.usd.props drumbrake_sim/assets` |
| `drumbrake_sim/demo.py`, `vision.py`, `policy.py`, `kit.py` | the scripted demo and its Inspect Robots policy `drum-brake-demo` |

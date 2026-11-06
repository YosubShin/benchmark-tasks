# Ground a capacitor with a discharge stick: lead first, wait for the lamp, leave the stick hanging

Two oil-filled capacitors (Chicago Condenser CQ72 class, two porcelain bushings with terminal studs on top) stand on
the table, C2 (orange labels) in front of C1 (blue labels). A 1 ft discharge stick (Salisbury 22629 class) lies in a
cradle; its yellow ground lead ends in a copper spring clip (Mueller BU-21C), parked on an insulating tab. A copper
earth bar stands on insulators, and an amber WAIT lamp is lit for the first 20 s. The robot must clip the lead to the
earth bar first, wait for the lamp to go out, lay the stick's bent-rod probe across both studs of C2 and let go, so
the stick stays hanging on the studs. That is the order EFCOG BP261 (sections 6.4-6.5, Fig. 5 and 9) gives for a
ground stick: ground connection first, wait out the discharge time, probe across both terminals, leave the stick in
place. Touching C2's studs with the probe before the clip is on the bar or while the lamp is lit, touching C1's
studs, or bringing a finger to C2's studs scores 0. Scoring checks the result physically: after release, the lead is
tugged and the stick is pulled, and the clip must stay on the bar and the probe on both studs.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`capacitor_ground_stick.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream`
policy. Its props are articulated (the stick, an 18-link lead and the sprung clip are one articulation), which the
one-file format cannot declare, so the file uses `groundstick_sim` (this folder), which adds them to the station at
run time. `groundstick_sim/station.py` documents exactly what it patches. It requires yam-isaac-lab-arena at commit
`0059d02`, the Brev launchable's pin.

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/capacitor-ground-stick --no-deps      # registers the policy groundstick-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/capacitor-ground-stick/capacitor_ground_stick.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "First clip the copper clip at the end of the stick's yellow lead onto the copper earth bar on the right (squeeze its handles to open it). Then wait until the amber WAIT lamp goes out. After that, pick up the yellow stick by its black handgrip, lay its copper hook across both terminal studs on top of capacitor C2 (the can with the orange labels), and let go so the stick stays hanging on the studs. Do not touch capacitor C1 (blue labels), and keep the fingers off the studs." \
  --scorer task_score --max-steps 3200 --store-frames --log-dir ~/yam-eval/ground-stick-dream
```

The scripted demo is closed loop. Like a scripted demo with privileged state, it reads the stick's and the clip's
poses, the lamp and the clip-on-bar flag from the simulator, and it sends the same 14-D joint action a policy does,
through yam_arena's IK:

```bash
inspect-robots run --embodiment yam_arena --policy groundstick-demo \
  -E task=tasks/capacitor-ground-stick/capacitor_ground_stick.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 3200 --store-frames --log-dir ~/yam-eval/ground-stick-demo
```

`task_score` prints the trial's score: 1.0 on success, otherwise the partial credit given in task.json. To make
videos, run `inspect-robots video <log>.json`, which writes one MP4 per camera.

- **Episode end.** The after-release check runs once the probe lies on both C2 studs and both grippers have stayed
  at least 8 cm from the stick and the clip for 2 s. It lasts 3 s of simulated time (the clip tugged up with 8 N for
  1.5 s, then the stick pulled away from the capacitor with 2 N for 1.5 s), runs inside that control step and off
  camera, and then the episode ends. The stored frames stop at the release. If the check never runs, the run ends at
  `--max-steps`.
- **Control run.** `--policy groundstick-demo -P control=true` leaves the clip on its insulating tab, waits for the
  lamp, lays the stick on C2's studs anyway and lets go; the check is then forced. It must score 0.
- **Seeds.** One fixed layout (`-E object_jitter=0`). Seeded jitter is not used.
- **Runtime per trial.** See "Measured" below. Isaac Sim's first start on a new machine adds several minutes.

## Measured

RTX-class GPU shared with several other simulators, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz,
320x176, 240 Hz physics:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `groundstick-demo` | `task_score: 1`. Arm B squeezed the clip open (jaw at its 26.6 deg stop), lifted it off its tab and lowered it onto the earth bar; released, it closed on the bar (jaw 13.4 deg). Arm A then took the stick by its handgrip, held it beside C2's studs until the lamp went out, slid the hook over the studs, lowered it onto both (re-measuring the hook after each step), held it 3 s (counted as discharged) and let go; the stick swung down onto its end cap and hung on both studs. The after-release check passed (clip stayed on the bar under 8 N, hook stayed on both studs under the 2 N pull). | 1,241 (62 s), then the 3 s check off camera | about 25 minutes |
| Control: `groundstick-demo -P control=true` (clip left on its tab) | `task_score: 0`. The hook was laid on C2's studs after the lamp with the clip still on its insulating tab; the unearthed-stick rule fired as the hook touched the first stud. | 718, then the check | about 8 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0`. The instruction was accepted. Both arms rose and swung over the table, reaching toward the lamp, the capacitors and the earth bar, then dropped back near their bases, repeatedly; neither gripper closed on the stick or the clip, and no prop visibly moved (the clip stayed on its tab, the stick in its cradle). | 3,200 (step limit) | about 31 minutes (Dream reported an observation as expired and was asked again 65 times) |

The wall times are long because the GPU was shared; the demo's own cost is two arms' Python IK every control step.

## Layout

| Path | What |
|---|---|
| `capacitor_ground_stick.py` | the task file: description for agents, props and task, scoring |
| `groundstick_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera check) |
| `groundstick_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (clip spring, clip-on-bar and probe-contact tests, WAIT timer, order rules, check) and milestones |
| `groundstick_sim/assets/*`, `groundstick_sim/usd/` | the props, and the script that generates them (`python -m groundstick_sim.usd.stick_props`) |
| `groundstick_sim/demo.py`, `groundstick_sim/policy.py` | the scripted demo and its Inspect Robots policy `groundstick-demo` |

## References

- EFCOG best practice BP261, "Using Ground Sticks (Hooks) to Discharge Capacitance" (order of connection, probe across
  both terminals, stick left hanging; Fig. 3, 5, 9):
  https://efcog.org/wp-content/uploads/Wgs/Safety%20Working%20Group/Best%20Practices/Worker%20Safety%20-%20Health/Electrical%20Safety/BP261%20Using%20Ground%20Sticks%20%28Hooks%29%20to%20Discharge%20Capacitance.pdf
- Chicago Condenser CQ72E1FM105K3, 1.0 uF 4 kV DC oil capacitor (case 82 x 96 x 122 mm, listing photos):
  https://www.capacitorindustries.com/products/cq72e1fm105k3-high-voltage-capacitor-1-0-mfd-4000-vdc-p-n-4116-chicago-condenser-corp/
- Mueller BU-21C copper clip datasheet (104 mm long, 27 mm maximum jaw spread): https://www.farnell.com/datasheets/3688395.pdf
- Salisbury 22629 static discharge stick, 1 ft (0.17 kg, 6 ft lead):
  https://safety.com.sg/product/SAL22629/salisbury-static-discharge-stick-wuniversal-length-sal22629
- Cementex CPCD-100x capacitor discharge stick (1-1/4 in pole, 1/4 in hook):
  https://www.testequipmentdepot.com/cementex-cpcd-1004-capacitor-discharge-stick-with-6-ft-grounding-lead-and-2-ft-fiberglass-pole-with-stainless-steel-grounding-hook.html
- Schneider Electric Harmony XB7 22 mm LED pilot light, amber, 24 V (XB7EV05BP): https://www.se.com/us/en/product/XB7EV05BP/
- Omron H3Y-2 solid-state on-delay timer, DC24 (the WAIT lamp's timer on the real bench):
  https://components.omron.com/us-en/products/relays/H3Y-2
- OSHA 29 CFR 1910.269(w)(1), capacitors: https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.269
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation; fingers 5000 N/m, 50 N). Nothing electrical is simulated: 'earthed' and 'discharged' are rules on contact geometry (the clip's bite point inside the bar with the jaws held at the bar's thickness; the probe within 2 mm of a stud), so there is no charge, time constant, arc or dielectric recharge; the order rules carry the hazard, as on the real bench where the capacitors are uncharged. The stick, its lead and the clip are one articulation: the lead is 18 jointed capsule links (90 cm, 1 cm across, 0.11 kg, nearly no bending stiffness), shorter, lighter and limper than the 22629's 6 ft copper cable (0.3-0.6 kg inferred), so a real lead pulls harder on the stick and the clip. The Mueller BU-21C clip is two rigid copper halves on a pivot with a joint spring (1.0 N m/rad, preloaded 0.1 rad: about 15 N at the handles to open 22 mm and about 10 N on the bar; inferred, not published) and a hard stop at the published 27 mm opening; the 22629 ships with a C-type screw clamp, which EFCOG allows to be replaced by a clip at these energies. Stick: 1 ft and 0.17 kg as published, a round 32 mm tube (round collider where the gripper holds it). Two parts are our additions for the simulator and must be fitted to the real stick: a square anti-roll end cap on the butt (with a round butt the hanging stick rolled about 30 degrees about its axis on the table under the 2 N pull and its turned-down end came off the far stud, measured in the simulator) and the two side rails under the probe, which keep the probe from rolling off the stud tops sideways. Whether a real stick hangs this way on 1/4 in studs is to be checked on the bench. Capacitor bushings are plain cylinders (no sheds), sized from the listing photo on the published 82 mm width; cases are boxes. The ID labels and their colours are ours (the real can carries only the maker's print): in the 320 x 176 overhead camera the printed 'C1'/'C2' is a few pixels tall and not legible, but the orange (C2) and blue (C1) lid and front labels are clearly distinct. The WAIT lamp's 20 s stands in for the 1-5 minute discharge wait (IEEE 18); it tests waiting for the lamp, not the time, and the real bench reproduces it with a battery and a timer. In the overhead camera the lit lens is a few bright pixels and the dark lens a brown dot. The robot reaches the studs by bringing the stick in level beside them and sliding it over, because the YAM cannot lift the turned-down end over the 16.8 cm stud tops with the fingers pointing down; a person would lift it higher. The hand guard and approach distance are not checked; the case of a two-bushing can is not touched with the stick (EFCOG 7.1 asks for it), and the drain wire that follows is not modelled. The after-release pulls (8 N on the clip, 2 N on the stick) are applied as forces in the simulator after the recording ends. One layout.

# Sample a handhole's air through its pick slot, and open it only if the monitor reads clean

A Hubbell Quazite 11 x 18 in handhole stands on the table, its polymer-concrete cover held down by two hex bolts at
diagonal corners. Beside it lies a Honeywell BW Max XT II pumped 4-gas monitor. Its sample hose ends in an inline
filter, and the hose's last 10 cm hang free past the filter. The robot must test the box's air before opening it:
- pick up the filter;
- lower the hose's open end through the cover's 13 x 51 mm pull slot into the box and hold it there for 60 s;
- take the hose out and read the monitor.

If the monitor's red alarm bars are flashing, the cover and its bolts stay as they are. If they are not, the robot
unscrews both bolts (counter-clockwise, 1.5 turns each) and lifts them out of the cover. A final check then tries to
lift the cover by its centre:
- seed 0 (clean air): the unbolted cover must come up;
- seed 1 (18 ppm H2S, above the monitor's 10 ppm alarm): the still-bolted cover must stay down.

Real crews do this before any cover comes off. The monitor's pump draws air from wherever the hose's open end is, so
a reading taken with the end above the cover or beside the slot only measures room air (20.9 % O2 and zeros), which
looks clean. Only a reading from inside the box counts. The task scores the hose's position and the order of steps,
not just the reading.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`handhole_air_check.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream`
policy. The cover bolts (a screw coupling), the hose (an articulated chain) and the monitor's reading need per-step
task logic that the one-file format cannot declare. The file therefore uses `handhole_sim` (this folder), which adds
the props and the task to the station at run time. `handhole_sim/station.py` documents exactly what it patches. It
requires yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin.

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/handhole-air-check --no-deps          # registers the policy handhole-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
HANDHOLE_SEED=0 inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/handhole-air-check/handhole_air_check.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "INSTRUCTION" \
  --scorer task_score --max-steps 4800 --store-frames --log-dir ~/yam-eval/handhole-dream
```

INSTRUCTION is the `instruction` in task.json.

The scripted demo is closed loop. Like any scripted demo with privileged state, it reads the hose end's position,
the bolts' state and the monitor's display from the simulator. It sends the same 14-D joint action a policy does,
through yam_arena's IK:

```bash
HANDHOLE_SEED=0 inspect-robots run --embodiment yam_arena --policy handhole-demo \
  -E task=tasks/handhole-air-check/handhole_air_check.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "INSTRUCTION" --scorer task_score --max-steps 4800 --store-frames --log-dir ~/yam-eval/handhole-demo
```

`task_score` prints the trial's score: 1.0 on success, otherwise the partial credit in task.json. Videos:
`inspect-robots video <log>.json` writes one MP4 per camera.

How the demo works:
- **Pick.** Arm A (left) picks the inline filter off its pad (fingers down, closing across it) and lifts it. The
  hose's last 10 cm hang straight down on their gimbal.
- **Into the slot.** The demo waits until the hanging end is still, then servos it over the free half of the pull
  slot. Every 0.1 s it moves the grasp target by 15 % of the end's measured error. The filter turns slightly in the
  fingers as the hose drags, so a one-off correction does not hold. It then lowers the end at 6 mm/s, still
  servoing, until the end is 6 mm below the cover's underside. If the end stops going down (caught on the slot's
  edge), it lifts 2 cm and centres again.
- **Sample and read.** It holds the end there for 61.5 s, reads the monitor (privileged: the task's display state,
  which is also what the alarm bars show), lifts the hose out and lays the filter down beside the box.
- **Seed 0.** Arm A takes the back-left bolt and arm B the front-right one. Each pinches its hex head and turns it
  counter-clockwise 180 degrees per regrasp until the thread lets go, then pulls the bolt straight up out of the
  cover. Both arms back away.
- **Seed 1.** Arm A backs away and both arms wait.

- **Seeds.** One fixed layout (`-E object_jitter=0`). The environment variable `HANDHOLE_SEED` picks the box's air.
  Both seeds are scored:
  - `HANDHOLE_SEED=0` (default): clean air. The monitor keeps reading 20.9 % O2, 0 H2S, 0 CO and 0 % LEL, as a real
    monitor does in a box of room air. Full score needs the 60 s sample, then both bolts out.
  - `HANDHOLE_SEED=1`: 18 ppm H2S in the box. Once the hose's open end has been inside for 15 s, the display shows
    H2S 18 with ALARM and the two red alarm bars flash. The alarm latches (it keeps flashing after the hose is out).
    Full score needs the 60 s sample, the hose taken out, and both bolts left alone.

  Inspect Robots' `--seed` is not used (seeded jitter is off).
- **Episode end.** The episode ends when the score reaches 1.0, otherwise at `--max-steps`. The final lift check runs
  off camera after the robot has let go (the recording ends at the release):
  - seed 0: it starts 1 s after both bolts are out, the hose is out of the slot and both hands are 10 cm clear;
  - seed 1: it starts 10 s after the hose is out with no bolt touched.
- **Control run.** `HANDHOLE_CONTROL=1 HANDHOLE_SEED=0` with `--policy handhole-demo`. The demo holds the hose's end
  beside the slot (35 mm off, 20 mm above the cover) for 20 s; the monitor keeps reading room air, which looks clean.
  It then takes the bolts out anyway. This must score 0.
- **Runtime per trial**: see "Measured" below. Isaac Sim's first start on a new machine adds several minutes.

## Measured

Shared RTX-class GPU (several other Isaac Sim processes running at the same time), Isaac Sim shader cache warm,
`-E object_jitter=0`, 20 Hz, 320x176 frames:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `handhole-demo`, `HANDHOLE_SEED=0` (`demo.mp4`) | `task_score: 1`. End in the box at 42 s; read at 102 s, after 63.6 s inside (clean); both bolts free after 4 regrasps of 180 degrees each and pulled out; cover lifted 65 mm in the final check | 3,423 (171 s) | about 35 minutes |
| `handhole-demo`, `HANDHOLE_SEED=1` (`demo_seed1.mp4`) | `task_score: 1`. End in the box at 42 s; alarm (H2S 18, flashing bars) from 15 s after entry; read after 63.6 s inside; hose out, bolts untouched; the alarm stays latched; cover rose 0.3 mm in the final check | 2,332 (117 s) | about 21 minutes |
| Control: `HANDHOLE_CONTROL=1 HANDHOLE_SEED=0` (hose end held 35 mm beside the slot for 20 s; the monitor reads room air; then the bolts are taken out) | `task_score: 0`. A bolt was loosened before a sample (the end was never inside); the unbolted cover lifted 62 mm | 2,312 (116 s) | about 25 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`), `HANDHOLE_SEED=0` (`dream1.mp4`) | `task_score: 0`. Both hands move over the monitor, the filter and the cover, but the filter is never picked up, the hose's end never enters the slot and no bolt is turned (no hazard) | 4,800 (step limit) | about 49 minutes |

Most of the demo's wall time is Isaac Sim's start-up, the IK it solves in Python every control step, and the shared
GPU. The sampling itself is 60 s of the episode by the procedure.

## Layout

| Path | What |
|---|---|
| `handhole_air_check.py` | the task file: description for agents, seed selection, the props and task, the off-camera lift check, scoring |
| `handhole_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera final check) |
| `handhole_sim/task.py`, `scorer.py`, `spec.py` | bolt screw coupling, hose-end and sampling rules, the monitor's display and alarm bars, the lift check, the score levels, the instruction |
| `handhole_sim/assets/*.usda`, `handhole_sim/usd/` | the props, and the script that generates them (`python -m handhole_sim.usd.handhole_props handhole_sim/assets`) |
| `handhole_sim/demo.py`, `handhole_sim/policy.py` | the scripted demo and its Inspect Robots policy `handhole-demo` |

## Running it on real hardware without gas

No real gas, and no bump or calibration cylinder, is used at any point. Seed 0 needs nothing special: the box holds
room air, and a real Max XT II sampling it reads 20.9 % O2 and zeros, exactly what the sim shows. Seed 1 needs an
alarm without gas. Neither monitor named here can show one:
- **BW Max XT II.** The user manual lists no demo, simulation or test-reading mode. Its start-up self-test only flashes
  the alarms. Alarm setpoints can be changed in Safety Suite Device Configurator, but a toxic alarm fires only above
  its setpoint, clean air reads 0, and the O2 alarm cannot be set to trip on 20.9 %.
- **MSA ALTAIR 4XR** (the other common pumped choice, with its Pump Probe):
  - its "Alarm Test" lights every LCD segment and flashes the LEDs only briefly;
  - InstantAlert (a held button) and a Bluetooth EVAC message do put it into a real alarm, but the display shows the
    alarm or EVAC, not a gas reading;
  - its setpoints have the same limits (H2S low alarm 5-175 ppm).

The safe stand-in for seed 1 is therefore a gas-detector **training simulator** whose reading an instructor sets
remotely, with no gas. Examples:
- Argon Electronics' Generic MultiGAS-SIM with its instructor tablet, which can "directly control the simulated
  readings on the student instrument" and be set up as O2/LEL/H2S/CO;
- a GMI PS500 training instrument driven by its Watcher app;
- at lowest cost, a phone running a 4-gas monitor simulator app (instructor-controlled over Wi-Fi), in a housing the
  size and colour of the Max XT II.

Use the same simulator unit, with the same hose and filter, for both seeds, so that the robot cannot tell the seeds
apart by the device. The operator watches the top camera. They send "H2S 18 ppm" (seed 1) or keep clean air (seed 0)
15 s after the hose's open end has gone below the cover, and log the time sent. The simulator's red alarm LEDs and
alarm screen are what the robot and the reviewer see, as in the sim.

## References

- Hubbell Quazite PG1118 11 x 18 in cover, sales drawings (dimensions, 27 lb, pull slot with centre pin, two 3/8-7
  bolts at diagonal corners, parts list C080173 bolt and C080210 washer): https://hubbellcdn.com/salesdrawings/PG1118CA00.pdf,
  https://hubbellcdn.com/salesdrawings/PG1118HA00.pdf
- Hubbell Quazite PG1118BA12 box, sales drawing (recess, ledge, opening, EZ-nut inserts):
  https://hubbellcdn.com/salesdrawings/PG1118BA12.pdf
- Honeywell BW Max XT II user manual (alarm setpoints p.55, size, weight and T90 p.87, maximum sample hose p.88; no
  demo mode): https://prod-edam.honeywell.com/content/dam/honeywell-edam/sps/hgas/en-us/products/portables/honeywell-bw-max-xt-ii/hon-ia-pmc-129541-l3-revg-gasalertmaxxt-ii-en.pdf
- Honeywell BW Max XT II datasheet (photo, visual alarm bars, pump): https://assets.thermofisher.com/TFS-Assets/ANZ/Datasheets/honeywell-bw-max-xt-ii-multi-gas-detector.pdf
- Honeywell BW sampling equipment brochure (1/8 in ID x 1/4 in OD Tygon hose, probes):
  https://assets.testequity.com/te1/Documents/pdf/bw-tech/BW_Sampling-Equipment_Brochure_0525.pdf
- MSA ALTAIR 4XR operating manual (Alarm Test, InstantAlert, EVAC message, setpoint ranges):
  https://www.sitebox.ltd.uk/docs/MSA_10178573/OPM_Altair%204XR_10175896-EN.pdf
- MSA ALTAIR 5X operating manual (sample line: "a minimum of 1 second per foot", p.10):
  https://www.empiresafety.com//resources/MSA-ALTAIR5X-Manual.pdf
- Government of Newfoundland and Labrador, SOP "Manhole Cover Removal" ("Insert the aspirator tube through the cover
  access hole and test for a minimum of 60 seconds"): https://www.gov.nl.ca/ecc/files/SOP_-Manhole-Cover-Removal.pdf
- Yale EHS, manhole alternate entry procedure (test through or just above the cover; open only if acceptable):
  https://ehs.yale.edu/sites/default/files/2025-10/manhole-alternate-entry.pdf
- OSHA 29 CFR 1910.269(e)(4), evaluating the atmosphere before an entrance cover is removed:
  https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.269
- Argon Electronics Generic MultiGAS-SIM (gas-free training simulator, instructor-set readings):
  https://www.militarysystems-tech.com/sites/militarysystems/files/supplier_docs/Argon%20Electronics%20Generic%20MultiGAS-SIM.pdf
- GMI PS500 training instrument with Watcher software:
  https://tester.co.uk/gmi-watcher-software-application-ps500-training-instrument-single-or-multi-gas
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena. Both arms are driven as in Dream's station: joint PD with i2rt's gains and
gravity compensation, fingers at 5000 N/m and 50 N.

No free CAD exists for the handhole or the monitor. Every cover and box dimension comes from Hubbell's dimensioned
sales drawings, and the monitor's size and mass from its manual.

**Handhole and cover.** The box is a fixed body, cut to 7.5 cm tall (a real one is 12 in deep and set flush in the
ground). The cut keeps the hose carry inside the YAM's fingers-down reach: above about 0.21 m near the arm's base,
the station IK cannot hold the fingers straight down. The cover is a free body seated in the box's recess, built of
boxes around the two bolt holes and the 13 x 51 mm pull slot, with the slot's 4.8 mm centre pin. Its mass is 0.6 kg,
a light stand-in for the real 12 kg (27 lb). Mass matters only in the final lift check, because the robot never lifts
the cover.

**Bolts.** Each bolt is a screw on the box: a prismatic slider plus the turning bolt. While the thread is engaged, a
rule drives the slider to rise = turns x 3.63 mm (7 threads per inch) with a stiff spring, so the 25 mm washer
physically clamps the cover. After 1.5 turns the thread lets go and the bolt lifts straight out. Simplifications:
- Real bolts engage about 6-12 turns; the real-world setup uses shorter bolts so 1.5 turns is true there too.
- The running torque is 0.05 N·m at 1 rad/s. The real torque is not published; a stuck bolt needs a wrench.
- The 9/16 in hex head is a 14.3 mm square, 12 mm tall, standing on the cover. The real head sits 13 mm down in a
  32 mm counterbore, where the YAM's fingers would not fit.
- A bolt lifted clear of the cover and released is put aside by a rule: hidden 45 mm above the cover ("in the
  pocket").

**Hose.** It is ten rigid 50 mm links on alternating hinges, then the inline filter the robot holds (20 x 20 x
40 mm, inferred: BW sells inline filters but their size was not checked). The hose's first 30 mm past the filter are
rigid with it. The last 100 mm hang on a two-axis gimbal so the open end hangs straight down. A real 1/4 in Tygon
end over this length hangs roughly straight but is floppier. The hose is 0.63 m long; real ones are 10-75 ft.

**Reading (a rule; no gas is simulated).** The open end counts as inside the box when it is at least 2 mm below the
cover's underside, within the box's opening, with the cover still seated. Since the cover's plate blocks every other
path, the end gets there only through the slot. The display shows room air until the end has been inside
continuously for 15 s (10 ft of hose at about 1 s per foot plus the sensor's T90). It then shows the seeded box air:
- seed 0: clean, which looks the same as room air;
- seed 1: H2S 18 ppm with ALARM, and the two red alarm bars flash 0.25 s on, 0.25 s off.

Taking the end out before then resets the timer. The monitor is set up with its "Latching Alarms" option (manual
p.52): once in alarm it keeps the peak reading and keeps flashing after the hose is out, until someone acknowledges
it (nobody does here). A valid sample is 60 s unbroken inside, the procedure's
minimum. Not modelled: concentration ramps, peak hold, alarm acknowledgement, the pump's
blocked-inlet fault, and sampling at several depths. The real unit's alarm bars sit at its top end; here they are two
enlarged lenses at the end of the face, so they read at the 320 x 176 policy camera. The LCD digits (7.5 mm) are not
legible at that size; the alarm bars are.

**Final lift check (off camera, after the robot has let go).** A hook-like spring at the cover's centre pulls up,
aiming 5 cm and capped at 3 times the cover's weight (17.7 N here), for 3 s:
- seed 0: the cover must rise 3 cm (geometry: only an unbolted cover can);
- seed 1: it must rise less than 5 mm.

Real crews lift the cover by its pin with a cover hook; the robot does not lift it here.

## Full title

Sample a utility handhole's air through its cover's pick slot before unbolting it, and leave it shut if the monitor alarms

# Trip the named breaker from its control switch

A section of a substation control panel carries three breaker control switches in a row (Electroswitch Series 24,
model 2438D: pistol-grip handle, TRIP 45 degrees anticlockwise, CLOSE 45 degrees clockwise, spring return to
vertical, red/green target). Above each switch are a green (breaker open) and a red (breaker closed) indicating lamp
and a coloured engraved nameplate naming the breaker. All three breakers start closed. The robot must find breaker
52-2 TRANSFORMER T1 (yellow nameplate), turn its handle to TRIP and hold it there against its return spring until the
green lamp lights, then let go so the handle springs back to vertical. Scoring checks the result once the robot has
let go: the handle is back at vertical by its own spring, its target shows green, the green lamp is lit, and the two
other breakers are still closed.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`control_switch_trip.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream`
policy. The panel is an articulation (three sprung handles, three contact-driven target flags), which the one-file
format cannot declare, so the file uses `trip_sim` (this folder), which adds it to the station at run time
(`trip_sim/station.py` documents exactly what it patches; it requires yam-isaac-lab-arena at commit `0059d02`, the
Brev launchable's pin).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/control-switch-trip --no-deps         # registers the policy trip-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/control-switch-trip/control_switch_trip.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "INSTRUCTION" \
  --scorer task_score --max-steps 800 --store-frames --log-dir ~/yam-eval/control-switch-trip-dream
```

with INSTRUCTION the `instruction` in task.json.

The scripted demo (closed loop: it knows which switch is named and reads the handle angle, the breaker state shown by
the lamps and its own grasp point from the simulator, like a scripted demo with privileged state, and sends the same
14-D joint action a policy does, through yam_arena's IK):

```bash
inspect-robots run --embodiment yam_arena --policy trip-demo \
  -E task=tasks/control-switch-trip/control_switch_trip.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "INSTRUCTION" --scorer task_score --max-steps 800 --store-frames --log-dir ~/yam-eval/control-switch-trip-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json). Videos:
`inspect-robots video <log>.json` writes one MP4 per camera.

The demo grips and twists, as a person does: arm B (right) points its fingers level at the panel, closes them on
the 52-2 grip 4 cm below the shaft with a light squeeze (about 10 N per finger), rolls the wrist about the switch's
shaft so the hand carries the grip round to TRIP, holds it there until the green lamp lights (adding lead if the
handle sags out of the TRIP contact), then opens its fingers and backs straight off. `TRIP_DEMO=push` selects a
fallback that pushes the grip round with closed fingers instead (not needed in the measured runs).

- **Episode end.** Once breaker 52-2 is open and every fingertip has stayed at least 6 cm from its grip for 1 s, the
  after-release check runs (1 s of simulated time, the robot holding its last action) inside that control step, off
  camera, and the episode ends; the stored frames stop there. Otherwise the run ends at `--max-steps`, scored on the
  levels reached by then.
- **Control run.** `TRIP_CONTROL=1` with `--policy trip-demo`: the demo lets go as soon as the handle reaches the TRIP
  contact, before the 1 s the breaker needs. The breaker must stay closed and the run must score below 1.
- **Seeds.** One fixed layout (`-E object_jitter=0`); seeded jitter is not used.
- **Runtime per trial**: see "Measured" below; Isaac Sim's first start on a new machine adds several minutes.

## Measured

Shared RTX-class GPU (other Isaac Sim processes running at the same time), Isaac Sim shader cache warm,
`-E object_jitter=0`, 20 Hz, 320x176 frames:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `trip-demo` | `task_score: 1` (52-2 held at -42.6 degrees, green lamp after 1.0 s of TRIP contact, handle back at 0 degrees by its spring, target green, 52-1 and 52-3 closed at 0 degrees; after-release check passed) | 267 (13.4 s), then the 1 s check off camera | about 3.6 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (both arms move about in front of the panel and the right hand closes near its right edge, but no handle is turned 10 degrees and every lamp stays red; no hazard) | 800 (step limit) | about 10 minutes |
| Control: `TRIP_CONTROL=1`, `trip-demo`, lets go at the TRIP contact | `task_score: 0.5` (handle reached -40.3 degrees, was let go after 0.05 s of contact and sprang back; the target shows green but the lamp stays red and breaker 52-2 stays closed) | 800 (step limit) | about 9.7 minutes |

Most of the wall time is Isaac Sim's start-up and the demo's IK, which it solves in Python every control step.

## Layout

| Path | What |
|---|---|
| `control_switch_trip.py` | the task file: description for agents, the panel and task, scoring |
| `trip_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera after-release check) |
| `trip_sim/task.py`, `scorer.py`, `spec.py` | the panel articulation, the breaker logic and lamps, the after-release check, the score levels, the breakers and instruction |
| `trip_sim/assets/panel.usda`, `assets/tex/`, `trip_sim/usd/` | the panel and its engraving textures, and the script that generates them (`python -m trip_sim.usd.panel_props trip_sim/assets`) |
| `trip_sim/demo.py`, `trip_sim/policy.py` | the scripted demo and its Inspect Robots policy `trip-demo` |

## References

- Electroswitch Series 24 control switches (product page): https://www.electroswitch.com/products/utility-power-switches-relays/manual-standard-instrument-and-control-switches/series-24
- Electroswitch Utility catalog (Series 24 dimensions and operating torque p11, 2438D breaker control p17, target
  nameplate p89): https://www.electroswitch.com/es_download_handler?f=2023%252d12%252f2023_Utility_U_2_F_full%252epdf
- Electroswitch Series 24 circuit-breaker control switch list: https://www.electroswitch.com/sites/psr/files/resources/files/2019-01/CIRCUIT%20BREAKER%20CONTROL%20SWITCHES-24.pdf
- Electroswitch construction details (45 degree throw, spring return to vertical): https://www.electroswitch.com/sites/psr/files/resources/files/2018-12/I.3%20page%2051%20-%20Construction%20Details%20-%20Detent%20Sw.pdf
- GE SB-1 control switch bulletin (target shows the last operation: green after TRIP, red after CLOSE): https://www.electro-meters.com/wp-content/uploads/pdf/ge/sb1.pdf
- Schneider Electric Harmony XB7 22 mm LED pilot lights, 24 V: green https://www.se.com/us/en/product/XB7EV03BP/ , red https://www.se.com/us/en/product/XB7EV04BP/
- Omron H3Y-2 on-delay timer (DC24, 0.1-1 s range): https://automationtrader.com/en/products/omron/h3y-2-dc24-1s/downloads/h3y-2-dc24-1s_datasheet_en_automation-trader.pdf
- Omron G2RK-2 DC24 two-coil latching relay: https://uk.rs-online.com/web/p/power-relays/6838769
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation; fingers 5000 N/m, 50 N). Rigid bodies on joints at real size. Each handle turns on a revolute joint with hard stops at +-45 degrees (the stop screws), and its return spring is a joint spring toward vertical, 0.9 N m at the stop (Electroswitch's published operating torque for a one-deck Series 24, 8 in-lb), linear from zero; a real return spring is preloaded, so the real handle pushes back harder near vertical (not measured). The pistol grip is a straight box, 1.9 x 1.5 cm, 7.8 cm long, sized from photos and the catalog drawing; the real grip is tapered and rounded. The target flag is driven by contact: a pin on the shaft pushes it through two lugs with lost motion, so TRIP leaves it green and the spring return does not move it back; its travel and lost-motion angles are chosen to give the published behaviour, as the real linkage is not published, and the window is square, not the real trapezoid. The breaker and lamps follow a rule that copies the real-world stand-in circuit in the setup (TRIP contact made within 5 degrees of the stop, 1.0 s on-delay, latching state, lamps by state); the contact angle on the real 2438D is not published (we assume the last 5 degrees). A real breaker opens in 3-5 cycles and its lamp changes at once, so the 1.0 s hold stands for the operator watching the lamp; the stand-in circuit makes the same 1.0 s true on the real bench. The lamps are plain cylinders with lit and dark lenses. Nameplate engraving and colours are ours (laminated nameplates come in these colours; no standard assigns them). In the 320 x 176 overhead camera each nameplate is about 15 x 5 pixels, so its text cannot be read there, but the white, yellow and blue plates are clearly distinct and only 52-2's is yellow; in the right wrist camera, near the panel, the engravings are legible. The demo grips and twists the handle: it closes the fingers on the grip with about 10 N each and rolls the wrist about the switch's shaft; under the spring load the grip slips about 10 degrees inside the fingers (the simulated fingers are independently position-controlled), so the hand leads the handle by about 10 degrees; this has not been tried on a real YAM. Collision shapes are boxes. One layout.

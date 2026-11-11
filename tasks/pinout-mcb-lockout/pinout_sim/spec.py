"""Pin-out MCB lockout: lock out the named miniature circuit breaker with a spring-pin lockout and a padlock, two
hands. No simulator imports.

One layout: a row of five ABB S201 breakers, a Brady pin-out lockout (push-button, spring-held pins) and a Master
Lock 410 padlock. The named breaker (blue label) is OFF, the others ON.

Mechanism (geometry unless marked RULE; see pinout_sim/usd/breaker_props.py and task.py):
- Each breaker toggle is a lever on a revolute joint with an over-centre detent (RULE).
- The lockout's pins swing outward only while its button is held in (mimic joint = the cam; return spring = the
  button's drive). One hand holds the body, the other presses and holds the button; the pins' feet then pass through
  the breaker's two holes; on release they swing in behind the lips between the holes (geometry).
- The body's pocket encloses the OFF toggle (geometry). The shackle through the stem's hole meets the body's end face
  before the button has moved far enough to free the pins (geometry).
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "pinout-mcb-lockout"
EPISODE_SECONDS = 150.0
NUM_SEEDS = 1

BOARD_X = 0.372
"""Desk-frame x of the back plate's front face (rail back)."""
ROW_Y = -0.03
"""Middle breaker's centre."""
RAIL_Z = 0.200
"""Rail centre = the breakers' window centre, in the YAM's rolled-hand reach."""
NUM_BREAKERS = 5
BREAKER_COLORS = ("blue", "yellow", "green", "white", "orange")
PARK_Z = -0.30


def instruction(color: str = "blue") -> str:
    return (
        f"Lock out the breaker with the {color} label: hold the red lockout by its body with one hand and press and "
        "hold its black button with the other, push it onto the breaker's switch, then let go of the button so its "
        "pins catch. Put the padlock's shackle through the hole in the lockout's stem and snap it shut. Leave the "
        "other breakers as they are."
    )


@dataclass(frozen=True)
class LockoutSeed:
    seed: int
    named_slot: int
    named_color: str
    breaker: str = "abb_s201"
    lockout: str = "brady_pinout"
    padlock: str = "ml410"

    @property
    def slot_colors(self) -> tuple[str, ...]:
        rest = [c for c in BREAKER_COLORS if c != self.named_color]
        return tuple(rest[: self.named_slot] + [self.named_color] + rest[self.named_slot:])

    @property
    def instruction(self) -> str:
        return instruction(self.named_color)


def make_seed(seed: int = 0) -> LockoutSeed:
    """One fixed layout: slot 0 (blue) is named."""
    return LockoutSeed(0, 0, "blue")

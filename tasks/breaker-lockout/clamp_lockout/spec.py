"""Clamp-on breaker lockout: lock out the named circuit of a small US load centre with a Master Lock 493B Grip Tight
clamp-on lockout and a Master Lock 410 padlock. No simulator imports.

One layout: a two-column, three-row Square D QO deadfront (circuits 1/3/5 left, 2/4/6 right, numbered on the centre
strip with colour-coded labels). Circuit 4 (blue label) is OFF, the others ON.

Mechanism (geometry unless marked RULE; see clamp_lockout/usd/panel_props.py):
- Each breaker handle is a lever on a revolute joint with an over-centre detent (RULE: the drive target flips to the
  nearer end once the handle passes the middle of its throw by DETENT_FLIP_DEG). ON leans toward the panel centre.
- The 493B is a steel channel pressed straight onto the handle from the front. Its thumbscrew (knob) drives a
  serrated jaw along the channel that squeezes the handle against a fixed serrated jaw (screw-coupled joint: RULE
  drive p = -pitch x turns, thread friction a Coulomb drive; the squeeze and the holding are contact + friction).
- Clamped, the 74 mm channel lies along the throw over the deadfront: the handle cannot pivot toward ON without
  driving the channel's far end into the deadfront (geometry).
- The hinged lever closes over the channel's open front and covers the knob; its tip tab sits between the channel's
  two end loops with the holes in line. A padlock shackle through the three holes stops the lever opening (geometry),
  so the knob cannot be reached and the clamp cannot be loosened.
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "breaker-lockout"
EPISODE_SECONDS = 200.0
NUM_SEEDS = 1

PANEL_X = 0.310
"""Desk-frame x of the deadfront's front face."""
PANEL_Y = 0.030
"""Desk-frame y of the panel's centre line (the middle of the numbered centre strip)."""
ROW_Z = 0.200
"""Desk-frame z of the middle row of breakers (the named one is in it)."""

CIRCUITS = (1, 2, 3, 4, 5, 6)
"""Breaker k: circuit CIRCUITS[k]; odd circuits in the left column (robot's +y), top to bottom."""
LABEL_COLORS = {1: "white", 2: "yellow", 3: "green", 4: "blue", 5: "orange", 6: "white"}
NAMED_CIRCUIT = 4
PARK_Z = -0.30


def column(k: int) -> int:
    """+1 left column (odd circuits, robot's +y side), -1 right column."""
    return 1 if CIRCUITS[k] % 2 == 1 else -1


def row(k: int) -> int:
    """0 top, 1 middle, 2 bottom."""
    return (CIRCUITS[k] - 1) // 2


def instruction(circuit: int = NAMED_CIRCUIT) -> str:
    color = LABEL_COLORS[circuit]
    return (
        f"Lock out circuit {circuit}, the breaker with the {color} label: press the red clamp-on lockout onto its "
        "handle, turn the lockout's black knob until it grips the handle, close the lockout's lever, then put the "
        "padlock's shackle through the holes at the lever's end and snap it shut. Leave the other breakers as they "
        "are."
    )


@dataclass(frozen=True)
class LockoutSeed:
    seed: int
    named_circuit: int = NAMED_CIRCUIT
    breaker: str = "square_d_qo120"
    lockout: str = "master_lock_493b"
    padlock: str = "ml410"

    @property
    def named_slot(self) -> int:
        return CIRCUITS.index(self.named_circuit)

    @property
    def instruction(self) -> str:
        return instruction(self.named_circuit)


def make_seed(seed: int = 0) -> LockoutSeed:
    """One fixed layout: circuit 4 is named."""
    return LockoutSeed(0)

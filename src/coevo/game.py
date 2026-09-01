"""Koza's 32-outcome extensive-form game (1992).

Players alternate L/R. X moves 1st, 3rd and 5th; O moves 2nd and 4th.
Payoffs are to X (zero-sum: O receives the negation).

Leaf order is the 32 terminals of the binary tree with L=0, R=1 packed as
    bits = XM1 OM1 XM2 OM2 XM3
chosen so that Koza's published minimax facts hold:

* game value = 12
* X minimax vs the four O move-pairs scores 32, 16, 28, 12 (sum 88, 4 hits)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Move = Literal["L", "R"]
MaybeMove = Optional[Move]  # None == Koza's U (undefined)

# 32 terminal payoffs to X, indexed by bits XM1|OM1|XM2|OM2|XM3 with L=0, R=1.
PAYOFFS: tuple[int, ...] = (
    32, 31, 15, 16,  7,  8, 24, 23,
     3,  4, 20, 19, 28, 27, 11, 12,
     2,  1, 18, 17, 26, 25,  9, 10,
    30, 29, 13, 14,  5,  6, 22, 21,
)

GAME_VALUE = 12


def _bit(move: Move) -> int:
    return 0 if move == "L" else 1


def payoff(xm1: Move, om1: Move, xm2: Move, om2: Move, xm3: Move) -> int:
    idx = (
        (_bit(xm1) << 4)
        | (_bit(om1) << 3)
        | (_bit(xm2) << 2)
        | (_bit(om2) << 1)
        | _bit(xm3)
    )
    return PAYOFFS[idx]


@dataclass
class History:
    """Partial play. Undefined slots are None (Koza's U)."""

    xm1: MaybeMove = None
    om1: MaybeMove = None
    xm2: MaybeMove = None
    om2: MaybeMove = None
    xm3: MaybeMove = None

    def ply(self) -> int:
        for i, m in enumerate((self.xm1, self.om1, self.xm2, self.om2, self.xm3)):
            if m is None:
                return i
        return 5

    def whose_turn(self) -> Literal["X", "O"]:
        return "X" if self.ply() % 2 == 0 else "O"

    def with_move(self, move: Move) -> "History":
        slots = [self.xm1, self.om1, self.xm2, self.om2, self.xm3]
        p = self.ply()
        if p >= 5:
            raise ValueError("game already finished")
        slots[p] = move
        return History(*slots)

    def finished(self) -> bool:
        return self.ply() == 5

    def terminal_payoff(self) -> int:
        if not self.finished():
            raise ValueError("game not finished")
        return payoff(self.xm1, self.om1, self.xm2, self.om2, self.xm3)  # type: ignore[arg-type]


def play(strategy_x, strategy_o) -> int:
    """Run one complete game. Each strategy is callable(history) -> Move."""
    h = History()
    while not h.finished():
        mover = strategy_x if h.whose_turn() == "X" else strategy_o
        h = h.with_move(mover(h))
    return h.terminal_payoff()


def minimax_value(h: History | None = None) -> int:
    """Perfect-information value from *h* (payoff to X)."""
    if h is None:
        h = History()
    if h.finished():
        return h.terminal_payoff()
    if h.whose_turn() == "X":
        return max(minimax_value(h.with_move(m)) for m in ("L", "R"))
    return min(minimax_value(h.with_move(m)) for m in ("L", "R"))


def best_move(h: History) -> Move:
    if h.whose_turn() == "X":
        return max(("L", "R"), key=lambda m: minimax_value(h.with_move(m)))
    return min(("L", "R"), key=lambda m: minimax_value(h.with_move(m)))


def minimax_strategy(h: History) -> Move:
    return best_move(h)

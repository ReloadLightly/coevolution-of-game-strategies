"""Iterated Prisoner's Dilemma on Koza-style GP trees.

Payoffs are Axelrod's canonical matrix (row player):

            C         D
        +---------+---------+
     C  |  3 / 3  |  0 / 5  |
        +---------+---------+
     D  |  5 / 0  |  1 / 1  |
        +---------+---------+

A program is an S-expression over terminals {C, D} and CASE functions
that inspect the last two rounds plus a grim bit (opponent ever defected).

    IOPP1 / ISELF1 / IOPP2 / ISELF2 / IGRIM
        each: (if-undefined, if-C-or-nice, if-D-or-burned)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

Move = Literal["C", "D"]
MaybeMove = Move | None

R, S, T, P = 3, 0, 5, 1
PAYOFF = {
    ("C", "C"): (R, R),
    ("C", "D"): (S, T),
    ("D", "C"): (T, S),
    ("D", "D"): (P, P),
}

FUNCTIONS: tuple[str, ...] = ("IOPP1", "ISELF1", "IOPP2", "ISELF2", "IGRIM")
TERMINALS: tuple[Move, ...] = ("C", "D")


@dataclass
class IPDHistory:
    self1: MaybeMove = None
    opp1: MaybeMove = None
    self2: MaybeMove = None
    opp2: MaybeMove = None
    opp_ever_d: bool = False

    def after(self, mine: Move, theirs: Move) -> "IPDHistory":
        return IPDHistory(
            self1=mine,
            opp1=theirs,
            self2=self.self1,
            opp2=self.opp1,
            opp_ever_d=self.opp_ever_d or theirs == "D",
        )

    def slot(self, name: str) -> MaybeMove | bool:
        if name == "IOPP1":
            return self.opp1
        if name == "ISELF1":
            return self.self1
        if name == "IOPP2":
            return self.opp2
        if name == "ISELF2":
            return self.self2
        if name == "IGRIM":
            if self.opp1 is None:
                return None
            return "D" if self.opp_ever_d else "C"
        raise KeyError(name)


@dataclass
class Node:
    name: str
    children: tuple["Node", ...] = ()

    def is_terminal(self) -> bool:
        return self.name in TERMINALS

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def sexp(self) -> str:
        if self.is_terminal():
            return self.name
        inner = " ".join(c.sexp() for c in self.children)
        return f"({self.name} {inner})"

    def nodes(self) -> list["Node"]:
        out = [self]
        for c in self.children:
            out.extend(c.nodes())
        return out

    def copy(self) -> "Node":
        return Node(self.name, tuple(c.copy() for c in self.children))

    def evaluate(self, h: IPDHistory) -> Move:
        if self.is_terminal():
            return self.name  # type: ignore[return-value]
        slot = h.slot(self.name)
        if slot is None:
            return self.children[0].evaluate(h)
        if slot == "C":
            return self.children[1].evaluate(h)
        return self.children[2].evaluate(h)

    def as_strategy(self) -> Callable[[IPDHistory], Move]:
        def strat(h: IPDHistory) -> Move:
            return self.evaluate(h)

        return strat


def parse_sexp(text: str) -> Node:
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()

    def read(i: int) -> tuple[Node, int]:
        tok = tokens[i]
        if tok == "(":
            name = tokens[i + 1]
            kids: list[Node] = []
            j = i + 2
            while tokens[j] != ")":
                kid, j = read(j)
                kids.append(kid)
            return Node(name, tuple(kids)), j + 1
        if tok in TERMINALS:
            return Node(tok), i + 1
        raise ValueError(f"unexpected token {tok!r}")

    node, end = read(0)
    if end != len(tokens):
        raise ValueError("trailing tokens")
    return node


ALLC = parse_sexp("C")
ALLD = parse_sexp("D")
TFT = parse_sexp("(IOPP1 C C D)")
GRIM = parse_sexp("(IGRIM C C D)")
PAVLOV = parse_sexp("(ISELF1 C (IOPP1 C C D) (IOPP1 D D C))")

CLASSICS: dict[str, Node] = {
    "ALLC": ALLC,
    "ALLD": ALLD,
    "TFT": TFT,
    "GRIM": GRIM,
    "PAVLOV": PAVLOV,
}


def play(
    a: Callable[[IPDHistory], Move],
    b: Callable[[IPDHistory], Move],
    rounds: int,
    noise: float = 0.0,
    rng: random.Random | None = None,
) -> tuple[int, int]:
    rng = rng or random.Random(0)
    ha, hb = IPDHistory(), IPDHistory()
    sa = sb = 0
    for _ in range(rounds):
        ma, mb = a(ha), b(hb)
        if noise and rng.random() < noise:
            ma = "D" if ma == "C" else "C"
        if noise and rng.random() < noise:
            mb = "D" if mb == "C" else "C"
        pa, pb = PAYOFF[(ma, mb)]
        sa += pa
        sb += pb
        ha = ha.after(ma, mb)
        hb = hb.after(mb, ma)
    return sa, sb


def play_trees(a: Node, b: Node, rounds: int, noise: float = 0.0, seed: int = 0) -> tuple[int, int]:
    return play(a.as_strategy(), b.as_strategy(), rounds, noise, random.Random(seed))


def random_tree(rng: random.Random, max_depth: int, method: str = "grow") -> Node:
    if max_depth <= 0:
        return Node(rng.choice(TERMINALS))
    if method == "grow":
        n_term, n_func = len(TERMINALS), len(FUNCTIONS)
        if rng.random() < n_term / (n_term + n_func):
            return Node(rng.choice(TERMINALS))
    name = rng.choice(FUNCTIONS)
    kids = tuple(random_tree(rng, max_depth - 1, method) for _ in range(3))
    return Node(name, kids)


def ramped_half_and_half(
    rng: random.Random, min_depth: int = 2, max_depth: int = 5
) -> Node:
    depth = rng.randint(min_depth, max_depth)
    method = rng.choice(("grow", "full"))
    return random_tree(rng, depth, method)


def _replace(root: Node, target: Node, replacement: Node) -> Node:
    if root is target:
        return replacement.copy()
    return Node(root.name, tuple(_replace(c, target, replacement) for c in root.children))


def crossover(a: Node, b: Node, rng: random.Random) -> tuple[Node, Node]:
    pa = rng.choice(a.nodes())
    pb = rng.choice(b.nodes())
    return _replace(a, pa, pb), _replace(b, pb, pa)


def mutate(node: Node, rng: random.Random, max_depth: int = 3) -> Node:
    target = rng.choice(node.nodes())
    return _replace(node, target, random_tree(rng, max_depth, "grow"))


def tournament(
    pop: Sequence[Node], fitness: Sequence[float], k: int, rng: random.Random
) -> Node:
    idxs = rng.sample(range(len(pop)), k=min(k, len(pop)))
    best = max(idxs, key=lambda i: fitness[i])
    return pop[best]

"""Koza-style genetic programming trees for the 32-outcome game.

Function set F = {CXM1, COM1, CXM2, COM2}
Terminal set T = {L, R}

Each CASE function takes three children (if-U, if-L, if-R) and inspects one
history variable. Evaluating a program against a History yields a move.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from .game import History, Move

FuncName = Literal["CXM1", "COM1", "CXM2", "COM2"]
FUNCTIONS: tuple[FuncName, ...] = ("CXM1", "COM1", "CXM2", "COM2")
TERMINALS: tuple[Move, ...] = ("L", "R")

_SLOT: dict[FuncName, str] = {
    "CXM1": "xm1",
    "COM1": "om1",
    "CXM2": "xm2",
    "COM2": "om2",
}


@dataclass
class Node:
    """S-expression node. `name` is a function or a terminal L/R."""

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

    def evaluate(self, h: History) -> Move:
        if self.is_terminal():
            return self.name  # type: ignore[return-value]
        slot = getattr(h, _SLOT[self.name])  # type: ignore[index]
        if slot is None:
            return self.children[0].evaluate(h)
        if slot == "L":
            return self.children[1].evaluate(h)
        return self.children[2].evaluate(h)

    def as_strategy(self) -> Callable[[History], Move]:
        def strat(h: History) -> Move:
            return self.evaluate(h)

        return strat


def parse_sexp(text: str) -> Node:
    """Parse a Koza-style S-expression into a Node."""
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
        if tok in TERMINALS or tok == "$":
            return Node("L" if tok == "$" else tok), i + 1
        raise ValueError(f"unexpected token {tok!r}")

    node, end = read(0)
    if end != len(tokens):
        raise ValueError("trailing tokens")
    return node


X_MINIMAX = parse_sexp("(COM2 (COM1 L L R) L R)")
O_MINIMAX = parse_sexp("(CXM2 (CXM1 L R L) L R)")


def random_terminal(rng: random.Random) -> Node:
    return Node(rng.choice(TERMINALS))


def random_tree(rng: random.Random, max_depth: int, method: str = "grow") -> Node:
    if max_depth <= 0:
        return random_terminal(rng)
    if method == "grow":
        n_term, n_func = len(TERMINALS), len(FUNCTIONS)
        if rng.random() < n_term / (n_term + n_func):
            return random_terminal(rng)
    name = rng.choice(FUNCTIONS)
    kids = tuple(random_tree(rng, max_depth - 1, method) for _ in range(3))
    return Node(name, kids)


def ramped_half_and_half(
    rng: random.Random, min_depth: int = 2, max_depth: int = 6
) -> Node:
    depth = rng.randint(min_depth, max_depth)
    method = rng.choice(("grow", "full"))
    return random_tree(rng, depth, method)


def _replace(root: Node, target: Node, replacement: Node) -> Node:
    if root is target:
        return replacement.copy()
    return Node(
        root.name,
        tuple(_replace(c, target, replacement) for c in root.children),
    )


def crossover(a: Node, b: Node, rng: random.Random) -> tuple[Node, Node]:
    pa = rng.choice(a.nodes())
    pb = rng.choice(b.nodes())
    child_a = _replace(a, pa, pb)
    child_b = _replace(b, pb, pa)
    return child_a, child_b


def mutate(node: Node, rng: random.Random, max_depth: int = 4) -> Node:
    target = rng.choice(node.nodes())
    fresh = random_tree(rng, max_depth, "grow")
    return _replace(node, target, fresh)


def tournament(pop: Sequence[Node], fitness: Sequence[float], k: int, rng: random.Random) -> Node:
    idxs = rng.sample(range(len(pop)), k=min(k, len(pop)))
    best = max(idxs, key=lambda i: fitness[i])
    return pop[best]

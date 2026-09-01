"""Two-state stochastic game (Shapley 1953 / Hilbe et al. Nature 2018)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Literal

from .ipd import IPDHistory, Move

State = Literal["RICH", "POOR"]

RICH_PAYOFF = {
    ("C", "C"): (4, 4),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}
POOR_PAYOFF = {
    ("C", "C"): (2, 2),
    ("C", "D"): (0, 3),
    ("D", "C"): (3, 0),
    ("D", "D"): (1, 1),
}
PAYOFFS = {"RICH": RICH_PAYOFF, "POOR": POOR_PAYOFF}

DEFAULT_Q = {
    ("C", "C"): 1.0,
    ("C", "D"): 0.0,
    ("D", "C"): 0.0,
    ("D", "D"): 0.0,
}
FIXED_RICH_Q = {k: 1.0 for k in DEFAULT_Q}
FIXED_POOR_Q = {k: 0.0 for k in DEFAULT_Q}

FUNCTIONS_3 = ("IOPP1", "ISELF1", "IOPP2", "ISELF2", "IGRIM")
FUNCTIONS_2 = ("ISTATE",)
TERMINALS = ("C", "D")
ARITY = {**{n: 3 for n in FUNCTIONS_3}, **{n: 2 for n in FUNCTIONS_2}}


@dataclass
class SGHistory:
    ipd: IPDHistory
    state: State = "RICH"

    def after(self, mine: Move, theirs: Move, next_state: State) -> "SGHistory":
        return SGHistory(ipd=self.ipd.after(mine, theirs), state=next_state)


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

    def evaluate(self, h: SGHistory) -> Move:
        if self.is_terminal():
            return self.name  # type: ignore[return-value]
        if self.name == "ISTATE":
            branch = self.children[0] if h.state == "RICH" else self.children[1]
            return branch.evaluate(h)
        slot = h.ipd.slot(self.name)
        if slot is None:
            return self.children[0].evaluate(h)
        if slot == "C":
            return self.children[1].evaluate(h)
        return self.children[2].evaluate(h)

    def as_strategy(self) -> Callable[[SGHistory], Move]:
        def strat(h: SGHistory) -> Move:
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
            expected = ARITY.get(name)
            if expected is not None and len(kids) != expected:
                raise ValueError(f"{name} expects {expected} args, got {len(kids)}")
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
PEACE_TFT = parse_sexp("(ISTATE (IOPP1 C C D) D)")
PEACE_GRIM = parse_sexp("(ISTATE (IGRIM C C D) D)")

CLASSICS = {
    "ALLC": ALLC,
    "ALLD": ALLD,
    "TFT": TFT,
    "GRIM": GRIM,
    "PAVLOV": PAVLOV,
    "PEACE_TFT": PEACE_TFT,
    "PEACE_GRIM": PEACE_GRIM,
}


def play(
    a: Callable[[SGHistory], Move],
    b: Callable[[SGHistory], Move],
    rounds: int,
    q: dict[tuple[str, str], float] | None = None,
    start: State = "RICH",
    rng: random.Random | None = None,
) -> tuple[int, int, float]:
    q = q or DEFAULT_Q
    rng = rng or random.Random(0)
    state: State = start
    ha = SGHistory(IPDHistory(), state)
    hb = SGHistory(IPDHistory(), state)
    sa = sb = 0
    rich_rounds = 0
    for _ in range(rounds):
        if state == "RICH":
            rich_rounds += 1
        ma, mb = a(ha), b(hb)
        pa, pb = PAYOFFS[state][(ma, mb)]
        sa += pa
        sb += pb
        nxt: State = "RICH" if rng.random() < q[(ma, mb)] else "POOR"
        ha = ha.after(ma, mb, nxt)
        hb = hb.after(mb, ma, nxt)
        state = nxt
    return sa, sb, rich_rounds / rounds


def play_trees(
    a: Node,
    b: Node,
    rounds: int,
    q: dict[tuple[str, str], float] | None = None,
    seed: int = 0,
    start: State = "RICH",
) -> tuple[int, int, float]:
    return play(a.as_strategy(), b.as_strategy(), rounds, q, start, random.Random(seed))


def random_tree(rng: random.Random, max_depth: int, method: str = "grow") -> Node:
    if max_depth <= 0:
        return Node(rng.choice(TERMINALS))
    names = FUNCTIONS_3 + FUNCTIONS_2
    if method == "grow":
        n_term, n_func = len(TERMINALS), len(names)
        if rng.random() < n_term / (n_term + n_func):
            return Node(rng.choice(TERMINALS))
    name = rng.choice(names)
    kids = tuple(random_tree(rng, max_depth - 1, method) for _ in range(ARITY[name]))
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


def tournament(pop, fitness, k, rng):
    idxs = rng.sample(range(len(pop)), k=min(k, len(pop)))
    best = max(idxs, key=lambda i: fitness[i])
    return pop[best]

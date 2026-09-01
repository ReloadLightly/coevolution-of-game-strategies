"""Evolve GP strategies on the two-state stochastic game."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .ipd_evolve import IPDParams
from .stoch import (
    CLASSICS,
    DEFAULT_Q,
    FIXED_RICH_Q,
    Node,
    crossover,
    mutate,
    play_trees,
    ramped_half_and_half,
    tournament,
)


@dataclass
class SGParams(IPDParams):
    regime: str = "collapse"

    def q(self):
        if self.regime == "fixed-rich":
            return FIXED_RICH_Q
        if self.regime == "fixed-poor":
            from .stoch import FIXED_POOR_Q

            return FIXED_POOR_Q
        return DEFAULT_Q


@dataclass
class SGStat:
    generation: int
    best_fitness: float
    mean_fitness: float
    best_vs_tft: float
    best_vs_alld: float
    best_rich_frac: float
    best_sexp: str


@dataclass
class SGRun:
    best: Node
    history: list[SGStat] = field(default_factory=list)


def _limit(node: Node, fallback: Node, max_depth: int) -> Node:
    return node if node.depth() <= max_depth else fallback


def _breed(pop, fitness, params: SGParams, rng: random.Random) -> list[Node]:
    nxt: list[Node] = []
    while len(nxt) < params.pop_size:
        r = rng.random()
        if r < params.reproduction_prob:
            nxt.append(tournament(pop, fitness, params.tournament_k, rng).copy())
        elif r < params.reproduction_prob + params.mutation_prob:
            p = tournament(pop, fitness, params.tournament_k, rng)
            nxt.append(_limit(mutate(p, rng), p, params.max_depth))
        else:
            p1 = tournament(pop, fitness, params.tournament_k, rng)
            p2 = tournament(pop, fitness, params.tournament_k, rng)
            c1, c2 = crossover(p1, p2, rng)
            nxt.append(_limit(c1, p1, params.max_depth))
            if len(nxt) < params.pop_size:
                nxt.append(_limit(c2, p2, params.max_depth))
    return nxt


def score_vs(prog: Node, opp: Node, params: SGParams, seed: int) -> tuple[float, float]:
    s, _, rich = play_trees(prog, opp, params.rounds, params.q(), seed)
    return s / params.rounds, rich


def vs_classics(prog: Node, params: SGParams, seed: int = 0) -> dict[str, tuple[float, float]]:
    return {name: score_vs(prog, node, params, seed) for name, node in CLASSICS.items()}


def fitness_vs_classics(prog: Node, params: SGParams, seed: int) -> float:
    return sum(v[0] for v in vs_classics(prog, params, seed).values()) / len(CLASSICS)


def evolve_sg(params: SGParams | None = None) -> SGRun:
    params = params or SGParams()
    rng = random.Random(params.seed)
    pop = [
        ramped_half_and_half(rng, params.min_init_depth, params.max_init_depth)
        for _ in range(params.pop_size)
    ]
    history: list[SGStat] = []
    best = pop[0]
    for gen in range(params.generations + 1):
        fit = [fitness_vs_classics(ind, params, params.seed + gen) for ind in pop]
        bi = max(range(len(pop)), key=lambda i: fit[i])
        best = pop[bi]
        vs = vs_classics(best, params, params.seed)
        history.append(
            SGStat(
                generation=gen,
                best_fitness=fit[bi],
                mean_fitness=sum(fit) / len(fit),
                best_vs_tft=vs["TFT"][0],
                best_vs_alld=vs["ALLD"][0],
                best_rich_frac=vs["TFT"][1],
                best_sexp=best.sexp(),
            )
        )
        if gen == params.generations:
            break
        pop = _breed(pop, fit, params, rng)
    return SGRun(best=best, history=history)


def compare_regimes(params: SGParams | None = None) -> dict[str, SGRun]:
    base = params or SGParams()
    out = {}
    for regime in ("fixed-rich", "fixed-poor", "collapse"):
        p = SGParams(**{**base.__dict__, "regime": regime})
        out[regime] = evolve_sg(p)
    return out

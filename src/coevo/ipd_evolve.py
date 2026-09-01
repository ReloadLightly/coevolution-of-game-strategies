"""IPD evolution: against a fixed classic environment, or co-evolution."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .evolve import GPParams
from .ipd import (
    CLASSICS,
    Node,
    crossover,
    mutate,
    play_trees,
    ramped_half_and_half,
    tournament,
)


@dataclass
class IPDParams(GPParams):
    rounds: int = 50
    noise: float = 0.0
    sample_opponents: int = 0


@dataclass
class IPDStat:
    generation: int
    best_fitness: float
    mean_fitness: float
    best_vs_tft: float
    best_vs_alld: float
    best_vs_allc: float
    coop_rate_vs_tft: float
    best_sexp: str


@dataclass
class IPDRun:
    best: Node
    history: list[IPDStat] = field(default_factory=list)


def _limit(node: Node, fallback: Node, max_depth: int) -> Node:
    return node if node.depth() <= max_depth else fallback


def _next_generation(
    pop: list[Node],
    fitness: list[float],
    params: IPDParams,
    rng: random.Random,
) -> list[Node]:
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


def _per_round(score: int, rounds: int) -> float:
    return score / rounds


def score_vs(prog: Node, opp: Node, params: IPDParams, seed: int) -> float:
    s, _ = play_trees(prog, opp, params.rounds, params.noise, seed)
    return _per_round(s, params.rounds)


def vs_classics(prog: Node, params: IPDParams, seed: int = 0) -> dict[str, float]:
    return {name: score_vs(prog, node, params, seed) for name, node in CLASSICS.items()}


def fitness_vs_classics(prog: Node, params: IPDParams, seed: int) -> float:
    return sum(vs_classics(prog, params, seed).values()) / len(CLASSICS)


def evolve_vs_classics(params: IPDParams | None = None) -> IPDRun:
    params = params or IPDParams()
    rng = random.Random(params.seed)
    pop = [
        ramped_half_and_half(rng, params.min_init_depth, params.max_init_depth)
        for _ in range(params.pop_size)
    ]
    history: list[IPDStat] = []
    best = pop[0]

    for gen in range(params.generations + 1):
        fit = [fitness_vs_classics(ind, params, params.seed + gen) for ind in pop]
        bi = max(range(len(pop)), key=lambda i: fit[i])
        best = pop[bi]
        vs = vs_classics(best, params, params.seed)
        history.append(
            IPDStat(
                generation=gen,
                best_fitness=fit[bi],
                mean_fitness=sum(fit) / len(fit),
                best_vs_tft=vs["TFT"],
                best_vs_alld=vs["ALLD"],
                best_vs_allc=vs["ALLC"],
                coop_rate_vs_tft=vs["TFT"] / 3.0,
                best_sexp=best.sexp(),
            )
        )
        if gen == params.generations:
            break
        pop = _next_generation(pop, fit, params, rng)
    return IPDRun(best=best, history=history)


def coevolve_ipd(params: IPDParams | None = None) -> IPDRun:
    params = params or IPDParams()
    rng = random.Random(params.seed)
    pop = [
        ramped_half_and_half(rng, params.min_init_depth, params.max_init_depth)
        for _ in range(params.pop_size)
    ]
    history: list[IPDStat] = []
    best = pop[0]

    for gen in range(params.generations + 1):
        n = len(pop)
        totals = [0.0] * n
        counts = [0] * n
        if params.sample_opponents and params.sample_opponents < n - 1:
            opponents = []
            for i in range(n):
                choices = [j for j in range(n) if j != i]
                opponents.append(rng.sample(choices, params.sample_opponents))
        else:
            opponents = [[j for j in range(n) if j != i] for i in range(n)]

        seen: set[tuple[int, int]] = set()
        for i in range(n):
            for j in opponents[i]:
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                seen.add(key)
                sa, sb = play_trees(
                    pop[i], pop[j], params.rounds, params.noise, params.seed + gen + i + j
                )
                totals[i] += sa
                totals[j] += sb
                counts[i] += 1
                counts[j] += 1
        fit = [(totals[i] / max(counts[i], 1)) / params.rounds for i in range(n)]
        bi = max(range(n), key=lambda i: fit[i])
        best = pop[bi]
        vs = vs_classics(best, params, params.seed)
        history.append(
            IPDStat(
                generation=gen,
                best_fitness=fit[bi],
                mean_fitness=sum(fit) / len(fit),
                best_vs_tft=vs["TFT"],
                best_vs_alld=vs["ALLD"],
                best_vs_allc=vs["ALLC"],
                coop_rate_vs_tft=vs["TFT"] / 3.0,
                best_sexp=best.sexp(),
            )
        )
        if gen == params.generations:
            break
        pop = _next_generation(pop, fit, params, rng)
    return IPDRun(best=best, history=history)


def horizon_experiment(
    short_rounds: int = 1,
    long_rounds: int = 50,
    params: IPDParams | None = None,
) -> tuple[IPDRun, IPDRun]:
    base = params or IPDParams()
    short = IPDParams(**{**base.__dict__, "rounds": short_rounds})
    long = IPDParams(**{**base.__dict__, "rounds": long_rounds})
    return evolve_vs_classics(short), evolve_vs_classics(long)

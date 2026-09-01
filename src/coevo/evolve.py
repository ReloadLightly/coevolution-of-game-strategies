"""Evolution and co-evolution loops matching Koza 1992 ch. 15–16."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Literal

from .game import GAME_VALUE, History, play
from .gp import (
    O_MINIMAX,
    X_MINIMAX,
    Node,
    crossover,
    mutate,
    ramped_half_and_half,
    tournament,
)


def all_move_seqs(n: int) -> list[tuple[str, ...]]:
    seqs: list[tuple[str, ...]] = [()]
    for _ in range(n):
        seqs = [s + (m,) for s in seqs for m in ("L", "R")]
    return seqs


O_SEQS = all_move_seqs(2)
X_SEQS = all_move_seqs(3)


def scripted(moves: tuple[str, ...]) -> Callable:
    def strat(h: History):
        own_index = h.ply() // 2
        return moves[own_index]

    return strat


def absolute_fitness_x(prog: Node) -> tuple[float, int, list[int]]:
    scores = [play(prog.as_strategy(), scripted(seq)) for seq in O_SEQS]
    hits = sum(1 for s in scores if s >= GAME_VALUE)
    return float(sum(scores)), hits, scores


def absolute_fitness_o(prog: Node) -> tuple[float, int, list[int]]:
    x_scores = [play(scripted(seq), prog.as_strategy()) for seq in X_SEQS]
    hits = sum(1 for s in x_scores if s <= GAME_VALUE)
    o_sum = sum(32 - s for s in x_scores)
    return float(o_sum), hits, x_scores


def vs_minimax_x(prog: Node) -> int:
    return play(prog.as_strategy(), O_MINIMAX.as_strategy())


def vs_minimax_o(prog: Node) -> int:
    return play(X_MINIMAX.as_strategy(), prog.as_strategy())


@dataclass
class GPParams:
    pop_size: int = 300
    generations: int = 40
    crossover_prob: float = 0.90
    reproduction_prob: float = 0.10
    mutation_prob: float = 0.00
    tournament_k: int = 7
    min_init_depth: int = 2
    max_init_depth: int = 6
    max_depth: int = 17
    seed: int = 42


@dataclass
class GenStat:
    generation: int
    best_relative: float
    mean_relative: float
    best_vs_minimax: int
    best_abs_hits: int
    best_sexp: str
    n_at_minimax: int


@dataclass
class RunResult:
    role: Literal["X", "O"]
    best: Node
    history: list[GenStat] = field(default_factory=list)


def _init_pop(params: GPParams, rng: random.Random) -> list[Node]:
    return [
        ramped_half_and_half(rng, params.min_init_depth, params.max_init_depth)
        for _ in range(params.pop_size)
    ]


def _limit_depth(node: Node, fallback: Node, max_depth: int) -> Node:
    return node if node.depth() <= max_depth else fallback


def _next_generation(
    pop: list[Node],
    fitness: list[float],
    params: GPParams,
    rng: random.Random,
) -> list[Node]:
    nxt: list[Node] = []
    while len(nxt) < params.pop_size:
        r = rng.random()
        if r < params.reproduction_prob:
            nxt.append(tournament(pop, fitness, params.tournament_k, rng).copy())
        elif r < params.reproduction_prob + params.mutation_prob:
            p = tournament(pop, fitness, params.tournament_k, rng)
            child = mutate(p, rng)
            nxt.append(_limit_depth(child, p, params.max_depth))
        else:
            p1 = tournament(pop, fitness, params.tournament_k, rng)
            p2 = tournament(pop, fitness, params.tournament_k, rng)
            c1, c2 = crossover(p1, p2, rng)
            nxt.append(_limit_depth(c1, p1, params.max_depth))
            if len(nxt) < params.pop_size:
                nxt.append(_limit_depth(c2, p2, params.max_depth))
    return nxt


def evolve_against_minimax(role: Literal["X", "O"], params: GPParams | None = None) -> RunResult:
    params = params or GPParams()
    rng = random.Random(params.seed)
    pop = _init_pop(params, rng)
    history: list[GenStat] = []
    best_node = pop[0]

    for gen in range(params.generations + 1):
        rel, hits_list, vs = [], [], []
        for ind in pop:
            if role == "X":
                raw, hits, _ = absolute_fitness_x(ind)
                vs.append(vs_minimax_x(ind))
            else:
                raw, hits, _ = absolute_fitness_o(ind)
                vs.append(vs_minimax_o(ind))
            rel.append(raw)
            hits_list.append(hits)

        bi = max(range(len(pop)), key=lambda i: (rel[i], hits_list[i]))
        best_node = pop[bi]
        if role == "X":
            n_mm = sum(1 for v in vs if v >= GAME_VALUE)
        else:
            n_mm = sum(1 for v in vs if v <= GAME_VALUE)
        history.append(
            GenStat(
                generation=gen,
                best_relative=rel[bi],
                mean_relative=sum(rel) / len(rel),
                best_vs_minimax=vs[bi],
                best_abs_hits=hits_list[bi],
                best_sexp=best_node.sexp(),
                n_at_minimax=n_mm,
            )
        )
        if gen == params.generations:
            break
        pop = _next_generation(pop, rel, params, rng)

    return RunResult(role=role, best=best_node, history=history)


def coevolve(params: GPParams | None = None) -> tuple[RunResult, RunResult]:
    params = params or GPParams()
    rng = random.Random(params.seed)
    pop_x = _init_pop(params, rng)
    pop_o = _init_pop(params, rng)
    hx: list[GenStat] = []
    ho: list[GenStat] = []
    best_x, best_o = pop_x[0], pop_o[0]

    for gen in range(params.generations + 1):
        matrix = [
            [play(xi.as_strategy(), oj.as_strategy()) for oj in pop_o] for xi in pop_x
        ]
        fit_x = [sum(row) / len(row) for row in matrix]
        fit_o = [
            sum(32 - matrix[i][j] for i in range(len(pop_x))) / len(pop_x)
            for j in range(len(pop_o))
        ]

        abs_x = [absolute_fitness_x(ind) for ind in pop_x]
        abs_o = [absolute_fitness_o(ind) for ind in pop_o]
        vsx = [vs_minimax_x(ind) for ind in pop_x]
        vso = [vs_minimax_o(ind) for ind in pop_o]

        ix = max(range(len(pop_x)), key=lambda i: fit_x[i])
        io = max(range(len(pop_o)), key=lambda i: fit_o[i])
        best_x, best_o = pop_x[ix], pop_o[io]

        hx.append(
            GenStat(
                generation=gen,
                best_relative=fit_x[ix],
                mean_relative=sum(fit_x) / len(fit_x),
                best_vs_minimax=vsx[ix],
                best_abs_hits=abs_x[ix][1],
                best_sexp=best_x.sexp(),
                n_at_minimax=sum(1 for v in vsx if v >= GAME_VALUE),
            )
        )
        ho.append(
            GenStat(
                generation=gen,
                best_relative=fit_o[io],
                mean_relative=sum(fit_o) / len(fit_o),
                best_vs_minimax=vso[io],
                best_abs_hits=abs_o[io][1],
                best_sexp=best_o.sexp(),
                n_at_minimax=sum(1 for v in vso if v <= GAME_VALUE),
            )
        )

        if gen == params.generations:
            break
        pop_x = _next_generation(pop_x, fit_x, params, rng)
        pop_o = _next_generation(pop_o, fit_o, params, rng)

    return (
        RunResult(role="X", best=best_x, history=hx),
        RunResult(role="O", best=best_o, history=ho),
    )

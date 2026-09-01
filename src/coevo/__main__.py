"""CLI: python -m coevo --help"""

from __future__ import annotations

import argparse
import json
import sys

from .evolve import GPParams, coevolve, evolve_against_minimax
from .game import GAME_VALUE, minimax_value, play
from .gp import O_MINIMAX, X_MINIMAX


def _print_hist(label: str, result) -> None:
    print(f"\n=== {label} ===")
    print(
        f"{'gen':>4}  {'best_rel':>10}  {'mean_rel':>10}  "
        f"{'vs_mm':>6}  {'hits':>4}  {'n_mm':>5}  program"
    )
    for s in result.history:
        print(
            f"{s.generation:4d}  {s.best_relative:10.3f}  {s.mean_relative:10.3f}  "
            f"{s.best_vs_minimax:6d}  {s.best_abs_hits:4d}  {s.n_at_minimax:5d}  "
            f"{s.best_sexp}"
        )
    print(f"best program: {result.best.sexp()}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Koza 1992 co-evolution of game strategies (32-outcome tree)."
    )
    p.add_argument(
        "mode",
        choices=("check", "evolve-x", "evolve-o", "coevolve"),
        help="check = verify game + published programs; "
        "evolve-x/o = ch.15 vs minimax opponent; "
        "coevolve = ch.16 two-population co-evolution",
    )
    p.add_argument("--pop", type=int, default=80, help="population size (Koza used 300)")
    p.add_argument("--gens", type=int, default=20, help="generations (Koza showed ~38)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cx", type=float, default=0.90, help="crossover probability")
    p.add_argument("--mut", type=float, default=0.00, help="mutation probability")
    p.add_argument("--json", type=str, default="", help="write run stats to this path")
    args = p.parse_args(argv)

    if args.mode == "check":
        v = minimax_value()
        px = play(X_MINIMAX.as_strategy(), O_MINIMAX.as_strategy())
        print(f"game value (minimax backup): {v}")
        print(f"published X vs published O:  {px}")
        print(f"expected game value:         {GAME_VALUE}")
        from .evolve import absolute_fitness_o, absolute_fitness_x

        raw_x, hits_x, scores_x = absolute_fitness_x(X_MINIMAX)
        raw_o, hits_o, scores_o = absolute_fitness_o(O_MINIMAX)
        print(f"X vs 4 O-scripts: scores={scores_x} sum={raw_x} hits={hits_x} (want 88 / 4)")
        print(f"O vs 8 X-scripts: X-payoffs={scores_o} hits={hits_o}")
        ok = v == GAME_VALUE and px == GAME_VALUE and raw_x == 88 and hits_x == 4
        print("OK" if ok else "MISMATCH")
        return 0 if ok else 1

    params = GPParams(
        pop_size=args.pop,
        generations=args.gens,
        crossover_prob=args.cx,
        mutation_prob=args.mut,
        reproduction_prob=max(0.0, 1.0 - args.cx - args.mut),
        seed=args.seed,
    )

    if args.mode == "evolve-x":
        rx = evolve_against_minimax("X", params)
        _print_hist("evolve X against O-minimax (ch. 15)", rx)
        results = {"X": [s.__dict__ for s in rx.history]}
    elif args.mode == "evolve-o":
        ro = evolve_against_minimax("O", params)
        _print_hist("evolve O against X-minimax (ch. 15)", ro)
        results = {"O": [s.__dict__ for s in ro.history]}
    else:
        rx, ro = coevolve(params)
        _print_hist("co-evolve X (ch. 16)", rx)
        _print_hist("co-evolve O (ch. 16)", ro)
        results = {
            "X": [s.__dict__ for s in rx.history],
            "O": [s.__dict__ for s in ro.history],
        }

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

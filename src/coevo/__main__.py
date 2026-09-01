"""CLI: python -m coevo --help"""

from __future__ import annotations

import argparse
import json
import sys


def _print_koza(label: str, result) -> None:
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


def _print_ipd(label: str, result) -> None:
    print(f"\n=== {label} ===")
    print(
        f"{'gen':>4}  {'fit':>7}  {'mean':>7}  "
        f"{'vsTFT':>6}  {'vsALLD':>7}  {'vsALLC':>7}  program"
    )
    for s in result.history:
        print(
            f"{s.generation:4d}  {s.best_fitness:7.3f}  {s.mean_fitness:7.3f}  "
            f"{s.best_vs_tft:6.3f}  {s.best_vs_alld:7.3f}  {s.best_vs_allc:7.3f}  "
            f"{s.best_sexp}"
        )
    print(f"best program: {result.best.sexp()}")


def _koza_params(args):
    from .evolve import GPParams

    return GPParams(
        pop_size=args.pop,
        generations=args.gens,
        crossover_prob=args.cx,
        mutation_prob=args.mut,
        reproduction_prob=max(0.0, 1.0 - args.cx - args.mut),
        seed=args.seed,
    )


def _ipd_params(args):
    from .ipd_evolve import IPDParams

    return IPDParams(
        pop_size=args.pop,
        generations=args.gens,
        crossover_prob=args.cx,
        mutation_prob=args.mut,
        reproduction_prob=max(0.0, 1.0 - args.cx - args.mut),
        seed=args.seed,
        rounds=args.rounds,
        noise=args.noise,
        sample_opponents=args.sample,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Koza 1992 game-strategy GP + iterated Prisoner's Dilemma."
    )
    p.add_argument(
        "mode",
        choices=(
            "check",
            "evolve-x",
            "evolve-o",
            "coevolve",
            "ipd-check",
            "ipd-evolve",
            "ipd-coevolve",
            "ipd-horizon",
        ),
    )
    p.add_argument("--pop", type=int, default=60)
    p.add_argument("--gens", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cx", type=float, default=0.90)
    p.add_argument("--mut", type=float, default=0.05)
    p.add_argument("--rounds", type=int, default=50, help="IPD iterations")
    p.add_argument("--noise", type=float, default=0.0, help="IPD move-flip probability")
    p.add_argument("--sample", type=int, default=0, help="IPD coevo opponents per individual (0=all)")
    p.add_argument("--short", type=int, default=1, help="ipd-horizon one-shot length")
    p.add_argument("--long", type=int, default=50, help="ipd-horizon long length")
    p.add_argument("--json", type=str, default="")
    args = p.parse_args(argv)

    results: dict = {}

    if args.mode == "check":
        from .evolve import absolute_fitness_o, absolute_fitness_x
        from .game import GAME_VALUE, minimax_value, play
        from .gp import O_MINIMAX, X_MINIMAX

        v = minimax_value()
        px = play(X_MINIMAX.as_strategy(), O_MINIMAX.as_strategy())
        print(f"game value (minimax backup): {v}")
        print(f"published X vs published O:  {px}")
        raw_x, hits_x, scores_x = absolute_fitness_x(X_MINIMAX)
        raw_o, hits_o, scores_o = absolute_fitness_o(O_MINIMAX)
        print(f"X vs 4 O-scripts: scores={scores_x} sum={raw_x} hits={hits_x}")
        print(f"O vs 8 X-scripts: X-payoffs={scores_o} hits={hits_o}")
        ok = v == GAME_VALUE and px == GAME_VALUE and raw_x == 88 and hits_x == 4
        print("OK" if ok else "MISMATCH")
        return 0 if ok else 1

    if args.mode == "ipd-check":
        from .ipd import ALLC, ALLD, CLASSICS, GRIM, PAVLOV, TFT, play_trees

        pairs = [
            ("TFT", TFT, "TFT", TFT, 10, (30, 30)),
            ("ALLD", ALLD, "ALLC", ALLC, 10, (50, 0)),
            ("TFT", TFT, "ALLD", ALLD, 10, (9, 14)),
            ("GRIM", GRIM, "TFT", TFT, 10, (30, 30)),
            ("PAVLOV", PAVLOV, "TFT", TFT, 10, (30, 30)),
        ]
        ok = True
        for n1, a, n2, b, n, expect in pairs:
            got = play_trees(a, b, n)
            flag = "OK" if got == expect else "FAIL"
            if got != expect:
                ok = False
            print(f"{n1:7} vs {n2:7}  {n} rounds  {got}  expected {expect}  {flag}")
        print("classics:", ", ".join(CLASSICS))
        print("OK" if ok else "MISMATCH")
        return 0 if ok else 1

    if args.mode in {"evolve-x", "evolve-o", "coevolve"}:
        from .evolve import coevolve, evolve_against_minimax

        params = _koza_params(args)
        if args.mode == "evolve-x":
            rx = evolve_against_minimax("X", params)
            _print_koza("evolve X against O-minimax (ch. 15)", rx)
            results = {"X": [s.__dict__ for s in rx.history]}
        elif args.mode == "evolve-o":
            ro = evolve_against_minimax("O", params)
            _print_koza("evolve O against X-minimax (ch. 15)", ro)
            results = {"O": [s.__dict__ for s in ro.history]}
        else:
            rx, ro = coevolve(params)
            _print_koza("co-evolve X (ch. 16)", rx)
            _print_koza("co-evolve O (ch. 16)", ro)
            results = {
                "X": [s.__dict__ for s in rx.history],
                "O": [s.__dict__ for s in ro.history],
            }

    elif args.mode == "ipd-evolve":
        from .ipd_evolve import evolve_vs_classics

        run = evolve_vs_classics(_ipd_params(args))
        _print_ipd(f"IPD vs classics  rounds={args.rounds} noise={args.noise}", run)
        results = {"ipd": [s.__dict__ for s in run.history]}

    elif args.mode == "ipd-coevolve":
        from .ipd_evolve import coevolve_ipd

        run = coevolve_ipd(_ipd_params(args))
        _print_ipd(f"IPD co-evolution  rounds={args.rounds} noise={args.noise}", run)
        results = {"ipd": [s.__dict__ for s in run.history]}

    elif args.mode == "ipd-horizon":
        from .ipd_evolve import horizon_experiment

        short, long = horizon_experiment(args.short, args.long, _ipd_params(args))
        _print_ipd(f"IPD horizon SHORT rounds={args.short}", short)
        _print_ipd(f"IPD horizon LONG  rounds={args.long}", long)
        print("\nshadow-of-the-future contrast (best program vs probes, per-round):")
        print(
            f"  short vs TFT={short.history[-1].best_vs_tft:.3f}  "
            f"vs ALLD={short.history[-1].best_vs_alld:.3f}  {short.best.sexp()}"
        )
        print(
            f"  long  vs TFT={long.history[-1].best_vs_tft:.3f}  "
            f"vs ALLD={long.history[-1].best_vs_alld:.3f}  {long.best.sexp()}"
        )
        results = {
            "short": [s.__dict__ for s in short.history],
            "long": [s.__dict__ for s in long.history],
        }

    if args.json and results:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

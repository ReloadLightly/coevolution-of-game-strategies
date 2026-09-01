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
        description="Koza 1992 GP + IPD + two-state stochastic games."
    )
    p.add_argument(
        "mode",
        choices=(
            "check", "evolve-x", "evolve-o", "coevolve",
            "ipd-check", "ipd-evolve", "ipd-coevolve", "ipd-horizon",
            "sg-check", "sg-evolve", "sg-compare",
        ),
    )
    p.add_argument("--pop", type=int, default=60)
    p.add_argument("--gens", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cx", type=float, default=0.90)
    p.add_argument("--mut", type=float, default=0.05)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--short", type=int, default=1)
    p.add_argument("--long", type=int, default=50)
    p.add_argument("--regime", choices=("collapse", "fixed-rich", "fixed-poor"), default="collapse")
    p.add_argument("--json", type=str, default="")
    args = p.parse_args(argv)
    results: dict = {}

    if args.mode == "check":
        from .evolve import absolute_fitness_o, absolute_fitness_x
        from .game import GAME_VALUE, minimax_value, play
        from .gp import O_MINIMAX, X_MINIMAX
        v = minimax_value()
        px = play(X_MINIMAX.as_strategy(), O_MINIMAX.as_strategy())
        raw_x, hits_x, scores_x = absolute_fitness_x(X_MINIMAX)
        raw_o, hits_o, scores_o = absolute_fitness_o(O_MINIMAX)
        print(f"game value: {v}  X vs O: {px}")
        print(f"X vs 4 O-scripts: {scores_x} sum={raw_x} hits={hits_x}")
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
            print(f"{n1:7} vs {n2:7}  {got}  expected {expect}  {'OK' if got == expect else 'FAIL'}")
            ok = ok and got == expect
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
            results = {"X": [s.__dict__ for s in rx.history], "O": [s.__dict__ for s in ro.history]}

    elif args.mode == "ipd-evolve":
        from .ipd_evolve import evolve_vs_classics
        run = evolve_vs_classics(_ipd_params(args))
        _print_ipd(f"IPD vs classics  rounds={args.rounds}", run)
        results = {"ipd": [s.__dict__ for s in run.history]}
    elif args.mode == "ipd-coevolve":
        from .ipd_evolve import coevolve_ipd
        run = coevolve_ipd(_ipd_params(args))
        _print_ipd(f"IPD co-evolution  rounds={args.rounds}", run)
        results = {"ipd": [s.__dict__ for s in run.history]}
    elif args.mode == "ipd-horizon":
        from .ipd_evolve import horizon_experiment
        short, long = horizon_experiment(args.short, args.long, _ipd_params(args))
        _print_ipd(f"IPD horizon SHORT rounds={args.short}", short)
        _print_ipd(f"IPD horizon LONG  rounds={args.long}", long)
        results = {"short": [s.__dict__ for s in short.history], "long": [s.__dict__ for s in long.history]}
    elif args.mode == "sg-check":
        from .stoch import ALLD, DEFAULT_Q, FIXED_RICH_Q, TFT, play_trees
        a, b, r = play_trees(TFT, TFT, 10, DEFAULT_Q, seed=1)
        print(f"TFT vs TFT  collapse   ({a},{b}) rich={r:.2f}")
        a, b, r = play_trees(ALLD, ALLD, 10, DEFAULT_Q, seed=1)
        print(f"ALLD vs ALLD collapse  ({a},{b}) rich={r:.2f}")
        a, b, r = play_trees(TFT, TFT, 10, FIXED_RICH_Q, seed=1)
        print(f"TFT vs TFT  fixed-rich ({a},{b}) rich={r:.2f}")
        ok = play_trees(TFT, TFT, 10, DEFAULT_Q, seed=1) == (40, 40, 1.0)
        print("OK" if ok else "MISMATCH")
        return 0 if ok else 1
    elif args.mode == "sg-evolve":
        from .stoch_evolve import SGParams, evolve_sg
        params = SGParams(
            pop_size=args.pop, generations=args.gens, crossover_prob=args.cx,
            mutation_prob=args.mut, reproduction_prob=max(0.0, 1.0 - args.cx - args.mut),
            seed=args.seed, rounds=args.rounds, regime=args.regime,
        )
        run = evolve_sg(params)
        print(f"\n=== stochastic game  regime={args.regime} ===")
        print(f"{'gen':>4}  {'fit':>7}  {'vsTFT':>6}  {'vsALLD':>7}  {'rich%':>6}  program")
        for s in run.history:
            print(f"{s.generation:4d}  {s.best_fitness:7.3f}  {s.best_vs_tft:6.3f}  {s.best_vs_alld:7.3f}  {100*s.best_rich_frac:5.1f}%  {s.best_sexp}")
        print(f"best program: {run.best.sexp()}")
        results = {"sg": [s.__dict__ for s in run.history]}
    elif args.mode == "sg-compare":
        from .stoch_evolve import SGParams, compare_regimes
        params = SGParams(
            pop_size=args.pop, generations=args.gens, crossover_prob=args.cx,
            mutation_prob=args.mut, reproduction_prob=max(0.0, 1.0 - args.cx - args.mut),
            seed=args.seed, rounds=args.rounds,
        )
        runs = compare_regimes(params)
        print("\n=== Hilbe contrast: same GP, three kernels ===")
        print(f"{'regime':<12}  {'fit':>7}  {'vsTFT':>6}  {'vsALLD':>7}  {'rich%':>6}  program")
        for name, run in runs.items():
            s = run.history[-1]
            print(f"{name:<12}  {s.best_fitness:7.3f}  {s.best_vs_tft:6.3f}  {s.best_vs_alld:7.3f}  {100*s.best_rich_frac:5.1f}%  {s.best_sexp}")
        results = {k: [s.__dict__ for s in v.history] for k, v in runs.items()}

    if args.json and results:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

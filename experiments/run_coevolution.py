#!/usr/bin/env python3
"""Book-sized co-evolution run (Koza: pop 300, ~40 generations)."""

from coevo.evolve import GPParams, coevolve


def main() -> None:
    params = GPParams(pop_size=80, generations=15, seed=1)
    rx, ro = coevolve(params)
    print("X best:", rx.best.sexp())
    print("O best:", ro.best.sexp())
    print("final X vs-minimax / hits:", rx.history[-1].best_vs_minimax, rx.history[-1].best_abs_hits)
    print("final O vs-minimax / hits:", ro.history[-1].best_vs_minimax, ro.history[-1].best_abs_hits)


if __name__ == "__main__":
    main()

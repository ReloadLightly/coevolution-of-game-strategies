# Fidelity to Koza 1992

## What is reconstructed exactly

From *Genetic evolution and co-evolution of game strategies* (ICGT 1992) and GP book ch. 15–16:

| Piece | Here |
|---|---|
| Game: X three moves, O two moves, L/R, 32 terminals, value 12 | `game.py` |
| Terminals `{L, R}` | `gp.py` |
| Functions `CXM1, COM1, CXM2, COM2` as 3-way CASE on one history slot (U / L / R) | `gp.py` |
| Published X program `(COM2 (COM1 L L R) L R)` | `X_MINIMAX` |
| Published O program `(CXM2 (CXM1 $ R L) L R)` | `O_MINIMAX` (`$` → unused first child) |
| X fitness vs 4 O-scripts, O fitness vs 8 X-scripts | `absolute_fitness_*` |
| Hit = payoff at least as good as the game value | same |
| Co-evolution: relative fitness = mean payoff vs the *current* opposing population | `coevolve()` |
| Ramped half-and-half initialisation | `ramped_half_and_half` |
| Subtree crossover, tournament selection | `evolve.py` |
| Optional mutation (Koza's reported runs used none) | `--mut 0` default |
| Depth cap 17 | `GPParams.max_depth` |

## What is approximated

- **Leaf labels.** The paper prints the 32 payoffs as a figure, not a table. The vector in `game.PAYOFFS` is arranged so that every *published numeric consequence* of the minimax programs is true (value 12; X vs four O-pairs = 32, 16, 28, 12). If you have a scan of Figure 1 and a pair is swapped, send a PR.
- **Selection.** Koza used fitness-proportionate reproduction. This repo defaults to tournament(k=7), which is stabler in small modern runs. Easy to swap.
- **Population.** CLI default is smaller than 300 so a laptop finishes. Pass `--pop 300 --gens 40` for the book-sized run.
- **Lisp.** Individuals print as S-expressions but are Python dataclasses, not eval'd Lisp.

## How the CASE functions work

At any ply only some history slots are defined:

| ply | about to move | defined |
|---|---|---|
| 0 | X | — |
| 1 | O | XM1 |
| 2 | X | XM1, OM1 |
| 3 | O | XM1, OM1, XM2 |
| 4 | X | XM1, OM1, XM2, OM2 |

`CXM1(u, l, r)` evaluates `u` if XM1 is still undefined, `l` if XM1 was L, `r` if XM1 was R. Same pattern for the other three.

So `(COM2 (COM1 L L R) L R)` means:

- if OM2 undefined and OM1 undefined → L   (X's first move)
- if OM2 undefined and OM1 = L → L         (X's second move)
- if OM2 undefined and OM1 = R → R
- if OM2 = L → L                           (X's third move)
- if OM2 = R → R

Which is Koza's simplified minimax for X.

## Co-evolution fitness

X is maximising payoff. O is minimising it. Relative fitness of an O-individual is therefore the mean of `(32 - payoff)` against the current X-population (32 is the largest leaf). Selection never sees the true minimax player; `best_vs_minimax` in the log is a probe, the way Koza reported "absolute fitness" separately from "relative fitness".

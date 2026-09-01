# coevolution-of-game-strategies

A from-scratch Python reconstruction of John R. Koza’s **genetic programming** experiments on evolving (and co-evolving) strategies for a 32-outcome two-person game.

Source chapters / paper:

- Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. MIT Press. **Ch. 15 Evolution of Strategy, Ch. 16 Co-Evolution.**
- Koza, J. R. (1992). *Genetic evolution and co-evolution of game strategies.* International Conference on Game Theory and Its Applications. [PDF](http://www.genetic-programming.com/jkpdf/icgt1992.pdf)

This is **genetic programming**, not a fixed-length genetic algorithm: individuals are S-expression trees in Koza’s function set `{CXM1, COM1, CXM2, COM2}` over terminals `{L, R}`.

## The game

Zero-sum, perfect information, extensive form.

- Players **X** and **O** alternate `L` / `R`.
- X moves 1st, 3rd, 5th (three moves). O moves 2nd and 4th (two moves).
- 2⁵ = **32 terminal payoffs** (to X; O gets the negation).
- Game value under minimax play: **12**.

Koza’s published X-minimax, written `(COM2 (COM1 L L R) L R)`, against the four possible O move-pairs scores **32, 16, 28, 12** (sum 88, four “hits”). That identity is a unit test in this repo.

```
X first move L
  if O's first move was L → X second L, else R
  if O's second move was L → X third L, else R
```

That is exactly what the three-line S-expression does.

## Why co-evolution

Chapter 15 evolves one side against a *known* minimax opponent. Chapter 16 removes that oracle: two populations, each is the fitness environment of the other. Relative fitness only. Absolute fitness vs. the true minimax player is logged, not used for selection.

That is the interesting part if you care about competition under anarchy — neither side is handed the equilibrium strategy.

## Install

```bash
git clone https://github.com/ReloadLightly/coevolution-of-game-strategies.git
cd coevolution-of-game-strategies
python -m pip install -e .
```

No third-party dependencies. Python 3.10+.

## Run

```bash
# verify the reconstructed tree + published programs
python -m coevo check

# chapter 15: evolve X against frozen O-minimax
python -m coevo evolve-x --pop 80 --gens 20 --seed 1

# chapter 15: evolve O against frozen X-minimax
python -m coevo evolve-o --pop 80 --gens 20 --seed 1

# chapter 16: co-evolve both populations
python -m coevo coevolve --pop 40 --gens 15 --seed 1
```

Koza used populations of **300** and showed a co-evolution run through generation **38**. Full 300×300 relative-fitness evaluation is O(pop²) games per generation; start small.

```bash
python -m coevo coevolve --pop 300 --gens 40 --seed 42 --json run.json
```

## Tests

```bash
python -m pip install pytest
PYTHONPATH=src pytest -q
```

## Mapping onto IR theory (optional reading)

The machinery is Koza’s. The interpretation is yours.

| knob in this repo | IR idea |
|---|---|
| one-sided evolution vs frozen minimax | adapting to a known rival doctrine |
| co-evolution, relative fitness only | self-help under anarchy; no referee |
| `best_vs_minimax` vs `best_relative` | absolute gains vs relative gains |
| hits (≥ 12 for X, ≤ 12 for O) | “did we reach the security equilibrium?” |

A later experiment (not in Koza’s chapter, easy to add) is to swap this zero-sum tree for iterated Prisoner's Dilemma and watch reciprocity appear when the horizon lengthens.

## Layout

```
src/coevo/
  game.py      # 32-leaf tree, minimax backup, play()
  gp.py        # trees, CASE functions, crossover, published programs
  evolve.py    # ch.15 one-pop and ch.16 two-pop loops
  __main__.py  # CLI
docs/koza.md   # notes on fidelity to the 1992 paper
```

## Fidelity notes

See [docs/koza.md](docs/koza.md). The leaf-payoff figure is reconstructed so that every *numeric claim* Koza makes about the minimax programs holds (value 12, X-hits 32/16/28/12 summing to 88). Internal node labels in the paper's ASCII figure are not needed at runtime.

## License

MIT. Koza's books and papers remain with their publishers; this is an independent reimplementation of the published experimental setup.

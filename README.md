# coevolution-of-game-strategies

A from-scratch Python reconstruction of John R. Koza’s **genetic programming** experiments on evolving (and co-evolving) strategies for a 32-outcome two-person game, plus iterated Prisoner's Dilemma on the same tree GP.

Source chapters / paper:

- Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. MIT Press. **Ch. 15 Evolution of Strategy, Ch. 16 Co-Evolution.**
- Koza, J. R. (1992). *Genetic evolution and co-evolution of game strategies.* International Conference on Game Theory and Its Applications. [PDF](http://www.genetic-programming.com/jkpdf/icgt1992.pdf)

This is **genetic programming**, not a fixed-length genetic algorithm: individuals are S-expression trees.

## The 32-outcome game

Zero-sum, perfect information, extensive form.

- Players **X** and **O** alternate `L` / `R`.
- X moves 1st, 3rd, 5th (three moves). O moves 2nd and 4th (two moves).
- 2⁵ = **32 terminal payoffs** (to X; O gets the negation).
- Game value under minimax play: **12**.

Koza’s published X-minimax, written `(COM2 (COM1 L L R) L R)`, against the four possible O move-pairs scores **32, 16, 28, 12** (sum 88, four “hits”). That identity is a unit test in this repo.

## Why co-evolution

Chapter 15 evolves one side against a *known* minimax opponent. Chapter 16 removes that oracle: two populations, each is the fitness environment of the other. Relative fitness only. Absolute fitness vs. the true minimax player is logged, not used for selection.

## Install

```bash
git clone https://github.com/ReloadLightly/coevolution-of-game-strategies.git
cd coevolution-of-game-strategies
python -m pip install -e .
```

No third-party dependencies. Python 3.10+.

## Run — Koza 32-outcome game

```bash
python -m coevo check
python -m coevo evolve-x --pop 80 --gens 20 --seed 1
python -m coevo evolve-o --pop 80 --gens 20 --seed 1
python -m coevo coevolve --pop 40 --gens 15 --seed 1
```

Koza used populations of **300** and showed a co-evolution run through generation **38**. Full 300×300 evaluation is O(pop²) games per generation; start small.

## Run — Iterated Prisoner's Dilemma

Same tree GP, Axelrod payoffs. A strategy is a program over `{C, D}` and CASE functions `IOPP1`, `ISELF1`, `IOPP2`, `ISELF2`, `IGRIM`. Details in [docs/ipd.md](docs/ipd.md).

```bash
python -m coevo ipd-check

# evolve against ALLC, ALLD, TFT, GRIM, PAVLOV
python -m coevo ipd-evolve --pop 40 --gens 12 --rounds 40 --seed 1

# co-evolve: fitness = score vs current peers (no oracle)
python -m coevo ipd-coevolve --pop 24 --gens 8 --rounds 20 --sample 8 --seed 1

# shadow of the future: one-shot vs long horizon
python -m coevo ipd-horizon --pop 30 --gens 8 --short 1 --long 40 --seed 1
```

`vsTFT` / `vsALLD` are per-round payoffs (`3.0` = perpetual mutual cooperation). One-shot runs lean `D`; long-horizon runs can keep `vsTFT` near 3.

## Tests

```bash
python -m pip install pytest
PYTHONPATH=src pytest -q
```

## Mapping onto IR theory

| knob in this repo | IR idea |
|---|---|
| one-sided evolution vs frozen minimax | adapting to a known rival doctrine |
| co-evolution, relative fitness only | self-help under anarchy; no referee |
| `best_vs_minimax` vs `best_relative` | absolute gains vs relative gains |
| IPD `--rounds 1` vs `--rounds 50` | one-shot anarchy vs shadow of the future |
| IPD `--noise` | misperception; why monitoring institutions matter |

## Layout

```
src/coevo/
  game.py       # 32-leaf tree, minimax backup, play()
  gp.py         # Koza CASE language for the 32-outcome game
  evolve.py     # ch.15 one-pop and ch.16 two-pop loops
  ipd.py        # IPD payoffs, history, CASE language, classics
  ipd_evolve.py # vs-classics, co-evolution, horizon experiment
  __main__.py   # CLI
docs/koza.md    # fidelity notes for the 1992 paper
docs/ipd.md     # IPD language and IR reading
```

## License

MIT. Koza’s books and papers remain with their publishers; this is an independent reimplementation of the published experimental setup.

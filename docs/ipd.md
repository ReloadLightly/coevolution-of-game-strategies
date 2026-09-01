# Iterated Prisoner's Dilemma

Same GP idea as Koza ch. 15–16: a strategy is a program. The game changes.

## Payoffs (Axelrod 1984)

|   | C | D |
|---|---|---|
| **C** | 3, 3 | 0, 5 |
| **D** | 5, 0 | 1, 1 |

Temptation > Reward > Punishment > Sucker, and `2R > T+S`, so mutual cooperation beats taking turns exploiting.

## Language

Terminals: `C`, `D`.

Functions (3-way CASE, same shape as Koza's `CXM1` / `COM1`):

| fn | inspects | child 0 | child 1 | child 2 |
|---|---|---|---|---|
| `IOPP1` | opponent last move | first round | was C | was D |
| `ISELF1` | own last move | first round | was C | was D |
| `IOPP2` / `ISELF2` | two rounds back | same | same | same |
| `IGRIM` | opponent ever defected | first round | never | at least once |

Published seeds:

```
ALLC    C
ALLD    D
TFT     (IOPP1 C C D)
GRIM    (IGRIM C C D)
PAVLOV  (ISELF1 C (IOPP1 C C D) (IOPP1 D D C))
```

TFT vs TFT scores `3` per round. TFT vs ALLD scores `(0 + 1·(n-1)) / n`.

## Experiments

`ipd-evolve` — one population, environment = the five classics. Closest to Axelrod's GA tournament, but the individual is a Koza tree.

`ipd-coevolve` — one population, fitness = mean score against current peers. Koza ch. 16 applied to IPD: no oracle. Watch `best_vs_tft` (probe) come apart from `best_fitness` (relative).

`ipd-horizon` — identical GP, `rounds=1` vs `rounds=50`.

- One-shot: D dominates. Structural-realist prediction.
- Long horizon: reciprocity can pay. Axelrod / neoliberal-institutionalist prediction (“shadow of the future”).

`--noise 0.05` makes TFT brittle and gives Pavlov / generous variants a reason to exist.

## IR reading

| knob | claim |
|---|---|
| `rounds=1` | anarchy + one interaction → defect |
| `rounds=50` | iteration can sustain cooperation among egoists |
| co-evolution | the rival is not a fixed type; doctrines adapt |
| noise | misperception / imperfect monitoring; institutions as error-correction |
| `best_vs_alld` vs `best_vs_tft` | security against exploiters vs gains from cooperation |

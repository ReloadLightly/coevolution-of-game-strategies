# Stochastic games

IPD repeats *one* stage game. A **stochastic game** (Shapley 1953; also called
a Markov game) lets the stage game itself move.

## Definition

A two-player stochastic game is

- a finite set of **states** S
- actions A_i in each state
- a payoff u_i(s, a_i, a_j)
- a transition kernel q(s' | s, a_i, a_j)
- a start state and a discount / finite horizon

After each joint action the world is redrawn. Strategies may condition on the
current state as well as on history.

Repeated games are the special case |S|=1. MDPs are the special case of one
player. That is why multi-agent RL talks about Markov games.

## What is in this repo

Two states, both Prisoner's Dilemmas, different surplus:

|   | RICH CC / CD / DC / DD | POOR |
|---|---|---|
| payoffs to row | 4 / 0 / 5 / 1 | 2 / 0 / 3 / 1 |

Default transitions (`regime=collapse`, Hilbe et al. Nature 2018):

```
P(next = RICH | CC) = 1
P(next = RICH | any D) = 0
```

Mutual cooperation keeps or restores the good game. Any defection dumps both
players into the poor game. Start in RICH.

The IPD we already had is `regime=fixed-rich`: q ≡ 1.

New primitive: `ISTATE(rich_branch, poor_branch)`.

State-aware seeds:

```
PEACE_TFT   (ISTATE (IOPP1 C C D) D)   # reciprocate only in peace
PEACE_GRIM  (ISTATE (IGRIM C C D) D)
```

## Why this is the IR model IPD is not

| object | IPD | stochastic game |
|---|---|---|
| the relationship | fixed payoff matrix | a state that players co-produce |
| security dilemma | always the same temptation | stakes change after a crisis |
| arms race / detente | iteration only | actions move you between high-tension and low-tension games |
| institutions | shadow of the future (horizon) | also the transition kernel q |
| commons / climate / trade | same PD every year | cooperation preserves the good game |

Hilbe's result, which `sg-compare` is built to replay: iteration alone
(fixed-rich IPD) and a changing world without reciprocity each produce less
cooperation than both together — repeated play plus action-dependent transitions.

Structural realism lives in the collapse kernel: one defection can lock you
into the poor game. Institutionalism is the claim that actors can rewrite q
(make restoration after CD possible, add verification that lowers the chance
a tremble sends you to POOR).

## Commands

```bash
python -m coevo sg-check
python -m coevo sg-evolve --pop 30 --gens 8 --rounds 40 --seed 1
python -m coevo sg-compare --pop 24 --gens 6 --rounds 30 --seed 1
```

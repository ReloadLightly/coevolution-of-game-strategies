"""Two-state stochastic game."""

from coevo.stoch import (
    ALLC,
    ALLD,
    DEFAULT_Q,
    FIXED_RICH_Q,
    PEACE_TFT,
    TFT,
    parse_sexp,
    play_trees,
)


def test_tft_vs_tft_stays_rich():
    sa, sb, rich = play_trees(TFT, TFT, rounds=20, q=DEFAULT_Q, seed=1)
    assert rich == 1.0
    assert sa == sb == 4 * 20


def test_alld_collapses_commons():
    sa, sb, rich = play_trees(ALLD, ALLD, rounds=20, q=DEFAULT_Q, seed=1)
    assert rich == 1 / 20
    assert sa == sb == 20


def test_alld_vs_allc_collapses_after_first():
    sa, sb, rich = play_trees(ALLD, ALLC, rounds=10, q=DEFAULT_Q, seed=1)
    assert rich == 1 / 10
    assert sa == 5 + 9 * 3
    assert sb == 0


def test_fixed_rich_is_ipd_with_rich_payoffs():
    sa, sb, rich = play_trees(TFT, TFT, rounds=10, q=FIXED_RICH_Q, seed=1)
    assert rich == 1.0
    assert sa == 40


def test_peace_tft_defects_in_poor():
    sa, sb, rich = play_trees(PEACE_TFT, ALLD, rounds=5, q=DEFAULT_Q, seed=1)
    assert rich == 1 / 5


def test_istate_parse():
    n = parse_sexp("(ISTATE C D)")
    assert n.name == "ISTATE"
    assert n.children[0].name == "C"
    assert n.children[1].name == "D"

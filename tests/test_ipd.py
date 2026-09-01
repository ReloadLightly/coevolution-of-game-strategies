"""IPD payoffs and canonical programs."""

from coevo.ipd import ALLC, ALLD, GRIM, PAVLOV, TFT, parse_sexp, play_trees


def test_tft_vs_tft_is_reward():
    a, b = play_trees(TFT, TFT, rounds=10)
    assert (a, b) == (30, 30)


def test_alld_exploits_allc():
    a, b = play_trees(ALLD, ALLC, rounds=10)
    assert (a, b) == (50, 0)


def test_tft_vs_alld():
    a, b = play_trees(TFT, ALLD, rounds=10)
    assert a == 0 + 9 * 1
    assert b == 5 + 9 * 1


def test_grim_punishes_forever():
    a, b = play_trees(GRIM, ALLD, rounds=10)
    assert a == 0 + 9 * 1
    assert b == 5 + 9 * 1
    a, b = play_trees(GRIM, TFT, rounds=10)
    assert (a, b) == (30, 30)


def test_pavlov_vs_tft_cooperates():
    a, b = play_trees(PAVLOV, TFT, rounds=10)
    assert (a, b) == (30, 30)


def test_parse_tft():
    assert parse_sexp("(IOPP1 C C D)").sexp() == TFT.sexp()

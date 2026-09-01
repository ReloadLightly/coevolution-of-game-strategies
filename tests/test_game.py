"""Sanity checks for the reconstructed 32-outcome game."""

from coevo.evolve import absolute_fitness_o, absolute_fitness_x
from coevo.game import GAME_VALUE, minimax_value, play
from coevo.gp import O_MINIMAX, X_MINIMAX, parse_sexp


def test_game_value_is_12():
    assert minimax_value() == GAME_VALUE == 12


def test_published_programs_play_to_value():
    assert play(X_MINIMAX.as_strategy(), O_MINIMAX.as_strategy()) == 12


def test_x_minimax_hits():
    raw, hits, scores = absolute_fitness_x(X_MINIMAX)
    assert scores == [32, 16, 28, 12] or set(scores) == {32, 16, 28, 12}
    assert raw == 88
    assert hits == 4


def test_parse_roundtrip():
    s = "(COM2 (COM1 L L R) L R)"
    assert parse_sexp(s).sexp() == s


def test_o_holds_x_to_value_against_minimax_x():
    raw, hits, x_scores = absolute_fitness_o(O_MINIMAX)
    assert play(X_MINIMAX.as_strategy(), O_MINIMAX.as_strategy()) == 12
    assert hits >= 1

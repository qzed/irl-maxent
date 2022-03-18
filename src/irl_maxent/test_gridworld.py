"""
Tests for gridworld.
"""

import pytest
from itertools import product

from . import gridworld


def check_zero_probability(world, sf, st, a):
    """
    Check that the transition-probability for the given transition is only
    non-zero in case of neighboring states or a transition of a state to
    itself via an edge.

    Args:
        world: The world to test this on.
        sf: The state from which the transition originates.
        st: The state to which the transition should go.
        a: The action of the transition.

    Raises:
        AssertionError: In case this check fails.
    """
    at = world.actions[a]

    # if states are neither neighbors or the same state, probability must be zero
    if abs(sf[0] - st[0]) + abs(sf[1] - st[1]) > 1:
        f = world.state_point_to_index(sf)
        t = world.state_point_to_index(st)
        assert world.p_transition[f, t, a] == 0.0

    # if states are the same, we can only move there via an edge
    if sf == st and not 0 <= sf[0] + at[0] < world.size and not 0 <= st[1] + at[1] < world.size:
        assert world.p_transition[f, t, a] == 0.0


def check_zero_probabilities(world):
    """
    Check that all transition-probabilities of the given world are only
    non-zero in case of neighboring states or a transition of a state to
    itself via an edge.

    Args:
        world: The world to test this on.

    Raises:
        AssertionError: In case this check fails.
    """
    for sf in product(range(world.size), range(world.size)):
        for st in product(range(world.size), range(world.size)):
            for a in range(world.n_actions):
                check_zero_probability(world, sf, st, a)


def test_probabilities_gridworld(size=5):
    """
    Check transition-probabilities for GridWorld

    Args:
        size: The size of the world to be used for testing.
    """
    check_zero_probabilities(gridworld.GridWorld(size))


def test_probabilities_icy_gridworld(size=5):
    """
    Check transition-probabilities for IcyGridWorld

    Args:
        size: The size of the world to be used for testing.
    """
    check_zero_probabilities(gridworld.IcyGridWorld(size))

import pytest
from itertools import product

import gridworld


def check_zero_probability(world, sf, st, a):
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
    for sf in product(range(world.size), range(world.size)):
        for st in product(range(world.size), range(world.size)):
            for a in range(world.n_actions):
                check_zero_probability(world, sf, st, a)


def test_probabilities_gridworld(size=5):
    check_zero_probabilities(gridworld.GridWorld(size))


def test_probabilities_icy_gridworld(size=5):
    check_zero_probabilities(gridworld.IcyGridWorld(size))

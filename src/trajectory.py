"""
Trajectories representing expert demonstrations and automated generation
thereof.
"""

import numpy as np
from itertools import chain


class Trajectory:
    """
    A trajectory consisting of states, corresponding actions, and outcomes.

    Args:
        transitions: The transitions of this trajectory as an array of
            tuples `(state_from, action, state_to)`. Note that `state_to` of
            an entry should always be equal to `state_from` of the next
            entry.
    """
    def __init__(self, transitions):
        self._t = transitions

    def transitions(self):
        """
        The transitions of this trajectory.

        Returns:
            All transitions in this trajectory as array of tuples
            `(state_from, action, state_to)`.
        """
        return self._t

    def states(self):
        """
        The states visited in this trajectory.

        Returns:
            All states visited in this trajectory as iterator in the order
            they are visited. If a state is being visited multiple times,
            the iterator will return the state multiple times according to
            when it is visited.
        """
        return map(lambda x: x[0], chain(self._t, [(self._t[-1][2], 0, 0)]))

    def __repr__(self):
        return "Trajectory({})".format(repr(self._t))

    def __str__(self):
        return "{}".format(self._t)


def generate_trajectory(world, policy, start, final):
    """
    Generate a single trajectory.

    Args:
        world: The world for which the trajectory should be generated.
        policy: A function (state: Integer) -> (action: Integer) mapping a
            state to an action, specifying which action to take in which
            state. This function may return different actions for multiple
            invokations with the same state, i.e. it may make a
            probabilistic decision and will be invoked anew every time a
            (new or old) state is visited (again).
        start: The starting state (as Integer index).
        final: A collection of terminal states. If a trajectory reaches a
            terminal state, generation is complete and the trajectory is
            returned.

    Returns:
        A generated Trajectory instance adhering to the given arguments.
    """

    state = start

    trajectory = []
    while state not in final:
        action = policy(state)

        next_s = range(world.n_states)
        next_p = world.p_transition[state, :, action]

        next_state = np.random.choice(next_s, p=next_p)

        trajectory += [(state, action, next_state)]
        state = next_state

    return Trajectory(trajectory)


def generate_trajectories(n, world, policy, start, final):
    """
    Generate multiple trajectories.

    Args:
        n: The number of trajectories to generate.
        world: The world for which the trajectories should be generated.
        policy: A function `(state: Integer) -> action: Integer` mapping a
            state to an action, specifying which action to take in which
            state. This function may return different actions for multiple
            invokations with the same state, i.e. it may make a
            probabilistic decision and will be invoked anew every time a
            (new or old) state is visited (again).
        start: The starting state (as Integer index), a list of starting
            states (with uniform probability), or a list of starting state
            probabilities, mapping each state to a probability. Iff the
            length of the provided list is equal to the number of states, it
            is assumed to be a probability distribution over all states.
            Otherwise it is assumed to be a list containing all starting
            state indices, an individual state is then chosen uniformly.
        final: A collection of terminal states. If a trajectory reaches a
            terminal state, generation is complete and the trajectory is
            complete.

    Returns:
        A generator expression generating `n` `Trajectory` instances
        adhering to the given arguments.
    """
    start_states = np.atleast_1d(start)

    def _generate_one():
        if len(start_states) == world.n_states:
            s = np.random.choice(range(world.n_states), p=start_states)
        else:
            s = np.random.choice(start_states)

        return generate_trajectory(world, policy, s, final)

    return (_generate_one() for _ in range(n))


def policy_adapter(policy):
    """
    A policy adapter for deterministic policies.

    Adapts a deterministic policy given as array or map
    `policy[state] -> action` for the trajectory-generation functions.

    Args:
        policy: The policy as map/array
            `policy[state: Integer] -> action: Integer`
            representing the policy function p(state).

    Returns:
        A function `(state: Integer) -> action: Integer` acting out the
        given policy.
    """
    return lambda state: policy[state]


def stochastic_policy_adapter(policy):
    """
    A policy adapter for stochastic policies.

    Adapts a stochastic policy given as array or map
    `policy[state, action] -> probability` for the trajectory-generation
    functions.

    Args:
        policy: The stochastic policy as map/array
            `policy[state: Integer, action: Integer] -> probability`
            representing the probability distribution p(action | state) of
            an action given a state.

    Returns:
        A function `(state: Integer) -> action: Integer` acting out the
        given policy, choosing an action randomly based on the distribution
        defined by the given policy.
    """
    return lambda state: np.random.choice([*range(policy.shape[1])], p=policy[state, :])

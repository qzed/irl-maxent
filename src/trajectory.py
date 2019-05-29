import numpy as np
from itertools import chain


class Trajectory:
    def __init__(self, transitions):
        self._t = transitions

    def transitions(self):
        return self._t

    def states(self):
        return map(lambda x: x[0], chain(self._t, [(self._t[-1][2], 0, 0)]))


def generate_trajectory(world, policy, start, final):
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
    return (generate_trajectory(world, policy, start, final) for _ in range(n))


def policy_adapter(policy):
    return lambda state: policy[state]


def stochastic_policy_adapter(policy):
    return lambda state: np.random.choice([*range(policy.shape[1])], p=policy[state, :])

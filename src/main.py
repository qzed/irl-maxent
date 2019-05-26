#!/usr/bin/env python

import gridworld as gw
import plotting

import numpy as np
import matplotlib.pyplot as plt


def main():
    style = {
        'cmap': 'gray',
        'border': {'color': 'red', 'linewidth': 0.5},
    }

    world = gw.GridWorld(5)

    ax = plt.figure().add_subplot(111)
    plotting.plot_transition_probabilities(world, ax, **style)
    plt.show()

    reward = np.zeros(world.n_states)
    reward[-1] = 1.0
    value = gw.stochastic_value_iteration(world.p_transition, reward, 0.8)
    policy = gw.optimal_policy_from_value(world, value)

    ax = plt.figure().add_subplot(111)
    plotting.plot_state_values(world, value, ax, **style)
    plotting.plot_deterministic_policy(world, policy, ax)
    plt.show()

    policy = gw.stochastic_policy_from_value(world, value, w=lambda x: x**2)

    ax = plt.figure().add_subplot(111)
    plotting.plot_stochastic_policy(world, policy, ax, **style)
    plt.show()

    for _ in range(5):
        print(gw.generate_trajectory(world, gw.stochastic_policy_adapter(policy), 0, [24]))


if __name__ == '__main__':
    main()

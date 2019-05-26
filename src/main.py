#!/usr/bin/env python

import gridworld
import plotting

import numpy as np
import matplotlib.pyplot as plt


def main():
    style = {
        'cmap': 'gray',
        'border': {'color': 'red', 'linewidth': 0.5},
    }

    world = gridworld.GridWorld(5)

    for _ in range(5):
        print(gridworld.generate_trajectory(world, lambda s: 0, 0, [4, 9, 14, 19, 24]))

    ax = plt.figure().add_subplot(111)
    plotting.plot_transition_probabilities(world, ax, **style)
    plt.show()

    reward = np.zeros(world.n_states)
    reward[-1] = 1.0
    value = gridworld.stochastic_value_iteration(world.p_transition, reward, 0.8)
    policy = gridworld.optimal_policy_from_value(world, value)

    ax = plt.figure().add_subplot(111)
    plotting.plot_state_values(world, value, ax, **style)
    plotting.plot_deterministic_policy(world, policy, ax)
    plt.show()

    policy = gridworld.stochastic_policy_from_value(world, value, w=lambda x: x**2)

    ax = plt.figure().add_subplot(111)
    plotting.plot_stochastic_policy(world, policy, ax, **style)
    plt.show()


if __name__ == '__main__':
    main()

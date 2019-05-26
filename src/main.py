#!/usr/bin/env python

import gridworld
import plotting

import numpy as np
import matplotlib.pyplot as plt


def main():
    cmap = 'gray'
    border_style = {'color': 'red', 'linewidth': 0.5}

    world = gridworld.GridWorld(5)

    for _ in range(5):
        print(gridworld.generate_trajectory(world, lambda s: 0, 0, [4, 9, 14, 19, 24]))

    ax = plt.figure().add_subplot(111)
    plotting.plot_transition_probabilities(world, ax, cmap=cmap, border=border_style)
    plt.show()

    reward = np.zeros(world.n_states)
    reward[-1] = 1.0
    value = gridworld.value_iteration(world.p_transition, reward, 0.8)

    ax = plt.figure().add_subplot(111)
    plotting.plot_state_values(world, value, ax, cmap=cmap, border=border_style)
    plt.show()


if __name__ == '__main__':
    main()

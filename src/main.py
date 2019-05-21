#!/usr/bin/env python

import gridworld
import plotting

import matplotlib.pyplot as plt


def main():
    world = gridworld.IcyGridWorld(5)

    for _ in range(5):
        print(gridworld.generate_trajectory(world, lambda s: 0, 0, [4, 9, 14, 19, 24]))

    ax = plt.figure().add_subplot(111)
    plotting.plot_transition_probabilities(world, ax, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()

#!/usr/bin/env python

import gridworld
import plotting

import matplotlib.pyplot as plt


def main():
    world = gridworld.IcyGridWorld(5)

    ax = plt.figure().add_subplot(111)
    plotting.plot_transition_probabilities(world, ax, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()

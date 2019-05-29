#!/usr/bin/env python

import gridworld as gw
import maxent
import plot
import trajectory as T

import numpy as np
import matplotlib.pyplot as plt


def main():
    style = {
        'border': {'color': 'red', 'linewidth': 0.5},
    }

    world = gw.IcyGridWorld(5)

    ax = plt.figure().add_subplot(111)
    plot.plot_transition_probabilities(ax, world, **style)
    plt.show()

    reward = np.zeros(world.n_states)
    reward[-1] = 1.0

    initial = np.zeros(world.n_states)
    initial[0] = 1.0

    value = gw.value_iteration(world.p_transition, reward, 0.8)
    policy = gw.optimal_policy_from_value(world, value)

    pla = maxent.local_action_probabilities(world.p_transition, [24], reward)
    svf = maxent.expected_svf_from_policy(world.p_transition, initial, [24], pla)

    ax = plt.figure().add_subplot(111)
    plot.plot_stochastic_policy(ax, world, pla, **style)
    plt.show()

    ax = plt.figure().add_subplot(111)
    plot.plot_state_values(ax, world, svf, **style)
    plt.show()

    ax = plt.figure().add_subplot(111)
    plot.plot_state_values(ax, world, value, **style)
    plot.plot_deterministic_policy(ax, world, policy)
    plt.show()

    policy = gw.stochastic_policy_from_value(world, value, w=lambda x: x**2)

    ax = plt.figure().add_subplot(111)
    plot.plot_stochastic_policy(ax, world, policy, **style)

    ts = [*T.generate_trajectories(200, world, T.stochastic_policy_adapter(policy), 0, [24])]
    for t in ts:
        plot.plot_trajectory(ax, world, t, color='yellow', alpha=0.025)

    plt.show()

    features = gw.state_features(world)
    f_expect = maxent.feature_expectation_from_trajectories(features, ts)

    ax = plt.figure().add_subplot(111)
    plot.plot_state_values(ax, world, f_expect, **style)
    plt.show()

    irl_reward = maxent.irl(world.p_transition, features, [24], ts, 20, 0.2)
    value = gw.value_iteration(world.p_transition, reward, 0.8)
    ts = [*T.generate_trajectories(200, world, T.stochastic_policy_adapter(policy), 0, [24])]

    ax = plt.figure().add_subplot(111)
    plot.plot_state_values(ax, world, irl_reward, **style)
    for t in ts:
        plot.plot_trajectory(ax, world, t, color='yellow', alpha=0.025)
    plt.show()


if __name__ == '__main__':
    main()

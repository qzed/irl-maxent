#!/usr/bin/env python

import gridworld as gw
import maxent as me
import plot as P
import trajectory as T
import solver as S
import optimizer as opt

import numpy as np
import matplotlib.pyplot as plt


def main():
    style = {
        'border': {'color': 'red', 'linewidth': 0.5},
    }

    world = gw.IcyGridWorld(5)

    ax = plt.figure().add_subplot(111)
    P.plot_transition_probabilities(ax, world, **style)
    plt.show()

    reward = np.zeros(world.n_states)
    reward[-1] = 1.0

    initial = np.zeros(world.n_states)
    initial[0] = 1.0

    value = S.value_iteration(world.p_transition, reward, 0.8)
    policy = S.optimal_policy_from_value(world, value)

    pla = me.local_action_probabilities(world.p_transition, [24], reward)
    svf = me.expected_svf_from_policy(world.p_transition, initial, [24], pla)

    ax = plt.figure().add_subplot(111)
    P.plot_stochastic_policy(ax, world, pla, **style)
    plt.show()

    ax = plt.figure().add_subplot(111)
    P.plot_state_values(ax, world, svf, **style)
    plt.show()

    ax = plt.figure().add_subplot(111)
    P.plot_state_values(ax, world, value, **style)
    P.plot_deterministic_policy(ax, world, policy)
    plt.show()

    policy = S.stochastic_policy_from_value(world, value, w=lambda x: x**2)

    ax = plt.figure().add_subplot(111)
    P.plot_stochastic_policy(ax, world, policy, **style)

    ts = [*T.generate_trajectories(200, world, T.stochastic_policy_adapter(policy), 0, [24])]
    for t in ts:
        P.plot_trajectory(ax, world, t, color='yellow', alpha=0.025)

    plt.show()

    features = gw.state_features(world)
    f_expect = me.feature_expectation_from_trajectories(features, ts)

    ax = plt.figure().add_subplot(111)
    P.plot_state_values(ax, world, f_expect, **style)
    plt.show()

    init = opt.Constant(fn=lambda n: 1.0 / n)
    optim = opt.ExpSga(lr=0.2)
    # optim = opt.ExpSga(lr=opt.linear_decay(0.2))
    # optim = opt.Sga(lr=opt.power_decay(0.2))
    irl_reward = me.irl(world.p_transition, features, [24], ts, optim, init, 20)

    irl_reward -= irl_reward.min()
    irl_reward /= irl_reward.sum()

    value = S.value_iteration(world.p_transition, irl_reward, 0.8)
    policy = S.stochastic_policy_from_value(world, value, w=lambda x: x**2)
    ts = [*T.generate_trajectories(200, world, T.stochastic_policy_adapter(policy), 0, [24])]

    ax = plt.figure().add_subplot(111)
    P.plot_state_values(ax, world, irl_reward, **style)
    for t in ts:
        P.plot_trajectory(ax, world, t, color='yellow', alpha=0.025)
    plt.show()


if __name__ == '__main__':
    main()

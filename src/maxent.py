"""
Maximum Entropy Inverse Reinforcement Learning.

Based on the corresponding paper by B. Ziebart et al. (2008).
"""

import numpy as np
from itertools import product


def feature_expectation_from_trajectories(features, trajectories):
    """
    Compute the feature expectation of the given trajectories.

    Simply counts the number of visitations to each feature-instance and
    divides them by the number of trajectories.

    Args:
        features: The feature-matrix (e.g. as numpy array), mapping states
            to features, i.e. a matrix of shape (n_states x n_features).
        trajectories: A list or iterator of `Trajectory` instances.

    Returns:
        The feature-expectation of the provided trajectories as map
        `[state: Integer] -> feature_expectation: Float`.
    """
    n_states, n_features = features.shape

    fe = np.zeros(n_features)

    for t in trajectories:
        for s in t.states():
            fe += features[s, :]

    return fe / len(trajectories)


def initial_probabilities_from_trajectories(n_states, trajectories):
    """
    Compute the probability of a state being a starting state using the
    given trajectories.

    Args:
        n_states: The number of states.
        trajectories: A list or iterator of `Trajectory` instances.

    Returns:
        The probability of a state being a starting-state as map
        `[state: Integer] -> probability: Float`.
    """
    p = np.zeros(n_states)

    for t in trajectories:
        p[t.transitions()[0][0]] += 1.0

    return p / len(trajectories)


def local_action_probabilities(p_transition, terminal, reward):
    """
    Compute the local action probabilities required for the edge frequency
    calculation.

    This is the backward pass of Algorithm 1 of the Maximum Entropy IRL
    paper by Ziebart et al. (2008).

    Args:
        p_transition: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        terminal: A set/list of terminal states.
        reward: The reward signal per state as table
            `[state: Integer] -> reward: Float`.

    Returns:
        The local action probabilities as map
        `[state: Integer, action: Integer] -> probability: Float`
    """
    n_states, _, n_actions = p_transition.shape

    er = np.exp(reward)
    p = [np.array(p_transition[:, :, a]) for a in range(n_actions)]

    # initialize at terminal states
    zs = np.zeros(n_states)
    for s in terminal:
        zs[s] = 1.0

    # perform backward pass
    # This does not converge, instead we iterate a fixed number of steps. The
    # number of steps is chosen to reflect the maximum steps required to
    # guarantee propagation from any state to any other state and back in an
    # arbitrary MDP defined by p_transition.
    for _ in range(2 * n_states):
        za = np.array([er * p[a].dot(zs) for a in range(n_actions)]).T
        zs = za.sum(axis=1)

    # compute local action probabilities
    return za / zs[:, None]


def expected_svf_from_policy(p_transition, p_initial, terminal, p_action, eps=1e-5):
    """
    Compute the expected state visitation frequency using the given local
    action probabilities.

    This is the forward pass of Algorithm 1 of the Maximum Entropy IRL paper
    by Ziebart et al. (2008). Alternatively, it can also be found as
    Algorithm 9.3 in in Ziebart's thesis (2010).

    It has been slightly adapted for convergence, by forcing transition
    probabilities from terminal stats to be zero.

    Args:
        p_transition: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        p_initial: The probability of a state being an initial state as map
            `[state: Integer] -> probability: Float`.
        terminal: A list of terminal states.
        p_action: Local action probabilities as map
            `[state: Integer, action: Integer] -> probability: Float`
            as returned by `local_action_probabilities`.
        eps: The threshold to be used as convergence criterion. Convergence
            is assumed if the expected state visitation frequency changes
            less than the threshold on all states in a single iteration.

    Returns:
        The expected state visitation frequencies as map
        `[state: Integer] -> svf: Float`.
    """
    n_states, _, n_actions = p_transition.shape

    # 'fix' our transition probabilities to allow for convergence
    # we will _never_ leave any terminal state
    p_transition = np.copy(p_transition)
    p_transition[terminal, :, :] = 0.0

    # set-up transition matrices for each action
    p_transition = [np.array(p_transition[:, :, a]) for a in range(n_actions)]

    # actual forward-computation of state expectations
    d = np.zeros(n_states)

    delta = np.inf
    while delta > eps:
        d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)]
        d_ = p_initial + np.array(d_).sum(axis=0)

        delta, d = np.max(np.abs(d_ - d)), d_

    return d


def compute_expected_svf(p_transition, p_initial, terminal, reward, eps=1e-5):
    """
    Compute the expected state visitation frequency.

    This is an implementation of Algorithm 1 of the Maximum Entropy IRL paper
    by Ziebart et al. (2008).

    This function combines the backward pass implemented in
    `local_action_probabilities` with the forward pass implemented in
    `expected_svf_from_policy`.

    Args:
        p_transition: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        p_initial: The probability of a state being an initial state as map
            `[state: Integer] -> probability: Float`.
        terminal: A list of terminal states.
        reward: The reward signal per state as table
            `[state: Integer] -> reward: Float`.
        eps: The threshold to be used as convergence criterion for the
            expected state-visitation frequency. Convergence is assumed if
            the expected state visitation frequency changes less than the
            threshold on all states in a single iteration.

    Returns:
        The expected state visitation frequencies as map
        `[state: Integer] -> svf: Float`.
    """
    p_action = local_action_probabilities(p_transition, terminal, reward)
    return expected_svf_from_policy(p_transition, p_initial, terminal, p_action, eps)


def irl(p_transition, features, terminal, trajectories, optim, init, eps=1e-4, eps_esvf=1e-5):
    """
    Compute the reward signal given the demonstration trajectories using the
    maximum entropy inverse reinforcement learning algorithm proposed in the
    corresponding paper by Ziebart et al. (2008).

    Args:
        p_transition: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        features: The feature-matrix (e.g. as numpy array), mapping states
            to features, i.e. a matrix of shape (n_states x n_features).
        terminal: A list of terminal states.
        trajectories: A list of `Trajectory` instances representing the
            expert demonstrations.
        optim: The `Optimizer` instance to use for gradient-based
            optimization.
        init: The `Initializer` to use for initialization of the reward
            function parameters.
        eps: The threshold to be used as convergence criterion for the
            reward parameters. Convergence is assumed if all changes in the
            scalar parameters are less than the threshold in a single
            iteration.
        eps_svf: The threshold to be used as convergence criterion for the
            expected state-visitation frequency. Convergence is assumed if
            the expected state visitation frequency changes less than the
            threshold on all states in a single iteration.

    Returns:
        The reward per state as table `[state: Integer] -> reward: Float`.
    """
    n_states, _, n_actions = p_transition.shape
    _, n_features = features.shape

    # compute static properties from trajectories
    e_features = feature_expectation_from_trajectories(features, trajectories)
    p_initial = initial_probabilities_from_trajectories(n_states, trajectories)

    # basic gradient descent
    theta = init(n_features)
    delta = np.inf

    optim.reset(theta)
    while delta > eps:
        theta_old = theta.copy()

        # compute per-state reward
        reward = features.dot(theta)

        # compute the gradient
        e_svf = compute_expected_svf(p_transition, p_initial, terminal, reward, eps_esvf)
        grad = e_features - features.T.dot(e_svf)

        # perform optimization step and compute delta for convergence
        optim.step(grad)
        delta = np.max(np.abs(theta_old - theta))

    # re-compute per-state reward and return
    return features.dot(theta)

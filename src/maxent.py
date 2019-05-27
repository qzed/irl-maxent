import numpy as np


def feature_expectation_from_trajectories(features, trajectories):
    n_states, n_features = features.shape

    fe = np.zeros(n_features)

    for t in trajectories:
        for s in t.states():
            fe += features[s]

    return fe / len(trajectories)


def local_action_probabilities(p_transition, terminal, reward):
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
        za = np.array([np.multiply(er, np.dot(p[a], zs)) for a in range(n_actions)]).T
        zs = np.sum(za, axis=1)

    # compute local action probabilities
    return za / zs[:, None]

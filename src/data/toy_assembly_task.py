import numpy as np
from itertools import product               # Cartesian product for iterators
import optimizer as O                       # stochastic gradient descent optimizer

# ------------------------------------------------ Canonical Task ---------------------------------------------------- #

actions = [0,  # screw at location A
           1,  # weld at location A
           2]  # hammer at location A

# # s = [screwed, welded, hammered, previous screw, previous weld, previous hammer]
# features = [[0, 0, 0, 0, 0, 0],  # nothing done, no previous tool
#             [1, 0, 0, 1, 0, 0],  # screwed, previous tool was screw
#             [0, 1, 0, 0, 1, 0],  # welded, previous tool was weld
#             [0, 0, 1, 0, 0, 1],
#             [1, 1, 0, 1, 0, 0],
#             [1, 1, 0, 0, 1, 0],
#             [1, 0, 1, 1, 0, 0],
#             [1, 0, 1, 0, 0, 1],
#             [0, 1, 1, 0, 1, 0],
#             [0, 1, 1, 0, 0, 1],
#             [1, 1, 1, 1, 0, 0],
#             [1, 1, 1, 0, 1, 0],
#             [1, 1, 1, 0, 0, 1]]
#
# terminal = [10, 11, 12]
#
# trajectories = [[(0, 0, 1), (1, 1, 5), (5, 2, 12)]]


features = [[0, 0, 0],  # nothing done, no previous tool
            [1, 0, 0],  # screwed, previous tool was screw
            [0, 1, 0],  # welded, previous tool was weld
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]]

terminal = [7]

trajectories = [[(0, 0, 1), (1, 1, 4), (4, 2, 7)]]


def p_transition(s_from, s_to, a):
    f_from = list(features[s_from])
    f_from[a] = 1
    # f_from[3:] = [0, 0, 0]
    # f_from[3 + a] = 1
    f_to = list(features[s_to])
    if f_from == f_to:
        return 1.0
    else:
        return 0.0


def feature_expectation_from_trajectories(features, trajectories):
    n_states, n_features = features.shape

    fe = np.zeros(n_features)

    for t in trajectories:                  # for each trajectory
        for s, a, sp in t:                  # for each state in trajectory
            fe += features[s, :]            # sum-up features
        fe += features[sp, :]

    return fe / len(trajectories)           # average over trajectories


def initial_probabilities_from_trajectories(n_states, trajectories):
    p = np.zeros(n_states)

    for t in trajectories:                  # for each trajectory
        p[t[0][0]] += 1.0                   # increment starting state

    return p / len(trajectories)            # normalize


def compute_expected_svf(n_states, n_actions, p_initial, terminal, reward, eps=1e-5):
    nonterminal = set(range(n_states)) - set(terminal)  # nonterminal states

    # Backward Pass
    # 1. initialize at terminal states
    zs = np.zeros(n_states)  # zs: state partition function
    zs[terminal] = 1.0

    # 2. perform backward pass
    for _ in range(2 * n_states):  # longest trajectory: n_states
        # reset action values to zero
        za = np.zeros((n_states, n_actions))  # za: action partition function

        # for each state-action pair
        for s_from, a in product(range(n_states), range(n_actions)):

            # sum over s_to
            for s_to in range(n_states):
                za[s_from, a] += p_transition(s_from, s_to, a) * np.exp(reward[s_from]) * zs[s_to]

        # sum over all actions
        zs = za.sum(axis=1)

    # 3. compute local action probabilities
    p_action = za / zs[:, None]

    # Forward Pass
    # 4. initialize with starting probability
    d = np.zeros((n_states, 2 * n_states))  # d: state-visitation frequencies
    d[:, 0] = p_initial

    # 5. iterate for N steps
    for t in range(1, 2 * n_states):  # longest trajectory: n_states

        # for all states
        for s_to in range(n_states):

            # sum over nonterminal state-action pairs
            for s_from, a in product(nonterminal, range(n_actions)):
                d[s_to, t] += d[s_from, t - 1] * p_action[s_from, a] * p_transition(s_from, s_to, a)

    # 6. sum-up frequencies
    return d.sum(axis=1)


def maxent_irl(actions, features, terminal, trajectories, optim, init, eps=1e-4):
    n_actions = len(actions)
    n_states, n_features = features.shape

    # compute feature expectation from trajectories
    e_features = feature_expectation_from_trajectories(features, trajectories)

    # compute starting-state probabilities from trajectories
    p_initial = initial_probabilities_from_trajectories(n_states, trajectories)

    # gradient descent optimization
    omega = init(n_features)  # initialize our parameters
    delta = np.inf  # initialize delta for convergence check

    optim.reset(omega)  # re-start optimizer
    while delta > eps:  # iterate until convergence
        omega_old = omega.copy()

        # compute per-state reward from features
        reward = features.dot(omega)

        # compute gradient of the log-likelihood
        e_svf = compute_expected_svf(n_states, n_actions, p_initial, terminal, reward)
        grad = e_features - features.T.dot(e_svf)

        # perform optimization step and compute delta for convergence
        optim.step(grad)

        # re-compute detla for convergence check
        delta = np.max(np.abs(omega_old - omega))

    # re-compute per-state reward and return
    return features.dot(omega), omega


# set up features: we use one feature vector per state
features = np.array(features)

# choose our parameter initialization strategy:
# initialize parameters with constant
init = O.Constant(1.0)

# choose our optimization strategy:
#   we select exponentiated stochastic gradient descent with linear learning-rate decay
optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

# actually do some inverse reinforcement learning
reward_maxent, weights = maxent_irl(actions, features, terminal, trajectories, optim, init)
print("Canonical Task Done")

# -------------------------------------------------- Actual Task ----------------------------------------------------- #
actions = [0,  # screw at location A
           1,  # screw at location A
           2,  # weld at location A
           3,  # hammer at location A
           4]  # hammer at location A

features = [[0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 2],
            [2, 1, 0],
            [2, 0, 1],
            [1, 1, 1],
            [1, 0, 2],
            [0, 1, 2],
            [2, 1, 1],
            [2, 0, 2],
            [1, 1, 2],
            [2, 1, 2]]
features = np.array(features)

rewards = features.dot(weights)
print("Actual Task Done")

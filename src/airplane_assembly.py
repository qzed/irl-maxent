from copy import deepcopy
import numpy as np
from itertools import product               # Cartesian product for iterators
import optimizer as O                       # stochastic gradient descent optimizer

# ------------------------------------------------ Complex Task ---------------------------------------------------- #

# actions = [0,   # insert main wing
#            1,   # insert tail wing
#            2,   # insert right wing tip
#            3,   # insert left wing tip
#            4,   # insert long bolt into main wing 1
#            5,   # insert long bolt into main wing 2
#            6,   # insert long bolt into main wing 3
#            7,   # insert long bolt into main wing 4
#            8,   # insert long bolt into tail wing 1
#            9,   # insert long bolt into tail wing 2
#            10,  # screw long bolt into main wing 1
#            11,  # screw long bolt into main wing 2
#            12,  # screw long bolt into main wing 3
#            13,  # screw long bolt into main wing 4
#            14,  # screw long bolt into tail wing 1
#            15,  # screw long bolt into tail wing 2
#            16,  # screw propeller 1
#            17,  # screw propeller 2
#            18,  # screw propeller 3
#            19,  # screw propeller 4
#            20,  # screw propeller base
#            21]  # screw propeller cap

actions = [0,   # insert main wing
           1,   # insert tail wing
           2,   # insert right wing tip
           3,   # insert left wing tip
           4,   # insert long bolt into main wing
           5,   # insert long bolt into tail wing
           6,   # screw long bolt into main wing
           7,   # screw long bolt into tail wing
           8,   # screw propeller
           9,   # screw propeller base
           10]  # screw propeller cap

# factor = [physical_effort, mental_effort]
factors = [[3.6, 2.6],  # insert main wing
           [2.4, 2.2],  # insert tail wing
           [1.8, 1.6],  # insert right wing tip
           [1.8, 1.6],  # insert left wing tip
           [1.6, 2.0],  # insert long bolt into main wing
           [1.4, 1.4],  # insert long bolt into tail wing
           [2.8, 1.8],  # screw long bolt into main wing
           [2.0, 1.8],  # screw long bolt into tail wing
           [3.8, 2.6],  # screw propeller
           [2.2, 1.6],  # screw propeller base
           [2.2, 2.4]]  # screw propeller cap

s_start = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
s_end = [1, 1, 1, 1, 4, 2, 4, 2, 4, 1, 1]

terminal = [7]

trajectories = [[(0, 0, 1), (1, 1, 4), (4, 2, 7)]]

states = []


def rollout_states(prev_states):
    next_states = []
    for state in prev_states:
        if not state:
            print("Damn")
        for action in actions:
            next_state = transition(state, action)
            if next_state:
                next_states.append(next_state)


        states.append(next_states)
        rollout_states(next_states)


def feature_vector(state):
    n_actions, n_features = factors.shape
    feature_value = np.zeros(n_features)
    for action, executed in enumerate(state):
        feature_value += executed * np.array(factors[action])

    return feature_value


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


def transition(s_from, a):

    # preconditions
    if a in [0, 1, 2, 3] and s_from[a] < 1:
        p = 1.0
    elif a == 4 and s_from[a] < 4 and s_from[0] == 1:
        p = 1.0
    elif a == 5 and s_from[a] < 2 and s_from[1] == 1:
        p = 1.0
    elif a == 6 and s_from[a] < 4 and s_from[a] + 1 <= s_from[a-2]:
        p = 1.0
    elif a == 7 and s_from[a] < 2 and s_from[a] + 1 <= s_from[a-2]:
        p = 1.0
    elif a == 8 and s_from[a] < 4:
        p = 1.0
    elif a == 9 and s_from[a] < 1 and s_from[a-1] == 4:
        p = 1.0
    elif a == 10 and s_from[a] < 1 and s_from[a-1]:
        p = 1.0
    else:
        p = 0.0

    # transition to next state
    if p == 1.0:
        s_to = deepcopy(s_from)
        s_to[a] += 1
        return s_to
    else:
        return None


def feature_expectation_from_trajectories(trajectories):
    n_actions, n_features = factors.shape

    fe = np.zeros(n_features)

    for t in trajectories:                  # for each trajectory
        for s, a, sp in t:                  # for each state in trajectory
            fe += feature_vector(s)         # sum-up features
        fe += feature_vector(sp)

    return fe / len(trajectories)           # average over trajectories


def init_probs(state):
    n_actions, n_features = factors.shape
    if np.array(state) == np.zeros(n_actions):
        return 1.0
    else:
        return 0.0


# def initial_probabilities_from_trajectories(n_states, trajectories):
#     p = np.zeros(n_states)
#
#     for t in trajectories:                  # for each trajectory
#         p[t[0][0]] += 1.0                   # increment starting state
#
#     return p / len(trajectories)            # normalize


def compute_expected_svf(n_actions, p_initial, terminal, omega, eps=1e-5):
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


def maxent_irl(features, terminal, trajectories, optim, init, eps=1e-4):
    n_actions, n_features = factors.shape

    # compute feature expectation from trajectories
    e_features = feature_expectation_from_trajectories(trajectories)

    # compute starting-state probabilities from trajectories
    p_initial = init_probs

    # gradient descent optimization
    omega = init(n_features)  # initialize our parameters
    delta = np.inf  # initialize delta for convergence check

    optim.reset(omega)  # re-start optimizer
    while delta > eps:  # iterate until convergence
        omega_old = omega.copy()

        # compute per-state reward from features
        # reward = features.dot(omega)

        # compute gradient of the log-likelihood
        e_svf = compute_expected_svf(n_actions, p_initial, terminal, omega)
        grad = e_features - features.T.dot(e_svf)

        # perform optimization step and compute delta for convergence
        optim.step(grad)

        # re-compute detla for convergence check
        delta = np.max(np.abs(omega_old - omega))

    # re-compute per-state reward and return
    return features.dot(omega), omega


rollout_states([s_start])

# choose our parameter initialization strategy:
# initialize parameters with constant
init = O.Constant(1.0)

# choose our optimization strategy:
#   we select exponentiated stochastic gradient descent with linear learning-rate decay
optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

# actually do some inverse reinforcement learning
reward_maxent, weights = maxent_irl(features, terminal, trajectories, optim, init)
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

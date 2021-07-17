from copy import deepcopy
import numpy as np
from itertools import product  # Cartesian product for iterators
import optimizer as O  # stochastic gradient descent optimizer

# ------------------------------------------------- Complex Task ---------------------------------------------------- #

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

# actions that can be taken in the complex task
actions = [0,  # insert main wing
           1,  # insert tail wing
           2,  # insert right wing tip
           3,  # insert left wing tip
           4,  # insert long bolt into main wing
           5,  # insert long bolt into tail wing
           6,  # screw long bolt into main wing
           7,  # screw long bolt into tail wing
           8,  # screw propeller
           9,  # screw propeller base
           10]  # screw propeller cap

# feature values for each action = [physical_effort, mental_effort]
features = [[3.6, 2.6],  # insert main wing
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

# start state of the assembly task (none of the actions have been performed)
s_start = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# terminal state of the assembly (each action has been performed)
s_end = [1, 1, 1, 1, 4, 2, 4, 2, 4, 1, 1]

# list of all states
all_states = []

# user demonstration (action sequence)
demos = [[8, 8, 8, 8, 0, 1, 2, 3, 5, 4, 6, 7, 9, 10]]


# -------------------------------------------------- Functions ----------------------------------------------------- #


def transition(s_from, a):
    # preconditions
    if a in [0, 1, 2, 3] and s_from[a] < 1:
        p = 1.0
    elif a == 4 and s_from[a] < 4 and s_from[0] == 1:
        p = 1.0
    elif a == 5 and s_from[a] < 2 and s_from[1] == 1:
        p = 1.0
    elif a == 6 and s_from[a] < 4 and s_from[a] + 1 <= s_from[a - 2]:
        p = 1.0
    elif a == 7 and s_from[a] < 2 and s_from[a] + 1 <= s_from[a - 2]:
        p = 1.0
    elif a == 8 and s_from[a] < 4:
        p = 1.0
    elif a == 9 and s_from[a] < 1 and s_from[a - 1] == 4:
        p = 1.0
    elif a == 10 and s_from[a] < 1 and s_from[a - 1]:
        p = 1.0
    else:
        p = 0.0

    # transition to next state
    if p == 1.0:
        s_to = deepcopy(s_from)
        s_to[a] += 1
        return p, s_to
    else:
        return p, None


def get_trajectories(all_states, demos):
    trajectories = []
    for demo in demos:
        s = s_start
        trajectory = []
        for action in demo:
            p, sp = transition(s, action)
            s_idx, sp_idx = all_states.index(s), all_states.index(sp)
            trajectory.append((s_idx, action, sp_idx))
            s = sp
        trajectories.append(trajectory)

    return trajectories


def feature_vector(state):
    n_actions, n_features = np.array(features).shape
    feature_value = np.zeros(n_features)
    for action, executed in enumerate(state):
        feature_value += executed * np.array(features[action])

    return feature_value


def feature_expectation_from_trajectories(features, trajectories):
    n_actions, n_features = features.shape
    fe = np.zeros(n_features)

    for t in trajectories:  # for each trajectory
        for s_idx, a, sp_idx in t:  # for each state in trajectory
            fe += feature_vector(all_states[sp_idx])  # sum-up features

    return fe / len(trajectories)  # average over trajectories


def initial_probabilities_from_trajectories(trajectories):
    n_states = len(all_states)
    p = np.zeros(n_states)

    for t in trajectories:  # for each trajectory
        p[t[0][0]] += 1.0  # increment starting state

    return p / len(trajectories)  # normalize


def p_transition(s_idx, sp_idx, a):
    p, s_to = transition(all_states[s_idx], a)
    if s_to == all_states[sp_idx]:
        return p
    else:
        return 0.0


def reward(s_idx, omega):
    f = feature_vector(all_states[s_idx])
    return f.dot(omega)


def back_transition(s_to, a):
    # preconditions
    if s_to[a] > 0:
        if a == 0 and s_to[4] < 1:
            p = 1.0
        elif a == 1 and s_to[5] < 1:
            p = 1.0
        elif a in [4, 5] and s_to[a] > s_to[a+2]:
            p = 1.0
        elif a in [8, 9] and s_to[a+1] < 1:
            p = 1.0
        elif a in [2, 3, 6, 7, 10]:
            p = 1.0
        else:
            p = 0.0
    else:
        p = 0.0

    # transition to next state
    if p == 1.0:
        s_from = deepcopy(s_to)
        s_from[a] -= 1
        return p, s_from
    else:
        return p, None


def compute_expected_svf(n_actions, p_initial, terminal, omega, eps=1e-5):
    n_states = len(all_states)
    nonterminal = set(range(n_states)) - set(terminal)  # nonterminal states

    # Backward Pass
    # 1. initialize at terminal states
    zs = np.zeros(n_states)  # zs: state partition function
    zs[terminal] = 1.0

    # 2. perform backward pass
    max_iters = 100
    for i in range(max_iters):
        za = np.zeros((n_states, n_actions))  # za: action partition function
        for s_idx in range(n_states):
            for a in range(n_actions):
                prob, sp = transition(all_states[s_idx], a)
                if prob:
                    sp_idx = all_states.index(sp)
                    za[s_idx, a] += np.exp(reward(s_idx, omega)) * zs[sp_idx]

        zs = za.sum(axis=1)

    # for _ in range(2 * n_states):  # longest trajectory: n_states
    #     # reset action values to zero
    #     za = np.zeros((n_states, n_actions))  # za: action partition function
    #
    #     # for each state-action pair
    #     for s_from, a in product(range(n_states), range(n_actions)):
    #
    #         # sum over s_to
    #         for s_to in range(n_states):
    #             za[s_from, a] += p_transition(s_from, s_to, a) * np.exp(reward(s_to, omega)) * zs[s_to]
    #
    #     # sum over all actions
    #     zs = za.sum(axis=1)

    # 3. compute local action probabilities
    p_action = za / zs[:, None]

    # Forward Pass
    # 4. initialize with starting probability
    max_iters = 100
    d = np.zeros((n_states, max_iters))  # d: state-visitation frequencies
    d[:, 0] = p_initial

    # 5. iterate for N steps
    for t in range(1, max_iters):  # longest trajectory: n_states
        for sp_idx in range(n_states):
            for a in range(n_actions):
                prob, s = back_transition(all_states[sp_idx], a)
                if prob:
                    s_idx = all_states.index(s)
                    d[sp_idx, a] += d[s_idx, t - 1] * p_action[s_idx, a]

        # # for all states
        # for s_to in range(n_states):
        #
        #     # sum over nonterminal state-action pairs
        #     for s_from, a in product(nonterminal, range(n_actions)):
        #         d[s_to, t] += d[s_from, t - 1] * p_action[s_from, a] * p_transition(s_from, s_to, a)

    # 6. sum-up frequencies
    return d.sum(axis=1)


def maxent_irl(features, terminal, trajectories, optim, init, eps=1e-4):
    # number of actions and features
    n_actions, n_features = features.shape

    # compute feature expectation from trajectories
    e_features = feature_expectation_from_trajectories(features, trajectories)

    # compute starting-state probabilities from trajectories
    p_initial = initial_probabilities_from_trajectories(trajectories)

    # gradient descent optimization
    omega = init(n_features)  # initialize our parameters
    delta = np.inf  # initialize delta for convergence check

    state_features = np.array([feature_vector(s) for s in all_states])
    optim.reset(omega)  # re-start optimizer
    while delta > eps:  # iterate until convergence
        omega_old = omega.copy()

        # compute per-state reward from features
        # reward = features.dot(omega)

        # compute gradient of the log-likelihood
        e_svf = compute_expected_svf(n_actions, p_initial, terminal, omega)
        grad = e_features - state_features.T.dot(e_svf)

        # perform optimization step and compute delta for convergence
        optim.step(grad)

        # re-compute detla for convergence check
        delta = np.max(np.abs(omega_old - omega))

    # re-compute per-state reward and return
    return state_features.dot(omega), omega


# ------------------------------------------------------------------------------------------------------------------- #

# list all states in the complex task
prev_states = [s_start]
all_states += prev_states
while prev_states:
    next_states = []
    for state in prev_states:
        for action in actions:
            p, next_state = transition(state, action)
            if next_state and (next_state not in next_states) and (next_state not in all_states):
                next_states.append(next_state)

    prev_states = next_states
    all_states += prev_states

# index of the terminal state
terminal = [len(all_states) - 1]

# demonstrations
trajectories = get_trajectories(all_states, demos)

# choose our parameter initialization strategy:
# initialize parameters with constant
init = O.Constant(1.0)

# choose our optimization strategy:
#   we select exponentiated stochastic gradient descent with linear learning-rate decay
optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

# actually do some inverse reinforcement learning
reward_maxent, weights = maxent_irl(np.array(features), terminal, trajectories, optim, init)
print("Weights have been learned for the complex task! Hopefully ...")

# ------------------------------------------------ Alternate Task --------------------------------------------------- #
# actions = [0,  # screw at location A
#            1,  # screw at location A
#            2,  # weld at location A
#            3,  # hammer at location A
#            4]  # hammer at location A
#
# features = [[0, 0, 0],
#             [1, 0, 0],
#             [0, 1, 0],
#             [0, 0, 1],
#             [2, 0, 0],
#             [1, 1, 0],
#             [1, 0, 1],
#             [0, 1, 1],
#             [0, 0, 2],
#             [2, 1, 0],
#             [2, 0, 1],
#             [1, 1, 1],
#             [1, 0, 2],
#             [0, 1, 2],
#             [2, 1, 1],
#             [2, 0, 2],
#             [1, 1, 2],
#             [2, 1, 2]]
# features = np.array(features)
#
# rewards = features.dot(weights)
# print("Actual Task Done")

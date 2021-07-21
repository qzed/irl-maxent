from copy import deepcopy
import numpy as np
from itertools import product  # Cartesian product for iterators
import optimizer as O  # stochastic gradient descent optimizer

# ----------------------------------------------- Canonical Task ----------------------------------------------------- #
# actions that can be taken in the complex task
actions = [0,  # insert long bolt
           1,  # insert short bolt
           2,  # insert wire
           3,  # screw long bolt
           4,  # screw short bolt
           5]  # screw wire

# feature values for each action = [physical_effort, mental_effort]
features = [[2.0, 2.0],  # insert long bolt
            [2.0, 2.0],  # insert short bolt
            [5.0, 5.0],  # insert wire
            [6.0, 1.0],  # screw long bolt
            [4.0, 1.0],  # screw short bolt
            [1.0, 1.0]]  # screw wire

features = np.array(features)
# features = (features - np.min(features))/(np.max(features) - np.min(features))

# start state of the assembly task (none of the actions have been performed)
s_start = [0, 0, 0, 0, 0, 0]

# terminal state of the assembly (each action has been performed)
s_end = [1, 1, 1, 1, 1, 1]

# list of all states
all_states = []

# user demonstration (action sequence)
demos = [[2, 5, 1, 0, 3, 4]]


# -------------------------------------------------- Functions ----------------------------------------------------- #


def transition(s_from, a):
    # preconditions
    if s_from[a] < 1:
        if a in [0, 1, 2]:
            prob = 1.0
        elif a in [3, 4, 5] and s_from[a-3] == 1:
            prob = 1.0
        else:
            prob = 0.0
    else:
        prob = 0.0

    # transition to next state
    if prob == 1.0:
        s_to = deepcopy(s_from)
        s_to[a] += 1
        return prob, s_to
    else:
        return prob, None


def feature_vector(state):
    n_actions, n_features = np.array(features).shape
    feature_value = np.zeros(n_features)
    for action, executed in enumerate(state):
        feature_value += executed * np.array(features[action])

    return feature_value


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


def feature_expectation_from_trajectories(s_features, trajectories):
    n_states, n_features = s_features.shape

    fe = np.zeros(n_features)
    for t in trajectories:  # for each trajectory
        for s_idx, a, sp_idx in t:  # for each state in trajectory
            fe += s_features[sp_idx]  # sum-up features

    return fe / len(trajectories)  # average over trajectories


def initial_probabilities_from_trajectories(n_states, trajectories):
    prob = np.zeros(n_states)

    for t in trajectories:  # for each trajectory
        prob[t[0][0]] += 1.0  # increment starting state

    return prob / len(trajectories)  # normalize


def p_transition(s_idx, sp_idx, a):
    p, s_to = transition(all_states[s_idx], a)
    if s_to == all_states[sp_idx]:
        return p
    else:
        return 0.0


# def reward(s_idx, omega):
#     f = feature_vector(all_states[s_idx])
#     return f.dot(omega)


def back_transition(s_to, a):
    # preconditions
    if s_to[a] > 0:
        if a in [0, 1, 2] and s_to[a+3] < 1:
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


def compute_expected_svf(n_states, n_actions, p_initial, terminal, reward, eps=1e-5):
    nonterminal = set(range(n_states)) - set(terminal)  # nonterminal states

    # Backward Pass
    # 1. initialize at terminal states
    zs = np.zeros(n_states)  # zs: state partition function

    # 2. perform backward pass
    max_iters = 44
    for i in range(max_iters):
        zs[terminal] = 1.0
        za = np.zeros((n_states, n_actions))  # za: action partition function
        for s_idx in range(n_states):
            for a in range(n_actions):
                prob, sp = transition(all_states[s_idx], a)
                if prob:
                    sp_idx = all_states.index(sp)
                    if zs[sp_idx] > 0.0:
                        za[s_idx, a] += np.exp(reward[s_idx]) * zs[sp_idx]

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
    # p_action = np.nan_to_num(p_action)

    # Forward Pass
    # 4. initialize with starting probability
    d = np.zeros((n_states, max_iters))  # d: state-visitation frequencies
    d[:, 0] = p_initial

    # 5. iterate for N steps
    for t in range(1, max_iters):  # longest trajectory: n_states
        d[0, t-1] = 1.0
        for sp_idx in range(n_states):
            for a in range(n_actions):
                prob, s = back_transition(all_states[sp_idx], a)
                if prob:
                    s_idx = all_states.index(s)
                    d[sp_idx, t] += d[s_idx, t - 1] * p_action[s_idx, a]

        # # for all states
        # for s_to in range(n_states):
        #
        #     # sum over nonterminal state-action pairs
        #     for s_from, a in product(nonterminal, range(n_actions)):
        #         d[s_to, t] += d[s_from, t - 1] * p_action[s_from, a] * p_transition(s_from, s_to, a)

    # 6. sum-up frequencies
    return d.sum(axis=1)


def maxent_irl(n_actions, s_features, terminal, trajectories, optim, init, eps=1e-4):

    # number of actions and features
    n_states, n_features = s_features.shape

    # compute feature expectation from trajectories
    e_features = feature_expectation_from_trajectories(s_features, trajectories)

    # compute starting-state probabilities from trajectories
    p_initial = initial_probabilities_from_trajectories(n_states, trajectories)

    # gradient descent optimization
    omega = init(n_features)  # initialize our parameters
    delta = np.inf  # initialize delta for convergence check

    optim.reset(omega)  # re-start optimizer
    while delta > eps:  # iterate until convergence
        omega_old = omega.copy()

        # compute per-state reward from features
        reward = s_features.dot(omega)

        # compute gradient of the log-likelihood
        e_svf = compute_expected_svf(n_states, n_actions, p_initial, terminal, reward)
        grad = e_features - s_features.T.dot(e_svf)

        # perform optimization step and compute delta for convergence
        optim.step(grad)

        # re-compute detla for convergence check
        delta = np.max(np.abs(omega_old - omega))

    # re-compute per-state reward and return
    return s_features.dot(omega), omega


# ----------------------------------------- Training: Learn weights ------------------------------------------------- #

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
terminal_idx = [len(all_states) - 1]

# features for each state
state_features = np.array([feature_vector(state) for state in all_states])

# demonstrations
trajectories = get_trajectories(all_states, demos)

# choose our parameter initialization strategy:
# initialize parameters with constant
init = O.Constant(1.0)

# choose our optimization strategy:
# we select exponentiated stochastic gradient descent with linear learning-rate decay
optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

# actually do some inverse reinforcement learning
reward_maxent, weights = maxent_irl(len(actions), state_features, terminal_idx, trajectories, optim, init)
print("Weights have been learned for the canonical task! Hopefully ...")


# ------------------------------------------ Testing: Reproduce demo ------------------------------------------------ #

# value iteration
vf = {s: 0 for s in range(len(all_states))}  # values
op_actions = {s: 0 for s in range(len(all_states))}  # optimal actions

qf = {s: {a: 0 for a in actions} for s in range(len(all_states))}

delta = 1e-4
for i in range(44):
    vf_temp = {s: 0 for s in range(len(all_states))}

    for j_state in vf:
        max_action = -1
        max_action_val = -np.inf
        for k_action in actions:
            # Check if terminal state
            if j_state in terminal_idx:  # keep the value function of the target 0
                vf_temp[j_state] = 1
                qf[j_state][k_action] = 1
                continue

            prob_ns, ns = transition(all_states[j_state], k_action)
            qf[j_state][k_action] = reward_maxent[j_state]
            if ns:
                int_ns = all_states.index(ns)
                qf[j_state][k_action] += prob_ns * vf[int(int_ns)]

            # Select max value v = max_a q(s, a)
            if max_action_val < qf[j_state][k_action]:
                max_action = k_action
                max_action_val = qf[j_state][k_action]

        # Update the value of the state
        vf_temp[j_state] = max_action_val

        # Simultaneously store the best action for the state
        op_actions[j_state] = max_action

    # After iterating over all states check if values have converged
    np_v = []
    np_v_temp = []
    for s in vf:
        np_v.append(vf[s])
        np_v_temp.append(vf_temp[s])
    np_v = np.array(np_v)
    np_v_temp = np.array(np_v_temp)
    change = np.linalg.norm((np_v - np_v_temp))
    vf = vf_temp
    if change < delta:
        print("VI converged after %d iterations" % (i))
        break

if change >= delta:
    print("VI did not converge after %d iterations (delta=%.2f)" % (i, change))

s, available_actions = 0, demos[0].copy()
generated_sequence = []
while len(available_actions) > 0:
    max_action_val = -np.inf
    candidates = []
    for a in available_actions:
        p, sp = transition(all_states[s], a)
        if sp:
            if qf[s][a] > max_action_val:
                candidates = [a]
                max_action_val = qf[s][a]
            elif qf[s][a] == max_action_val:
                candidates.append(a)
                max_action_val = qf[s][a]

    take_action = np.random.choice(candidates)
    generated_sequence.append(take_action)
    p, sp = transition(all_states[s], take_action)
    s = all_states.index(sp)
    available_actions.remove(take_action)

print(demos)
print(generated_sequence)

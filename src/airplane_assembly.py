from copy import deepcopy
import numpy as np
from itertools import product  # Cartesian product for iterators
import optimizer as O  # stochastic gradient descent optimizer
from vi import value_iteration

# ------------------------------------------------- Optimization ---------------------------------------------------- #

# choose our parameter initialization strategy:
# initialize parameters with constant
init = O.Constant(1.0)

# choose our optimization strategy:
# we select exponentiated stochastic gradient descent with linear learning-rate decay
optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))


# ------------------------------------------------- Complex Task ---------------------------------------------------- #


class ComplexTask:
    """
    Complex task parameters.
    """

    def __init__(self):
        # actions that can be taken in the complex task
        self.actions = [0,  # insert main wing
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
        self.min_value, self.max_value = 1.0, 7.0  # rating are on 1-7 Likert scale
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

        self.features = (np.array(features) - self.min_value) / (self.max_value - self.min_value)

        # start state of the assembly task (none of the actions have been performed)
        self.s_start = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # terminal state of the assembly (each action has been performed)
        self.s_end = [1, 1, 1, 1, 4, 2, 4, 2, 4, 1, 1]

    def transition(self, s_from, a):
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

    def back_transition(self, s_to, a):
        # preconditions
        if s_to[a] > 0:
            if a == 0 and s_to[4] < 1:
                p = 1.0
            elif a == 1 and s_to[5] < 1:
                p = 1.0
            elif a in [4, 5] and s_to[a] > s_to[a + 2]:
                p = 1.0
            elif a in [8, 9] and s_to[a + 1] < 1:
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


# ----------------------------------------------- Canonical Task ----------------------------------------------------- #

class CanonicalTask:
    """
    Canonical task parameters.
    """

    def __init__(self):
        # actions that can be taken in the complex task
        self.actions = [0,  # insert long bolt
                        1,  # insert short bolt
                        2,  # insert wire
                        3,  # screw long bolt
                        4,  # screw short bolt
                        5]  # screw wire

        # feature values for each action = [physical_effort, mental_effort]
        self.min_value, self.max_value = 1.0, 7.0  # rating are on 1-7 Likert scale
        features = [[1.1, 1.1],  # insert long bolt
                    [1.1, 1.1],  # insert short bolt
                    [5.0, 5.0],  # insert wire
                    [6.9, 5.0],  # screw long bolt
                    [2.0, 2.0],  # screw short bolt
                    [2.0, 1.1]]  # screw wire

        self.features = (np.array(features) - self.min_value) / (self.max_value - self.min_value)

        # start state of the assembly task (none of the actions have been performed)
        self.s_start = [0, 0, 0, 0, 0, 0]

        # terminal state of the assembly (each action has been performed)
        self.s_end = [1, 1, 1, 1, 1, 1]

    def transition(self, s_from, a):
        # preconditions
        if s_from[a] < 1:
            if a in [0, 1, 2]:
                prob = 1.0
            elif a in [3, 4, 5] and s_from[a - 3] == 1:
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

    def back_transition(self, s_to, a):
        # preconditions
        if s_to[a] > 0:
            if a in [0, 1, 2] and s_to[a + 3] < 1:
                p = 1.0
            elif a in [3, 4, 5]:
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


# -------------------------------------------------- Functions ----------------------------------------------------- #


def enumerate_states(start_state, list_of_actions, transition_function):
    states = [start_state]
    prev_states = states.copy()
    while prev_states:
        next_states = []
        for state in prev_states:
            for action in list_of_actions:
                p, next_state = transition_function(state, action)
                if next_state and (next_state not in next_states) and (next_state not in states):
                    next_states.append(next_state)

        prev_states = next_states
        states += prev_states

    return states


def feature_vector(state, action_features):
    n_actions, n_features = np.array(action_features).shape
    feature_value = np.zeros(n_features)
    for action, executed in enumerate(state):
        feature_value += executed * np.array(action_features[action])

    return feature_value


def get_trajectories(states, demonstrations, transition_function):
    trajectories = []
    for demo in demonstrations:
        s = states[0]
        trajectory = []
        for action in demo:
            p, sp = transition_function(s, action)
            s_idx, sp_idx = states.index(s), states.index(sp)
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


def initial_probabilities_from_trajectories(states, trajectories):
    n_states = len(states)
    prob = np.zeros(n_states)

    for t in trajectories:  # for each trajectory
        prob[t[0][0]] += 1.0  # increment starting state

    return prob / len(trajectories)  # normalize


def compute_expected_svf(states, terminal, actions, transition_function, back_transition_function,
                         p_initial, reward, max_iters, eps=1e-5):
    n_states, n_actions = len(states), len(actions)

    # Backward Pass
    # 1. initialize at terminal states
    zs = np.zeros(n_states)  # zs: state partition function
    zs[terminal] = 1.0

    # 2. perform backward pass
    for i in range(max_iters):
        za = np.zeros((n_states, n_actions))  # za: action partition function
        for s_idx in range(n_states):
            for a in actions:
                prob, sp = transition_function(states[s_idx], a)
                if sp:
                    sp_idx = states.index(sp)
                    if zs[sp_idx] > 0.0:
                        za[s_idx, a] += np.exp(reward[s_idx]) * zs[sp_idx]

        zs = za.sum(axis=1)
        zs[terminal] = 1.0

    # 3. compute local action probabilities
    p_action = za / zs[:, None]

    # Forward Pass
    # 4. initialize with starting probability
    d = np.zeros((n_states, max_iters))  # d: state-visitation frequencies
    d[:, 0] = p_initial

    # 5. iterate for N steps
    for t in range(1, max_iters):  # longest trajectory: n_states
        for sp_idx in range(n_states):
            for a in actions:
                prob, s = back_transition_function(states[sp_idx], a)
                if s:
                    s_idx = states.index(s)
                    d[sp_idx, t] += d[s_idx, t - 1] * p_action[s_idx, a]

    # 6. sum-up frequencies
    return d.sum(axis=1)


def maxent_irl(states, actions, transition_function, back_transition_function,
               s_features, terminal, trajectories, optim, init, eps=1e-3):
    # number of actions and features
    n_states, n_features = s_features.shape

    # length of each demonstration
    _, demo_length, _ = np.shape(trajectories)

    # compute feature expectation from trajectories
    e_features = feature_expectation_from_trajectories(s_features, trajectories)

    # compute starting-state probabilities from trajectories
    p_initial = initial_probabilities_from_trajectories(states, trajectories)

    # gradient descent optimization
    omega = init(n_features)  # initialize our parameters
    delta = np.inf  # initialize delta for convergence check

    optim.reset(omega)  # re-start optimizer
    while delta > eps:  # iterate until convergence
        omega_old = omega.copy()

        # compute per-state reward from features
        reward = s_features.dot(omega)

        # compute gradient of the log-likelihood
        e_svf = compute_expected_svf(states, terminal, actions, transition_function, back_transition_function,
                                     p_initial, reward, demo_length)
        grad = e_features - s_features.T.dot(e_svf)

        # perform optimization step and compute delta for convergence
        optim.step(grad)

        # re-compute detla for convergence check
        delta = np.max(np.abs(omega_old - omega))
        # print(delta)

    # re-compute per-state reward and return
    return s_features.dot(omega), omega


def rollout_trajectory(qf, states, demos, transition_function):

    s, available_actions = 0, demos[0].copy()
    generated_sequence = []
    while len(available_actions) > 0:
        max_action_val = -np.inf
        candidates = []
        for a in available_actions:
            p, sp = transition_function(states[s], a)
            if sp:
                if qf[s][a] > max_action_val:
                    candidates = [a]
                    max_action_val = qf[s][a]
                elif qf[s][a] == max_action_val:
                    candidates.append(a)
                    max_action_val = qf[s][a]

        if not candidates:
            print(s)
        take_action = np.random.choice(candidates)
        generated_sequence.append(take_action)
        p, sp = transition_function(states[s], take_action)
        s = states.index(sp)
        available_actions.remove(take_action)

    return generated_sequence


# ----------------------------------------- Training: Learn weights ------------------------------------------------- #

# initialize canonical task
canonical_task = CanonicalTask()
s_start = canonical_task.s_start
actions = canonical_task.actions

# list all states
canonical_states = enumerate_states(s_start, actions, canonical_task.transition)

# index of the terminal state
terminal_idx = [len(canonical_states) - 1]

# features for each state
state_features = np.array(canonical_states)
abstract_features = np.array([feature_vector(state, canonical_task.features) for state in canonical_states])


# demonstrations
canonical_demo = [[1, 0, 4, 3, 2, 5]]
demo_trajectories = get_trajectories(canonical_states, canonical_demo, canonical_task.transition)

print("Training ...")

# using true features
canonical_rewards_true, canonical_weights_true = maxent_irl(canonical_states,
                                                            actions,
                                                            canonical_task.transition,
                                                            canonical_task.back_transition,
                                                            state_features,
                                                            terminal_idx,
                                                            demo_trajectories,
                                                            optim, init)

# using abstract features
canonical_rewards_abstract, canonical_weights_abstract = maxent_irl(canonical_states,
                                                                    actions,
                                                                    canonical_task.transition,
                                                                    canonical_task.back_transition,
                                                                    abstract_features,
                                                                    terminal_idx,
                                                                    demo_trajectories,
                                                                    optim, init)

print("Weights have been learned for the canonical task! Hopefully.")

# ----------------------------------------- Verifying: Reproduce demo ----------------------------------------------- #

qf_true, _, _ = value_iteration(canonical_states, actions, canonical_task.transition,
                                          canonical_rewards_true, terminal_idx)
generated_sequence_true = rollout_trajectory(qf_true, canonical_states, canonical_demo, canonical_task.transition)

qf_abstract, _, _ = value_iteration(canonical_states, actions, canonical_task.transition,
                                          canonical_rewards_abstract, terminal_idx)
generated_sequence_abstract = rollout_trajectory(qf_abstract, canonical_states, canonical_demo, canonical_task.transition)

print("Canonical task:")
print("       demonstration -", canonical_demo)
print("    generated (true) -", generated_sequence_true)
print("generated (abstract) -", generated_sequence_abstract)
print("\n")

# ------------------------------------------ Testing: Predict complex ----------------------------------------------- #

# initialize complex task
complex_task = ComplexTask()
s_start = complex_task.s_start
actions = complex_task.actions

# list all states
complex_states = enumerate_states(s_start, actions, complex_task.transition)

# index of the terminal state
terminal_idx = [len(complex_states) - 1]

# features for each state
state_features = np.array([feature_vector(state, complex_task.features) for state in complex_states])

# demonstrations
complex_demo = [[8, 8, 8, 8, 0, 1, 2, 3, 5, 5, 4, 4, 4, 4, 6, 6, 6, 6, 7, 7, 9, 10]]
demo_trajectories = get_trajectories(complex_states, complex_demo, complex_task.transition)

# transfer rewards to complex task
transfer_rewards_abstract = state_features.dot(canonical_weights_abstract)

# rollout trajectory
qf_abstract, _, _ = value_iteration(complex_states, actions, complex_task.transition,
                                    transfer_rewards_abstract, terminal_idx)
predicted_sequence_abstract = rollout_trajectory(qf_abstract, complex_demo, complex_task.transition)

print("Complex task:")
print("       demonstration -", complex_demo)
print("predicted (abstract) -", predicted_sequence_abstract)
print("\n")

# ----------------------------------------- Verifying: Reproduce demo ----------------------------------------------- #

# print("Training ...")
# # inverse reinforcement learning
# complex_rewards, weights = maxent_irl(complex_states,
#                                       actions,
#                                       complex_task.transition,
#                                       complex_task.back_transition,
#                                       state_features,
#                                       terminal_idx,
#                                       demo_trajectories,
#                                       optim, init)
#
# print("Weights have been learned for the complex task! Hopefully.")

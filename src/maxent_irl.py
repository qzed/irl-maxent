import numpy as np


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


def feature_vector(state, terminal_state, action_features):
    max_phase = sum(terminal_state[:-1])
    phase = sum(state[:-1]) / max_phase
    e_p, e_m = action_features[state[-1]]
    feature_value = np.array([phase * e_p, phase * e_m, (1 - phase) * e_p, (1 - phase) * e_m])

    # n_actions, n_features = np.array(action_features).shape
    # feature_value = np.zeros(n_features)
    # for action, executed in enumerate(state):
    #     feature_value += executed * np.array(action_features[action])

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


def compute_expected_svf(states, terminal, actions, transition_function, prev_states_function,
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
            parents = prev_states_function(states[sp_idx])
            if parents:
                for s in parents:
                    s_idx = states.index(s)
                    a = states[sp_idx][-1]
                    d[sp_idx, t] += d[s_idx, t - 1] * p_action[s_idx, a]

    # 6. sum-up frequencies
    return d.sum(axis=1)


def maxent_irl(states, actions, transition_function, prev_states_function,
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
        e_svf = compute_expected_svf(states, terminal, actions, transition_function, prev_states_function,
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


def predict_trajectory(qf, states, demos, transition_function):

    demo = demos[0]
    s, available_actions = 0, demo.copy()

    generated_sequence, score = [], []
    for take_action in demo:
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

        predict_action = np.random.choice(candidates)
        score.append(predict_action == take_action)

        generated_sequence.append(take_action)
        p, sp = transition_function(states[s], take_action)
        s = states.index(sp)
        available_actions.remove(take_action)

    return generated_sequence, score


def random_trajectory(states, demos, transition_function):
    """
    random predicted trajectory
    """

    demo = demos[0]
    s, available_actions = 0, demo.copy()

    generated_sequence, score = [], []
    for take_action in demo:
        candidates = []
        for a in available_actions:
            p, sp = transition_function(states[s], a)
            if sp:
                candidates.append(a)

        if not candidates:
            print(s)

        predict_action = np.random.choice(candidates)
        score.append(predict_action == take_action)

        generated_sequence.append(take_action)
        p, sp = transition_function(states[s], take_action)
        s = states.index(sp)
        available_actions.remove(take_action)

    return generated_sequence, score

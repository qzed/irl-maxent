import numpy as np
from copy import deepcopy


class AssemblyTask:

    def __init__(self, features):

        self.num_actions, self.num_features = np.shape(features)
        self.actions = np.array(range(self.num_actions))
        self.features = np.array(features)

        self.min_value, self.max_value = 0., 1.0

        # start state of the assembly task (none of the actions have been performed)
        self.s_start = [0] * self.num_actions
        self.s_end = []

        self.states = [self.s_start]
        self.terminal_idx = []

    def scale_features(self):
        self.features = (np.array(self.features) - self.min_value) / (self.max_value - self.min_value)

    def convert_to_rankings(self):
        nominal_features = self.features
        for feature_idx in range(self.num_features):
            feature_values = self.features[:, feature_idx]
            sorted_values = [x for x, _, _ in sorted(zip(self.actions, feature_values, nominal_features)
                                                     , key=lambda y: (y[1], y[2]))]
            feature_ranks = np.array(sorted_values).argsort() + 1
            self.features[:, feature_idx] = feature_ranks

    def set_end_state(self, user_demo):
        self.s_end.append(list(np.bincount(user_demo)))

    def enumerate_states(self):
        prev_states = self.states.copy()
        while prev_states:
            next_states = []
            for state in prev_states:
                for action in self.actions:
                    p, next_state = self.transition(state, action)
                    if next_state and (next_state not in next_states) and (next_state not in self.states):
                        next_states.append(next_state)

            prev_states = next_states
            self.states += prev_states

    def enumerate_trajectories(self, demos):

        n_demos, n_steps = np.shape(demos)
        all_traj = [[(-1, -1, 0)]]
        for t in range(n_steps):
            all_traj_new = []
            for traj in all_traj:
                prev_state = traj[-1][2]
                for action in self.actions:
                    p, next_state = self.transition(self.states[prev_state], action)
                    if next_state:
                        new_traj = deepcopy(traj)
                        new_traj.append((prev_state, action, self.states.index(next_state)))
                        all_traj_new.append(new_traj)

            all_traj = deepcopy(all_traj_new)
        all_traj = np.array(all_traj)

        return all_traj[:, 1:, :]

    def set_terminal_idx(self):
        self.terminal_idx = [self.states.index(s_terminal) for s_terminal in self.s_end]

    def get_features(self, state, new_feature=False):

        feature_values = [a*self.features[i] for i, a in enumerate(state)]
        feature_value = np.sum(feature_values, axis=0)

        return feature_value

    def prev_states(self, s_to):
        previous_states = []

        curr_a = s_to[-2]
        if curr_a >= 0:
            s_from = deepcopy(s_to[:-2])
            s_from[curr_a] -= 1
            prev_a = s_to[-1]

            previous_state = deepcopy(s_from)
            previous_state.append(prev_a)
            if prev_a >= 0:
                hist_s = deepcopy(s_from)
                hist_s[prev_a] -= 1
                hist_actions = [a for a, s in enumerate(hist_s) if s >= 1]
                if hist_actions:
                    for hist_a in hist_actions:
                        prob, s = self.back_transition(hist_s, hist_a)
                        if s:
                            prev_s = deepcopy(previous_state)
                            prev_s.append(hist_a)
                            previous_states.append(prev_s)
                else:
                    hist_a = -1
                    previous_state.append(hist_a)
                    previous_states.append(previous_state)
            else:
                hist_a = -1
                previous_state.append(hist_a)
                previous_states.append(previous_state)

        return previous_states


# ----------------------------------------------- Canonical Task ----------------------------------------------------- #

class CanonicalTask(AssemblyTask):
    """
    Actions:

    feature values for each action = [physical_effort, mental_effort, movement]
    """

    @staticmethod
    def transition(s_from, a):
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

    @staticmethod
    def back_transition(s_to, a):
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


# ------------------------------------------------ Complex Task ----------------------------------------------------- #

class ComplexTask(AssemblyTask):
    """
    Actions:

    """

    @staticmethod
    def transition(s_from, a):
        # preconditions
        if s_from[a] < 1:
            if a in [0, 1, 2, 3, 4]:
                prob = 1.0
            elif a in [5, 6, 7, 8, 9] and s_from[a - 5] == 1:
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

    @staticmethod
    def back_transition(s_to, a):
        # preconditions
        if s_to[a] > 0:
            if a in [0, 1, 2, 3, 4] and s_to[a + 5] < 1:
                p = 1.0
            elif a in [5, 6, 7, 8, 9]:
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


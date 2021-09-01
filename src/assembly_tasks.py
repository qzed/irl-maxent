import numpy as np
from copy import deepcopy


class AssemblyTask:

    def __init__(self, features):

        self.num_actions, self.num_features = np.shape(features)
        self.actions = np.array(range(self.num_actions))
        self.features = np.array(features)

        self.min_value, self.max_value = 1.0, 7.0  # rating are on 1-7 Likert scale

        # start state of the assembly task (none of the actions have been performed)
        self.s_start = [0] * self.num_actions + [-1, -1]
        self.s_end = []

        self.states = [self.s_start]
        self.terminal_idx = []

    def scale_features(self):
        self.features = (np.array(self.features) - self.min_value) / (self.max_value - self.min_value)

    def convert_to_rankings(self):
        for feature_idx in range(self.num_features):
            feature_values = self.features[:, feature_idx]
            sorted_values = [x for x, _, _ in sorted(zip(self.actions, feature_values, self.nominal_features)
                                                     , key=lambda y: (y[1], y[2]))]
            feature_ranks = np.array(sorted_values).argsort()
            self.features[:, feature_idx] = feature_ranks

    def set_end_state(self, user_demo):
        terminal_state = list(np.bincount(user_demo))

        for curr_a in self.actions:
            _, prev_s = self.back_transition(terminal_state, curr_a)
            if prev_s:
                rem_actions = list(user_demo)
                rem_actions.remove(curr_a)
                for prev_a in set(rem_actions):
                    _, s = self.back_transition(prev_s, prev_a)
                    if s:
                        self.s_end.append(terminal_state + [curr_a, prev_a])

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

    def set_terminal_idx(self):
        self.terminal_idx = [self.states.index(s_terminal) for s_terminal in self.s_end]

    def get_features(self, state):

        # calculate current phase
        terminal_state = self.s_end[-1]
        max_phase = sum(terminal_state[:-2])
        phase = sum(state[:-2]) / max_phase

        curr_a, prev_a = state[-2], state[-1]

        if curr_a >= 0:
            e_p, e_m = self.features[curr_a]
        else:
            e_p, e_m = 0.0, 0.0

        if prev_a >= 0:
            c_part = self.part_similarity[prev_a][curr_a]
            c_tool = self.tool_similarity[prev_a][curr_a]
        else:
            c_part, c_tool = 0.0, 0.0

        feature_value = np.array([phase * e_p, phase * e_m, (1.0 - phase) * e_p, (1.0 - phase) * e_m, c_part, c_tool])

        # n_actions, n_features = np.array(action_features).shape
        # feature_value = np.zeros(n_features)
        # for action, executed in enumerate(state):
        #     feature_value += executed * np.array(action_features[action])

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
    0 - insert long bolt
    1 - insert short bolt
    2 - insert wire (short)
    3 - screw long bolt
    4 - screw short bolt
    5 - screw wire / insert wire (long)

    feature values for each action = [physical_effort, mental_effort]
    """

    nominal_features = [[1.2, 1.1],  # insert long bolt
                        [1.1, 1.1],  # insert short bolt
                        [4.0, 6.0],  # insert wire (short)
                        [6.0, 2.0],  # screw long bolt
                        [2.0, 2.0],  # screw short bolt
                        [5.0, 6.9]]  # insert wire (long)

    part_similarity = [[1, 0, 0, 1, 0, 1],
                       [0, 1, 1, 0, 1, 0],
                       [0, 1, 1, 0, 1, 0],
                       [1, 0, 0, 1, 0, 1],
                       [0, 1, 1, 0, 1, 0],
                       [1, 0, 0, 1, 0, 1]]

    tool_similarity = [[1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, 1, 0],
                       [0, 0, 0, 1, 1, 0],
                       [0, 0, 0, 0, 0, 1]]

    @staticmethod
    def transition(s_from, a):
        # preconditions
        if s_from[a] < 1:
            if a in [0, 1, 2, 5]:
                prob = 1.0
            elif a in [3, 4] and s_from[a - 3] == 1:
                prob = 1.0
            else:
                prob = 0.0
        else:
            prob = 0.0

        # transition to next state
        if prob == 1.0:
            s_to = deepcopy(s_from)
            s_to[a] += 1
            s_to[-1] = s_from[-2]
            s_to[-2] = a
            return prob, s_to
        else:
            return prob, None

    @staticmethod
    def back_transition(s_to, a):
        # preconditions
        if s_to[a] > 0:
            if a in [0, 1] and s_to[a + 3] < 1:
                p = 1.0
            elif a in [2, 3, 4, 5]:
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
    0 - insert main wing
    1 - insert tail wing
    2 - insert long bolt into main wing
    3 - insert long bolt into tail wing
    4 - screw long bolt into main wing
    5 - screw long bolt into tail wing
    6 - screw propeller
    7 - screw propeller base

    """

    nominal_features = [[3.5, 3.5],  # insert main wing
                        [2.0, 3.0],  # insert tail wing
                        [1.2, 1.1],  # insert long bolt into main wing
                        [1.1, 1.1],  # insert long bolt into tail wing
                        [2.1, 2.1],  # screw long bolt into main wing
                        [2.0, 2.0],  # screw long bolt into tail wing
                        [3.5, 6.0],  # screw propeller
                        [2.0, 3.5]]  # screw propeller base

    part_similarity = [[1, 0, 1, 0, 1, 0, 0, 0],
                       [0, 1, 0, 1, 0, 1, 0, 0],
                       [1, 0, 1, 0, 1, 0, 0, 0],
                       [0, 1, 0, 1, 0, 1, 0, 0],
                       [1, 0, 1, 0, 1, 0, 0, 0],
                       [0, 1, 0, 1, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 0, 0, 1, 1]]

    tool_similarity = [[1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1, 0],
                       [0, 0, 0, 0, 1, 1, 1, 0],
                       [0, 0, 0, 0, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1]]

    @staticmethod
    def transition(s_from, a):
        # preconditions
        if a in [0, 1] and s_from[a] < 1:
            p = 1.0
        elif a == 2 and s_from[a] < 4 and s_from[0] == 1:
            p = 1.0
        elif a == 3 and s_from[a] < 1 and s_from[1] == 1:
            p = 1.0
        elif a == 4 and s_from[a] < 4 and s_from[a] + 1 <= s_from[a - 2]:
            p = 1.0
        elif a == 5 and s_from[a] < 1 and s_from[a] + 1 <= s_from[a - 2]:
            p = 1.0
        elif a == 6 and s_from[a] < 4:
            p = 1.0
        elif a == 7 and s_from[a] < 1 and s_from[a - 1] == 4:
            p = 1.0
        else:
            p = 0.0

        # transition to next state
        if p == 1.0:
            s_to = deepcopy(s_from)
            s_to[a] += 1
            s_to[-1] = s_from[-2]
            s_to[-2] = a
            return p, s_to
        else:
            return p, None

    @staticmethod
    def back_transition(s_to, a):
        # preconditions
        if s_to[a] > 0:
            if a == 0 and s_to[2] < 1:
                p = 1.0
            elif a == 1 and s_to[3] < 1:
                p = 1.0
            elif a in [2, 3] and s_to[a] > s_to[a + 2]:
                p = 1.0
            elif a in [6] and s_to[a + 1] < 1:
                p = 1.0
            elif a in [4, 5, 7]:
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

    # ----------------------------------------------- Event sequence ------------------------------------------------ #
    # @staticmethod
    # def transition(self, s_from, a):
    #     # preconditions
    #     if a in [0, 1] and s_from[a] < 1:
    #         p = 1.0
    #     elif a == 2 and s_from[a] < 1 and s_from[0] == 1:
    #         p = 1.0
    #     elif a == 3 and s_from[a] < 1 and s_from[1] == 1:
    #         p = 1.0
    #     elif a == 4 and s_from[a] < 1 and s_from[a] + 1 <= s_from[a - 2]:
    #         p = 1.0
    #     elif a == 5 and s_from[a] < 1 and s_from[a] + 1 <= s_from[a - 2]:
    #         p = 1.0
    #     elif a == 6 and s_from[a] < 1:
    #         p = 1.0
    #     elif a == 7 and s_from[a] < 1 and s_from[a - 1] == 1:
    #         p = 1.0
    #     else:
    #         p = 0.0
    #
    #     # transition to next state
    #     if p == 1.0:
    #         s_to = deepcopy(s_from)
    #         s_to[a] += 1
    #         s_to[-1] = a
    #         return p, s_to
    #     else:
    #         return p, None
    #
    # @staticmethod
    # def back_transition(self, s_to, a):
    #     # preconditions
    #     if s_to[a] > 0:
    #         if a == 0 and s_to[2] < 1:
    #             p = 1.0
    #         elif a == 1 and s_to[3] < 1:
    #             p = 1.0
    #         elif a in [2, 3] and s_to[a] > s_to[a + 2]:
    #             p = 1.0
    #         elif a in [6] and s_to[a + 1] < 1:
    #             p = 1.0
    #         elif a in [4, 5, 7]:
    #             p = 1.0
    #         else:
    #             p = 0.0
    #     else:
    #         p = 0.0
    #
    #     # transition to next state
    #     if p == 1.0:
    #         s_from = deepcopy(s_to)
    #         s_from[a] -= 1
    #         return p, s_from
    #     else:
    #         return p, None

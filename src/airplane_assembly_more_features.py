import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from copy import deepcopy
from itertools import product  # Cartesian product for iterators

import optimizer as O  # stochastic gradient descent optimizer
from vi import value_iteration
from maxent_irl import *

root_path = "data/"
canonical_path = root_path + "canonical_demos.csv"
complex_path = root_path + "complex_demos.csv"
feature_path = root_path + "survey_data.csv"

ca_df = pd.read_csv(canonical_path, header=None)
com_df = pd.read_csv(complex_path, header=None)

canonical_data = []
complex_data = []

for i in range(0, ca_df.shape[1]):
    ca_seq = []

    for j in range(0, len(ca_df)):
        ca_seq.append(ca_df.iloc[j][i])

    canonical_data.append(ca_seq.copy())

print('canonical_data:', canonical_data)

for i in range(0, com_df.shape[1]):
    com_seq = []

    for j in range(0, len(com_df)):
        com_seq.append(com_df.iloc[j][i])

    complex_data.append(com_seq.copy())

print('complex_data:', complex_data)


# input features
def process_val(x):
    if '1' in x:
        x = 1.1
    elif '7' in x:
        x = 6.9
    else:
        x = float(x)

    return x


fea_df = pd.read_csv(feature_path)

canonical_features = []
complex_features = []

for i in range(0, len(fea_df)):
    if i == 0 or i == 1:
        continue

    fea_mat = []

    for j in [1, 3, 5, 2, 4, 6]:
        phy_col = "Q7_" + str(j)
        men_col = "Q8_" + str(j)
        fam_col = "Q6_" + str(j)

        phy_val = process_val(fea_df[phy_col][i])
        men_val = process_val(fea_df[men_col][i])
        fam_val = process_val(fea_df[fam_col][i])

        fea_mat.append([phy_val, men_val, fam_val])

    canonical_features.append(fea_mat.copy())


for i in range(0, len(fea_df)):
    if i == 0 or i == 1:
        continue

    fea_mat = []

    for j in [1, 3, 7, 8, 2, 4, 5, 6]:
        phy_col = "Q14_" + str(j)
        men_col = "Q15_" + str(j)
        fam_col = "Q13_" + str(j)

        phy_val = process_val(fea_df[phy_col][i])
        men_val = process_val(fea_df[men_col][i])
        fam_val = process_val(fea_df[fam_col][i])

        fea_mat.append([phy_val, men_val, fam_val])

    complex_features.append(fea_mat.copy())

# ------------------------------------------------- Optimization ---------------------------------------------------- #

# choose our parameter initialization strategy:
# initialize parameters with constant
init = O.Constant(1.0)

# choose our optimization strategy:
# we select exponentiated stochastic gradient descent with linear learning-rate decay
optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))


# ----------------------------------------------- Canonical Task ----------------------------------------------------- #

class CanonicalTask:
    """
    Canonical task parameters.
    """

    def __init__(self, actions, features):
        # actions that can be taken in the complex task

        # self.actions = [0,  # insert long bolt
        #                 1,  # insert short bolt
        #                 2,  # insert wire
        #                 3,  # screw long bolt
        #                 4,  # screw short bolt
        #                 5]  # screw wire

        self.actions = actions

        # feature values for each action = [physical_effort, mental_effort]
        self.min_value, self.max_value = 1.0, 7.0  # rating are on 1-7 Likert scale

        # features = [[1.2, 1.1],  # insert long bolt
        #             [1.1, 1.1],  # insert short bolt
        #             [4.0, 6.0],  # insert wire
        #             [6.0, 2.0],  # screw long bolt
        #             [2.0, 2.0],  # screw short bolt
        #             [1.1, 2.0]]  # screw wire

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


# ----------------------------------------------- Complex Task ----------------------------------------------------- #


class ComplexTask:
    """
    Complex task parameters.
    """

    def __init__(self, actions, features):
        # actions that can be taken in the complex task
        # self.actions = [0,  # insert main wing
        #                 1,  # insert tail wing
        #                 2,  # insert long bolt into main wing
        #                 3,  # insert long bolt into tail wing
        #                 4,  # screw long bolt into main wing
        #                 5,  # screw long bolt into tail wing
        #                 6,  # screw propeller
        #                 7]  # screw propeller base

        self.actions = actions

        # feature values for each action = [physical_effort, mental_effort]
        self.min_value, self.max_value = 1.0, 7.0  # rating are on 1-7 Likert scale

        # features = [[3.5, 3.5],  # insert main wing
        #             [2.0, 3.0],  # insert tail wing
        #             [1.2, 1.1],  # insert long bolt into main wing
        #             [1.1, 1.1],  # insert long bolt into tail wing
        #             [2.1, 2.1],  # screw long bolt into main wing
        #             [2.0, 2.0],  # screw long bolt into tail wing
        #             [3.5, 6.0],  # screw propeller
        #             [2.0, 3.5]]  # screw propeller base

        self.features = (np.array(features) - self.min_value) / (self.max_value - self.min_value)

        # start state of the assembly task (none of the actions have been performed)
        self.s_start = [0, 0, 0, 0, 0, 0, 0, 0]

        # terminal state of the assembly (each action has been performed)
        # self.s_end = [1, 1, 4, 2, 4, 2, 4, 1]
        self.s_end = [1, 1, 1, 1, 1, 1, 1, 1]

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
    #         return p, s_to
    #     else:
    #         return p, None
    #
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

    def transition(self, s_from, a):
        # preconditions
        if a in [0, 1] and s_from[a] < 1:
            p = 1.0
        elif a == 2 and s_from[a] < 4 and s_from[0] == 1:
            p = 1.0
        elif a == 3 and s_from[a] < 2 and s_from[1] == 1:
            p = 1.0
        elif a == 4 and s_from[a] < 4 and s_from[a] + 1 <= s_from[a - 2]:
            p = 1.0
        elif a == 5 and s_from[a] < 2 and s_from[a] + 1 <= s_from[a - 2]:
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
            return p, s_to
        else:
            return p, None

    def back_transition(self, s_to, a):
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


# ------------------------------------------- Training: Learn weights ----------------------------------------------- #

def feature_vector_3f(state, action_features):
    n_actions, n_features = np.array(action_features).shape
    feature_value = np.zeros(n_features)
    for action, executed in enumerate(state):
        feature_value += (executed/10) * np.array(action_features[action])

    return feature_value

match_scores, predict_scores = [], []

# loop over all users
for i in range(0, len(canonical_data)):

    # ---------------------------------------- Training: Learn weights ---------------------------------------------- #

    # initialize canonical task
    canonical_task = CanonicalTask(canonical_data[i], canonical_features[i])
    s_start = canonical_task.s_start
    actions = canonical_task.actions

    # list all states
    canonical_states = enumerate_states(s_start, actions, canonical_task.transition)

    # index of the terminal state
    terminal_idx = [len(canonical_states) - 1]

    # features for each state
    state_features = np.array(canonical_states)
    abstract_features = np.array([feature_vector_3f(state, canonical_task.features) for state in canonical_states])

    # demonstrations
    canonical_demo = [canonical_data[i]]
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

    # --------------------------------------- Verifying: Reproduce demo --------------------------------------------- #

    qf_true, _, _ = value_iteration(canonical_states, actions, canonical_task.transition,
                                    canonical_rewards_true, terminal_idx)
    generated_sequence_true = rollout_trajectory(qf_true, canonical_states, canonical_demo, canonical_task.transition)

    qf_abstract, _, _ = value_iteration(canonical_states, actions, canonical_task.transition,
                                        canonical_rewards_abstract, terminal_idx)
    generated_sequence_abstract = rollout_trajectory(qf_abstract, canonical_states, canonical_demo,
                                                     canonical_task.transition)

    print("\n")
    print("Canonical task:")
    print("       demonstration -", canonical_demo)
    print("    generated (true) -", generated_sequence_true)
    print("generated (abstract) -", generated_sequence_abstract)

    # ----------------------------------------- Testing: Predict complex -------------------------------------------- #

    # initialize complex task
    complex_task = ComplexTask(complex_data[i], complex_features[i])
    s_start = complex_task.s_start
    actions = complex_task.actions

    # list all states
    complex_states = enumerate_states(s_start, actions, complex_task.transition)

    # index of the terminal state
    terminal_idx = [len(complex_states) - 1]

    # features for each state
    state_features = np.array([feature_vector_3f(state, complex_task.features) for state in complex_states])

    # demonstrations
    complex_demo = [complex_data[i]]
    demo_trajectories = get_trajectories(complex_states, complex_demo, complex_task.transition)

    # transfer rewards to complex task
    transfer_rewards_abstract = state_features.dot(canonical_weights_abstract)

    # rollout trajectory
    qf_abstract, _, _ = value_iteration(complex_states, actions, complex_task.transition,
                                        transfer_rewards_abstract, terminal_idx)
    rolled_sequence_abstract = rollout_trajectory(qf_abstract, complex_states, complex_demo, complex_task.transition)

    # complex_rewards_true, complex_weights_true = maxent_irl(complex_states,
    #                                                         actions,
    #                                                         complex_task.transition,
    #                                                         complex_task.back_transition,
    #                                                         state_features,
    #                                                         terminal_idx,
    #                                                         demo_trajectories,
    #                                                         optim, init)
    # qf_true, _, _ = value_iteration(complex_states, actions, complex_task.transition,
    #                                 complex_rewards_true, terminal_idx)
    # predicted_sequence_true = rollout_trajectory(qf_true, complex_states, complex_demo, complex_task.transition)

    print("\n")
    print("Complex task:")
    print("       demonstration -", complex_demo)
    print("rolled (abstract) -", rolled_sequence_abstract)

    match_score = (np.array(complex_demo[0]) == np.array(rolled_sequence_abstract))
    match_scores.append(match_score)

    _, predict_score = predict_trajectory(qf_abstract, complex_states, complex_demo, complex_task.transition)
    predict_scores.append(predict_score)

    # ------------------------------------------------- Results ----------------------------------------------------- #
    # sm_abstract = edit_distance.SequenceMatcher(a=complex_demo[0], b=predicted_sequence_abstract)
    # sm_true = edit_distance.SequenceMatcher(a=complex_demo[0], b=predicted_sequence_true)
    # print(sm_abstract.distance(), sm_true.distance())

match_accuracy = np.sum(match_scores, axis=0)/len(match_scores)
np.savetxt("match_short.csv", match_accuracy)

predict_accuracy = np.sum(predict_scores, axis=0)/len(predict_scores)
np.savetxt("predict_short.csv", predict_accuracy)

sns.set(style="darkgrid", context="paper")
steps = range(1, len(match_accuracy)+1)

plt.figure()
plt.bar(steps, match_accuracy)
plt.ylim(0, 1)
plt.xticks(steps)
plt.savefig("trajectory_matching_fixed.jpg")

plt.figure()
plt.bar(steps, predict_accuracy)
plt.ylim(0, 1)
plt.xticks(steps)
plt.savefig("trajectory_prediction_fixed.jpg")

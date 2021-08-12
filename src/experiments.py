# import python libraries
import numpy as np
from copy import deepcopy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import functions
import optimizer as O  # stochastic gradient descent optimizer
from vi import value_iteration
from maxent_irl import *
from assembly_tasks import *

# -------------------------------------------------- Load data ------------------------------------------------------ #
# paths
root_path = "data/"
canonical_path = root_path + "canonical_demos.csv"
complex_path = root_path + "complex_demos.csv"
feature_path = root_path + "survey_data.csv"

# load user demonstrations
canonical_df = pd.read_csv(canonical_path, header=None)
complex_df = pd.read_csv(complex_path, header=None)
canonical_demos = canonical_df.to_numpy().T
complex_demos = complex_df.to_numpy().T

# load user responses
ratings_df = pd.read_csv(feature_path)


# pre-process feature value
def process_val(x):
    if x == "1 (No effort at all)":
        x = 1.1
    elif x == "7 (A lot of effort)":
        x = 6.9
    else:
        x = float(x)

    return x


# load user ratings
def load_features(data, feature_idx, action_idx):
    user_features = []
    for i in range(2, len(ratings_df)):
        fea_mat = []
        for j in action_idx:
            fea_vec = []
            for k in feature_idx:
                fea_col = k + str(j)
                fea_val = process_val(ratings_df[fea_col][i])
                fea_vec.append(fea_val)
            fea_mat.append(fea_vec)
        user_features.append(fea_mat.copy())
    return user_features


# user ratings for features
canonical_q, complex_q = ["Q7_", "Q8_"], ["Q14_", "Q15_"]  # ["Q6_", "Q7_", "Q8_"], ["Q13_", "Q14_", "Q15_"]
canonical_features = load_features(ratings_df, canonical_q, [1, 3, 5, 2, 4, 6])
complex_features = load_features(ratings_df, complex_q, [1, 3, 7, 8, 2, 4, 5, 6])

# ------------------------------------------------- Optimization ---------------------------------------------------- #

# choose our parameter initialization strategy:
# initialize parameters with constant
init = O.Constant(1.0)

# choose our optimization strategy:
# we select exponentiated stochastic gradient descent with linear learning-rate decay
optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

# ------------------------------------------- Training: Learn weights ----------------------------------------------- #

match_scores, predict_scores, random_scores = [], [], []

# loop over all users
for i in range(len(canonical_demos)):

    print("=======================")
    print("User:", i)

    # ---------------------------------------- Training: Learn weights ---------------------------------------------- #

    # initialize canonical task
    canonical_task = CanonicalTask(canonical_features[i])
    canonical_task.set_end_state(canonical_demos[i])
    canonical_task.scale_features()

    # list all states
    canonical_states = enumerate_states(canonical_task.s_start, canonical_task.actions, canonical_task.transition)

    # index of the terminal state
    canonical_terminal_idx = [canonical_states.index(s_terminal) for s_terminal in canonical_task.s_end]

    # features for each state
    canonical_state_features = np.array(canonical_states)/np.max(canonical_states)
    canonical_action_features = np.vstack((canonical_task.features, [[0., 0.]]))
    canonical_abstract_features = np.array([feature_vector(state, canonical_states[-1], canonical_action_features)
                                            for state in canonical_states])

    # demonstrations
    canonical_user_demo = [list(canonical_demos[i])]
    canonical_trajectories = get_trajectories(canonical_states, canonical_user_demo, canonical_task.transition)

    print("Training ...")

    # using true features
    canonical_rewards_true, canonical_weights_true = maxent_irl(canonical_states,
                                                                canonical_task.actions,
                                                                canonical_task.transition,
                                                                canonical_task.prev_states,
                                                                canonical_state_features,
                                                                canonical_terminal_idx,
                                                                canonical_trajectories,
                                                                optim, init)

    # using abstract features
    canonical_rewards_abstract, canonical_weights_abstract = maxent_irl(canonical_states,
                                                                        canonical_task.actions,
                                                                        canonical_task.transition,
                                                                        canonical_task.prev_states,
                                                                        canonical_abstract_features,
                                                                        canonical_terminal_idx,
                                                                        canonical_trajectories,
                                                                        optim, init)

    print("Weights have been learned for the canonical task! Hopefully.")

    # --------------------------------------- Verifying: Reproduce demo --------------------------------------------- #

    qf_true, _, _ = value_iteration(canonical_states, canonical_task.actions, canonical_task.transition,
                                    canonical_rewards_true, canonical_terminal_idx)
    generated_sequence_true = rollout_trajectory(qf_true, canonical_states, canonical_user_demo,
                                                 canonical_task.transition)

    qf_abstract, _, _ = value_iteration(canonical_states, canonical_task.actions, canonical_task.transition,
                                        canonical_rewards_abstract, canonical_terminal_idx)
    generated_sequence_abstract = rollout_trajectory(qf_abstract, canonical_states, canonical_user_demo,
                                                     canonical_task.transition)

    print("\n")
    print("Canonical task:")
    print("       demonstration -", canonical_user_demo)
    print("    generated (true) -", generated_sequence_true)
    print("generated (abstract) -", generated_sequence_abstract)

    # ----------------------------------------- Testing: Predict complex -------------------------------------------- #

    # initialize complex task
    complex_task = ComplexTask(complex_features[i])
    complex_task.set_end_state(complex_demos[i])
    complex_task.scale_features()

    # list all states
    complex_states = enumerate_states(complex_task.s_start, complex_task.actions, complex_task.transition)

    # index of the terminal state
    complex_terminal_idx = [complex_states.index(s_terminal) for s_terminal in complex_task.s_end]

    # features for each state
    complex_state_features = np.array(complex_states)/np.max(complex_states)
    complex_action_features = np.vstack((complex_task.features, [[0., 0.]]))
    complex_abstract_features = np.array([feature_vector(state, complex_states[-1], complex_action_features)
                                          for state in complex_states])

    # demonstrations
    complex_user_demo = [list(complex_demos[i])]
    complex_trajectories = get_trajectories(complex_states, complex_user_demo, complex_task.transition)

    # transfer rewards to complex task
    transfer_rewards_abstract = complex_abstract_features.dot(canonical_weights_abstract)

    # rollout trajectory
    qf_abstract, _, _ = value_iteration(complex_states, complex_task.actions, complex_task.transition,
                                        transfer_rewards_abstract, complex_terminal_idx)
    rolled_sequence_abstract = rollout_trajectory(qf_abstract, complex_states, complex_user_demo,
                                                  complex_task.transition)

    # using true features
    complex_rewards_true, complex_weights_true = maxent_irl(complex_states,
                                                            complex_task.actions,
                                                            complex_task.transition,
                                                            complex_task.prev_states,
                                                            complex_state_features,
                                                            complex_terminal_idx,
                                                            complex_trajectories,
                                                            optim, init)

    # using abstract features
    complex_rewards_abstract, complex_weights_abstract = maxent_irl(complex_states,
                                                                    complex_task.actions,
                                                                    complex_task.transition,
                                                                    complex_task.prev_states,
                                                                    complex_abstract_features,
                                                                    complex_terminal_idx,
                                                                    complex_trajectories,
                                                                    optim, init)

    print("\n")
    print("Complex task:")
    print("       demonstration -", complex_user_demo)
    print("rolled (abstract) -", rolled_sequence_abstract)

    _, random_score = random_trajectory(complex_states, complex_user_demo, complex_task.transition)
    random_scores.append(random_score)

    match_score = (np.array(complex_user_demo[0]) == np.array(rolled_sequence_abstract))
    match_scores.append(match_score)

    _, predict_score = predict_trajectory(qf_abstract, complex_states, complex_user_demo, complex_task.transition)
    predict_scores.append(predict_score)

# ---------------------------------------------------- Results ------------------------------------------------------ #
random_accuracy = np.sum(random_scores, axis=0)/len(random_scores)
np.savetxt("results/random.csv", random_accuracy)

match_accuracy = np.sum(match_scores, axis=0)/len(match_scores)
np.savetxt("results/match.csv", match_accuracy)

predict_accuracy = np.sum(predict_scores, axis=0)/len(predict_scores)
np.savetxt("results/predict.csv", predict_accuracy)

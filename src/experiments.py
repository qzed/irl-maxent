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
from visualize import *

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
optim = O.ExpSga(lr=O.linear_decay(lr0=0.6))

# ------------------------------------------- Training: Learn weights ----------------------------------------------- #

rank_features = False
scale_weights = False
run_proposed = False
run_random_baseline = False

match_scores, predict_scores, random_scores = [], [], []

# loop over all users
for i in range(len(canonical_demos)):

    print("=======================")
    print("User:", i)

    # ---------------------------------------- Training: Learn weights ---------------------------------------------- #

    # initialize canonical task
    C = CanonicalTask(canonical_features[i])
    C.set_end_state(canonical_demos[i])
    C.enumerate_states()
    C.set_terminal_idx()
    if rank_features:
        C.convert_to_rankings()

    # demonstrations
    canonical_user_demo = [list(canonical_demos[i])]
    visualize_demo(C, canonical_user_demo[0], i)
    canonical_trajectories = get_trajectories(C.states, canonical_user_demo, C.transition)

    if run_proposed:
        print("Training ...")

        # using abstract features
        abstract_features = np.array([C.get_features(state) for state in C.states])
        norm_abstract_features = abstract_features / np.linalg.norm(abstract_features, axis=0)
        canonical_rewards_abstract, canonical_weights_abstract = maxent_irl(C, norm_abstract_features,
                                                                            canonical_trajectories,
                                                                            optim, init)

        print("Weights have been learned for the canonical task! Hopefully.")
        print("Weights -", canonical_weights_abstract)

        # scale weights
        if scale_weights:
            canonical_weights_abstract /= max(canonical_weights_abstract)

    # --------------------------------------- Verifying: Reproduce demo --------------------------------------------- #

    # canonical_rewards_true = norm_abstract_features.dot(np.array([1., 0., 0., 0., 1., 0.]))
    # qf_true, _, _ = value_iteration(C.states, C.actions, C.transition, canonical_rewards_true, C.terminal_idx)
    # generated_sequence_true = rollout_trajectory(qf_true, C.states, canonical_user_demo, C.transition)
    #
    # qf_abstract, _, _ = value_iteration(C.states, C.actions, C.transition, canonical_rewards_abstract, C.terminal_idx)
    # predict_sequence_canonical, _ = predict_trajectory(qf_abstract, C.states, canonical_user_demo, C.transition)
    #
    # print("\n")
    # print("Canonical task:")
    # print("     demonstration -", canonical_user_demo)
    # print("  generated (true) -", generated_sequence_true)
    # print("predict (abstract) -", predict_sequence_canonical)

    # ----------------------------------------- Testing: Predict complex -------------------------------------------- #

    # initialize complex task
    X = ComplexTask(complex_features[i])
    X.set_end_state(complex_demos[i])
    X.enumerate_states()
    X.set_terminal_idx()
    if rank_features:
        X.convert_to_rankings()

    # demonstrations
    complex_user_demo = [list(complex_demos[i])]
    complex_trajectories = get_trajectories(X.states, complex_user_demo, X.transition)

    # using abstract features
    complex_abstract_features = np.array([X.get_features(state) for state in X.states])
    complex_abstract_features /= np.linalg.norm(complex_abstract_features, axis=0)

    if run_proposed:
        # transfer rewards to complex task
        canonical_weights_abstract = np.ones(6)
        transfer_rewards_abstract = complex_abstract_features.dot(canonical_weights_abstract)

        # score for predicting the action based on transferred rewards based on abstract features
        qf_transfer, _, _ = value_iteration(X.states, X.actions, X.transition, transfer_rewards_abstract, X.terminal_idx)
        predict_sequence, predict_score = predict_trajectory(qf_transfer, X.states, complex_user_demo, X.transition,
                                                             sensitivity=0.02)
        predict_scores.append(predict_score)

    # -------------------------------- Training: Learn weights from complex demo ------------------------------------ #

    # using true features
    # complex_state_features = np.array(X.states) / np.linalg.norm(X.states, axis=0)
    # complex_rewards_true, complex_weights_true = maxent_irl(X, complex_state_features, complex_trajectories,
    #                                                         optim, init, eps=1e-2)

    # using abstract features
    # complex_rewards_abstract, complex_weights_abstract = maxent_irl(X, complex_abstract_features,
    #                                                                 complex_trajectories,
    #                                                                 optim, init, eps=1e-2)

    # ----------------------------------------- Testing: Random baselines ------------------------------------------- #
    # if run_random_baseline:
    #     print("Assuming random weights ...")
    #     random_score = []
    #     for _ in range(100):
    #         # score for selecting actions based on random weights
    #         random_weights = np.random.rand(6)  # np.random.shuffle(canonical_weights_abstract)
    #         random_rewards_abstract = complex_abstract_features.dot(random_weights)
    #         qf_random, _, _ = value_iteration(X.states, X.actions, X.transition, random_rewards_abstract, X.terminal_idx)
    #         predict_sequence, r_score = predict_trajectory(qf_random, X.states, complex_user_demo, X.transition)
    #
    #         # score for randomly selecting an action
    #         # _, r_score = random_trajectory(X.states, complex_user_demo, X.transition)
    #
    #         random_score.append(r_score)
    #
    #     random_score = np.mean(random_score, axis=0)
    #     random_scores.append(random_score)

    # print("\n")
    # print("Complex task:")
    # print("     demonstration -", complex_user_demo)
    # print("predict (abstract) -", predict_sequence)

# -------------------------------------------------- Save results --------------------------------------------------- #
if run_proposed:
    np.savetxt("results_new_vi/predict11_normalized_features_uniform_weights_sensitivity2.csv", predict_scores)

if run_random_baseline:
    np.savetxt("results_new_vi/random11_normalized_features_random_weights.csv", random_scores)


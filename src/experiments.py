# import python libraries
import numpy as np
from copy import deepcopy
import pandas as pd

# import functions
import optimizer as O  # stochastic gradient descent optimizer
from maxent_irl import *
from assembly_tasks import *
from visualize import *


# pre-process feature value
def process_val(val):
    if val == "1 (No effort at all)":
        fea_val = 1.0
    elif val == "7 (A lot of effort)":
        fea_val = 7.0
    else:
        fea_val = float(val)

    return fea_val


# load user ratings
def load_features(data_df, feature_idx, action_idx):
    user_features = []
    for user_idx in range(2, len(data_df)):
        fea_mat = []
        for j in action_idx:
            fea_vec = []
            for k in feature_idx:
                fea_col = k + str(j)
                fea_val = process_val(data_df[fea_col][user_idx])
                fea_vec.append(fea_val)
            fea_mat.append(fea_vec)
        user_features.append(fea_mat.copy())
    return user_features


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
canonical_q, complex_q = ["Q7_", "Q8_"], ["Q14_", "Q15_"]
canonical_features = load_features(ratings_df, canonical_q, [1, 3, 5, 2, 4, 6])
complex_features = load_features(ratings_df, complex_q, [1, 3, 7, 8, 2, 4, 5, 6])

# ------------------------------------------------- Optimization ---------------------------------------------------- #

# initialize optimization parameters with constant
init = O.Constant(1.0)

# choose our optimization strategy: exponentiated stochastic gradient descent with linear learning-rate decay
optim = O.ExpSga(lr=O.linear_decay(lr0=0.6))

# ------------------------------------------- Training: Learn weights ----------------------------------------------- #

# select experiment
run_proposed = False
run_random_baseline = True
visualize = False
# rank_features = False

# initialize list of scores
match_scores, predict_scores, random_scores = [], [], []
weights, decision_pts = [], []

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
    # if rank_features:
    #     C.convert_to_rankings()

    # demonstrations
    canonical_user_demo = [list(canonical_demos[i])]
    canonical_trajectories = get_trajectories(C.states, canonical_user_demo, C.transition)

    if visualize:
        visualize_rel_actions(C, canonical_user_demo[0], i, "canonical")

    if run_proposed:
        print("Training ...")

        # using abstract features
        abstract_features = np.array([C.get_features(state) for state in C.states])
        norm_abstract_features = abstract_features / np.sum(abstract_features, axis=0)
        canonical_rewards_abstract, canonical_weights_abstract = maxent_irl(C, norm_abstract_features,
                                                                            canonical_trajectories,
                                                                            optim, init)

        print("Weights have been learned for the canonical task! Hopefully.")
        print("Weights -", canonical_weights_abstract)
        weights.append(canonical_weights_abstract)

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
    # if rank_features:
    #     X.convert_to_rankings()

    # demonstrations
    complex_user_demo = [list(complex_demos[i])]
    complex_trajectories = get_trajectories(X.states, complex_user_demo, X.transition)

    # using abstract features
    complex_abstract_features = np.array([X.get_features(state) for state in X.states])
    complex_abstract_features /= np.sum(complex_abstract_features, axis=0)

    if run_proposed:
        # transfer rewards to complex task
        transfer_rewards_abstract = complex_abstract_features.dot(canonical_weights_abstract)

        # score for predicting the action based on transferred rewards based on abstract features
        qf_transfer, _, _ = value_iteration(X.states, X.actions, X.transition, transfer_rewards_abstract,
                                            X.terminal_idx)
        predict_sequence, predict_score, decisions = predict_trajectory(qf_transfer, X.states, complex_user_demo,
                                                                        X.transition,
                                                                        sensitivity=0.0,
                                                                        consider_options=False)
        predict_scores.append(predict_score)
        # decision_pts.append(decisions)

        if visualize:
            visualize_rel_actions(X, complex_user_demo[0], i, "actual", predict_sequence, complex_user_demo[0])

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
    if run_random_baseline:
        print("Assuming random weights ...")
        random_score = []
        for _ in range(20):
            # score for selecting actions based on random weights
            random_weights = np.random.rand(6)  # np.random.shuffle(canonical_weights_abstract)
            random_rewards_abstract = complex_abstract_features.dot(random_weights)
            qf_random, _, _ = value_iteration(X.states, X.actions, X.transition, random_rewards_abstract, X.terminal_idx)
            predict_sequence, r_score, _ = predict_trajectory(qf_random, X.states, complex_user_demo, X.transition,
                                                              sensitivity=0.0, consider_options=False)

            # score for randomly selecting an action
            # predict_sequence, r_score = random_trajectory(X.states, complex_user_demo, X.transition)

            random_score.append(r_score)

        random_score = np.mean(random_score, axis=0)
        random_scores.append(random_score)

    print("\n")
    print("Complex task:")
    print("   demonstration -", complex_user_demo)
    print("     predictions -", predict_sequence)

# -------------------------------------------------- Save results --------------------------------------------------- #
if run_proposed:
    # np.savetxt("results/decide19.csv", decision_pts)
    np.savetxt("results/weights19_normalized_features.csv", weights)
    np.savetxt("results/predict19_normalized_features_test_sum.csv", predict_scores)

if run_random_baseline:
    np.savetxt("results/random19_normalized_features_random_test.csv", random_scores)

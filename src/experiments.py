# import python libraries
import numpy as np
from copy import deepcopy
import pandas as pd

# import functions
import optimizer as O  # stochastic gradient descent optimizer
from maxent_irl import *
from toy_assembly import *
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

# -------------------------------------------------- Load data ------------------------------------------------------ #

canonical_demos = [[0, 2, 3, 1, 4, 5],
                   [2, 5, 1, 4, 0, 3],
                   [0, 3, 2, 1, 4, 5]]

complex_demos = [[0, 4, 3, 8, 2, 7, 1, 6, 5, 9],
                 [2, 7, 3, 8, 4, 0, 5, 1, 6, 9],
                 [1, 6, 4, 3, 8, 9, 2, 7, 0, 5]]

canonical_features = [[0.837, 0.244, 0.282],
                      [0.212, 0.578, 0.018],
                      [0.712, 0.911, 0.418],
                      [0.462, 0.195, 0.882],
                      [0.962, 0.528, 0.618],
                      [0.056, 0.861, 0.218]]

complex_features = [[ 0.95, 0.033,  0.18],
                    [0.044, 0.367,   0.9],
                    [0.544,   0.7,  0.38],
                    [0.294, 0.145,  0.58],
                    [0.794, 0.478,  0.78],
                    [0.169, 0.811, 0.041],
                    [0.669, 0.256,  0.98],
                    [0.419, 0.589, 0.241],
                    [0.919, 0.922, 0.441],
                    [0.106, 0.095, 0.641]]

# ------------------------------------------------- Optimization ---------------------------------------------------- #

# initialize optimization parameters with constant
init = O.Constant(1.0)

# choose our optimization strategy: exponentiated stochastic gradient descent with linear learning-rate decay
optim = O.ExpSga(lr=O.linear_decay(lr0=0.6))

# ------------------------------------------- Training: Learn weights ----------------------------------------------- #

# select experiment
run_proposed = True
run_random_baseline = False
visualize = False
# rank_features = False

# initialize list of scores
match_scores, predict_scores, random_scores = [], [], []
weights, decision_pts = [], []

# # loop over all users
# for i in range(len(canonical_demos)):
#
#     print("=======================")
#     print("User:", i)
i = 0

# initialize canonical task
C = CanonicalTask(canonical_features)
C.set_end_state(canonical_demos[i])
C.enumerate_states()
C.set_terminal_idx()
# if rank_features:
#     C.convert_to_rankings()
all_canonical_trajectories = C.enumerate_trajectories([canonical_demos[i]])

# initialize actual task
X = ComplexTask(complex_features)
X.set_end_state(complex_demos[i])
X.enumerate_states()
X.set_terminal_idx()
# if rank_features:
#     X.convert_to_rankings()
all_complex_trajectories = X.enumerate_trajectories([complex_demos[i]])

# loop over all users
for i in range(len(canonical_demos)):

    print("=======================")
    print("User:", i)

    # demonstrations
    canonical_user_demo = [list(canonical_demos[i])]
    canonical_trajectories = get_trajectories(C.states, canonical_user_demo, C.transition)
    if visualize:
        visualize_rel_actions(C, canonical_user_demo[0], i, "canonical")

    # demonstrations
    complex_user_demo = [list(complex_demos[i])]
    complex_trajectories = get_trajectories(X.states, complex_user_demo, X.transition)

    # using abstract features
    complex_abstract_features = np.array([X.get_features(state) for state in X.states])
    complex_abstract_features /= np.linalg.norm(complex_abstract_features, axis=0)

    # ---------------------------------------- Training: Learn weights ---------------------------------------------- #

    if run_proposed:
        print("Training ...")

        # using abstract features
        abstract_features = np.array([C.get_features(state) for state in C.states])
        norm_abstract_features = abstract_features / np.linalg.norm(abstract_features, axis=0)
        _, n_features = np.shape(abstract_features)

        # initialize prior over weights
        n_samples = 10000
        samples = []
        priors = []
        max_likelihood = - np.inf
        max_reward = 0
        for _ in range(n_samples):
            u = np.random.uniform(0., 1., n_features)
            d = np.sum(u)  # np.sum(u ** 2) ** 0.5
            canonical_weights_prior = u/d

            likelihood_all_trajectories, _ = boltzman_likelihood(norm_abstract_features, all_canonical_trajectories,
                                                                 canonical_weights_prior)
            likelihood_user_demo, demo_reward = boltzman_likelihood(norm_abstract_features, canonical_trajectories,
                                                                    canonical_weights_prior)
            likelihood_user_demo = likelihood_user_demo/np.sum(likelihood_all_trajectories)

            samples.append(canonical_weights_prior)
            priors.append(likelihood_user_demo[0])

            if likelihood_user_demo > max_likelihood:
                max_likelihood = likelihood_user_demo
                max_reward = demo_reward
                canonical_weights_abstract = canonical_weights_prior

        # _, canonical_weights_abstract_irl = maxent_irl(C, norm_abstract_features,
        #                                                canonical_trajectories,
        #                                                optim, init)

        priors = priors / np.sum(priors)
        print("Weights have been learned for the canonical task! Hopefully.")
        print("Weights -", canonical_weights_abstract, max_reward)
        weights.append(canonical_weights_abstract)

    # --------------------------------------- Verifying: Reproduce demo --------------------------------------------- #

    # qf_abstract, _, _ = value_iteration(C.states, C.actions, C.transition, canonical_rewards_abstract, C.terminal_idx)
    # predict_sequence_canonical, _ = predict_trajectory(qf_abstract, C.states, canonical_user_demo, C.transition)
    #
    # print("\n")
    # print("Canonical task:")
    # print("     demonstration -", canonical_user_demo)
    # print("predict (abstract) -", predict_sequence_canonical)

    # ----------------------------------------- Testing: Predict complex -------------------------------------------- #

    if run_proposed:
        predict_score = []

        # ws = []
        # for _ in range(50):
        #     weight_idx = np.random.choice(range(len(samples)), size=1, p=priors)[0]
        #     complex_weights_abstract = samples[weight_idx]
        #     ws.append(complex_weights_abstract)

        # transfer rewards to complex task
        transfer_rewards_abstract = complex_abstract_features.dot(canonical_weights_abstract)

        # score for predicting the action based on transferred rewards based on abstract features
        qf_transfer, _, _ = value_iteration(X.states, X.actions, X.transition, transfer_rewards_abstract,
                                            X.terminal_idx)
        predict_sequence, p_score, decisions = predict_trajectory(qf_transfer, X.states, complex_user_demo,
                                                                  X.transition,
                                                                  sensitivity=0.0,
                                                                  consider_options=False)

        # predict_sequence, p_score, decisions = online_predict_trajectory(X, complex_user_demo,
        #                                                                  all_complex_trajectories,
        #                                                                  canonical_weights_abstract,
        #                                                                  complex_abstract_features,
        #                                                                  samples, priors,
        #                                                                  sensitivity=0.0,
        #                                                                  consider_options=False)
        predict_score.append(p_score)

        predict_score = np.mean(predict_score, axis=0)
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
        n_samples = 100
        max_likelihood = - np.inf
        for _ in range(n_samples):
            u = np.random.uniform(0., 1., n_features)
            d = np.sum(u)  # np.sum(u ** 2) ** 0.5
            random_weights = u / d  # np.random.shuffle(canonical_weights_abstract)
            random_rewards_abstract = complex_abstract_features.dot(random_weights)
            qf_random, _, _ = value_iteration(X.states, X.actions, X.transition, random_rewards_abstract, X.terminal_idx)
            predict_sequence, r_score, _ = predict_trajectory(qf_random, X.states, complex_user_demo, X.transition,
                                                              sensitivity=0.0, consider_options=False)

            # predict_sequence, r_score, _ = predict_trajectory(X, optim, init,
            #                                                   qf_random, complex_user_demo,
            #                                                   sensitivity=0.0,
            #                                                   consider_options=False)

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
    np.savetxt("results/toy/weights19_normalized_features_bayesian.csv", weights)
    # np.savetxt("results/toy/predict19_normalized_features_bayesian_online.csv", predict_scores)

if run_random_baseline:
    np.savetxt("results/toy/random19_normalized_features_bayesian.csv", random_scores)

print("Done.")

# import python libraries
import pdb
import numpy as np
from copy import deepcopy
import pandas as pd
import pickle

# import functions
import optimizer as O  # stochastic gradient descent optimizer
from vi import value_iteration
from maxent_irl import *
from assembly_tasks import *
from import_qualtrics import get_qualtrics_survey

# ----------------------------------------------- Load data ---------------------------------------------------- #

# download data from qualtrics
learning_survey_id = "SV_8eoX63z06ZhVZRA"
get_qualtrics_survey(dir_save_survey="", survey_id=learning_survey_id)

# paths
data_path = "/home/heramb/Git/irl-maxent/src/Human-Robot Assembly - Learning.csv"
save_path = "/home/heramb/ros_ws/src/assembly_demos/data/"

# load user data
df = pd.read_csv(data_path)


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
def load_features(data, user_idx, feature_idx, action_idx):
    fea_mat = []
    for j in action_idx:
        fea_vec = []
        for k in feature_idx:
            fea_col = k + str(j)
            fea_val = process_val(data[fea_col][user_idx])
            fea_vec.append(fea_val)
        fea_mat.append(fea_vec)
    return fea_mat

# ----------------------------------------------- Optimization -------------------------------------------------- #

# choose our parameter initialization strategy:
# initialize parameters with constant
init = O.Constant(1.0)

# choose our optimization strategy:
# we select exponentiated stochastic gradient descent with linear learning-rate decay
optim = O.ExpSga(lr=O.linear_decay(lr0=0.6))

# --------------------------------------------- User information ------------------------------------------------ #

rank_features = False
scale_weights = False

user_id = input("Enter user id: ")

print("=======================")
print("Calculating preference for user:", user_id)

idx = df.index[df['Q1'] == user_id][0]
canonical_survey_actions = [0, 3, 1, 4, 2, 5]
preferred_order = [df[q][idx] for q in ['Q9_1', 'Q9_2', 'Q9_3', 'Q9_4', 'Q9_5', 'Q9_6']]
canonical_demo = [a for _, a in sorted(zip(preferred_order, canonical_survey_actions))]

# user ratings for features
canonical_q, complex_q = ["Q6_", "Q7_"], ["Q13_", "Q14_"] 
canonical_features = load_features(df, idx, canonical_q, [2, 4, 6, 3, 5, 7])
complex_features = load_features(df, idx, complex_q, [3, 8, 15, 16, 4, 9, 10, 11])

# ---------------------------------------- Training: Learn weights ---------------------------------------------- #

# initialize canonical task
C = CanonicalTask(canonical_features)
C.set_end_state(canonical_demo)
C.enumerate_states()
C.set_terminal_idx()
if rank_features:
    C.convert_to_rankings()

# demonstrations
canonical_user_demo = [canonical_demo]
canonical_trajectories = get_trajectories(C.states, canonical_user_demo, C.transition)

print("Training ...")

# using abstract features
abstract_features = np.array([C.get_features(state) for state in C.states])
norm_abstract_features = abstract_features / np.linalg.norm(abstract_features, axis=0)
canonical_rewards_abstract, canonical_weights_abstract = maxent_irl(C, norm_abstract_features,
                                                                    canonical_trajectories,
                                                                    optim, init)

print("Weights have been learned for the canonical task! Fingers X-ed.")
print("Weights -", canonical_weights_abstract)

# scale weights
if scale_weights:
    canonical_weights_abstract /= max(canonical_weights_abstract)

# ----------------------------------------- Testing: Predict complex -------------------------------------------- #

sample_complex_demo = [1, 3, 5, 0, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 7]

# initialize complex task
X = ComplexTask(complex_features)
X.set_end_state(sample_complex_demo)
X.enumerate_states()
X.set_terminal_idx()
if rank_features:
    X.convert_to_rankings()

# using abstract features
complex_abstract_features = np.array([X.get_features(state) for state in X.states])
complex_abstract_features /= np.linalg.norm(complex_abstract_features, axis=0)

# transfer rewards to complex task
transfer_rewards_abstract = complex_abstract_features.dot(canonical_weights_abstract)

# score for predicting the action based on transferred rewards based on abstract features
qf_transfer, _, _ = value_iteration(X.states, X.actions, X.transition, transfer_rewards_abstract, X.terminal_idx)

pickle.dump(qf_transfer, open(save_path + "q_values_" + user_id + ".p", "wb"))
pickle.dump(X.states, open(save_path + "states_" + user_id + ".p", "wb"))
print("Q-values have been saved for the actual task!")

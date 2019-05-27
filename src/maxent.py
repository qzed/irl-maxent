import numpy as np


def feature_expectation_from_trajectories(features, trajectories):
    n_states, n_features = features.shape

    fe = np.zeros(n_features)

    for t in trajectories:
        for s in t.states():
            fe += features[s]

    return fe / len(trajectories)

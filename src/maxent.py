import numpy as np
from itertools import chain


def feature_expectation_from_trajectories(features, trajectories):
    n_states, n_features = features.shape

    fe = np.zeros(n_features)

    for t in trajectories:
        for s_from, _action, _s_to in chain(t, [(t[-1][2], 0, 0)]):
            fe += features[s_from]

    return fe / len(trajectories)

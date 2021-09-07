import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

predict_scores = np.loadtxt("results_new_vi/predict11_normalized_features.csv")
random_scores = np.loadtxt("results_new_vi/predict11_ranked_normalized_features.csv")
sns.set(style="darkgrid", context="talk")

n_users, n_steps = np.shape(predict_scores)

# check statistical difference
predict_users = np.sum(predict_scores, axis=1)/n_steps
random_users = np.sum(random_scores, axis=1)/n_steps
print(stats.ttest_rel(predict_users, random_users))

# accuracy at each time steps
predict_accuracy = np.sum(predict_scores, axis=0)/n_users
random_accuracy = np.sum(random_scores, axis=0)/n_users
steps = range(1, len(predict_accuracy)+1)

plt.figure(figsize=(10, 5))
plt.plot(steps, predict_accuracy, 'g', linewidth=3.5)
plt.plot(steps, random_accuracy, 'y-.', linewidth=3.5)
plt.ylim(-0.1, 1.1)
plt.xticks(steps)
plt.xlabel("Time step")
plt.ylabel("Accuracy")
plt.gcf().subplots_adjust(bottom=0.15)
plt.legend(["proposed (ranked features)", "random weights (ranked features)"], loc=4)
plt.show()
# plt.savefig("figures/results11_ranked_features_random_weights.jpg")

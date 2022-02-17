import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

file_path = os.path.dirname(__file__) + '/results/study_hr/'

predict_scores = np.loadtxt(file_path + "predict.csv")
random1_scores = np.loadtxt(file_path + "random_weights.csv")
random2_scores = np.loadtxt(file_path + "random.csv")
uniform_scores = np.loadtxt(file_path + "predict_complex.csv")

n_users, n_steps = np.shape(predict_scores)

# check statistical difference
predict_users = list(np.sum(predict_scores, axis=1)/n_steps)
random1_users = list(np.sum(random1_scores, axis=1)/n_steps)
random2_users = list(np.sum(random2_scores, axis=1)/n_steps)
uniform_users = list(np.sum(uniform_scores, axis=1)/n_steps)
print("Random weights:", stats.ttest_rel(predict_users, random1_users))
print("Random actions:", stats.ttest_rel(predict_users, random2_users))
print("Uniform weights:", stats.ttest_rel(predict_users, uniform_users))

# accuracy at each time steps
predict_accuracy = np.sum(predict_scores, axis=0)/n_users
random1_accuracy = np.sum(random1_scores, axis=0)/n_users
random2_accuracy = np.sum(random2_scores, axis=0)/n_users
uniform_accuracy = np.sum(uniform_scores, axis=0)/n_users
steps = range(1, len(predict_accuracy)+1)

# plotting
sns.set(style="darkgrid", context="talk")
plt.figure(figsize=(10, 5))
plt.plot(steps, random2_accuracy, 'r--', linewidth=3.7)
plt.plot(steps, random1_accuracy, 'b-.', linewidth=3.7)
plt.plot(steps, predict_accuracy, 'g', linewidth=4.0)
plt.plot(steps, uniform_accuracy, 'y:', linewidth=4)
plt.ylim(-0.1, 1.1)
plt.xticks(steps, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Time step", fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.gcf().subplots_adjust(bottom=0.15)
plt.legend(["random action", "random weights", "proposed", "actual"], loc=4, fontsize=20)
# plt.show()
plt.savefig("figures/results6_hr.png", bbox_inches='tight')

# plt.figure()
# Y = list(predict_scores[:, 0]) + uniform_users
# X = ["proposed"]*n_users + ["uniform weights"]*n_users
# sns.barplot(X, Y, palette=['g', 'y'], ci=68)
# plt.ylim(-0.1, 1.1)
# plt.ylabel("Accuracy")
# plt.gcf().subplots_adjust(left=0.15)
# plt.savefig("figures/results11_timestep1.jpg", bbox_inches='tight')
# plt.show()

# Sensitivity

# predict1_scores = np.loadtxt("results_new_vi/predict11_normalized_features_sensitivity2.csv")
# predict2_scores = np.loadtxt("results_new_vi/predict11_normalized_features_sensitivity5.csv")
# predict3_scores = np.loadtxt("results_new_vi/predict11_normalized_features_sensitivity10.csv")
#
# # accuracy at each time steps
# predict1_accuracy = np.sum(predict1_scores, axis=0)/n_users
# predict2_accuracy = np.sum(predict2_scores, axis=0)/n_users
# predict3_accuracy = np.sum(predict3_scores, axis=0)/n_users
# steps = range(1, len(predict_accuracy)+1)
#
#
# plt.figure(figsize=(10, 5))
# plt.plot(steps, predict_accuracy, 'g', linewidth=3.5)
# plt.plot(steps, predict1_accuracy, 'b-.', linewidth=3.5)
# plt.plot(steps, predict2_accuracy, 'r--', linewidth=3.5)
# plt.plot(steps, predict3_accuracy, 'y:', linewidth=3.5)
# plt.ylim(-0.1, 1.1)
# plt.xticks(steps)
# plt.xlabel("Time step")
# plt.ylabel("Accuracy")
# plt.gcf().subplots_adjust(bottom=0.15)
# plt.legend(["proposed", "2%", "5%", "10%"], loc=4)
# plt.show()
# # plt.savefig("figures/results11_sensitivity.jpg", bbox_inches='tight')

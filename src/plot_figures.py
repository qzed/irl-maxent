import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# plotting style
sns.set(style="darkgrid", context="talk")

dir_path = os.path.dirname(__file__)
canonical_p = np.loadtxt(dir_path + "/data/complex_ratings_physical.csv")
canonical_m = np.loadtxt(dir_path + "/data/complex_ratings_mental.csv")

n_users, n_actions = np.shape(canonical_p)

X, Y1, Y2 = [], [], []
for a in range(n_actions):
    y1 = [r[a] for r in canonical_p]
    y2 = [r[a] for r in canonical_m]
    Y1 += y1
    Y2 += y2
    X += [a]*len(y1)
df_dict = {"Actions": X, "Physical Effort": Y1, "Mental Effort": Y2}
df = pd.DataFrame(df_dict)

# plt.figure()
# sns.boxplot(x="Actions", y="Physical Effort", data=df)
# plt.gcf().subplots_adjust(bottom=0.175)
# plt.gcf().subplots_adjust(left=0.15)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("Actions", fontsize=24)
# plt.ylabel("Physical Effort", fontsize=24)
# # plt.savefig("figures/canonical_physical_ratings.png", bbox_inches='tight')
#
# plt.figure()
# sns.boxplot(x="Actions", y="Mental Effort", data=df)
# plt.gcf().subplots_adjust(bottom=0.175)
# plt.gcf().subplots_adjust(left=0.15)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel("Actions", fontsize=24)
# plt.ylabel("Mental Effort", fontsize=24)
# plt.savefig("figures/canonical_mental_ratings.png", bbox_inches='tight')
# plt.show()

file_path = dir_path + '/results/'
predict_scores = np.loadtxt(file_path + "predict19_normalized_features.csv")
random1_scores = np.loadtxt(file_path + "predict19_normalized_features_test_sum.csv")
random2_scores = np.loadtxt(file_path + "random19_normalized_features_random_test.csv")
decision_pts = np.loadtxt(file_path + "decide19.csv")

n_users, n_steps = np.shape(predict_scores)

# check statistical difference
predict_users = list(np.sum(predict_scores, axis=1)/n_steps)
random1_users = list(np.sum(random1_scores, axis=1)/n_steps)
random2_users = list(np.sum(random2_scores, axis=1)/n_steps)
print("Random weights:", stats.ttest_rel(predict_users, random1_users))
print("Random actions:", stats.ttest_rel(predict_users, random2_users))

# plt.figure()
# X1 = predict_users + random1_users
# Y = ["canonical weights"]*n_users + ["random weights"]*n_users
# df_dict = {"Y": X1, "X": Y}
# df = pd.DataFrame(df_dict)
# sns.barplot(x="X", y="Y", data=df)
# plt.gcf().subplots_adjust(bottom=0.15)
# plt.gcf().subplots_adjust(left=0.15)
# plt.title("Over all time steps")
# plt.savefig("figures/results19_random_weights.png", bbox_inches='tight')

# predict_users_new, random1_users_new, utility = [], [], []
# for i in range(n_users):
#     predict_new = [score for j, score in enumerate(predict_scores[i]) if decision_pts[i][j]]
#     random1_new = [score for j, score in enumerate(random1_scores[i]) if decision_pts[i][j]]
#     n_steps_new = len(predict_new)
#     utility.append(n_steps_new/n_steps)
#     predict_users_new.append(np.sum(predict_new)/n_steps_new)
#     random1_users_new.append(np.sum(random1_new)/n_steps_new)
# print("Random weights:", stats.ttest_rel(predict_users_new, random1_users_new))

# plt.figure()
# X2 = predict_users_new + random1_users_new
# df_dict = {"Y": X2, "X": Y}
# df = pd.DataFrame(df_dict)
# sns.barplot(x="X", y="Y", data=df)
# plt.gcf().subplots_adjust(bottom=0.15)
# plt.gcf().subplots_adjust(left=0.15)
# plt.title("Over all time steps")
# plt.savefig("figures/results19_random_weights_new.png", bbox_inches='tight')
# plt.show()

# accuracy at each time steps
predict_accuracy = np.sum(predict_scores, axis=0)/n_users
random1_accuracy = np.sum(random1_scores, axis=0)/n_users
random2_accuracy = np.sum(random2_scores, axis=0)/n_users
steps = list(range(len(predict_accuracy)))

plt.figure(figsize=(9, 5))

X, Y1, Y2 = [], [], []
for i in range(n_users):
    X += steps
    Y1 += list(predict_scores[i, :])
    Y2 += list(random2_scores[i, :])
df1 = pd.DataFrame({"Time step": X, "Accuracy": Y1})
df2 = pd.DataFrame({"Time step": X, "Accuracy": Y2})
# sns.lineplot(data=df2, x="Time step", y="Accuracy", color="r", linestyle="--", alpha=0.9)
# sns.lineplot(data=df1, x="Time step", y="Accuracy", color="g", linewidth=4, alpha=0.9)
plt.plot(steps, random2_accuracy, 'r--', linewidth=4.5, alpha=0.95)
plt.plot(steps, random1_accuracy, 'b-.', linewidth=4.5, alpha=0.95)
plt.plot(steps, predict_accuracy, 'g', linewidth=5.0, alpha=0.95)
plt.xlim(-1, 17)
plt.ylim(0.1, 1.1)
plt.xticks(steps, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Time step", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.gcf().subplots_adjust(bottom=0.15)
# plt.legend(["random action", "proposed"], loc=4, fontsize=16)
plt.show()
# plt.savefig("figures/results19_ci.png", bbox_inches='tight')

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

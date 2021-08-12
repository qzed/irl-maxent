import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

match_accuracy = np.loadtxt("match_new.csv")
predict_accuracy = np.loadtxt("predict_new.csv")
random_accuracy = np.loadtxt("random_new.csv")
sns.set(style="darkgrid", context="talk")

steps = range(1, len(match_accuracy)+1)

plt.figure(figsize=(10, 5))
plt.plot(steps, predict_accuracy, 'g', linewidth=3.5)
plt.plot(steps, random_accuracy, 'r--', linewidth=3.5)
plt.ylim(-0.1, 1.1)
plt.xticks(steps)
plt.xlabel("Action")
plt.ylabel("Accuracy")
plt.gcf().subplots_adjust(bottom=0.15)
plt.legend(["proposed", "random"], loc=4)
plt.savefig("results1.jpg")

# plt.figure(figsize=(10, 5))
# plt.bar(steps, predict_accuracy)
# plt.ylim(0, 1)
# plt.xticks(steps)
# plt.xlabel("Action")
# plt.ylabel("Accuracy")
# plt.gcf().subplots_adjust(bottom=0.15)
# plt.savefig("trajectory_prediction_fixed.jpg")

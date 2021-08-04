import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

match_accuracy = np.loadtxt("match_short.csv")
predict_accuracy = np.loadtxt("predict_short.csv")
sns.set(style="darkgrid", context="talk")

steps = range(1, len(match_accuracy)+1)

plt.figure(figsize=(10, 5))
plt.bar(steps, match_accuracy)
plt.ylim(0, 1)
plt.xticks(steps)
plt.xlabel("Action")
plt.ylabel("Accuracy")
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig("trajectory_matching_fixed.jpg")

plt.figure(figsize=(10, 5))
plt.bar(steps, predict_accuracy)
plt.ylim(0, 1)
plt.xticks(steps)
plt.xlabel("Action")
plt.ylabel("Accuracy")
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig("trajectory_prediction_fixed.jpg")

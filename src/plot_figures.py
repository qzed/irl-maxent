import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

match_scores = np.loadtxt("results/match.csv")
predict_scores = np.loadtxt("results/predict.csv")
random_scores = np.loadtxt("results/random.csv")
sns.set(style="darkgrid", context="talk")

match_accuracy = np.sum(match_scores, axis=0)/len(match_scores)
predict_accuracy = np.sum(predict_scores, axis=0)/len(predict_scores)
random_accuracy = np.sum(random_scores, axis=0)/len(random_scores)
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
plt.savefig("figures/results6.jpg")

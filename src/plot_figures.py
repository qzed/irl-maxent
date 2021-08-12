import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

match_accuracy = np.loadtxt("results/match.csv")
predict_accuracy = np.loadtxt("results/predict.csv")
random_accuracy = np.loadtxt("results/random.csv")
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
plt.savefig("figures/results1.jpg")

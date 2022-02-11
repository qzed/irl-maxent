# import python libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from import_qualtrics import get_qualtrics_survey

# ----------------------------------------------- Load data ---------------------------------------------------- #

# download data from qualtrics
learning_survey_id = "SV_8eoX63z06ZhVZRA"
execution_survey_id = "SV_29ILBswADgbr79Q"
get_qualtrics_survey(dir_save_survey="", survey_id=execution_survey_id)

# paths
data_path = "/home/heramb/Git/irl-maxent/src/Human-Robot Assembly - Execution.csv"
save_path = "/home/heramb/Git/irl-maxent/src/figures/human-robot/"

# load user data
# User ids are 1 - 20. First 10 users perform the reactive case first.
df = pd.read_csv(data_path, skiprows=[0, 2])
df1 = df[df["User ID"] >= 1]
df20 = df1[df1["User ID"] <= 20]
df10 = df20[df20["User ID"] <= 10]
df20 = df20[df20["User ID"] >= 11]

# q_id = input("Enter a question id (1-15): ")
for q_id in range(1, 16):
    q_id_1 = 14 + int(q_id)
    q_id_2 = 30 + int(q_id)
    q_col_1 = list(df.columns)[q_id_1]
    q_col_2 = list(df.columns)[q_id_2]

    reactive = list(df10[q_col_1]) + list(df20[q_col_2])
    proactive = list(df10[q_col_2]) + list(df20[q_col_1])

    x = ["reactive"]*len(reactive) + ["proactive"]*len(proactive)
    y = reactive + proactive
    plt.figure()
    sns.barplot(x=x, y=y)
    plt.title(q_col_1)
    plt.savefig(save_path + str(q_id) + ".png")

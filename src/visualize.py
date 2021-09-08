import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def visualize_demo_2(task, demo, idx, prefix):

    features, states, transition_function = task.features, task.states, task.transition
    # features = list((np.array(features) - 1.0) / (7.0 - 1.0))
    s, available_actions = 0, demo.copy()
    prev_a = -1

    sns.set(style="darkgrid", context="talk")
    fig = plt.figure(figsize=(12,5))
    plt.xlabel('Time steps')
    plt.ylabel('Action index')

    for step, take_action in enumerate(demo):
        #candidates = []
        candidates = set()
        for a in available_actions:
            p, sp = transition_function(states[s], a)
            if sp:
                #candidates.append(a)
                candidates.add(a)

        if len(candidates) < 1:
            print("Error: No candidate actions to pick from.")

        eps = [features[curr_a][0] for curr_a in candidates]
        ems = [features[curr_a][1] for curr_a in candidates]
        if prev_a >= 0:
            cps = [task.part_similarity[prev_a][curr_a] for curr_a in candidates]
            cts = [task.tool_similarity[prev_a][curr_a] for curr_a in candidates]
        else:
            cps, cts = [0.0]*len(candidates), [0.0]*len(candidates)

        for option, curr_a in enumerate(candidates):
            r_val = features[curr_a][0] / max(eps)
            b_val = features[curr_a][1] / max(ems)
            if prev_a >= 0 and max(cps) > 0.0:
                g_val = task.part_similarity[prev_a][curr_a] / max(cps)
            else:
                g_val = 0.0

            if prev_a >= 0 and max(cts) > 0.0:
                t_val = task.tool_similarity[prev_a][curr_a] / max(cts)
            else:
                t_val = 0.0

            marker_shape = "o"
            if g_val > 0.0:
                if t_val > 0.0:
                    marker_shape = "^"
                else:
                    marker_shape = "s"
            else:
                if t_val>0.0:
                    marker_shape = "d"


            plt.scatter([step], [curr_a], s=100, c=[[r_val, b_val, 0.0]], marker=marker_shape)
            if curr_a == take_action:
                if "complex" in prefix:
                    plt.plot([step + 0.4], [curr_a], "r*")
                else:
                    plt.plot([step + 0.1], [curr_a], "r*")

        p, sp = transition_function(states[s], take_action)
        s = states.index(sp)
        available_actions.remove(take_action)
        prev_a = take_action

    add_color_legend(fig)

    plt.savefig("visualizations/"+prefix+"_user" + str(idx) + ".jpg", bbox_inches='tight')
    print("visualizations/"+prefix+"_user" + str(idx) + ".jpg"+" finished")

    return

def visualize_demo(task, demo, idx):

    features, states, transition_function = task.features, task.states, task.transition
    # features = list((np.array(features) - 1.0) / (7.0 - 1.0))
    s, available_actions = 0, demo.copy()
    prev_a = -1

    sns.set(style="darkgrid", context="talk")
    plt.figure()

    for step, take_action in enumerate(demo):
        #candidates = []
        candidates = set()
        for a in available_actions:
            p, sp = transition_function(states[s], a)
            if sp:
                #candidates.append(a)
                candidates.add(a)

        if len(candidates) < 1:
            print("Error: No candidate actions to pick from.")

        eps = [features[curr_a][0] for curr_a in candidates]
        ems = [features[curr_a][1] for curr_a in candidates]
        if prev_a >= 0:
            cps = [task.part_similarity[prev_a][curr_a] for curr_a in candidates]
            cts = [task.tool_similarity[prev_a][curr_a] for curr_a in candidates]
        else:
            cps, cts = [0.0]*len(candidates), [0.0]*len(candidates)

        for option, curr_a in enumerate(candidates):
            r_val = features[curr_a][0] / max(eps)
            b_val = features[curr_a][1] / max(ems)
            if prev_a >= 0 and max(cps) > 0.0:
                g_val = task.part_similarity[prev_a][curr_a] / max(cps)
            else:
                g_val = 0.0

            marker_shape = "o"
            if g_val > 0.0:
                marker_shape = "^"

            plt.scatter([step], [option], s=100, c=[[r_val, b_val, 0.0]], marker=marker_shape)
            if curr_a == take_action:
                plt.plot([step + 0.2], [option], "r*")

        p, sp = transition_function(states[s], take_action)
        s = states.index(sp)
        available_actions.remove(take_action)
        prev_a = take_action

    plt.savefig("visualizations/complex_user" + str(idx) + ".jpg", bbox_inches='tight')
    print("visualizations/complex_user" + str(idx) + ".jpg"+" finished")

    return


def visualize_predictions(qf, states, demos, transition_function, sensitivity=0):

    demo = demos[0]
    s, available_actions = 0, demo.copy()

    generated_sequence, scores = [], []
    for take_action in demo:
        max_action_val = -np.inf
        candidates = []
        for a in available_actions:
            p, sp = transition_function(states[s], a)
            if sp:
                if qf[s][a] > (1 + sensitivity) * max_action_val:
                    candidates = [a]
                    max_action_val = qf[s][a]
                elif (1 - sensitivity) * max_action_val <= qf[s][a] <= (1 + sensitivity) * max_action_val:
                    candidates.append(a)
                    max_action_val = qf[s][a]

        if len(candidates) > 1:
            predict_iters = 1000
        elif len(candidates) == 1:
            predict_iters = 1
        else:
            print("Error: No candidate actions to pick from.")

        predict_score = []
        for _ in range(predict_iters):
            predict_action = np.random.choice(candidates)
            predict_score.append(predict_action == take_action)
        score = np.mean(predict_score)
        scores.append(score)

        generated_sequence.append(predict_action)
        p, sp = transition_function(states[s], take_action)
        s = states.index(sp)
        available_actions.remove(take_action)

    return generated_sequence, scores

def add_color_legend(fig):
        x = np.linspace(0.0, 1.0, 100)
        y = np.linspace(0.0, 1.0, 100)
        c = [[x_val, y_val, 0.0] for y_val in y for x_val in x]
        x, y = np.meshgrid(x, y)
        
        ax = fig.add_axes([1, 0.6, 0.1, 0.2])
        ax.set_xlabel('Physical Effort')
        ax.set_ylabel('Mental Effort')
        ax.scatter(x, y, s=2, c=c, marker='s')
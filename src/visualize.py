import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.lines as mlines


def visualize_rel_actions(task, demo, idx, prefix, predictions=None, ):

    features, states, transition_function = task.features, task.states, task.transition
    n_actions, n_steps = len(task.actions), len(demo)
    s, available_actions = 0, demo.copy()
    prev_a = -1

    eps = features[:, 0]
    ems = features[:, 1]

    ep_vals = eps/max(eps)
    em_vals = ems/max(ems)

    if prefix == "actual":
        fig_width = 12.75
    else:
        fig_width = 6

    sns.set(style="darkgrid", context="talk", rc={'axes.facecolor': '0.93'})
    plt.figure(figsize=(fig_width, 6))
    plt.xlabel('Time steps')
    plt.ylabel('Action')

    # plot user sequence
    plt.plot(range(len(demo)), demo, "k", zorder=1, alpha=0.23, linewidth=10)
    feat_order = 2

    # plot the features for each action
    for step, take_action in enumerate(demo):

        # plot predictions
        if predictions:
            pred_a = list(set(predictions[step]))
            scat = plt.scatter([step] * len(pred_a), pred_a, s=1100, facecolor=(0, 0, 1, 0.23), label="Prediction",
                               edgecolors=(0, 0, 0.1, 0.23), marker="o", zorder=2, linewidth=0.0)
            plt.legend(handles=[scat])
            feat_order = 3

        candidates = set()
        for a in available_actions:
            p, sp = transition_function(states[s], a)
            if sp:
                candidates.add(a)

        if len(candidates) < 1:
            print("Error: No candidate actions to pick from.")

        for option, curr_a in enumerate(candidates):
            ep_val = ep_vals[curr_a]
            em_val = em_vals[curr_a]
            if prev_a >= 0:
                p_val = task.part_similarity[prev_a][curr_a]
                t_val = task.tool_similarity[prev_a][curr_a]
            else:
                p_val, t_val = 0.0, 0.0

            marker_shape = "o"
            if p_val > 0.0:
                if t_val > 0.0:
                    marker_shape = "^"
                else:
                    marker_shape = "s"
            else:
                if t_val > 0.0:
                    marker_shape = "d"

            plt.scatter([step], [curr_a], s=400, c=[[ep_val, em_val, 0.0]], marker=marker_shape, zorder=feat_order,
                        alpha=0.97, linewidth=0.0)

        p, sp = transition_function(states[s], take_action)
        s = states.index(sp)
        available_actions.remove(take_action)
        prev_a = take_action

    # add_marker_legend()
    plt.title(prefix + " task")
    plt.xlim(-0.5, n_steps - 0.5)
    plt.ylim(-0.5, n_actions - 0.5)
    plt.xticks(range(n_steps))
    plt.gcf().subplots_adjust(bottom=0.15)

    # plt.show()
    plt.savefig("visualizations/"+prefix+"_user" + str(idx) + "_predictions.jpg", bbox_inches='tight')
    # print("visualizations/"+prefix+"_user" + str(idx) + ".jpg"+" finished")

    return


def visualize_rel_candidates(task, demo, idx, prefix):

    features, states, transition_function = task.features, task.states, task.transition
    # features = list((np.array(features) - 1.0) / (7.0 - 1.0))
    s, available_actions = 0, demo.copy()
    prev_a = -1

    sns.set(style="darkgrid", context="talk")
    fig = plt.figure(figsize=(12,5))
    plt.xlabel('Time steps')
    plt.ylabel('Action index')

    for step, take_action in enumerate(demo):
        candidates = set()
        for a in available_actions:
            p, sp = transition_function(states[s], a)
            if sp:
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

    plt.ylim(-0.5, len(task.actions) - 0.5)
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.show()
    # plt.savefig("visualizations/"+prefix+"_user" + str(idx) + ".jpg", bbox_inches='tight')
    # print("visualizations/"+prefix+"_user" + str(idx) + ".jpg"+" finished")

    return


# # Plot heatmap
# sns.set(style="white", context="talk")
# fig = plt.figure(figsize=(2.7, 2.7))
# x = np.linspace(0.0, 1.0, 10)
# y = np.linspace(0.0, 1.0, 10)
# c = [[x_val, y_val, 0.0] for y_val in y for x_val in x]
# x, y = np.meshgrid(x, y)
# plt.axis('equal')
# plt.xticks([0, 1])
# plt.yticks([0, 1])
# plt.xlabel('Physical Effort')
# plt.ylabel('Mental Effort')
# plt.scatter(x, y, s=200, c=c, marker='s', linewidth=0.0, alpha=0.97)
# plt.gcf().subplots_adjust(bottom=0.25)
# plt.gcf().subplots_adjust(left=0.25)
# # plt.savefig("visualizations/heatmap.jpg")
# # plt.show()
#
# # plot legend
# fig = plt.figure(figsize=(2.7, 2.7))
# no = mlines.Line2D([], [], color='k', marker='o', linestyle='None',
#                    markersize=10, label='Not same', alpha=0.72, linewidth=0)
# tool = mlines.Line2D([], [], color='k', marker='d', linestyle='None',
#                      markersize=10, label='Same tool', alpha=0.72, linewidth=0)
# part = mlines.Line2D([], [], color='k', marker='s', linestyle='None',
#                      markersize=10, label='Same part', alpha=0.72, linewidth=0)
# partool = mlines.Line2D([], [], color='k', marker='^', linestyle='None',
#                         markersize=10, label='Same tool & part', alpha=0.72, linewidth=0)
# plt.legend(handles=[partool, part, tool, no], loc="center")
# plt.axis("off")
# # plt.savefig("visualizations/legend.jpg")
# # plt.show()

# plot legend
fig = plt.figure()
proposed = mlines.Line2D([], [], markerfacecolor='b', marker='o', linestyle='None', markeredgecolor=(0, 0, 0.1, 0.23),
                         markersize=15, alpha=0.23, label='Prediction', linewidth=0)
plt.legend(handles=[proposed], loc="center", fontsize=18)
plt.axis("off")
plt.savefig("visualizations/legend2.jpg")
from itertools import product

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.tri as tri


def plot_transition_probabilities(world, ax, **kwargs):
    xy = [(x - 0.5, y - 0.5) for y, x in product(range(world.size + 1), range(world.size + 1))]
    xy += [(x, y) for y, x in product(range(world.size), range(world.size))]

    t, v = [], []
    for sy, sx in product(range(world.size), range(world.size)):
        state = world.state_point_to_index((sx, sy))
        state_r = world.state_point_to_index_clipped((sx + 1, sy))
        state_l = world.state_point_to_index_clipped((sx - 1, sy))
        state_t = world.state_point_to_index_clipped((sx, sy + 1))
        state_b = world.state_point_to_index_clipped((sx, sy - 1))

        # compute cell points
        bl, br = sy * (world.size + 1) + sx, sy * (world.size + 1) + sx + 1
        tl, tr = (sy + 1) * (world.size + 1) + sx, (sy + 1) * (world.size + 1) + sx + 1
        cc = (world.size + 1)**2 + sy * world.size + sx

        # compute triangles
        t += [(tr, cc, br)]                             # action = (1, 0)
        t += [(tl, bl, cc)]                             # action = (-1, 0)
        t += [(tl, cc, tr)]                             # action = (0, 1)
        t += [(bl, br, cc)]                             # action = (0, -1)

        # stack triangle values
        v += [world.p_transition[state, state_r, 0]]    # action = (1, 0)
        v += [world.p_transition[state, state_l, 1]]    # action = (-1, 0)
        v += [world.p_transition[state, state_t, 2]]    # action = (0, 1)
        v += [world.p_transition[state, state_b, 3]]    # action = (0, -1)

    x, y = zip(*xy)
    x, y = np.array(x), np.array(y)
    t, v = np.array(t), np.array(v)

    ax.set_aspect('equal')
    ax.set_xticks(range(world.size))
    ax.set_yticks(range(world.size))
    ax.set_xlim(-0.5, world.size - 0.5)
    ax.set_ylim(-0.5, world.size - 0.5)

    ax.tripcolor(x, y, t, facecolors=v, vmin=0.0, vmax=1.0, **kwargs)
    ax.triplot(x, y, t, color='red', linewidth=0.5)

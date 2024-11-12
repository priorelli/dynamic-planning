import numpy as np
import matplotlib.pyplot as plt
from pylab import tight_layout
import time
import sys
import config as c
import matplotlib.animation as animation


def record_video(log, width):
    plot_type = 2
    frame = c.n_steps - 1

    # Initialize 1st agent
    idxs_1st = {}
    ids_1st = {joint: j for j, joint in enumerate(c.joints_1st)}
    size_1st = np.zeros((c.n_joints_1st, 2))
    for joint in c.joints_1st:
        size_1st[ids_1st[joint]] = c.joints_1st[joint]['size']
        if c.joints_1st[joint]['link']:
            idxs_1st[ids_1st[joint]] = ids_1st[c.joints_1st[joint]['link']]
        else:
            idxs_1st[ids_1st[joint]] = -1

    # Initialize 2nd agent
    idxs_2nd = {}
    ids_2nd = {joint: j for j, joint in enumerate(c.joints_2nd)}
    size_2nd = np.zeros((c.n_joints_2nd, 2))
    for joint in c.joints_2nd:
        size_2nd[ids_2nd[joint]] = c.joints_2nd[joint]['size']
        if c.joints_2nd[joint]['link']:
            idxs_2nd[ids_2nd[joint]] = ids_2nd[c.joints_2nd[joint]['link']]
        else:
            idxs_2nd[ids_2nd[joint]] = -1

    # Load variables
    pos_1st = log['pos_1st'] + c.offset_1st
    est_pos_1st = log['est_pos_1st'] + c.offset_1st

    pos_2nd = log['pos_2nd'] + c.offset_2nd
    est_pos_2nd = log['est_pos_2nd'] + c.offset_2nd

    # Create plot
    scale = 0.8
    x_range = (-c.width / 2, c.width / 2)
    y_range = (-c.height / 2, c.height / 2)
    fig, axs = plt.subplots(1, figsize=(
        20, (y_range[1] - y_range[0]) * 20 / (x_range[1] - x_range[0])))

    def animate(n):
        if (n + 1) % 10 == 0:
            sys.stdout.write('\rStep: {:d}'.format(n + 1))
            sys.stdout.flush()

        # Clear plot
        axs.clear()
        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)
        axs.set_xlim(x_range)
        axs.set_ylim(y_range)
        tight_layout()

        # Draw text
        axs.text(x_range[0] + 50, y_range[0] + 50, '%d' % n, color='grey',
                 size=50, weight='bold')

        #######
        # 1st #
        #######

        for j in range(c.n_joints_1st):
            # Draw real body
            axs.plot(*np.array([pos_1st[n, idxs_1st[j] + 1],
                                pos_1st[n, j + 1]]).T,
                     lw=size_1st[j, 1] * scale, color='b', zorder=1)

            # Draw estimated bodies
            # axs.plot(*np.array([est_pos_1st[n, 0, idxs_1st[j] + 1],
            #                     est_pos_1st[n, 0, j + 1]]).T,
            #          lw=size_1st[j, 1] * scale, color='c', zorder=1)
            #
            # axs.plot(*np.array([est_pos_1st[n, 1, idxs_1st[j] + 1],
            #                     est_pos_1st[n, 1, j + 1]]).T,
            #          lw=size_1st[j, 1] * scale, color='navy', zorder=1)
        #
        # for j in range(c.n_joints_2nd):
        #     axs.plot(*np.array([est_pos_1st[n, 2, idxs_2nd[j] + 1],
        #                         est_pos_1st[n, 2, j + 1]]).T,
        #              lw=size_2nd[j, 1] * scale, color='m', zorder=2,
        #              alpha=1.0)

        # Draw real body trajectory
        axs.scatter(*pos_1st[n - (n % c.n_steps): n + 1, -1].T,
                    color='darkblue', zorder=2)

        #######
        # 2nd #
        #######

        for j in range(c.n_joints_2nd):
            # Draw real body
            axs.plot(*np.array([pos_2nd[n, idxs_2nd[j] + 1],
                                pos_2nd[n, j + 1]]).T,
                     lw=size_2nd[j, 1] * scale, color='r', zorder=1)

            # Draw estimated bodies
            axs.plot(*np.array([est_pos_2nd[n, 0, idxs_2nd[j] + 1],
                                est_pos_2nd[n, 0, j + 1]]).T,
                     lw=size_2nd[j, 1] * scale, color='orange', zorder=1)

            axs.plot(*np.array([est_pos_2nd[n, 1, idxs_2nd[j] + 1],
                                est_pos_2nd[n, 1, j + 1]]).T,
                     lw=size_2nd[j, 1] * scale, color='darkred', zorder=2,
                     alpha=1.0)

        for j in range(c.n_joints_1st):
            axs.plot(*np.array([est_pos_2nd[n, 2, idxs_1st[j] + 1],
                                est_pos_2nd[n, 2, j + 1]]).T,
                     lw=size_1st[j, 1] * scale, color='g', zorder=1)

        # Draw real body trajectory
        axs.scatter(*pos_2nd[n - (n % c.n_steps): n + 1, -1].T,
                    color='darkred', zorder=2)

    # Plot video
    if plot_type == 0:
        start = time.time()
        ani = animation.FuncAnimation(fig, animate, c.n_steps)
        writer = animation.writers['ffmpeg'](fps=60)
        ani.save('plots/video.mp4', writer=writer)
        print('\nTime elapsed:', time.time() - start)

    # Plot frame sequence
    elif plot_type == 1:
        for i in range(0, c.n_steps, c.n_steps // 10):
            animate(i)
            plt.savefig('plots/frame_%d' % i, bbox_inches='tight')

    # Plot single frame
    elif plot_type == 2:
        animate(frame)
        plt.savefig('plots/frame_%d' % frame, bbox_inches='tight')

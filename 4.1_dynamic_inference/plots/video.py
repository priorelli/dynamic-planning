import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pylab import tight_layout
import time
import sys
import config as c
import matplotlib.animation as animation


def record_video(log, width):
    plot_type = 1
    frame = c.n_steps - 1

    # Initialize body
    idxs = {}
    ids = {joint: j for j, joint in enumerate(c.joints)}
    size = np.zeros((c.n_joints, 2))
    for joint in c.joints:
        size[ids[joint]] = c.joints[joint]['size']
        if c.joints[joint]['link']:
            idxs[ids[joint]] = ids[c.joints[joint]['link']]
        else:
            idxs[ids[joint]] = -1

    # Load variables
    pos = log['pos']
    est_pos = log['est_pos']

    ball_pos = log['ball_pos']
    est_ball_pos = log['est_ball_pos']

    square_pos = log['square_pos']
    est_square_pos = log['est_square_pos']

    est_vel = log['est_vel']
    F_m = log['F_m']

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

        for j in range(c.n_joints):
            # Draw real body
            axs.plot(*np.array([pos[n, idxs[j] + 1], pos[n, j + 1]]).T,
                     lw=size[j, 1] * scale, color='b', zorder=1)

        # Draw real ball
        ball_size = c.ball_size * scale * 100
        axs.scatter(*ball_pos[n], color='r', s=ball_size, zorder=0)

        # Draw real square
        rect = patches.Rectangle(
            square_pos[n] - [c.square_size / 2, c.square_size / 2],
            c.square_size / 2, c.square_size / 2,
            color='g', zorder=0)
        axs.add_patch(rect)

        # Draw estimated square
        rect2 = patches.Rectangle(
            est_square_pos[n] - [c.square_size / 2, c.square_size / 2],
            c.square_size / 2, c.square_size / 2,
            color='olive', zorder=0)
        axs.add_patch(rect2)

        # Draw estimated ball
        axs.scatter(*est_ball_pos[n], color='purple', s=ball_size, zorder=0)

        # Draw real body trajectory
        # axs.scatter(*pos[n - (n % c.n_steps): n + 1, -1].T,
        #             color='darkblue', zorder=2)

        # Draw real ball trajectory
        # axs.scatter(*ball_pos[n - (n % c.n_steps): n + 1].T,
        #             color='darkred', zorder=2)

        # Draw real square trajectory
        # axs.scatter(*square_pos[n - (n % c.n_steps): n + 1].T,
        #             color='darkgreen', zorder=2)

        # Draw quivers
        x_est, u_est = pos[n, -1], est_vel[n]
        x_pred1, u_pred1 = pos[n, -1], F_m[n, 0]
        x_pred2, u_pred2 = pos[n, -1], F_m[n, 1]

        q = axs.quiver(*x_est.T, *u_est.T, angles='xy', color='navy',
                       width=0.006, scale=400)
        q = axs.quiver(*x_pred1.T, *u_pred1.T, angles='xy',
                       color='r', width=0.006, scale=700)
        q = axs.quiver(*x_pred2.T, *u_pred2.T, angles='xy',
                       color='g', width=0.006, scale=700)

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

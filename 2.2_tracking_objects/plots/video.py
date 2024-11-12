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

    # Create plot
    scale = 1.0
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

        # Draw real body trajectory
        axs.scatter(*pos[n - (n % c.n_steps): n + 1].T,
                    color='darkblue', zorder=2)

        # Draw real ball trajectory
        axs.scatter(*ball_pos[n - (n % c.n_steps): n + 1].T,
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

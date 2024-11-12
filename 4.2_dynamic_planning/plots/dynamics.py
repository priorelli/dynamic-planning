import numpy as np
import matplotlib.pyplot as plt
import utils
import config as c


def plot_dynamics(log, width):
    # Load variables
    angles = log['angles']
    est_angles = log['est_angles']

    grasp_pos = log['grasp_pos']
    est_grasp_pos = log['est_grasp_pos']

    ball_pos = log['ball_pos']
    est_ball_pos = log['est_ball_pos']

    causes_int = log['causes_int']

    disc_actions = log['disc_actions']

    # Initialize plots
    fig, axs = plt.subplots(2, figsize=(25, 15))

    axs[0].set_xlabel('Time')
    axs[1].set_xlabel('Time')

    # Draw actions
    for action, label in zip(disc_actions.T[1:], ['Track', 'Open', 'Close']):
        axs[0].plot(action, lw=width, label=label)

    # Draw intrinsic causes
    for int_, label in zip(causes_int.T, ['Open', 'Close']):
        axs[1].plot(int_, lw=width, label=label)

    # Draw error
    error = utils.normalize(np.linalg.norm(ball_pos - grasp_pos, axis=1),
                            c.norm_cart)
    axs[0].plot(error, 'r--', lw=width - 3)
    axs[1].plot(error, 'r--', lw=width - 3)

    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    fig.savefig('plots/dynamics_' + c.log_name, bbox_inches='tight')

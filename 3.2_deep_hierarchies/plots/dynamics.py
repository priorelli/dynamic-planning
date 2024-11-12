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

    square_pos = log['square_pos']
    est_square_pos = log['est_square_pos']

    # Initialize plots
    fig, axs = plt.subplots(2, figsize=(25, 15))

    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    fig.savefig('plots/dynamics_' + c.log_name, bbox_inches='tight')

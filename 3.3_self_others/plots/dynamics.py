import matplotlib.pyplot as plt
import config as c


def plot_dynamics(log, width):
    # Load variables
    angles_1st = log['angles_1st']
    est_angles_1st = log['est_angles_1st']

    angles_2nd = log['angles_2nd']
    est_angles_2nd = log['est_angles_2nd']

    pos_1st = log['pos_1st']
    est_pos_1st = log['est_pos_1st']

    pos_2nd = log['pos_2nd']
    est_pos_2nd = log['est_pos_2nd']

    # Initialize plots
    fig, axs = plt.subplots(2, figsize=(25, 15))

    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    fig.savefig('plots/dynamics_' + c.log_name, bbox_inches='tight')

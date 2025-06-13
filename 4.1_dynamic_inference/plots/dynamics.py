import numpy as np
import matplotlib.pyplot as plt
import config as c
import utils


def plot_dynamics(log, width):
    # Load variables
    pos = log['pos']
    est_pos = log['est_pos']

    ball_pos = log['ball_pos']
    est_ball_pos = log['est_ball_pos']

    causes_int = log['causes_int']
    causes_ext = log['causes_ext']

    est_vel = log['est_vel']
    F_m = log['F_m']

    L_ext = log['L_ext']
    L_softmax = np.zeros((len(L_ext), 2))
    for s, step in enumerate(L_ext):
        L_softmax[s] = utils.softmax(step * c.gain_evidence, c.w_bmc)

    # Initialize plots
    fig, axs = plt.subplots(1, figsize=(22, 16))
    axs.set_xlim(0, c.n_steps)
    axs.set_ylim(-0.05, 1.05)
    axs.set_ylabel('Prob.')
    axs.set_xlabel('Time')

    axs.plot(np.repeat(L_softmax[:, 0][::c.n_tau], c.n_tau),
             lw=width, label=r'$v_{t1}$', color='r')
    axs.plot(np.repeat(L_softmax[:, 1][::c.n_tau], c.n_tau),
             lw=width, label=r'$v_{t2}$', color='g')

    axs.legend()

    plt.tight_layout()
    fig.savefig('plots/dynamics_' + c.log_name, bbox_inches='tight')

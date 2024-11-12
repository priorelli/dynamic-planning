import numpy as np
import matplotlib.pyplot as plt
import config as c


def plot_dynamics(log, width):
    # Load variables
    angles = log['angles'][:150]
    est_angles = log['est_angles'][:150]

    pos = log['pos'][:150, -1]
    est_pos = log['est_pos'][:150, -1]

    ball_pos = log['ball_pos'][:150]
    est_ball_pos = log['est_ball_pos'][:150, -1]

    square_pos = log['square_pos'][:150]
    est_square_pos = log['est_square_pos'][:150, -1]

    causes = log['causes'][:150]

    E_i = log['E_i'][:150]
    F_m = log['F_m'][:150]

    mu_t = log['mu_t'][:150]
    o_t = log['o_t'][:150]

    # Initialize plots
    fig, axs = plt.subplots(4, figsize=(25, 21))

    # Draw angles
    axs[0].plot(np.linalg.norm(pos - est_pos, axis=1), 'b', lw=width,
                label=r'$||g_v(\mu_{x,0}) - x_h||$', ls='--')
    axs[0].plot(np.linalg.norm(ball_pos - est_ball_pos, axis=1), 'r', lw=width,
                label=r'$||g_v(\mu_{x,b}) - x_b||$', ls='--')
    axs[0].plot(np.linalg.norm(square_pos - est_square_pos, axis=1), 'g',
                lw=width, label=r'$||g_v(\mu_{x,s}) - x_s||$', ls='--')

    axs[0].plot(np.linalg.norm(pos - ball_pos, axis=1), 'darkred', lw=width,
                label=r'$||x_h - x_t||$')
    axs[0].plot(np.linalg.norm(pos - square_pos, axis=1), 'darkgreen', lw=width,
                label=r'$||x_h - x_s||$')

    axs[1].plot(causes.T[0], 'r', lw=width,
                label=r'$\mu_{v,ball}$')
    axs[1].plot(causes.T[1], 'g', lw=width,
                label=r'$\mu_{v,square}$')

    axs[2].plot(E_i[:, 0, 0], 'r', lw=width, ls='--',
                label=r'$e_{i,ball}$')
    axs[2].plot(E_i[:, 1, 0], 'g', lw=width, ls='--',
                label=r'$e_{i,square}$')

    axs[2].plot(F_m[:, 0, 0], 'darkred', lw=width,
                label=r'$f_{ball}$')
    axs[2].plot(F_m[:, 1, 0], 'darkgreen', lw=width,
                label=r'$f_{square}$')

    axs[3].plot(mu_t.T, 'olive', lw=width,
                label=r'$\mu_t$')
    axs[3].plot(o_t.T, 'teal', lw=width,
                label=r'$o_t$')

    for ax in axs:
        ax.legend(loc='lower right')
    axs[0].legend(loc='upper right', ncol=2)

    axs[0].set_xticklabels('')
    axs[1].set_xticklabels('')
    axs[2].set_xticklabels('')
    axs[3].set_xlabel('Time')

    axs[0].set_ylabel('L2 Norm (px)')
    axs[1].set_ylabel('a.u.')
    axs[2].set_ylabel('a.u.')
    axs[3].set_ylabel('a.u.')

    plt.tight_layout()
    fig.savefig('plots/dynamics_' + c.log_name, bbox_inches='tight')

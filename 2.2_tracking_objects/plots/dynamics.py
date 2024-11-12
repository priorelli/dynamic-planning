import numpy as np
import matplotlib.pyplot as plt
import config as c


def plot_dynamics(log, width):
    # Load variables
    angles = log['angles']
    est_angles = log['est_angles']

    mu_first = log['mu_first']
    mu_dot = log['mu_dot']

    ball_angle = log['ball_angle']
    est_ball_angle = log['est_ball_angle']

    grad_eps_o_v = log['grad_eps_o_v']
    grad_eps_x_v = log['grad_eps_x_v']

    # Initialize plots
    fig, axs = plt.subplots(3, figsize=(25, 17))

    # Draw angles
    axs[0].plot(est_angles.T[0], 'c', lw=width,
                label=r'$\mu_x$')
    axs[0].plot(angles.T[0], 'b', ls='--', lw=width,
                label=r'$x$')

    axs[0].plot(est_ball_angle.T[0], 'm', lw=width,
                label=r'$\mu_v$')
    axs[0].plot(ball_angle.T[0], 'r', ls='--', lw=width,
                label=r'$v$')

    axs[1].plot(mu_first.T[0], 'c', lw=width,
                label=r'$\mu_x^\prime$')
    axs[1].plot(mu_dot.T[0], 'b', lw=width, ls='--',
                label=r'$\dot{\mu}_x$')

    axs[2].plot(grad_eps_o_v.T[0], 'orange', lw=width,
                label=r'$\partial_{v} g_v^T \pi_{o,v} \varepsilon_{o,v}$')
    axs[2].plot(grad_eps_x_v.T[0], 'g', lw=width,
                label=r'$\partial_{v} f^T \pi_x \varepsilon_x$')

    for ax in axs:
        ax.legend(loc='lower right')

    axs[0].set_xticklabels('')
    axs[1].set_xticklabels('')
    axs[2].set_xlabel('Time')

    axs[0].set_ylabel('Angle')
    axs[1].set_ylabel('Ang. velocity')

    axs[2].set_ylim(-35, 10)

    plt.tight_layout()
    fig.savefig('plots/dynamics_' + c.log_name, bbox_inches='tight')

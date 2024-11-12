import numpy as np
import matplotlib.pyplot as plt
import config as c


def plot_dynamics(log, width):
    # Load variables
    angles = log['angles']
    est_angles = log['est_angles']

    e_x = log['e_x']

    mu_first = log['mu_first']
    mu_dot = log['mu_dot']

    grad_eps_o = log['grad_eps_o']
    grad_eps_x = log['grad_eps_x']

    rho = np.full(c.n_steps, 120)

    # Initialize plots
    fig, axs = plt.subplots(3, figsize=(25, 17))

    # Draw angles
    axs[0].plot(est_angles.T[0], 'c', lw=width,
                label=r'$\mu_x$')
    axs[0].plot(angles.T[0], 'b', ls='--', lw=width,
                label=r'$x$')
    axs[0].plot(rho, 'r', ls='--', lw=width,
                label=r'$\rho$')

    axs[1].plot(mu_first.T[0], 'c', lw=width,
                label=r'$\mu_x^\prime$')
    axs[1].plot(mu_dot.T[0], 'b', lw=width, ls='--',
                label=r'$\dot{\mu}_x$')

    axs[2].plot(mu_first.T[0], 'c', lw=width,
                label=r'$\mu_x^\prime$')
    axs[2].plot(grad_eps_o.T[0], 'orange', lw=width,
                label=r'$\partial_{x} g_p^T \pi_{o,p} \varepsilon_{o,p}$')
    axs[2].plot(grad_eps_x.T[0], 'g', lw=width,
                label=r'$\partial_{x} f^T \pi_x \varepsilon_x$')
    axs[2].plot(-e_x.T[0], 'violet', lw=width,
                label=r'$-\pi_x \varepsilon_x$')

    for ax in axs:
        ax.legend(loc='upper right')
    axs[2].legend(loc='lower right')

    axs[0].set_xticklabels('')
    axs[1].set_xticklabels('')
    axs[2].set_xlabel('Time')

    axs[0].set_ylabel('Angle')
    axs[1].set_ylabel('Ang. velocity')

    plt.tight_layout()
    fig.savefig('plots/dynamics_' + c.log_name, bbox_inches='tight')

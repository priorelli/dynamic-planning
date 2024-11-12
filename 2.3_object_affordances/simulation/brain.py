import torch
import numpy as np
import utils
import config as c
from simulation.unit import Unit, Obs


# Get proprioceptive prediction
def g_prop(x):
    return x[0, 0]


# Get visual prediction
def g_vis(x):
    lengths_norm = utils.normalize(c.lengths, c.norm_cart)
    return torch.stack([utils.kinematics(x_obj, lengths_norm, c.norm_polar)
                        for x_obj in x[0]])


# Get theta dynamics
def f_theta(x, v):
    # Reach ball
    i_b = torch.stack([x[1], x[1], x[2]], -2)  # eq. 12

    # Reach square
    i_s = torch.stack([x[2], x[1], x[2]], -2)  # eq. 12

    # Compute potential trajectories
    F_m = torch.stack([(i - x) * c.lambda_theta for i in [i_b, i_s]])  # eq. 14

    # Compute average trajectory
    f = torch.tensordot(v, F_m, dims=([0], [0]))  # eq. 15

    return f, F_m


# Define brain class
class Brain:
    def __init__(self):
        # Initialize units
        self.theta = Unit(dim=(c.n_orders, c.n_objects, c.n_joints),
                          inputs=utils.normalize(c.eta_x_theta, c.norm_polar),
                          eta_v=c.eta_v_theta, pi_eta_x=c.pi_eta_x_theta,
                          pi_eta_v=c.pi_eta_v_theta, pi_x=c.pi_x_theta,
                          lr=c.lr_theta, f=f_theta)

        self.prop = Obs(dim=c.n_joints, inputs=[self.theta],
                        pi_o=c.pi_prop, g=g_prop, lr=c.lr_a)

        self.vis = Obs(dim=(c.n_objects, 2), inputs=[self.theta],
                       pi_o=c.pi_vis, g=g_vis)

        # Initialize tactile belief
        self.tact = 0.0
        self.beta = 0.0

        self.units = [self.theta, self.prop, self.vis]

    # Initialize beliefs
    def init_belief(self, angles):
        theta_start = angles if c.x_theta_start is None else c.x_theta_start
        theta_start_norm = utils.normalize(theta_start, c.norm_polar)
        self.theta.x[0, :] = torch.tensor(theta_start_norm)

        self.theta.v = torch.tensor(c.eta_v_theta)

    # Run an inference step
    def inference_step(self, O):
        # Set observations
        self.prop.o = torch.tensor(O[0])
        self.vis.o = torch.tensor(O[1])

        # Perform message passing step
        for unit in self.units:
            unit.step()

        # Update all units
        for unit in self.units:
            unit.update(c.dt)

        # Update tactile belief
        e_tact = (O[2] - self.tact) * c.pi_tact
        self.tact += c.dt * e_tact
        self.tact = np.clip(self.tact, 0, 1)

        # Set prior over hidden causes
        self.beta = 1 / (1 + np.exp(- 10 * (self.beta + self.tact - 0.5)))
        self.theta.eta_v[0] = 1 - self.beta
        self.theta.eta_v[1] = self.beta

        return utils.denormalize(self.prop.actions, c.norm_polar)

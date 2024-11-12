import torch
import utils
import config as c
from simulation.unit import Unit, Obs


# Get proprioceptive prediction
def g_prop(x):
    return x[0]


# Get visual prediction
def g_vis(x):
    return x


# Get theta dynamics
def f_theta(x, v):
    e_i = (v - x) * c.lambda_theta  # eq. 8

    return e_i


# Define brain class
class Brain:
    def __init__(self):
        # Initialize units
        self.theta = Unit(dim=(c.n_orders, c.n_joints),
                          inputs=utils.normalize(c.eta_x_theta, c.norm_polar),
                          eta_v=utils.normalize(c.eta_v_theta, c.norm_polar),
                          pi_eta_x=c.pi_eta_x_theta,
                          pi_eta_v=c.pi_eta_v_theta, pi_x=c.pi_x_theta,
                          lr=c.lr_theta, f=f_theta)

        self.prop = Obs(dim=c.n_joints, inputs=[self.theta], kind='x',
                        pi_o=c.pi_prop, g=g_prop, lr=c.lr_a)

        self.vis = Obs(dim=c.n_joints, inputs=[self.theta], kind='v',
                       pi_o=c.pi_vis, g=g_vis)

        self.units = [self.theta, self.prop, self.vis]

    # Initialize beliefs
    def init_belief(self, angles):
        theta_start = angles if c.x_theta_start is None else c.x_theta_start
        theta_start_norm = utils.normalize(theta_start, c.norm_polar)
        self.theta.x[0] = torch.tensor(theta_start_norm)

        self.theta.v = torch.tensor(utils.normalize(angles, c.norm_polar))

    # Run an inference step
    def inference_step(self, O):
        # Set observations
        self.prop.o = torch.tensor(O[0][0])
        self.vis.o = torch.tensor(O[1][0])

        # Perform message passing step
        for unit in self.units:
            unit.step()

        # Update all units
        for unit in self.units:
            unit.update(c.dt)

        return (utils.denormalize(self.prop.actions, c.norm_polar),
                self.theta.x_dot)

import torch
import utils
import config as c
from simulation.unit import Unit, Obs
from simulation.discrete import Discrete


# Get proprioceptive prediction
def g_prop(x):
    return x[0, 0]


# Get visual prediction
def g_vis(x):
    return x


# Get extrinsic prediction
def g_ext(x):
    lengths_norm = utils.normalize(c.lengths, c.norm_cart)
    return torch.stack([utils.kinematics(x_obj, lengths_norm, c.norm_polar)
                        for x_obj in x])


# Stay
def f_0(x):
    return x * 0.0


# Open hand
def f_o(x):
    angle_open = torch.tensor(utils.normalize(c.angle_open, c.norm_polar),
                              dtype=torch.float32)

    i_o = torch.stack([torch.stack([*x[0, :4], *angle_open], -1), x[1]], -2)

    return (i_o - x) * c.lambda_int


# Close hand
def f_c(x):
    angle_closed = torch.tensor(utils.normalize(c.angle_closed, c.norm_polar),
                                dtype=torch.float32)

    i_o = torch.stack([torch.stack([*x[0, :4], *angle_closed], -1), x[1]], -2)

    return (i_o - x) * c.lambda_int


# Reach ball
def f_b(x):
    return (torch.stack([x[1], x[1]], -2) - x) * c.lambda_ext


# Define brain class
class Brain:
    def __init__(self):
        # Initialize discrete
        self.discrete = Discrete()

        # Initialize units
        self.int = Unit(dim=(c.n_orders, c.n_objects, c.n_joints),
                        inputs=utils.normalize(c.eta_x_int, c.norm_polar),
                        v=self.discrete.o_int, L=self.discrete.L_int,
                        pi_eta_x=c.pi_eta_x_int, p_x=c.p_x_int,
                        pi_x=c.pi_x_int, lr=c.lr_int, F_m=[f_o, f_c])

        self.ext = Unit(dim=(c.n_orders, c.n_objects, 2),
                        inputs=[self.int],
                        v=self.discrete.o_ext, L=self.discrete.L_ext,
                        pi_eta_x=c.pi_eta_x_ext, p_x=c.p_x_ext,
                        pi_x=c.pi_x_ext, lr=c.lr_ext,
                        F_m=[f_0, f_b], g=g_ext)

        self.prop = Obs(dim=c.n_joints, inputs=[self.int],
                        pi_o=c.pi_prop, g=g_prop, lr=c.lr_a)

        self.vis = Obs(dim=(c.n_orders, c.n_objects, 2), inputs=[self.ext],
                       pi_o=c.pi_vis, g=g_vis)

        self.units = [self.int, self.ext, self.prop, self.vis]

    # Initialize beliefs
    def init_belief(self, angles, pos):
        int_start = angles if c.x_int_start is None else c.x_int_start
        int_start_norm = utils.normalize(int_start, c.norm_polar)
        self.int.x[0, :] = torch.tensor(int_start_norm)

        ext_start = pos if c.x_int_start is None else g_ext(self.int.x[0])[0]
        ext_start_norm = utils.normalize(ext_start, c.norm_cart)
        self.ext.x[0, :] = torch.tensor(ext_start_norm)

    # Run an inference step
    def inference_step(self, O, step):
        # Run discrete step
        if (step + 1) % c.n_tau == 0:
            self.discrete.step(O[2])

        # Set observations
        self.prop.o = torch.tensor(O[0])
        self.vis.o = torch.tensor(O[1])

        # Perform message passing step
        for unit in self.units:
            unit.step()

        # Update all units
        for unit in self.units:
            unit.update(c.dt)

        return utils.denormalize(self.prop.actions, c.norm_polar)

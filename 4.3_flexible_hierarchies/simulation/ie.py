import torch
import numpy as np
import utils
import config as c
from simulation.unit import Unit, Obs


# Get proprioceptive prediction
def g_prop(x):
    return x[0, 0, 0]


# Get visual prediction
def g_vis(x):
    return x[:, :, :2]


# Get extrinsic prediction
def g_ext(int_, ext):
    return torch.stack([utils.kinematics(*int_obj, *ext_obj, c.norm_polar)
                        for int_obj, ext_obj in zip(int_, ext)])


# Define IE class
class IE:
    def __init__(self, input_ext, input_int, f_0, f_t, f_b,
                 o_int, L_int, o_ext, L_ext):
        # Set precisions and learning rates
        lr_int = np.full((c.n_orders, c.n_objects, 2), c.lr_int)
        lr_int[:, :, 1] = c.lr_len

        pi_eta_x_ext = np.full((c.n_objects, 3), c.pi_eta_x_ext)
        pi_eta_x_ext[:, -1] = c.pi_phi

        pi_vis = np.zeros((c.n_orders, c.n_objects, 2))
        pi_vis[:, 0] = c.pi_vis

        # Initialize units
        self.int = Unit(dim=(c.n_orders, c.n_objects, 2),
                        inputs=[input_int[0], input_int[1]],
                        v=o_int, L=L_int,
                        pi_eta_x=c.pi_eta_x_int, p_x=c.p_x_int,
                        pi_x=c.pi_x_int, lr=lr_int, F_m=[f_0],
                        lmbda=c.lambda_int)

        self.ext = Unit(dim=(c.n_orders, c.n_objects, 3),
                        inputs=[self.int, input_ext],
                        v=o_ext, L=L_ext,
                        pi_eta_x=pi_eta_x_ext, p_x=c.p_x_ext,
                        pi_x=c.pi_x_ext, lr=c.lr_ext, F_m=[f_0, f_0, f_0],
                        lmbda=c.lambda_ext, g=g_ext)

        self.prop = Obs(dim=1, inputs=[self.int],
                        pi_o=c.pi_prop, g=g_prop, lr=c.lr_a)

        self.vis = Obs(dim=(c.n_orders, c.n_objects, 2), inputs=[self.ext],
                       pi_o=pi_vis, g=g_vis)

        self.units = [self.int, self.ext, self.prop, self.vis]

    # Update IE module
    def update(self):
        # Perform message passing step
        for unit in self.units:
            unit.step()

        # Update all units
        for u, unit in enumerate(self.units):
            unit.update(c.dt)

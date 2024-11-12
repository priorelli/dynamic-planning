import torch
import numpy as np
import config as c
import utils
from simulation.unit import Unit
from simulation.ie import IE


# Stay
def f_0(x, lmbda):
    return x * 0.0


# Reach angle
def f_r(x, lmbda):
    angle = torch.tensor(utils.normalize(150, c.norm_polar),
                         dtype=torch.float32)

    return (torch.stack([angle, x[1]], -1) - x) * lmbda


# Reach point
def f_p(x, lmbda):
    pos = torch.tensor(utils.normalize(c.ball_pos, c.norm_cart),
                       dtype=torch.float32)

    return (torch.stack([*pos, x[2]], -1) - x) * lmbda


# Define brain class
class Brain:
    def __init__(self, idxs):
        # Initialize inputs
        ext0 = Unit(dim=(c.n_orders, 3),
                    inputs=np.zeros(3),
                    pi_eta_x=0.0, pi_x=0.0,
                    lr=0.0, f=None, lmbda=0.0)

        input_int = np.array([utils.normalize(c.eta_x_int, c.norm_polar),
                              utils.normalize(c.lengths, c.norm_cart)])

        # Initialize modules
        self.modules = []
        for j in range(c.n_joints):
            input_ext = self.modules[idxs[j]].ext \
                if idxs[j] >= 0 else ext0

            ie = IE(input_ext, input_int.T[j], f_0)
            self.modules.append(ie)

        # Link dynamics
        self.modules[-3].ext.f = f_p

    # Initialize beliefs
    def init_belief(self, angles, phi, pos):
        int_start = angles if c.x_int_start is None else c.x_int_start

        int_start_norm = utils.normalize(int_start, c.norm_polar)
        phi_norm = utils.normalize(phi, c.norm_polar)
        lengths_norm = utils.normalize(c.lengths, c.norm_cart)
        pos_norm = utils.normalize(pos, c.norm_cart)

        for m, module in enumerate(self.modules):
            module.int.x[0, 0] = torch.tensor(int_start_norm[m])
            module.int.x[0, 1] = torch.tensor(lengths_norm[m])
            module.ext.x[0, :2] = torch.tensor(pos_norm[m])
            module.ext.x[0, 2] = torch.tensor(phi_norm[m])

    # Run an inference step
    def inference_step(self, O):
        # Set observations
        for m, module in enumerate(self.modules):
            module.prop.o = torch.tensor(O[0][m])
            module.vis.o = torch.tensor(O[1][m])

        # Update modules
        for module in self.modules:
            module.update()

        actions = np.array([module.prop.actions for module in self.modules])

        return utils.denormalize(actions, c.norm_polar)

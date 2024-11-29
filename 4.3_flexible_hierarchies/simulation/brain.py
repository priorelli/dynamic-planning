import torch
import numpy as np
import config as c
import utils
from simulation.unit import Unit
from simulation.discrete import Discrete
from simulation.ie import IE


# Stay
def f_0(x, lmbda):
    return x * 0.0


# Keep tool angle
def f_int(x, lmbda):
    return (torch.stack([x[0], x[1], x[1]], -2) - x) * lmbda


# Reach tool
def f_t(x, lmbda):
    return (torch.stack([x[1], x[1], x[2]], -2) - x) * lmbda


# Reach ball
def f_b(x, lmbda):
    return (torch.stack([x[2], x[2], x[2]], -2) - x) * lmbda


# Define brain class
class Brain:
    def __init__(self, idxs):
        # Initialize inputs
        ext0 = Unit(dim=(c.n_orders, c.n_objects, 3),
                    inputs=np.zeros(3), v=np.zeros(1), L=None,
                    pi_eta_x=0.0, p_x=0.0, pi_x=0.0,
                    lr=0.0, F_m=None, lmbda=0.0)

        input_int = [utils.normalize([*c.eta_x_int, c.eta_x_int[-1]],
                                     c.norm_polar),
                     utils.normalize([*c.lengths, c.tool_length],
                                     c.norm_cart)]
        input_int = np.array(input_int)

        # Initialize discrete
        self.discrete = Discrete()

        # Initialize modules
        self.modules = []
        for j in range(c.n_joints):
            input_ext = self.modules[idxs[j]].ext if idxs[j] >= 0 else ext0

            ie = IE(input_ext, input_int.T[j], f_0, f_t, f_b,
                    self.discrete.o_int, self.discrete.L_int,
                    self.discrete.o_ext, self.discrete.L_ext[j])
            self.modules.append(ie)

        # Initialize tool module
        ie = IE(self.modules[-1].ext, input_int.T[-1], f_0, f_t, f_b,
                self.discrete.o_int, self.discrete.L_int,
                self.discrete.o_ext, self.discrete.L_ext[-1])
        self.modules.append(ie)

        # Link dynamics
        self.modules[-1].int.F_m[0] = f_int  # keep tool angle

        self.modules[-2].ext.F_m[1] = f_t  # reach tool

        self.modules[-2].ext.F_m[2] = f_b  # reach ball (hand level)
        self.modules[-1].ext.F_m[2] = f_b  # reach ball (virtual level)

        # Link objects to last joint
        self.modules[-2].vis.pi_o[:, 1] = c.pi_vis_obj  # tool's origin
        self.modules[-1].vis.pi_o[:, 1:] = c.pi_vis_obj  # tool's end/ball

        # Remove virtual level for actual body configuration
        self.modules[-1].prop.pi_o = 0
        self.modules[-1].vis.pi_o[:, 0] = 0
        self.modules[-1].ext.pi_eta_x[0] = 0

    # Initialize beliefs
    def init_belief(self, angles, phi, pos):
        int_start = angles if c.x_int_start is None else c.x_int_start

        int_start_norm = utils.normalize(
            [*int_start, int_start[-1]], c.norm_polar)
        phi_norm = utils.normalize([*phi, phi[-1]], c.norm_polar)
        lengths_norm = utils.normalize(
            [*c.lengths, c.tool_length], c.norm_cart)
        pos_norm = utils.normalize([*pos, pos[-1]], c.norm_cart)

        for m, module in enumerate(self.modules):
            module.int.x[0, :, 0] = torch.tensor(int_start_norm[m])
            module.int.x[0, :, 1] = torch.tensor(lengths_norm[m])
            module.ext.x[0, :, :2] = torch.tensor(pos_norm[m])
            module.ext.x[0, :, 2] = torch.tensor(phi_norm[m])

    # Run an inference step
    def inference_step(self, O, step):
        # Run discrete step
        if (step + 1) % c.n_tau == 0:
            self.discrete.step(O[2])

        # Set observations
        for m, module in enumerate(self.modules):
            module.prop.o = torch.tensor(O[0][m])
            module.vis.o = torch.tensor(O[1][:, :, m])

        # Update modules
        for module in self.modules:
            module.update()

        actions = np.array([module.prop.actions for
                            module in self.modules[:-1]])

        return utils.denormalize(actions, c.norm_polar)

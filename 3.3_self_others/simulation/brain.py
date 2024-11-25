import torch
import numpy as np
import config as c
import utils
from simulation.unit import Unit
from simulation.ie import IE


# Stay
def f_0(x, lmbda):
    return x * 0.0


# Reach other
def f_r(x, lmbda):
    return (torch.stack([x[1], x[1], x[2]], -2) - x) * lmbda


# Define brain class
class Brain:
    def __init__(self, idxs_1st, idxs_2nd):
        ref = utils.normalize(np.array(
            c.offset_2nd) - np.array(c.offset_1st), c.norm_cart)

        #######
        # 1st #
        #######

        # Initialize inputs
        ext0_1st = Unit(dim=(c.n_orders, c.n_objects, 3),
                        inputs=np.array([*c.offset_1st, 0]),
                        pi_eta_x=0.0, pi_x=0.0,
                        lr=0.0, f=None, lmbda=0.0)
        ext0_1st.x[0, 2, 0] = ref[0]
        ext0_1st.x[0, 2, 1] = ref[1]

        input_int_1st = [utils.normalize(c.eta_x_int, c.norm_polar),
                         utils.normalize([*c.lengths_1st, 0, 0], c.norm_cart)]
        input_int_1st = np.array(input_int_1st)

        # Initialize modules
        self.modules_1st = []
        for j in range(c.n_joints_2nd):
            input_ext_1st = self.modules_1st[idxs_2nd[j]].ext \
                if idxs_2nd[j] >= 0 else ext0_1st

            ie_1st = IE(input_ext_1st, input_int_1st.T[j], f_0)
            self.modules_1st.append(ie_1st)

        # Set dynamics
        self.modules_1st[-3].ext.f = f_r  # hand level
        for module in self.modules_1st:
            module.int.f = f_r

        # 1st agent has only 3 dof
        self.modules_1st[-1].vis.pi_o[:2] = 0.0
        self.modules_1st[-1].ext.pi_eta_x[:2] = 0.0
        self.modules_1st[-1].prop.pi_o = 0.0
        self.modules_1st[-2].vis.pi_o[:2] = 0.0
        self.modules_1st[-2].ext.pi_eta_x[:2] = 0.0
        self.modules_1st[-2].prop.pi_o = 0.0

        # Set observation of 2nd agent for hand level
        self.modules_1st[-3].vis.pi_o[1] = c.pi_vis_obj

        #######
        # 2nd #
        #######

        # Initialize inputs
        ext0_2nd = Unit(dim=(c.n_orders, c.n_objects, 3),
                        inputs=np.array([*c.offset_2nd, 0]),
                        pi_eta_x=0.0, pi_x=0.0,
                        lr=0.0, f=None, lmbda=0.0)
        ext0_2nd.x[0, 2, 0] = -ref[0]
        ext0_2nd.x[0, 2, 1] = -ref[1]

        input_int_2nd = [utils.normalize(c.eta_x_int, c.norm_polar),
                         utils.normalize(c.lengths_2nd, c.norm_cart)]
        input_int_2nd = np.array(input_int_2nd)

        # Initialize modules
        self.modules_2nd = []
        for j in range(c.n_joints_2nd):
            input_ext_2nd = self.modules_2nd[idxs_2nd[j]].ext \
                if idxs_2nd[j] >= 0 else ext0_2nd

            ie_2nd = IE(input_ext_2nd, input_int_2nd.T[j], f_0)
            self.modules_2nd.append(ie_2nd)

        # Set dynamics
        self.modules_2nd[-1].ext.f = f_r  # hand level
        for module in self.modules_2nd:
            module.int.f = f_r

        # 1st agent has only 3 dof
        self.modules_2nd[-1].vis.pi_o[2] = 0.0
        self.modules_2nd[-1].ext.pi_eta_x[2] = 0.0
        self.modules_2nd[-2].vis.pi_o[2] = 0.0
        self.modules_2nd[-2].ext.pi_eta_x[2] = 0.0

        # Set observation of 1st agent for hand level
        self.modules_2nd[-1].vis.pi_o[1] = c.pi_vis_obj

    # Initialize beliefs
    def init_belief(self, angles_1st, phi_1st, pos_1st, pos_2nd_1st,
                    angles_2nd, phi_2nd, pos_2nd, pos_1st_2nd):
        int_start_norm_1st = utils.normalize([*angles_1st, 0, 0], c.norm_polar)
        int_start_norm_2nd = utils.normalize(angles_2nd, c.norm_polar)

        phi_norm_1st = utils.normalize([*phi_1st, 0, 0], c.norm_polar)
        phi_norm_2nd = utils.normalize(phi_2nd, c.norm_polar)

        lengths_norm_1st = utils.normalize([*c.lengths_1st, 0, 0], c.norm_cart)
        lengths_norm_2nd = utils.normalize(c.lengths_2nd, c.norm_cart)

        pos_norm_1st = utils.normalize([*pos_1st, (0, 0), (0, 0)], c.norm_cart)
        pos_norm_2nd_1st = utils.normalize(pos_2nd_1st, c.norm_cart)

        pos_norm_2nd = utils.normalize(pos_2nd, c.norm_cart)
        pos_norm_1st_2nd = utils.normalize([*pos_1st_2nd, (0, 0), (0, 0)],
                                            c.norm_cart)

        # Initialize 1st agent
        for m, module in enumerate(self.modules_1st):
            module.int.x[0, :2, 0] = torch.tensor(int_start_norm_1st[m])
            module.int.x[0, :2, 1] = torch.tensor(lengths_norm_1st[m])
            module.ext.x[0, :2, :2] = torch.tensor(pos_norm_1st[m])
            module.ext.x[0, :2, 2] = torch.tensor(phi_norm_1st[m])

            module.int.x[0, 2, 0] = torch.tensor(int_start_norm_2nd[m])
            module.int.x[0, 2, 1] = torch.tensor(lengths_norm_2nd[m])
            module.ext.x[0, 2, :2] = torch.tensor(pos_norm_2nd_1st[m])
            module.ext.x[0, 2, 2] = torch.tensor(phi_norm_2nd[m])

        # Initialize 2nd agent
        for m, module in enumerate(self.modules_2nd):
            module.int.x[0, :2, 0] = torch.tensor(int_start_norm_2nd[m])
            module.int.x[0, :2, 1] = torch.tensor(lengths_norm_2nd[m])
            module.ext.x[0, :2, :2] = torch.tensor(pos_norm_2nd[m])
            module.ext.x[0, :2, 2] = torch.tensor(phi_norm_2nd[m])

            module.int.x[0, 2, 0] = torch.tensor(int_start_norm_1st[m])
            module.int.x[0, 2, 1] = torch.tensor(lengths_norm_1st[m])
            module.ext.x[0, 2, :2] = torch.tensor(pos_norm_1st_2nd[m])
            module.ext.x[0, 2, 2] = torch.tensor(phi_norm_1st[m])

    # Run an inference step
    def inference_step(self, O):
        # Set observations
        for m, module in enumerate(self.modules_1st):
            module.prop.o = torch.tensor(O[0][m])
            module.vis.o = torch.tensor(O[1][:, m])

        # Set observations
        for m, module in enumerate(self.modules_2nd):
            module.prop.o = torch.tensor(O[2][m])
            module.vis.o = torch.tensor(O[3][:, m])

        # Step modules
        for module in [*self.modules_1st, *self.modules_2nd]:
            module.step()

        # Update modules
        for module in [*self.modules_1st, *self.modules_2nd]:
            module.update()

        actions_1st = np.array([module.prop.actions for module
                                in self.modules_1st])
        actions_2nd = np.array([module.prop.actions for module
                                in self.modules_2nd])

        return (utils.denormalize(actions_1st, c.norm_polar),
                utils.denormalize(actions_2nd, c.norm_polar))

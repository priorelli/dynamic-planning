import torch
import utils
import config as c
from simulation.unit import Unit, Obs


# Get proprioceptive prediction
def g_prop(x):
    return x[0, 0]


# Get visual prediction
def g_vis(x):
    return x[0]


# Get extrinsic prediction (eq. 20)
def g_ext(x):
    lengths_norm = utils.normalize(c.lengths, c.norm_cart)
    return torch.stack([utils.kinematics(x_obj, lengths_norm, c.norm_polar)
                        for x_obj in x])


# Get intrinsic dynamics (eq. 22)
def f_int(x, v):
    # Reach ball
    i_b = torch.stack([x[1], x[1], x[2]], -2)

    # Compute potential trajectories
    F_m = torch.stack([(i - x) * c.lambda_int for i in [i_b]])

    # Compute average trajectory
    f = torch.tensordot(v, F_m, dims=([0], [0]))

    return f


# Get extrinsic dynamics (eq. 22)
def f_ext(x, v):
    # Reach ball
    i_b = torch.stack([x[1], x[1], x[2]], -2)
    f_attr = (i_b - x) * c.lambda_ext

    # Avoid square
    f_rep = get_rep_force(x[0], x[2])

    return f_attr * v[0] + f_rep * v[1]


# Compute repulsive force
def get_rep_force(hand_pos, obstacle_pos):
    avoid_dist = c.square_size + 150
    q_star = utils.normalize(avoid_dist, c.norm_cart)
    error_r = obstacle_pos - hand_pos
    error_r_norm = torch.norm(error_r)

    if error_r_norm > q_star:
        rep_force = torch.zeros(2)
    else:
        rep_force = c.k_rep * (1 / q_star - 1 / error_r_norm) \
                        * (1 / error_r_norm ** 2) * (error_r / error_r_norm)

    return rep_force


# Define brain class
class Brain:
    def __init__(self):
        # Initialize units
        self.int = Unit(dim=(c.n_orders, c.n_objects, c.n_joints),
                        inputs=utils.normalize(c.eta_x_int, c.norm_polar),
                        eta_v=c.eta_v_int, pi_eta_x=c.pi_eta_x_int,
                        pi_eta_v=c.pi_eta_v_int, pi_x=c.pi_x_int,
                        lr=c.lr_int, f=f_int)

        self.ext = Unit(dim=(c.n_orders, c.n_objects, 2),
                        inputs=[self.int],
                        eta_v=c.eta_v_ext, pi_eta_x=c.pi_eta_x_ext,
                        pi_eta_v=c.pi_eta_v_ext, pi_x=c.pi_x_ext,
                        lr=c.lr_ext, f=f_ext, g=g_ext)

        self.prop = Obs(dim=c.n_joints, inputs=[self.int],
                        pi_o=c.pi_prop, g=g_prop, lr=c.lr_a)

        self.vis = Obs(dim=(c.n_objects, 2), inputs=[self.ext],
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
        self.ext.x[0, 2] = torch.tensor(utils.normalize(
            c.square_pos, c.norm_cart))

        self.int.v = torch.tensor(c.eta_v_int)
        self.ext.v = torch.tensor(c.eta_v_ext)

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

        return utils.denormalize(self.prop.actions, c.norm_polar)

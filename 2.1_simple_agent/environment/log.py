import numpy as np
import config as c
import utils


# Define log class
class Log:
    def __init__(self):
        # Initialize logs
        self.angles = np.zeros((c.n_steps, c.n_joints))
        self.est_angles = np.zeros_like(self.angles)

        self.pos = np.zeros((c.n_steps, c.n_joints + 1, 2))
        self.est_pos = np.zeros_like(self.pos)

        self.ball_pos = np.zeros((c.n_steps, 2))

        self.mu_first = np.zeros((c.n_steps, c.n_joints))
        self.mu_dot = np.zeros((c.n_steps, c.n_joints))

        self.action = np.zeros((c.n_steps, c.n_joints))
        self.e_x = np.zeros((c.n_steps, c.n_joints))

        self.grad_eps_o = np.zeros((c.n_steps, c.n_joints))
        self.grad_eps_x = np.zeros((c.n_steps, c.n_joints))

    # Track logs for each iteration
    def track(self, step, brain, body, objects, action, x_dot):
        self.angles[step] = body.get_angles()
        est_angles = brain.prop.predict().detach().numpy()
        self.est_angles[step] = utils.denormalize(est_angles, c.norm_polar)

        self.pos[step, 1:] = body.get_pos()
        self.est_pos[step] = body.get_poses(self.est_angles[step],
                                            c.lengths)[:, :2]

        self.ball_pos[step] = objects.ball.get_pos()

        self.mu_first[step] = utils.denormalize(
            brain.theta.x[1].detach().numpy(), c.norm_polar)
        self.mu_dot[step] = utils.denormalize(x_dot, c.norm_polar)

        self.action[step] = action

        self.grad_eps_x[step] = utils.denormalize(
            brain.theta.grad_x.detach().numpy(), c.norm_polar)
        self.grad_eps_o[step] = utils.denormalize(
            brain.theta.grad_o[0].detach().numpy(), c.norm_polar)
        self.e_x[step] = utils.denormalize(
            brain.theta.eps_x.detach().numpy(), c.norm_polar)

    # Save log to file
    def save_log(self):
        np.savez_compressed('simulation/log_{}'.format(c.log_name),
                            angles=self.angles,
                            est_angles=self.est_angles,
                            pos=self.pos,
                            est_pos=self.est_pos,
                            ball_pos=self.ball_pos,
                            mu_first=self.mu_first,
                            mu_dot=self.mu_dot,
                            action=self.action,
                            e_x=self.e_x,
                            grad_eps_x=self.grad_eps_x,
                            grad_eps_o=self.grad_eps_o)

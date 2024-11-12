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

        self.grasp_pos = np.zeros((c.n_steps, 2))
        self.est_grasp_pos = np.zeros_like(self.grasp_pos)

        self.ball_pos = np.zeros((c.n_steps, 2))
        self.est_ball_pos = np.zeros_like(self.ball_pos)

        self.causes_int = np.zeros((c.n_steps, 2))
        self.causes_ext = np.zeros((c.n_steps, 2))

        self.disc_actions = np.zeros((c.n_steps, 4))

    # Track logs for each iteration
    def track(self, step, brain, body, objects):
        self.angles[step] = body.get_angles()
        est_angles = brain.prop.predict().detach().numpy()
        self.est_angles[step] = utils.denormalize(est_angles, c.norm_polar)

        self.pos[step, 1:] = body.get_pos()
        self.est_pos[step] = body.get_poses(self.est_angles[step],
                                            c.lengths)[:, :2]

        self.grasp_pos[step] = body.get_grasp()
        est_grasp_pos = brain.vis.predict()[0, 0].detach().numpy()
        self.est_grasp_pos[step] = utils.denormalize(est_grasp_pos, c.norm_cart)

        self.ball_pos[step] = objects.ball.get_pos()
        est_ball_pos = brain.vis.predict()[0, 1].detach().numpy()
        self.est_ball_pos[step] = utils.denormalize(est_ball_pos, c.norm_cart)

        self.causes_int[step] = brain.discrete.o_int
        self.causes_ext[step] = brain.discrete.o_ext

        self.disc_actions[step] = brain.discrete.P_u

    # Save log to file
    def save_log(self):
        np.savez_compressed('simulation/log_{}'.format(c.log_name),
                            angles=self.angles,
                            est_angles=self.est_angles,
                            pos=self.pos,
                            est_pos=self.est_pos,
                            grasp_pos=self.grasp_pos,
                            est_grasp_pos=self.est_grasp_pos,
                            ball_pos=self.ball_pos,
                            est_ball_pos=self.est_ball_pos,
                            causes_int=self.causes_int,
                            causes_ext=self.causes_ext,
                            disc_actions=self.disc_actions)

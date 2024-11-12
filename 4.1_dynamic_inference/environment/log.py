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
        self.est_ball_pos = np.zeros_like(self.ball_pos)

        self.square_pos = np.zeros((c.n_steps, 2))
        self.est_square_pos = np.zeros_like(self.square_pos)

        self.causes_int = np.zeros((c.n_steps, 2))
        self.causes_ext = np.zeros((c.n_steps, 2))

        self.est_vel = np.zeros((c.n_steps, 2))
        self.F_m = np.zeros((c.n_steps, 2, 2))
        self.L_ext = np.zeros((c.n_steps, 2))

    # Track logs for each iteration
    def track(self, step, brain, body, objects):
        self.angles[step] = body.get_angles()
        est_angles = brain.prop.predict().detach().numpy()
        self.est_angles[step] = utils.denormalize(est_angles, c.norm_polar)

        self.pos[step, 1:] = body.get_pos()
        self.est_pos[step] = body.get_poses(self.est_angles[step],
                                            c.lengths)[:, :2]

        self.ball_pos[step] = objects.ball.get_pos()
        est_ball_pos = brain.vis.predict()[0, 1].detach().numpy()
        self.est_ball_pos[step] = utils.denormalize(est_ball_pos, c.norm_cart)

        self.square_pos[step] = objects.square.get_pos()
        est_square_pos = brain.vis.predict()[0, 2].detach().numpy()
        self.est_square_pos[step] = utils.denormalize(est_square_pos,
                                                      c.norm_cart)

        self.causes_int[step] = brain.int.v
        self.causes_ext[step] = brain.discrete.o_ext

        self.est_vel[step] = utils.denormalize(
            brain.ext.x[1, 0].detach().numpy(), c.norm_cart)
        self.F_m[step] = utils.denormalize(
            brain.ext.Preds_x[:, 0].detach().numpy(), c.norm_cart)
        self.L_ext[step] = brain.discrete.L_ext

    # Save log to file
    def save_log(self):
        np.savez_compressed('simulation/log_{}'.format(c.log_name),
                            angles=self.angles,
                            est_angles=self.est_angles,
                            pos=self.pos,
                            est_pos=self.est_pos,
                            ball_pos=self.ball_pos,
                            est_ball_pos=self.est_ball_pos,
                            square_pos=self.square_pos,
                            est_square_pos=self.est_square_pos,
                            causes_int=self.causes_int,
                            causes_ext=self.causes_ext,
                            est_vel=self.est_vel,
                            F_m=self.F_m,
                            L_ext=self.L_ext)

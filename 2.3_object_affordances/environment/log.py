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
        self.est_ball_pos = np.zeros_like(self.est_pos)

        self.square_pos = np.zeros((c.n_steps, 2))
        self.est_square_pos = np.zeros_like(self.est_pos)

        self.causes = np.zeros((c.n_steps, 2))

        self.E_i = np.zeros((c.n_steps, 2, c.n_joints))
        self.F_m = np.zeros_like(self.E_i)

        self.beta = np.zeros(c.n_steps)
        self.mu_t = np.zeros_like(self.beta)
        self.o_t = np.zeros_like(self.beta)

    # Track logs for each iteration
    def track(self, step, brain, body, objects, o_tact):
        self.angles[step] = body.get_angles()
        est_angles = brain.prop.predict().detach().numpy()
        self.est_angles[step] = utils.denormalize(est_angles, c.norm_polar)

        self.pos[step, 1:] = body.get_pos()
        self.est_pos[step] = body.get_poses(self.est_angles[step],
                                            c.lengths)[:, :2]

        self.ball_pos[step] = objects.ball.get_pos()
        est_ball_angles = utils.denormalize(
            brain.theta.x[0, 1].detach().numpy(), c.norm_polar)
        self.est_ball_pos[step] = body.get_poses(
            est_ball_angles, c.lengths)[:, :2]

        self.square_pos[step] = objects.square.get_pos()
        est_square_angles = utils.denormalize(
            brain.theta.x[0, 2].detach().numpy(), c.norm_polar)
        self.est_square_pos[step] = body.get_poses(
            est_square_angles, c.lengths)[:, :2]

        self.causes[step] = brain.theta.v.detach().numpy()

        self.E_i[step] = brain.theta.Preds_x
        self.F_m[step, 0] = self.causes[step, 0] * brain.theta.Preds_x[0]
        self.F_m[step, 1] = self.causes[step, 1] * brain.theta.Preds_x[1]

        self.mu_t[step] = brain.tact
        self.o_t[step] = o_tact

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
                            causes=self.causes,
                            E_i=self.E_i,
                            F_m=self.F_m,
                            mu_t=self.mu_t,
                            o_t=self.o_t)

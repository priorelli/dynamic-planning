import numpy as np
import utils
import config as c


# Define log class
class Log:
    def __init__(self):
        # Initialize logs
        self.angles = np.zeros((c.n_steps, c.n_joints))
        self.est_angles = np.zeros_like(self.angles)

        self.pos = np.zeros((c.n_steps, c.n_joints + 1, 2))
        self.est_pos = np.zeros_like(self.pos)

        self.lengths = np.zeros((c.n_steps, c.n_joints))
        self.est_lengths = np.zeros_like(self.lengths)

        self.tool_pos = np.zeros((c.n_steps, 2, 2))
        self.est_tool_pos = np.zeros((c.n_steps, c.n_joints + 2, 2))

        self.ball_pos = np.zeros((c.n_steps, 2))
        self.est_ball_pos = np.zeros_like(self.est_tool_pos)

        self.causes_int = np.zeros((c.n_steps, 1))
        self.causes_ext = np.zeros((c.n_steps, 3))

        self.disc_actions = np.zeros((c.n_steps, 3))

        self.est_vel = np.zeros((c.n_steps, 2))
        self.F_m = np.zeros((c.n_steps, 3, 2))
        self.L_ext = np.zeros((c.n_steps, c.n_joints + 1, 3))

    # Track logs for each iteration
    def track(self, step, brain, body, objects):
        self.angles[step] = body.get_angles()
        est_angles = np.array([module.prop.predict().detach().numpy()
                               for module in brain.modules[:-1]])
        self.est_angles[step] = utils.denormalize(est_angles, c.norm_polar)

        self.pos[step, 1:] = body.get_pos()
        est_pos = np.array([module.vis.predict()[0, 0].detach().numpy()
                            for module in brain.modules[:-1]])
        self.est_pos[step, 1:] = utils.denormalize(est_pos, c.norm_cart)

        self.lengths[step] = c.lengths
        est_lengths = np.array([module.int.x[0, 0, 1].detach().numpy()
                                for module in brain.modules[:-1]])
        self.est_lengths[step] = utils.denormalize(est_lengths, c.norm_cart)

        self.tool_pos[step] = [objects.tool.get_pos(),
                               objects.tool.get_end()]
        est_tool_pos = np.array([module.vis.predict()[0, 1].detach().numpy()
                                 for module in brain.modules])
        self.est_tool_pos[step, 1:] = utils.denormalize(
            est_tool_pos, c.norm_cart)

        self.ball_pos[step] = objects.ball.get_pos()
        est_ball_pos = np.array([module.vis.predict()[0, 2].detach().numpy()
                                 for module in brain.modules])
        self.est_ball_pos[step, 1:] = utils.denormalize(
            est_ball_pos, c.norm_cart)

        self.causes_int[step] = brain.discrete.o_int
        self.causes_ext[step] = brain.discrete.o_ext

        self.disc_actions[step] = brain.discrete.P_u

        self.est_vel[step] = utils.denormalize(
            brain.modules[-2].ext.x[1, 0, :2].detach().numpy(), c.norm_cart)
        self.F_m[step] = utils.denormalize(
            brain.modules[-2].ext.Preds_x[:, 0, :2].detach().numpy(),
            c.norm_cart)
        self.L_ext[step] = brain.discrete.L_ext

    # Save log to file
    def save_log(self):
        np.savez_compressed('simulation/log_{}'.format(c.log_name),
                            angles=self.angles,
                            est_angles=self.est_angles,
                            pos=self.pos,
                            est_pos=self.est_pos,
                            tool_pos=self.tool_pos,
                            est_tool_pos=self.est_tool_pos,
                            ball_pos=self.ball_pos,
                            est_ball_pos=self.est_ball_pos,
                            causes_int=self.causes_int,
                            causes_ext=self.causes_ext,
                            disc_actions=self.disc_actions,
                            est_vel=self.est_vel,
                            F_m=self.F_m,
                            L_ext=self.L_ext)

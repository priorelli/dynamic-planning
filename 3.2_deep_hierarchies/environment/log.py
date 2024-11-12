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

    # Track logs for each iteration
    def track(self, step, brain, body):
        self.angles[step] = body.get_angles()
        est_angles = np.array([module.prop.predict().detach().numpy()
                               for module in brain.modules])
        self.est_angles[step] = utils.denormalize(est_angles, c.norm_polar)

        self.pos[step, 1:] = body.get_pos()
        est_pos = np.array([module.vis.predict().detach().numpy()
                            for module in brain.modules])
        self.est_pos[step, 1:] = utils.denormalize(est_pos, c.norm_cart)

        self.lengths[step] = c.lengths
        est_lengths = np.array([module.int.x[0, 1].detach().numpy()
                                for module in brain.modules])
        self.est_lengths[step] = utils.denormalize(est_lengths, c.norm_cart)

    # Save log to file
    def save_log(self):
        np.savez_compressed('simulation/log_{}'.format(c.log_name),
                            angles=self.angles,
                            est_angles=self.est_angles,
                            pos=self.pos,
                            est_pos=self.est_pos)

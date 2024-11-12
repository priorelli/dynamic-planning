import numpy as np
import utils
import config as c


# Define log class
class Log:
    def __init__(self):
        ref = np.array(c.offset_2nd) - np.array(c.offset_1st)

        # Initialize logs
        self.angles_1st = np.zeros((c.n_steps, c.n_joints_1st))
        self.est_angles_1st = np.zeros_like(self.angles_1st)

        self.angles_2nd = np.zeros((c.n_steps, c.n_joints_2nd))
        self.est_angles_2nd = np.zeros_like(self.angles_2nd)

        self.pos_1st = np.zeros((c.n_steps, c.n_joints_1st + 1, 2))
        self.est_pos_1st = np.zeros((c.n_steps, c.n_objects,
                                     c.n_joints_2nd + 1, 2))
        self.est_pos_1st[:, 2, 0] = ref

        self.pos_2nd = np.zeros((c.n_steps, c.n_joints_2nd + 1, 2))
        self.est_pos_2nd = np.zeros((c.n_steps, c.n_objects,
                                     c.n_joints_2nd + 1, 2))
        self.est_pos_2nd[:, 2, 0] = -ref

    # Track logs for each iteration
    def track(self, step, brain, body_1st, body_2nd):
        self.angles_1st[step] = body_1st.get_angles()
        est_angles_1st = np.array([module.prop.predict().detach().numpy()
                                   for module in brain.modules_1st])
        self.est_angles_1st[step] = utils.denormalize(
            est_angles_1st, c.norm_polar)[:-2]

        self.angles_2nd[step] = body_2nd.get_angles()
        est_angles_2nd = np.array([module.prop.predict().detach().numpy()
                                   for module in brain.modules_2nd])
        self.est_angles_2nd[step] = utils.denormalize(est_angles_2nd,
                                                      c.norm_polar)

        self.pos_1st[step, 1:] = body_1st.get_pos(c.offset_1st)
        est_pos_1st = np.array([module.vis.predict()[:, :2].detach().numpy()
                                for module in brain.modules_1st])

        for j in range(c.n_joints_2nd):
            self.est_pos_1st[step, :, j + 1] = utils.denormalize(
                est_pos_1st[j], c.norm_cart)

        self.pos_2nd[step, 1:] = body_2nd.get_pos(c.offset_2nd)
        est_pos_2nd = np.array([module.vis.predict()[:, :2].detach().numpy()
                                for module in brain.modules_2nd])
        for j in range(c.n_joints_2nd):
            self.est_pos_2nd[step, :, j + 1] = utils.denormalize(
                est_pos_2nd[j], c.norm_cart)

    # Save log to file
    def save_log(self):
        np.savez_compressed('simulation/log_{}'.format(c.log_name),
                            angles_1st=self.angles_1st,
                            est_angles_1st=self.est_angles_1st,
                            pos_1st=self.pos_1st,
                            est_pos_1st=self.est_pos_1st,
                            angles_2nd=self.angles_2nd,
                            est_angles_2nd=self.est_angles_2nd,
                            pos_2nd=self.pos_2nd,
                            est_pos_2nd=self.est_pos_2nd)

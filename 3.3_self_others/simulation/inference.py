import utils
import config as c
from environment.window import Window
from environment.log import Log
from simulation.brain import Brain


# Define inference class
class Inference(Window):
    def __init__(self):
        super().__init__()

        # Initialize brain
        self.brain = Brain(self.body_1st.idxs, self.body_2nd.idxs)

        self.brain.init_belief(self.body_1st.get_angles(),
                               self.body_1st.get_phi(),
                               self.body_1st.get_pos(c.offset_1st),
                               self.body_2nd.get_pos(c.offset_1st),
                               self.body_2nd.get_angles(),
                               self.body_2nd.get_phi(),
                               self.body_2nd.get_pos(c.offset_2nd),
                               self.body_1st.get_pos(c.offset_2nd))

        # Initialize error tracking
        self.log = Log()

    def update(self, dt):
        dt = 1 / c.fps

        # Track log
        self.log.track(self.step, self.brain, self.body_1st, self.body_2nd)

        # Get observations
        O = [self.get_prop_obs_1st(), self.get_visual_obs_1st(),
             self.get_prop_obs_2nd(), self.get_visual_obs_2nd()]

        # Perform free energy step
        actions = self.brain.inference_step(O)

        # Update bodies
        self.body_1st.update(actions[0])
        self.body_2nd.update(actions[1])

        # Update physics
        for i in range(c.phys_steps):
            self.space_1st.step(c.speed / (c.fps * c.phys_steps))
            self.space_2nd.step(c.speed / (c.fps * c.phys_steps))

        # Move sprites
        self.update_sprites()

        # Print info
        if (self.step + 1) % 100 == 0:
            utils.print_info(self.step, c.n_steps)

        # Stop simulation
        self.step += 1
        if self.step == c.n_steps:
            self.log.save_log()
            self.stop()

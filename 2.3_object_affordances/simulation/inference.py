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
        self.brain = Brain()
        self.brain.init_belief(self.body.get_angles())

        # Initialize error tracking
        self.log = Log()

    def update(self, dt):
        dt = 1 / c.fps

        # Get observations
        O = [self.get_prop_obs(), self.get_visual_obs(),
             self.get_tactile_obs()]

        # Track log
        self.log.track(self.step, self.brain, self.body, self.objects,
                       O[2])

        # Perform free energy step
        action = self.brain.inference_step(O)

        # Update body
        self.body.update(action)

        # Update physics
        for i in range(c.phys_steps):
            self.space.step(c.speed / (c.fps * c.phys_steps))

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

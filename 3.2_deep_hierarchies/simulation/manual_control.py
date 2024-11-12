import numpy as np
from pyglet.window import key
import utils
import config as c
from environment.window import Window


# Define manual control class
class ManualControl(Window):
    def __init__(self):
        super().__init__()
        self.objects.target.set_collision(1)

    def update(self, dt):
        dt = 1 / c.fps

        # Get action from user
        action = self.get_pressed()

        # Update body
        self.body.update(np.array(action) / 5)

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
            self.stop()

    # Get action from user input
    def get_pressed(self):
        return [(key.Z in self.keys) - (key.X in self.keys),
                (key.LEFT in self.keys) - (key.RIGHT in self.keys),
                (key.UP in self.keys) - (key.DOWN in self.keys),
                (key.A in self.keys) - (key.S in self.keys),
                (key.Q in self.keys) - (key.W in self.keys),
                (key.W in self.keys) - (key.Q in self.keys),
                (key.W in self.keys) - (key.Q in self.keys),
                (key.Q in self.keys) - (key.W in self.keys)]

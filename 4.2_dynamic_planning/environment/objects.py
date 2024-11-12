import config as c
import numpy as np
from environment.sprites import Ball, Wall


# Define objects class
class Objects:
    def __init__(self, batch, space):
        self.ball, self.walls = self.setup(batch, space)

        # Set collision type
        self.ball.shape.collision_type = 3

    def setup(self, batch, space):
        ball = Ball(batch, space, c.ball_size,
                    (200, 100, 0), c.ball_pos)

        walls = []
        corners = [(0, 0), (0, c.height), (c.width, c.height),
                   (c.width, 0), (0, 0)]

        for i in range(len(corners) - 1):
            walls.append(Wall(space, corners[i], corners[i + 1]))

        return ball, walls

    # Sample objects
    def sample(self):
        objs = [self.ball]
        positions = [c.ball_pos]
        vels = [c.ball_vel]
        dirs = [c.ball_dir]

        for obj, pos, vel, dir_ in zip(objs, positions, vels, dirs):
            # Set object position
            if pos == [0, 0]:
                pos = np.random.randint((-c.width / 2, -c.height / 2),
                                        (c.width / 2, c.height / 2))
                obj.set_pos(pos)

            # Set object velocity
            if vel != 0:
                dir_ = np.random.rand() * 2 * np.pi if dir_ \
                    is None else np.radians(dir_)
                obj.set_vel(np.cos(dir_), np.sin(dir_), vel)
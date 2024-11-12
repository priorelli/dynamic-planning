import config as c
import numpy as np
from environment.sprites import Ball, Wall, Tool


# Define objects class
class Objects:
    def __init__(self, batch, space):
        self.tool, self.ball, self.walls = self.setup(batch, space)

        # Set collision type
        self.ball.shape.collision_type = 3
        # self.ball.set_collision(1)

    def setup(self, batch, space):
        tool = Tool(batch, space, (c.tool_length,
                    c.tool_length // 6), c.tool_pos)
        ball = Ball(batch, space, c.ball_size,
                    (200, 100, 0), c.ball_pos)

        walls = []
        corners = [(0, 0), (0, c.height), (c.width, c.height),
                   (c.width, 0), (0, 0)]

        for i in range(len(corners) - 1):
            walls.append(Wall(space, corners[i], corners[i + 1]))

        return tool, ball, walls

    # Sample objects
    def sample(self):
        objs = [self.tool, self.ball]
        positions = [c.tool_pos, c.ball_pos]
        vels = [c.tool_vel, c.ball_vel]
        dirs = [c.tool_dir, c.ball_dir]

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

        angle = np.random.randint(0, 360) if \
            c.tool_angle is None else c.tool_angle

        self.tool.body.angle = np.radians(angle)

import config as c
import torch
from environment.sprites import Ball, Wall


# Define objects class
class Objects:
    def __init__(self, batch, space):
        self.joints = [torch.tensor(c.ball_joint)]

        self.vels = [c.ball_vel]

        self.ball, self.walls = self.setup(batch, space)

        # Set collision type
        self.ball.shape.collision_type = 3
        self.ball.set_collision(1)

    def setup(self, batch, space):
        ball = Ball(batch, space, c.ball_size,
                    (200, 100, 0), (0, 0))

        walls = []
        corners = [(0, 0), (0, c.height), (c.width, c.height),
                   (c.width, 0), (0, 0)]

        for i in range(len(corners) - 1):
            walls.append(Wall(space, corners[i], corners[i + 1]))

        return ball, walls

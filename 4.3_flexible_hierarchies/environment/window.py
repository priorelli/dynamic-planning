import numpy as np
import pyglet
import pymunk
import utils
import config as c
from environment.body import Body
from environment.objects import Objects


class Window(pyglet.window.Window):
    def __init__(self):
        super().__init__(c.width, c.height, 'Tool use',
                         vsync=False)
        # Start physics engine
        self.space = pymunk.Space()
        self.space.gravity = 0, 0

        self.keys = set()
        self.batch = pyglet.graphics.Batch()
        self.fps_display = pyglet.window.FPSDisplay(self)

        # Initialize body
        self.body = Body(self.batch, self.space)

        # Initialize objects
        self.objects = Objects(self.batch, self.space)

        # Initialize brain
        self.brain = None

        # Initialize simulation variables
        self.step = 0
        self.picked, self.touch = False, np.array([1, 0])

        # Sample objects
        self.objects.sample()
        self.update_sprites()

        # Set background
        pyglet.gl.glClearColor(1, 1, 1, 1)

    def on_key_press(self, sym, mod):
        self.keys.add(sym)

    def on_key_release(self, sym, mod):
        self.keys.remove(sym)

    def on_draw(self):
        self.clear()
        self.batch.draw()
        self.fps_display.draw()

    # Update function to override
    def update(self, dt):
        pass

    # Run simulation with custom update function
    def run(self):
        if c.fps == 0:
            pyglet.clock.schedule(self.update)
        else:
            pyglet.clock.schedule_interval(self.update, 1 / c.fps)
        pyglet.app.run()

    # Stop simulation
    def stop(self):
        pyglet.app.exit()
        self.close()

    # Update sprites rotation and position
    def update_sprites(self):
        sprites = [self.objects.tool, self.objects.ball]
        for sprite in self.body.links + self.body.joints + sprites:
            sprite.position = sprite.body.position
            sprite.rotation = -np.degrees(sprite.body.angle)

    # Get proprioceptive observation
    def get_prop_obs(self):
        return utils.normalize([*self.body.get_angles(), 0], c.norm_polar)

    # Get visual observation
    def get_visual_obs(self):
        hand_pos = [*self.body.get_pos(), self.body.get_pos()[-1]]

        tool_pos = np.zeros((c.n_joints + 1, 2))
        tool_pos[-2:] = [self.objects.tool.get_pos(),
                         self.objects.tool.get_end()]

        ball_pos = np.zeros((c.n_joints + 1, 2))
        ball_pos[-1] = self.objects.ball.get_pos()

        hand_vel = [*self.body.get_vel(), self.body.get_vel()[-1]]

        tool_vel = np.zeros((c.n_joints + 1, 2))
        tool_vel[-2:] = [self.objects.tool.get_vel(),
                         self.objects.tool.get_vel()]

        ball_vel = np.zeros((c.n_joints + 1, 2))
        ball_vel[-1] = self.objects.ball.get_vel()

        pos = [hand_pos, tool_pos, ball_pos]
        vel = [hand_vel, tool_vel, ball_vel]

        return utils.normalize([pos, vel], c.norm_cart)

    # Get tactile observation
    def get_tactile_obs(self):
        return self.touch

    # Check if tool is reached
    def tool_reached(self):
        dist = np.linalg.norm(self.objects.tool.get_pos() -
                              self.body.get_pos()[-1])

        return dist < c.reach_dist

    # Check if stick is picked
    def stick_picked(self):
        if not self.picked and self.tool_reached():
            self.picked = True
            self.touch = np.array([0, 1])

            self.space.add(pymunk.PinJoint(
                self.body.joints[-1].body, self.objects.tool.body))

            self.objects.tool.motor = pymunk.SimpleMotor(
                self.body.joints[-1].body, self.objects.tool.body, 0)
            self.objects.tool.motor.max_force = 2e10
            self.space.add(self.objects.tool.motor)

            self.space.add(pymunk.RotaryLimitJoint(
                self.body.joints[-1].body, self.objects.tool.body,
                -0.1, 0.1))

        if self.picked:
            self.objects.tool.set_vel(0, 0)
            self.objects.tool.motor.rate = 0

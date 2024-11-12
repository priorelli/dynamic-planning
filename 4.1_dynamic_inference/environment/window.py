import numpy as np
import pyglet
import pymunk
import utils
import config as c
from environment.body import Body
from environment.objects import Objects


class Window(pyglet.window.Window):
    def __init__(self):
        super().__init__(c.width, c.height, 'Dynamic inference '
                         'by model reduction', vsync=False)
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
        sprites = [self.objects.ball, self.objects.square]
        for sprite in self.body.links + self.body.joints + sprites:
            sprite.position = sprite.body.position
            sprite.rotation = -np.degrees(sprite.body.angle)

    # Get proprioceptive observation
    def get_prop_obs(self):
        return utils.normalize(self.body.get_angles(), c.norm_polar)

    # Get visual observation
    def get_visual_obs(self):
        pos = [self.body.get_pos()[-1], self.objects.ball.get_pos(),
               self.objects.square.get_pos()]
        vel = [self.body.joints[-1].get_vel(), self.objects.ball.get_vel(),
               self.objects.square.get_vel()]

        return utils.normalize([pos, vel], c.norm_cart)

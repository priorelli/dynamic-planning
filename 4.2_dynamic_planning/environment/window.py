import numpy as np
import pyglet
import pymunk
import utils
import config as c
from environment.body import Body
from environment.objects import Objects


class Window(pyglet.window.Window):
    def __init__(self):
        super().__init__(c.width, c.height, 'A discrete interface for '
                         'dynamic planning', vsync=False)
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
        self.picked, self.touch = False, np.zeros(2)

        # Start collision handlers
        self.handlers = []
        for i in range(2):
            handler = self.space.add_collision_handler(i + 1, 3)
            handler.begin = self.begin(i)
            handler.separate = self.separate(i)
            self.handlers.append(handler)

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

    def begin(self, i):
        def f(arbiter, space, data):
            self.touch[i] = 1
            return True
        return f

    def separate(self, i):
        def f(arbiter, space, data):
            self.touch[i] = 0
        return f

    # Update sprites rotation and position
    def update_sprites(self):
        sprites = [self.objects.ball]
        for sprite in self.body.links + self.body.joints + sprites:
            sprite.position = sprite.body.position
            sprite.rotation = -np.degrees(sprite.body.angle)

    # Get proprioceptive observation
    def get_prop_obs(self):
        return utils.normalize(self.body.get_angles(), c.norm_polar)

    # Get visual observation
    def get_visual_obs(self):
        pos = [self.body.get_grasp(), self.objects.ball.get_pos()]
        vel = [self.body.joints[-5].get_vel(), self.objects.ball.get_vel()]

        return utils.normalize([pos, vel], c.norm_cart)

    # Get tactile observation
    def get_tactile_obs(self):
        return np.array([not self.touch.all(),
                         self.touch.all()]).astype(int)

    # Check if ball is picked
    def ball_picked(self):
        self.picked = np.all(self.touch)

        # Change color
        if self.picked:
            self.objects.ball.color = (100, 100, 100)
        else:
            self.objects.ball.color = (200, 100, 0)

        collide = self.brain.discrete.o_int[1] > 0.3 and \
            (self.ball_reached() or self.picked)
        self.objects.ball.set_collision(collide)

    # Check if ball is reached
    def ball_reached(self):
        dist = np.linalg.norm(self.body.get_grasp() -
                              self.objects.ball.get_pos())

        return dist < c.reach_dist

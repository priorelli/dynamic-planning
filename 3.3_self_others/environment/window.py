import numpy as np
import pyglet
import pymunk
import utils
import config as c
from environment.body import Body


class Window(pyglet.window.Window):
    def __init__(self):
        super().__init__(c.width, c.height, 'The self, the objects, '
                         'and the others', vsync=False)
        # Start physics engine
        self.space_1st = pymunk.Space()
        self.space_1st.gravity = 0, 0

        self.space_2nd = pymunk.Space()
        self.space_2nd.gravity = 0, 0

        self.keys = set()
        self.batch = pyglet.graphics.Batch()
        self.fps_display = pyglet.window.FPSDisplay(self)

        # Initialize bodies
        self.body_1st = Body(self.batch, self.space_1st, c.joints_1st,
                             c.offset_1st, (0, 100, 200))
        self.body_2nd = Body(self.batch, self.space_2nd, c.joints_2nd,
                             c.offset_2nd, (0, 200, 200))

        # Initialize brain
        self.brain = None

        # Initialize simulation variables
        self.step = 0

        # Sample objects
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
        sprites = [*self.body_1st.links, *self.body_1st.joints,
                   *self.body_2nd.links, *self.body_2nd.joints]
        for sprite in sprites:
            sprite.position = sprite.body.position
            sprite.rotation = -np.degrees(sprite.body.angle)

    # Get proprioceptive observation of 1st agent
    def get_prop_obs_1st(self):
        return utils.normalize([*self.body_1st.get_angles(), 0, 0],
                               c.norm_polar)

    # Get proprioceptive observation of 2nd agent
    def get_prop_obs_2nd(self):
        return utils.normalize(self.body_2nd.get_angles(), c.norm_polar)

    # Get visual observation of 1st agent
    def get_visual_obs_1st(self):
        self_norm = np.zeros((c.n_joints_2nd, 2))
        self_norm[:-2] = utils.normalize(
            self.body_1st.get_pos(c.offset_1st), c.norm_cart)

        # Hand of 2nd agent at hand level
        other_self_norm = np.zeros((c.n_joints_2nd, 2))
        other_self_norm[-3] = utils.normalize(
            self.body_2nd.get_pos(c.offset_1st), c.norm_cart)[-1]

        other_norm = utils.normalize(
            self.body_2nd.get_pos(c.offset_1st), c.norm_cart)

        return np.array([self_norm, other_self_norm, other_norm])

    # Get visual observation of 2nd agent
    def get_visual_obs_2nd(self):
        self_norm = utils.normalize(self.body_2nd.get_pos(c.offset_2nd),
                                    c.norm_cart)

        # Elbow of 1st agent at hand level
        other_self_norm = np.zeros((c.n_joints_2nd, 2))
        other_self_norm[-1] = utils.normalize(
            self.body_1st.get_pos(c.offset_2nd), c.norm_cart)[-2]

        other_norm = np.zeros((c.n_joints_2nd, 2))
        other_norm[:-2] = utils.normalize(self.body_1st.get_pos(c.offset_2nd),
                                          c.norm_cart)

        return np.array([self_norm, other_self_norm, other_norm])

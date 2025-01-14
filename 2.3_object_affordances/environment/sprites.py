import numpy as np
import pyglet
import pymunk
import config as c

offset = pymunk.Vec2d(c.width / 2 + c.off_x, c.height / 2 + c.off_y)


class Ball(pyglet.shapes.Circle):
    def __init__(self, batch, space, radius, color, pos):
        super().__init__(*(offset + pos), radius, color=color,
                         batch=batch, group=pyglet.graphics.Group(1))
        self.body = pymunk.Body()
        self.body.position = offset + pos

        self.shape = pymunk.Circle(self.body, radius)
        self.shape.mass = 1
        self.shape.friction = 1
        self.shape.elasticity = 1

        self.shape.filter = pymunk.ShapeFilter(mask=0b11000,
                                               categories=0b00001)

        space.add(self.body, self.shape)

    def get_pos(self):
        return np.array(self.position - offset)

    def get_vel(self):
        return np.array(self.body.velocity)

    def set_pos(self, pos):
        self.body.position = pymunk.Vec2d(*pos) + offset

    def set_vel(self, x, y, w=0):
        self.body.velocity = pymunk.Vec2d(x, y) * w

    def set_radius(self, radius):
        self.radius = radius
        self.body.radius = radius
        self.shape.unsafe_set_radius(radius)

    def set_collision(self, mask):
        if mask:
            self.shape.filter = pymunk.ShapeFilter(mask=0b01110,
                                                   categories=0b00010)
        else:
            self.shape.filter = pymunk.ShapeFilter(mask=0b11000,
                                                   categories=0b00001)


class Square(pyglet.shapes.Rectangle):
    def __init__(self, batch, space, pos):
        super().__init__(*(offset + pos), c.square_size, c.square_size,
                         color=(100, 200, 100), batch=batch,
                         group=pyglet.graphics.Group(0))
        self.body = pymunk.Body()
        self.body.position = offset + pos

        self.shape = pymunk.Poly.create_box(
            self.body, (c.square_size, c.square_size))
        self.shape.mass = 1
        self.shape.friction = 1
        self.shape.elasticity = 1

        self.shape.filter = pymunk.ShapeFilter(mask=0b11000,
                                               categories=0b00001)

        self.anchor_x = self.width / 2
        self.anchor_y = self.height / 2

        space.add(self.body, self.shape)

    def get_pos(self):
        return np.array(self.position - offset)

    def get_vel(self):
        return np.array(self.body.velocity)

    def set_pos(self, pos):
        self.body.position = pymunk.Vec2d(*pos) + offset

    def set_vel(self, x, y, w=0):
        self.body.velocity = pymunk.Vec2d(x, y) * w


class Joint(pyglet.shapes.Circle):
    def __init__(self, batch, space, radius, pin, v, v_rot):
        super().__init__(*(offset + v_rot), radius, color=(0, 100, 200),
                         batch=batch, group=pyglet.graphics.Group(2))
        self.body = pymunk.Body()
        self.body.position = offset + v_rot

        self.shape = pymunk.Circle(self.body, radius)
        self.shape.mass = 2
        self.shape.friction = 1
        self.shape.elasticity = 0

        self.shape.filter = pymunk.ShapeFilter(group=1, mask=0b01110,
                                               categories=0b00010)

        self.motor = pymunk.SimpleMotor(pin, self.body, 0)
        self.motor.max_force = 2e10
        space.add(self.motor)

        space.add(pymunk.PinJoint(pin, self.body, v))

        space.add(self.body, self.shape)

    def get_pos(self):
        return np.array(self.position - offset)

    def get_vel(self):
        return np.array(self.body.velocity)


class Link(pyglet.shapes.Rectangle):
    def __init__(self, batch, space, size, pin, angle):
        super().__init__(*pin.position, *size, color=(0, 100, 200),
                         batch=batch, group=pyglet.graphics.Group(2))
        self.body = pymunk.Body()
        self.body.position = pin.position
        self.body.angle = np.radians(angle)

        w, h = size
        self.shape = pymunk.Segment(self.body, (0, 0), (w, 0), h / 2)
        self.shape.mass = 2
        self.shape.friction = 1
        self.shape.elasticity = 0

        self.shape.filter = pymunk.ShapeFilter(group=1, mask=0b01110,
                                               categories=0b00010)

        space.add(pymunk.PinJoint(pin, self.body))

        self.motor = pymunk.SimpleMotor(pin, self.body, 0)
        self.motor.max_force = 2e10
        space.add(self.motor)

        self.anchor_x = -h / 6
        self.anchor_y = h / 2

        space.add(self.body, self.shape)

    def get_pos(self):
        return np.array(self.position - offset)

    def get_end(self):
        v = pymunk.Vec2d(self.width, 0)
        return self.body.local_to_world(v) - offset

    def get_local(self, other):
        return other.body.world_to_local(self.get_end() + offset)


class Wall:
    def __init__(self, space, a, b):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)

        self.shape = pymunk.Segment(self.body, a, b, 1)
        self.shape.elasticity = 1

        space.add(self.body, self.shape)


class Origin:
    def __init__(self, space):
        self.body = space.static_body
        self.body.position = offset

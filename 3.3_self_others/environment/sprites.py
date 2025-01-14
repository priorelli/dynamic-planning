import numpy as np
import pyglet
import pymunk
import config as c

offset_main = pymunk.Vec2d(c.width / 2 + c.off_x, c.height / 2 + c.off_y)


class Joint(pyglet.shapes.Circle):
    def __init__(self, batch, space, radius, pin, v, v_rot, offset, color):
        super().__init__(*(offset_main + offset + v_rot), radius, color=color,
                         batch=batch, group=pyglet.graphics.Group(2))
        self.body = pymunk.Body()
        self.body.position = offset_main + offset + v_rot

        self.shape = pymunk.Circle(self.body, radius)
        self.shape.density = 2
        self.shape.friction = 1
        self.shape.elasticity = 0

        self.shape.filter = pymunk.ShapeFilter(group=1, mask=0b01110,
                                               categories=0b00010)

        self.motor = pymunk.SimpleMotor(pin, self.body, 0)
        self.motor.max_force = 2e10
        space.add(self.motor)

        space.add(pymunk.PinJoint(pin, self.body, v))

        space.add(self.body, self.shape)

    def get_pos(self, ref):
        return np.array(self.position - offset_main + ref)

    def get_vel(self):
        return np.array(self.body.velocity)


class Link(pyglet.shapes.Rectangle):
    def __init__(self, batch, space, size, pin, angle, color):
        super().__init__(*pin.position, *size, color=color, batch=batch,
                         group=pyglet.graphics.Group(2))
        self.body = pymunk.Body()
        self.body.position = pin.position
        self.body.angle = np.radians(angle)

        w, h = size
        self.shape = pymunk.Segment(self.body, (0, 0), (w, 0), h / 2)
        self.shape.density = 2
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

    def get_pos(self, ref):
        return np.array(self.position - offset_main + ref)

    def get_end(self, ref):
        v = pymunk.Vec2d(self.width, 0)
        return self.body.local_to_world(v) - offset_main - ref

    # def get_local(self, other):
    #     return other.body.world_to_local(self.get_end() + offset_main)


class Wall:
    def __init__(self, space, a, b):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)

        self.shape = pymunk.Segment(self.body, a, b, 1)
        self.shape.elasticity = 1

        space.add(self.body, self.shape)


class Origin:
    def __init__(self, space, offset):
        self.body = space.static_body
        self.body.position = offset_main + offset

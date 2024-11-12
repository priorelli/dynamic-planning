import numpy as np

from environment.sprites import Origin, Joint, Link


# Define body class
class Body:
    def __init__(self, batch, space, joints, offset, color):
        # Initialize body parameters
        ids = {joint: j for j, joint in enumerate(joints)}
        self.n_joints = len(joints)
        self.offset = offset
        self.color = color

        self.idxs = {}
        self.start = np.zeros(self.n_joints)
        self.size = np.zeros((self.n_joints, 2))

        for joint in joints:
            self.start[ids[joint]] = joints[joint]['angle']
            self.size[ids[joint]] = joints[joint]['size']

            if joints[joint]['link']:
                self.idxs[ids[joint]] = ids[joints[joint]['link']]
            else:
                self.idxs[ids[joint]] = -1

        self.joints, self.links = self.setup(batch, space)

    def setup(self, batch, space):
        # Initialize origin
        origin = Origin(space, self.offset)

        # Add joints and links
        joints = [Joint(batch, space, 30, origin.body, (0, 0), (0, 0),
                        self.offset, self.color)]
        links = []
        for j in range(self.n_joints):
            idx = self.idxs[j] + 1
            angle = np.sum(self.start[:j + 1])

            link = Link(batch, space, self.size[j], joints[idx].body, angle,
                        self.color)
            joint = Joint(batch, space, self.size[j, 1] / 2,
                          link.body, (link.width, 0),
                          link.get_end((0, 0)), (0, 0),
                          self.color)

            links.append(link)
            joints.append(joint)

        return joints, links

    # Set velocity
    def update(self, action):
        for j in range(self.n_joints):
            self.links[j].motor.rate = -action[j]

    # Get relative angles
    def get_angles(self):
        angles = np.zeros(self.n_joints)

        for l, link in enumerate(self.links):
            angles[l] = -link.rotation - np.sum(angles[:l])

        return angles

    # Get absolute orientations
    def get_phi(self):
        phi = np.zeros(self.n_joints)

        for l, link in enumerate(self.links):
            phi[l] = -link.rotation

        return phi

    # Get relative rates
    def get_rates(self):
        vel = np.zeros(self.n_joints)

        for j, link in enumerate(self.links):
            vel[j] = link.body.angular_velocity - np.sum(vel[:j])

        return np.degrees(vel)

    # Get relative torques
    def get_torques(self):
        return [link.body.torque for link in self.links]

    # Get link positions
    def get_pos(self, offset):
        return np.array([link.get_end(offset) for link in self.links])

    # Get joint velocities
    def get_vel(self):
        return np.array([joint.get_vel() for joint in self.joints[1:]])

    # Compute pose of every link
    def get_poses(self, angles, lengths):
        poses = np.zeros((self.n_joints + 1, 3))
        poses[0, :2] = self.offset

        for j in range(self.n_joints):
            old_pose = poses[self.idxs[j] + 1]

            position, phi = old_pose[:2], old_pose[2]
            new_phi = angles[j] + phi

            direction = np.array([np.cos(np.radians(new_phi)),
                                  np.sin(np.radians(new_phi))])
            new_position = position + lengths[j] * direction

            poses[j + 1] = *new_position, new_phi

        return poses

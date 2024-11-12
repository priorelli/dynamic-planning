import sys
import argparse
import torch
import numpy as np


# Parse arguments for simulation
def get_sim_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manual-control',
                        action='store_true', help='Start manual control')
    parser.add_argument('-i', '--inference',
                        action='store_true', help='Start inference')

    args = parser.parse_args()

    return args


# Parse arguments for plots
def get_plot_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dynamics',
                        action='store_true', help='Plot dynamics')
    parser.add_argument('-v', '--video',
                        action='store_true', help='Record video')

    args = parser.parse_args()
    return args


# Print simulation info
def print_info(step, n_steps):
    sys.stdout.write('\rStep: {:5d}/{:d}'
                     .format(step + 1, n_steps))
    sys.stdout.flush()


# Add Gaussian noise to array
def add_gaussian_noise(array, noise):
    sigma = noise ** 0.5
    return array + np.random.normal(0, sigma, np.shape(array))


# Normalize data
def normalize(x, limits, pyt=False, rng=True):
    limits = np.array(limits)
    if pyt:
        limits = torch.tensor(limits, dtype=torch.float32)

    x_norm = (x - limits[0]) / (limits[1] - limits[0])
    if rng:
        x_norm = x_norm * 2 - 1
    return x_norm


# Denormalize data
def denormalize(x, limits, pyt=False, rng=True):
    limits = np.array(limits)
    if pyt:
        limits = torch.tensor(limits, dtype=torch.float32)

    x_denorm = (x + 1) / 2 if rng else x
    x_denorm = x_denorm * (limits[1] - limits[0]) + limits[0]
    return x_denorm


# Compute forward kinematics
def kinematics(angles_norm, lengths, limits):
    angles = denormalize(angles_norm, limits, pyt=True)

    z = torch.tensor(0.0)
    i = torch.tensor(1.0)

    T_abs = torch.eye(3)

    for angle, length in zip(angles, lengths):
        l = torch.tensor(length, dtype=torch.float32)
        c = torch.cos(torch.deg2rad(angle))
        s = torch.sin(torch.deg2rad(angle))

        T_rel = torch.stack([torch.stack([c, -s, l * c], -1),
                            torch.stack([s, c, l * s], -1),
                            torch.stack([z, z, i], -1)], -2)

        T_abs = T_abs.matmul(T_rel)

    return T_abs[:2, 2]


# Shift (D) operator
def shift(array):
    if len(array) > 1:
        return torch.stack((*array[1:], torch.zeros_like(array[0])))
    else:
        return torch.zeros((1, len(array)))


# Transform angle to cos/sin
def to_cos_sin(angles):
    angles_rad = np.radians(angles)

    return np.array([np.cos(angles_rad), np.sin(angles_rad)]).T


# Transform cos/sin to angle
def to_angle(cos_sin):
    if isinstance(cos_sin[0], np.ndarray):
        angles_rad = np.arctan2(cos_sin[:, 1], cos_sin[:, 0])
    else:
        angles_rad = np.arctan2(cos_sin[1], cos_sin[0])

    return np.degrees(angles_rad)

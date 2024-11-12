# Window
width = 1600
height = 1000
off_x = 0
off_y = 0

fps = 60
speed = 10
phys_steps = 100

# Environment
dt = 0.3

# Simulation
ball_joint = [60.0]
ball_size = 20
ball_vel = 0.8

n_steps = 120
log_name = ''

# Brain
eta_x_theta = [0]
eta_v_theta = [0]
x_theta_start = None

pi_eta_x_theta = 0.0
pi_eta_v_theta = 0.0
pi_x_theta = 1.0
pi_prop = 1.0
pi_vis = 1.0

lambda_theta = 0.4

lr_theta = 1.0
lr_a = 0.4

n_orders = 2

# Body
start = [0]
lengths = [300]

joints = {}
joints['trunk'] = {'link': None, 'angle': start[0],
                   'size': (lengths[0], 50)}
n_joints = len(joints)

norm_polar = [-180.0, 180.0]
norm_cart = [-sum(lengths), sum(lengths)]

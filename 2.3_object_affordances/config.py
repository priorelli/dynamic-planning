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
ball_pos = [290, 200]
ball_size = 20
ball_vel = 10
ball_dir = None

square_size = 80
square_pos = [-290, 10]
square_vel = 10
square_dir = None

n_steps = 200
log_name = ''

# Brain
eta_x_theta = [0, 0, 0]
eta_v_theta = [1.0, 0.0]
x_theta_start = None

pi_eta_x_theta = 0.0
pi_eta_v_theta = 1.0
pi_x_theta = 1.0
pi_prop = 1.0
pi_vis = 0.5
pi_tact = 0.8

lambda_theta = 0.2

lr_theta = 1.0
lr_a = 1.0

n_orders = 2
n_objects = 3

# Body
start = [0, 0, 0]
lengths = [300, 200, 100]

joints = {}
joints['trunk'] = {'link': None, 'angle': start[0],
                   'size': (lengths[0], 50)}
joints['shoulder'] = {'link': 'trunk', 'angle': start[1],
                      'size': (lengths[1], 40)}
joints['elbow'] = {'link': 'shoulder', 'angle': start[2],
                   'size': (lengths[2], 26)}
n_joints = len(joints)

norm_polar = [-180.0, 180.0]
norm_cart = [-sum(lengths), sum(lengths)]

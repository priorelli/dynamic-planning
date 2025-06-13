# Window
width = 1000
height = 800
off_x = 0
off_y = 0

debug = 0
fps = 60
speed = 10
phys_steps = 100

# Environment
dt = 0.3

# Simulation
ball_pos = [-250, 0]
ball_size = 20
ball_vel = 10
ball_dir = 90

square_size = 80
square_pos = [250, 0]
square_vel = 10
square_dir = -90

n_steps = 1000
log_name = ''

# Brain
eta_x_int = [0]
x_int_start = None

pi_eta_x_int = 0.0
pi_x_int = 0.5
p_x_int = 1.0
pi_prop = 1.0

pi_eta_x_ext = 0.4
pi_x_ext = pi_x_int
p_x_ext = p_x_int
pi_vis = 1.0

lambda_int = 0.0
lambda_ext = 0.4

lr_int = 1.0
lr_ext = 1.0
lr_a = 1.0

n_orders = 2
n_objects = 3
n_policy = 1
n_tau = 30

gain_prior = 0.0
gain_evidence = 15.0
w_bmc = 2.0

# Body
start = [0]
lengths = [200]

joints = {}
joints['trunk'] = {'link': None, 'angle': start[0],
                   'size': (lengths[0], 50)}
n_joints = len(joints)

norm_polar = [-180.0, 180.0]
norm_cart = [-sum(lengths), sum(lengths)]

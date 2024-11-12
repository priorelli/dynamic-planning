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
ball_pos = [-160, 250]
ball_size = 20

n_steps = 100
log_name = ''

# Brain
eta_x_theta = [0]
eta_v_theta = [120]  # rho
x_theta_start = -40

pi_eta_x_theta = 0.0
pi_eta_v_theta = 1.0
pi_x_theta = 1.0
pi_prop = 1.0
pi_vis = 0.0

lambda_theta = 0.2

lr_theta = 1.0
lr_a = 0.2

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

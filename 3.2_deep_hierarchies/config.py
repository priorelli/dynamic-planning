# Window
width = 2000
height = 1400
off_x = 0
off_y = 0

fps = 60
speed = 10
phys_steps = 100

# Environment
dt = 0.3

# Simulation
ball_pos = [-300, 300]
ball_size = 24
ball_vel = 0
ball_dir = None

square_size = 80
square_pos = [-300, 0]
square_vel = 0
square_dir = None

n_steps = 800
log_name = ''

# Brain
eta_x_int = [0, 0, 0, 0, 0, 0, 0, 0]
x_int_start = None

pi_eta_x_int = 0.0
pi_x_int = 1.0
pi_prop = 1.0

pi_eta_x_ext = 0.5
pi_x_ext = pi_x_int
pi_vis = 0.1
pi_phi = 1.0

lambda_int = 0.0
lambda_ext = 0.8

lr_int = 1.0
lr_len = 0.0
lr_ext = 1.0
lr_a = 1.0

n_orders = 2

# Body
start = [0, 0, 0, 0, -30, 30, 0, 0]
lengths = [100, 140, 180, 40, 70, 70, 60, 60]

joints = {}
joints['trunk'] = {'link': None, 'angle': start[0],
                   'size': (lengths[0], 50)}
joints['shoulder'] = {'link': 'trunk', 'angle': start[1],
                      'size': (lengths[1], 40)}
joints['elbow'] = {'link': 'shoulder', 'angle': start[2],
                   'size': (lengths[2], 36)}
joints['wrist'] = {'link': 'elbow', 'angle': start[3],
                   'size': (lengths[3], 30)}
joints['thumb1'] = {'link': 'wrist', 'angle': start[4],
                    'size': (lengths[4], 10)}
joints['index1'] = {'link': 'wrist', 'angle': start[5],
                    'size': (lengths[5], 10)}
joints['thumb2'] = {'link': 'thumb1', 'angle': start[6],
                    'size': (lengths[6], 10)}
joints['index2'] = {'link': 'index1', 'angle': start[7],
                    'size': (lengths[7], 10)}
n_joints = len(joints)

norm_polar = [-180.0, 180.0]
norm_cart = [-sum(lengths), sum(lengths)]

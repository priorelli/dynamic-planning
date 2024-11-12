# Window
width = 2600
height = 1600
off_x = 0
off_y = 0

debug = 0
fps = 60
speed = 10
phys_steps = 100

# Environment
dt = 0.3

# Simulation
ball_pos = [0, 0]
ball_size = 24
ball_vel = 40
ball_dir = None

n_steps = 1500
log_name = ''

# Brain
eta_x_int = [0, 0, 0, 0, 0, 0, 0, 0]
x_int_start = None

pi_eta_x_int = 0.0
pi_x_int = 1.0
p_x_int = 2.0
pi_prop = 0.8

pi_eta_x_ext = 0.4
pi_x_ext = 1.0
p_x_ext = 2.0
pi_vis = 1.0

lambda_int = 0.3
lambda_ext = 0.6

lr_int = 1.0
lr_ext = 1.0
lr_a = 1.0

n_orders = 2
n_objects = 2
n_policy = 3
n_tau = 10

k_d = 0.0
gain_prior = 0.1
gain_evidence_int = 10.0
gain_evidence_ext = 30.0
w_bmc = 2.0

# Body
start = [0, 0, 0, 0, 0, 0, 0, 0]
lengths = [100, 140, 180, 40, 70, 70, 60, 60]

reach_dist = 50
angle_open = [50, -50, 0, 0]
angle_closed = [-30, 30, -30, 30]

joints = {}
joints['trunk'] = {'link': None, 'angle': start[0],
                   'size': (lengths[0], 50)}
joints['shoulder'] = {'link': 'trunk', 'angle': start[1],
                      'size': (lengths[1], 40)}
joints['elbow'] = {'link': 'shoulder', 'angle': start[2],
                   'size': (lengths[2], 36)}
joints['wrist'] = {'link': 'elbow', 'angle': start[3],
                   'size': (lengths[3], 36)}
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

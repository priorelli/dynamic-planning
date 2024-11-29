# Window
width = 1800
height = 1400
off_x = 0
off_y = 0

debug = 0
fps = 60
speed = 10
phys_steps = 100

# Environment
dt = 0.3

# Simulation
ball_pos = [-300, 300]
ball_size = 24
ball_vel = 10
ball_dir = None

tool_pos = [350, 300]
tool_length = 200
tool_vel = 5
tool_dir = None
tool_angle = 30

n_steps = 6000
log_name = ''

# Brain
eta_x_int = [0, 0, 0, 0]
x_int_start = None

pi_eta_x_int = 0.0
pi_x_int = 1.0
p_x_int = 2.0
pi_prop = 1.0

pi_eta_x_ext = 0.5
pi_x_ext = pi_x_int
p_x_ext = p_x_int
pi_vis = 0.1
pi_vis_obj = 1.0
pi_phi = 1.0

lambda_int = 0.0
lambda_ext = 0.7

lr_int = 1.0
lr_len = 0.0
lr_ext = 1.0
lr_a = 1.0

n_orders = 2
n_objects = 3
n_policy = 3
n_tau = 20

k_d = 0.0
gain_prior = 0.5
gain_evidence = 15.0
w_bmc = 2.0

# Body
start = [-40, 0, 0, 0]
lengths = [100, 140, 180, 80]
reach_dist = 15

joints = {}
joints['trunk'] = {'link': None, 'angle': start[0],
                   'size': (lengths[0], 50)}
joints['shoulder'] = {'link': 'trunk', 'angle': start[1],
                      'size': (lengths[1], 40)}
joints['elbow'] = {'link': 'shoulder', 'angle': start[2],
                   'size': (lengths[2], 36)}
joints['wrist'] = {'link': 'elbow', 'angle': start[3],
                   'size': (lengths[3], 30)}
n_joints = len(joints)

norm_polar = [-180.0, 180.0]
norm_cart = [-sum(lengths), sum(lengths)]

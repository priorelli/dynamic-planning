# Window
width = 2000
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
ball_pos = [-200, 200]
ball_size = 20
ball_vel = 0
ball_dir = None

square_size = 80
square_pos = [0, 380]
square_vel = 0
square_dir = None

n_steps = 500
log_name = ''

# Brain
eta_x_int = [0, 0, 0, 0]
eta_v_int = [1.0]  # reach ball
eta_v_ext = [0.0, 1.0]  # reach ball, avoid square
x_int_start = None

pi_eta_x_int = 0.0
pi_eta_v_int = 1.0
pi_x_int = 1.0
pi_prop = 1.0

pi_eta_x_ext = 0.5
pi_eta_v_ext = 1.0
pi_x_ext = pi_x_int
pi_vis = 1.0

lambda_int = 0.3
lambda_ext = 0.3
k_rep = 5e-4

lr_int = 1.0
lr_ext = 1.0
lr_a = 1.0

n_orders = 2
n_objects = 3

# Body
start = [0, 0, 0, 0]
lengths = [100, 140, 180, 80]

joints = {}
joints['trunk'] = {'link': None, 'angle': start[0],
                   'size': (lengths[0], 50)}
joints['shoulder'] = {'link': 'trunk', 'angle': start[1],
                      'size': (lengths[1], 40)}
joints['elbow'] = {'link': 'shoulder', 'angle': start[2],
                   'size': (lengths[2], 36)}
joints['wrist'] = {'link': 'elbow', 'angle': start[3],
                   'size': (lengths[3], 36)}
n_joints = len(joints)

norm_polar = [-180.0, 180.0]
norm_cart = [-sum(lengths), sum(lengths)]

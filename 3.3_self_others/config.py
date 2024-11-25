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
n_steps = 1000
log_name = ''

# Brain
eta_x_int = [0, 0, 0, 0, 0]
x_int_start = None

pi_eta_x_int = 0.0
pi_x_int = 1.0
pi_prop = 0.8

pi_eta_x_ext = 0.4
pi_x_ext = pi_x_int
pi_vis = 0.2
pi_vis_obj = 1.0
pi_phi = 1.0

lambda_int = 0.5
lambda_ext = 1.0

lr_int = 1.0
lr_len = 0.0
lr_ext = 1.0
lr_a = 1.0

n_orders = 2
n_objects = 3

# Body (1st agent)
start_1st = [-70, 10, 10]
lengths_1st = [100, 140, 180]
offset_1st = (300, 100)

joints_1st = {}
joints_1st['trunk'] = {'link': None, 'angle': start_1st[0],
                       'size': (lengths_1st[0], 50)}
joints_1st['elbow'] = {'link': 'trunk', 'angle': start_1st[1],
                       'size': (lengths_1st[1], 40)}
joints_1st['hand'] = {'link': 'elbow', 'angle': start_1st[2],
                      'size': (lengths_1st[2], 36)}
n_joints_1st = len(joints_1st)

# Body (2nd agent)
start_2nd = [110, -20, 10, 0, 0]
lengths_2nd = [100, 120, 200, 80, 120]
offset_2nd = (-300, -100)

joints_2nd = {}
joints_2nd['trunk'] = {'link': None, 'angle': start_2nd[0],
                       'size': (lengths_2nd[0], 50)}
joints_2nd['shoulder'] = {'link': 'trunk', 'angle': start_2nd[1],
                          'size': (lengths_2nd[1], 40)}
joints_2nd['elbow'] = {'link': 'shoulder', 'angle': start_2nd[2],
                       'size': (lengths_2nd[2], 36)}
joints_2nd['wrist'] = {'link': 'elbow', 'angle': start_2nd[3],
                       'size': (lengths_2nd[3], 30)}
joints_2nd['hand'] = {'link': 'wrist', 'angle': start_2nd[4],
                      'size': (lengths_2nd[4], 20)}
n_joints_2nd = len(joints_2nd)

norm_polar = [-180.0, 180.0]
norm_cart = [-sum(lengths_2nd), sum(lengths_2nd)]

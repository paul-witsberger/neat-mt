# TODO look into adding textures to a sphere to make a planet

from constants import *


gm = u_sun_km3s2
n_dim = 2
assert n_dim == 2 or n_dim == 3
# Create logical list for indices of 2D components
if n_dim == 2:
    ind_dim = [True, True, False, True, True, False, False]
else:
    ind_dim = [True] * 6 + [False]

# Define initial and final orbits
# Initial orbit parameters
a0_max, a0_min = a_earth_km, a_earth_km
e0_max, e0_min = e_earth, e_earth
i0_max, i0_min = 0, 0
w0_max, w0_min = w_earth_rad, w_earth_rad
om0_max, om0_min = 0, 0
f0_max, f0_min = 2 * np.pi / 3, 2 * np.pi / 3
# Final orbit parameters
af_max, af_min = a_mars_km, a_mars_km
ef_max, ef_min = e_mars, e_mars
if_max, if_min = 0, 0
wf_max, wf_min = w_mars_rad, w_mars_rad
omf_max, omf_min = 0, 0
ff_max, ff_min = np.pi / 2, np.pi / 2
# Flag that specifies if orbit is elliptical or circular
elliptical_initial = True if e0_max > 0 else False
elliptical_final = True if ef_max > 0 else False

# Specify spacecraft and engine parameters
m_dry = 10000
m_prop = 3000
m0 = m_dry + m_prop
variable_power = False
if variable_power:
    power_min, power_max = 3.4, 12.5 # kW
    solar_array_m2 = 1  # m^2
    power_reference = solar_constant * solar_array_m2  # kW, power available at 1 AU
    thrust_power_coef = np.array([-363.67, 225.49, -21.475, 0.7943, 0]) / 1000
    isp_power_coef = np.array([2274.5, -319.39, 61.817, -2.6802, 0])
    T_max_kN = thrust_power_coef * np.array([1, power_max, power_max ** 2, power_max ** 3, power_max ** 4]) * 1e-3
    Isp = isp_power_coef * np.array([1, power_max, power_max ** 2, power_max ** 3, power_max ** 4])
else:
    T_max_kN = 1.2 * 1e-3 # 2 x HERMeS engines with 37.5 kW  # 21.8 mg/s for one thruster
    Isp = 2780
Isp_chemical = 370 # for the final correction burns

# Define time of flight
t0 = 0.
tf = 2 * year_to_sec

# Define scales for the state vectors to non-dimensionalize later
input_frame = 'kep'  # 'kep', 'mee', 'car'
r_scale, v_scale = a_earth_km, vp_earth_kms
if input_frame == 'kep':
    if n_dim == 2:
        scales_in = np.array([r_scale, 1, np.pi, np.pi])
    else:
        scales_in = np.array([r_scale, 1, np.pi, np.pi, np.pi, np.pi])
elif input_frame == 'mee':
    if n_dim == 2:
        raise ValueError('Cannot use MEE as input frame for 2D case')
    else:
        scales_in = np.array([r_scale, 1, 1, 1, 1, np.pi])
elif input_frame == 'car':
    if n_dim == 2:
        scales_in = np.array([r_scale, r_scale, v_scale, v_scale])
    else:
        scales_in = np.array([r_scale, r_scale, r_scale, v_scale, v_scale, v_scale])
else:
    raise ValueError('Undefined input frame')
scales_in = np.hstack((scales_in, scales_in, 1., 1.))  # add twice for current plus target, then add 1 for mass, time
if n_dim == 2:
    scales_out = np.array([[-np.pi, np.pi], [0, 1]])  # thrust angle, thrust throttle
else:
    scales_out = np.array([[0, 2 * np.pi], [0, 2 * np.pi], [0, 1]])  # alpha, beta, throttle

# Specify output activation type (NOTE: this does not automatically change with the NEAT config file)
out_node_scales = np.array([[-1, 1], [0, 1]])

# Optionally specify a set of angle choices, and have an output node for each - categorical classification
use_multiple_angle_nodes = False
angle_choices = np.array([0, np.pi])

# For the GA (not NEAT), define network size
n_in = 10
n_hid = 4
n_out = 2

# Define integration parameters
tol = 1e-7
num_nodes = 50

# Specify missed thrust cases
num_cases = 1
num_outages = 0

# Choose to add a penalty for going too close to the central body in the fitness function
rp_penalty = True
min_allowed_rp = a_earth_km * 0.90
rp_penalty_multiplier = 5000

# Choose to add a penalty for not thrusting at all (just staying in initial orbit)
no_thrust_penalty = True

# Choose maximum energy to allow before stopping integration
max_energy = - u_sun_km3s2 / 2 / max(a0_max, af_max) * 0.8
min_energy = - u_sun_km3s2 / 2 / min(a0_min, af_min) * 1.2

# Choose whether missed thrust events occur or not, and scale time-between-events and recovery-duration
missed_thrust_allowed = False
missed_thrust_tbe_factor = 0.5
missed_thrust_rd_factor = 2

# Specify the indices of the input array that should be used
# input_indices = np.array([0, 1, 2, 3, 8, 9])
input_indices = None
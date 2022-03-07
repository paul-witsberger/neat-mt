########################################################################################################################
# Imports
########################################################################################################################
import numpy as np
import constants as c
from datetime import datetime
########################################################################################################################


########################################################################################################################
# Define general problem characteristics: initial, final, and central bodies; dimensionality; times
########################################################################################################################
gm = c.u_sun_km3s2
gm_target = c.u_mars_km3s2

n_dim = 2

# Define initial and final bodies and times
init_body = 'earth'
target_body = 'mars'
central_body = 'sun'
t0_str = '2024 Aug 24 00:00:00'
ckout_str = '2024 Sep 23 00:00:00'
tf_str = '2025 Nov 07 00:00:00'
fmt = '%Y %b %d %H:%M:%S'
ordinal_to_julian = 1721424.5
t0 = datetime.strptime(t0_str, fmt).toordinal() + ordinal_to_julian
ckout = datetime.strptime(ckout_str, fmt).toordinal() + ordinal_to_julian
tf = datetime.strptime(tf_str, fmt).toordinal() + ordinal_to_julian
times = np.array([t0, tf])
times_jd1950_jc = (times - c.reference_date_jd1950) * c.day_to_jc
tf = (tf - t0) * c.day_to_sec
t0 = 0
launch_window = 0 * c.day_to_jc
arrival_window = 0 * c.day_to_jc
########################################################################################################################


########################################################################################################################
# Create logical list for indices of state components excluding mass
########################################################################################################################
if n_dim == 2:
    # [x, y, z, vx, vy, vz, mass] -> [x, y, vx, vy]
    ind_dim = np.array([True, True, False, True, True, False, False])
else:
    # [x, y, z, vx, vy, vz, mass] -> [x, y, z, vx, vy, vz]
    ind_dim = np.array([True] * 6 + [False])
########################################################################################################################


########################################################################################################################
# Define initial and final orbits
########################################################################################################################
# Initial orbit parameters
# TODO define min and max parameters with respect to launch_window and arrival_window using analytical ephemerides
a0_max, a0_min = c.a_earth_km, c.a_earth_km
e0_max, e0_min = c.e_earth, c.e_earth
i0_max, i0_min = 0, 0
w0_max, w0_min = c.lp_earth_rad, c.lp_earth_rad
om0_max, om0_min = 0, 0
f0_ref = 259.7 * np.pi / 180
# f0_max, f0_min = 274.6 * np.pi / 180, 245.3 * np.pi / 180
f0_max, f0_min = f0_ref, f0_ref

# Final orbit parameters
af_max, af_min = c.a_mars_km, c.a_mars_km
ef_max, ef_min = c.e_mars, c.e_mars
if_max, if_min = 0, 0
wf_max, wf_min = c.lp_mars_rad, c.lp_mars_rad
omf_max, omf_min = 0, 0
ff_ref = 1 * np.pi / 180
# ff_max, ff_min = 10.5 * np.pi / 180, -8.5 * np.pi / 180
ff_max, ff_min = ff_ref, ff_ref
true_final_f = False

# Flag that specifies if orbit is elliptical or circular
elliptical_initial = True if e0_max > 0 else False
elliptical_final = True if ef_max > 0 else False
########################################################################################################################


########################################################################################################################
# Specify spacecraft and engine parameters
########################################################################################################################
m_dry = 10000
m_prop = 3000
m0 = m_dry + m_prop
fixed_step = False  # NOTE: on big runs, definitely use adaptive step
variable_power = False
if variable_power:
    power_min, power_max = 3.4, 12.5  # kW
    solar_array_m2 = 1  # m^2
    power_reference = c.solar_constant * solar_array_m2  # kW, power available at 1 AU
    thrust_power_coef = np.array([-363.67, 225.49, -21.475, 0.7943, 0]) / 1000
    isp_power_coef = np.array([2274.5, -319.39, 61.817, -2.6802, 0])
    T_max_kN = thrust_power_coef * np.array([1, power_max, power_max ** 2, power_max ** 3, power_max ** 4]) * 1e-3
    Isp = isp_power_coef * np.array([1, power_max, power_max ** 2, power_max ** 3, power_max ** 4])
else:
    T_max_kN = 1.2 * 1e-3  # 2 x HERMeS engines with 37.5 kW  # 21.8 mg/s for one thruster
    Isp = 2780
isp_chemical = 370  # for the final correction burns
########################################################################################################################


########################################################################################################################
# Specify the indices of the input array that should be used
########################################################################################################################
if n_dim == 2:
    input_indices = np.array([0, 1, 2, 3, 8, 9])  # ignore target, 2D [6 nodes]
    # input_indices = np.array([8, 9])  # mass and time, 2D [2]
else:
    # input_indices = np.array([0, 1, 3, 4, 6, 7, 9, 10, 12, 13])  # inertial inputs, ignore Z components, 3D [10 nodes]
    # input_indices = np.array([0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13])  # keplerian inputs, ignore inclination [12 nodes]
    input_indices = np.array([0, 1, 3, 4, 5, 12, 13])  # keplerian inputs, ignore inclination and target, 3D [7]
    # input_indices = np.array([12, 13])  # mass and time, 3D [2]
input_indices = np.arange(4 * n_dim + 2)  # all
########################################################################################################################


########################################################################################################################
# Define scale units for distance, time, mass and force for integration
########################################################################################################################
du = max(a0_max, af_max)
tu = (du ** 3 / gm) ** 0.5
mu = m0
fu = mu * du / tu / tu
if n_dim == 2:
    state_scales = [du, du, du / tu, du / tu, mu]
else:
    state_scales = [du, du, du, du / tu, du / tu, du / tu, mu]
########################################################################################################################


########################################################################################################################
# Define scales for the state vectors to non-dimensionalize the inputs and outputs of the neurocontroller
########################################################################################################################
input_frame = 'kep'  # 'kep', 'mee', 'car'
r_scale, v_scale = c.a_earth_km, c.vp_earth_kms  # TODO should the dimensionalization be with respect to something else?
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

max_thrust_angle = np.pi / 3
if n_dim == 2:
    # thrust angle, thrust throttle
    scales_out = np.array([[-max_thrust_angle, max_thrust_angle], [0, 1]])
else:
    # alpha, beta, throttle
    scales_out = np.array([[-max_thrust_angle, max_thrust_angle], [-max_thrust_angle, max_thrust_angle], [0, 1]])

# Specify output activation type (NOTE: this does not automatically change with the NEAT config file)
if n_dim == 2:
    out_node_scales = np.array([[-1, 1], [-1, 1]])
else:
    out_node_scales = np.array([[-1, 1], [-1, 1], [-1, 1]])

# Optionally specify a set of angle choices, and have an output node for each - categorical classification
use_multiple_angle_nodes = False
angle_choices = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
########################################################################################################################


########################################################################################################################
# For the GA (not NEAT), define network size
########################################################################################################################
n_in = 10
n_hid = 4
n_out = 2
########################################################################################################################


########################################################################################################################
# Define integration parameters
########################################################################################################################
rtol = 1e-8       # relative tolerance
atol = 1e-8       # absolute tolerance
n_steps = 20      # substeps between two nodes
segment_duration_sec = 7. * c.day_to_sec  # the time between thrust updates assuming no outages occur
########################################################################################################################


########################################################################################################################
# Define penalties used in fitness function
########################################################################################################################
# Choose to add a penalty for going too close to the central body in the fitness function
rp_penalty = True
min_allowed_rp = c.a_earth_km * 0.9
rp_penalty_multiplier = 10000

# Choose to add a penalty for not thrusting at all (just staying in initial orbit)
is_no_thrust_penalty = True
no_thrust_penalty = 1000
no_thrust_tol = 0.2  # ratio of distance away from initial orbit wrt final orbit

# Choose a penalty for trajectories that leave the allowable zone
big_penalty = 100000

# Choose maximum energy to allow before stopping integration
max_energy = - c.u_sun_km3s2 / 2 / (max(a0_max, af_max) * 1.5)
min_energy = - c.u_sun_km3s2 / 2 / (min(a0_min, af_min) * 0.5)
########################################################################################################################


########################################################################################################################
# Specify post-transfer capture maneuver settings
########################################################################################################################
do_terminal_lambert_arc = False
n_terminal_steps = 50
position_tol = 0.1  # outer non-dimensional position
capture_periapsis_alt_km = 100
capture_periapsis_radius_km = c.r_mars_km + capture_periapsis_alt_km
capture_period_day = 10
capture_low_not_high = False  # If true, capture into low circular orbit; if false, capture into high elliptic orbit
capture_current_not_optimal = True  # If true, capture at current location; if false, capture at optimal point on orbit
max_final_time = 20 * c.day_to_sec
r_limit_soi = 1  # Limit in SOI radii of how far away a capture maneuver can occur
vallado_rtol = 1e-12  # Tolerance for vallado() convergence
vallado_numiter = 50  # Number of iterations allowed in vallado()
capture_short = True
if capture_short:
    capture_time_low = 1  # minimum time of Lambert arc, days
    capture_time_high = 30 # maximum time of Lambert arc, days
else:
    capture_time_low = 300  # minimum time of Lambert arc, days
    capture_time_high = 1000  # maximum time of Lambert arc, days
########################################################################################################################


########################################################################################################################
# Specify thrust vector appearance
########################################################################################################################
arrows_with_heads = True
########################################################################################################################

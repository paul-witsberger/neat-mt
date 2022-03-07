from nnet import Neurocontroller
import boost_tbp
import neatfast as neat
import numpy as np
import traj_config as tc
import orbit_util as ou
import constants as c
import os
import pickle
from numba import njit, jit
from traj_config import gm
from constants import au_to_km
import warnings
import time


# Create 2-body integrator object and lambert function
tbp = boost_tbp.TBP()
lambert = boost_tbp.maneuvers().lambert
year_to_sec: float = c.year_to_sec
day_to_sec: float = c.day_to_sec


def eval_traj_neat(genome: neat.genome.DefaultGenome, config: neat.config.Config) -> float:
    """
    Evaluates a neural network's ability as a controller for the defined problem, using a NEAT-style network. Returns
    a scalar value representing the network's usefulness.
    :param genome:
    :param config:
    """
    # Create network and get thrust vector
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    nc = Neurocontroller.init_from_neat_net(net, tc.scales_in, tc.scales_out)
    thrust_fcn = nc.get_thrust_vec_neat

    # Initialize score vector
    f = np.empty(config.num_cases)
    f[:] = np.infty

    # Main loop - evaluate network on "num_cases" number of random cases
    for i in range(config.num_cases):
        # Create a new case based on the given boundary condition boundary conditions
        _y0, yf, times = compute_bcs()

        # Append mass to initial state
        y0 = np.empty(len(_y0) + 1)
        y0[:-1], y0[-1] = _y0, tc.m0

        # Integrate trajectory
        y, times, is_outage, full_traj, maneuvers = integrate_func_missed_thrust(thrust_fcn, y0, yf, times, config)

        # Check if integration was stopped early - if so, assign a large penalty
        if len(y.shape) == 0:
            f[i] = tc.big_penalty
            continue

        # # Get target final state
        yf_actual = y[-3, :-1]
        yf_target = yf[:]

        # Calculate final propellant mass ratio and final time ratio
        m_ratio = (y0[-1] - y[-1, -1]) / y0[-1]
        t_ratio = (times[-1] - times[-3]) / times[-3]

        # Get fitness
        f[i], dr, dv = traj_fit_func(yf_actual, yf_target, y0[:2 * tc.n_dim], m_ratio, t_ratio)

    # Calculate scalar fitness
    rdo = False
    rbdo = False
    normal = True
    if config.num_cases > 1:
        if rdo:
            # Robust Design Optimization
            alpha = 0.9  # weight to favor mean vs std
            f_mean = np.mean(f)
            f_std = np.std(f)
            f = alpha * f_mean + (1 - alpha) * f_std

        # Reliability-Based Design Optimization
        elif rbdo:
            f_threshold = 10
            cases_outside_bounds = f[f >= f_threshold]
            num_cases_outside_bounds = len(cases_outside_bounds)
            f_mean = np.mean(f)
            f_constraint_violation = num_cases_outside_bounds / config.num_cases
            c = 100.
            f = f_mean + c * f_constraint_violation

        # Mean
        elif normal:
            f = np.mean(f)

    else:
        f = f[0]

    return -f


# TODO enable variable launch and arrival start dates
def compute_bcs() -> (np.ndarray, np.ndarray):
    """
    Computes the locations of the initial and target bodies at the initial and final times in Cartesian coordinates.
    The times used are actual dates, and the states are based on analytic ephemeris.
    :return:
    """
    if tc.n_dim == 2:
        elems = ['a', 'e', 'w', 'O', 'M']
    else:
        elems = ['a', 'e', 'i', 'w', 'O', 'M']
    planets = [tc.init_body, tc.target_body]
    times = tc.times_jd1950_jc.copy()
    times[0] += (np.random.rand() - 0.5) * tc.launch_window
    times[1] += (np.random.rand() - 0.5) * tc.arrival_window

    # Get planet states at bounds
    states_coe = c.ephem(elems, planets, times)

    # Convert keplerian coordinates to inertial
    if tc.n_dim == 2:
        states_coe_2d = np.empty([4, 2, 2])
        states_coe_2d[0:2] = states_coe[0:2]
        states_coe_2d[2] = states_coe[2] + states_coe[3]
        states_coe_2d[3] = states_coe[4]
        state_0_i = ou.keplerian_to_inertial_2d(states_coe_2d[:, 0, 0], mean_or_true='mean')  # TODO verify
        state_f_i = ou.keplerian_to_inertial_2d(states_coe_2d[:, 1, 1], mean_or_true='mean')
    else:
        # states_coe[2, :, :] = 0.  # inclination
        # states_coe[4, :, :] = 0.  # longitude of ascending node
        state_0_i = ou.keplerian_to_inertial_3d(states_coe[:, 0, 0], mean_or_true='mean')
        state_f_i = ou.keplerian_to_inertial_3d(states_coe[:, 1, 1], mean_or_true='mean')

    # See if there's a non-zero launch C3
    if tc.launch_c3 > 0:
        v_inf = np.sqrt(tc.launch_c3)
        v_vec = state_0_i[tc.n_dim:]
        v_mag = ou.mag2(v_vec) if tc.n_dim == 2 else ou.mag3(v_vec)
        v_hat = v_vec / v_mag
        state_0_i[tc.n_dim:] = (v_inf + v_mag) * v_hat  # assume that launch C3 is in direction of planetary motion

    # See if there's a checkout period
    if tc.ckout_duration_sec > 0:
        eom_type = 3  # simple 2BP integration
        integrator_type = 1  # adaptive step
        traj = tbp.prop(list(state_0_i / tc.state_scales[:-1]), [0, tc.ckout_duration_sec / tc.tu], [], 2 * tc.n_dim, 2,
                        0, tc.rtol, tc.atol, 0.1, integrator_type, eom_type, tc.n_dim)
        state_0_i = np.array(traj)[-1, 1:] * tc.state_scales[:-1]

    return state_0_i, state_f_i, times


# @njit
def traj_fit_func(y: np.ndarray, yf: np.ndarray, y0: np.ndarray, m_ratio: float, t_ratio: float = 0.) \
        -> (float, float, float):
    """
    Calculates a scalar fitness value for a trajectory based on weighted sum of final state error plus the final mass.
    """

    # Convert states to keplerian elements
    if tc.n_dim == 2:
        a0, e0, w0, f0 = ou.inertial_to_keplerian_2d(y0, gm=gm)
        a1, e1, w1, f1 = ou.inertial_to_keplerian_2d(y, gm=gm)
        a2, e2, w2, f2 = ou.inertial_to_keplerian_2d(yf, gm=gm)
    else:
        a0, e0, i0, w0, om0, f0 = ou.inertial_to_keplerian_3d(y0, gm=gm)
        a1, e1, i1, w1, om1, f1 = ou.inertial_to_keplerian_3d(y, gm=gm)
        a2, e2, i2, w2, om2, f2 = ou.inertial_to_keplerian_3d(yf, gm=gm)

    # Get periapsis and apoapsis radii
    rp0 = a0 * (1 - e0)
    rp1 = a1 * (1 - e1)
    rp2 = a2 * (1 - e2)
    ra0 = a0 * (1 + e0)
    ra1 = a1 * (1 + e1)
    ra2 = a2 * (1 + e2)

    # Calculate error in states
    dr_tol_close = c.r_soi_mars / c.au_to_km
    dr_tol_far = 30 * dr_tol_close  # TODO set the multiplier in traj_config
    mag = ou.mag2 if tc.n_dim == 2 else ou.mag3
    yf_mag = mag(yf[tc.n_dim:])
    dr = mag(yf[:tc.n_dim] - y[:tc.n_dim]) / c.au_to_km
    dv = mag(yf[tc.n_dim:] - y[tc.n_dim:]) / yf_mag
    drp = abs(rp2 - rp1) / c.au_to_km
    dra = abs(ra2 - ra1) / c.au_to_km
    da = abs(a2 - a1) / c.au_to_km
    de = abs(e2 - e1)
    dw = abs(ou.fix_angle(w2 - w1, np.pi, -np.pi) / np.pi)
    df = abs(ou.fix_angle(f2 - f1, np.pi, -np.pi) / np.pi)
    dwf = abs(ou.fix_angle(f2 + w2 - f1 - w1, np.pi, -np.pi) / np.pi)
    close = False
    if dr > dr_tol_far:
        # Far away
        # penalty[:] = 100
        states = np.array([drp, dra, dw, df])
        penalty = np.array([20, 20, 20, 20], dtype=np.float64)
        # states = np.array([dr, dwf])
        # penalty = np.array([25, 25], dtype=np.float64)
    elif dr > dr_tol_close:
        # Intermediate
        # print('Intermediate')
        # penalty[:] = 20
        # states = np.array([drp, dra, dw, df])
        # penalty = np.array([10, 10, 10, 10], dtype=np.float64)
        states = np.array([dr, dv])
        penalty = np.array([15, 15], dtype=np.float64)
    else:
        # Within sphere-of-influence
        # penalty[:] = 0
        # states = np.array([drp, dra, dw, df], dtype=np.float64)
        # penalty = np.array([1, 1, 1, 1], dtype=np.float64)
        states = np.array([dr, dv])
        penalty = np.array([0, 0])
        close = True

    # Set cost function based on final trajectory type
    if tc.elliptical_final:
        state_weights = penalty
    else:
        state_weights = np.array([penalty[0], penalty[1], 0., 0.])
    mass_weight = 1
    time_weight = 50  # Lambert arc t_ratio = ~0.02
    # Observation: making mass_weight and/or time_weight too high relative to the penalties will prevent the algorithm
    # from going from intermediate to close

    weighted = states * state_weights
    squares = np.square(weighted)
    abses = np.abs(weighted)
    f = 0.
    for i in range(len(states)):
        f += max(squares[i], abses[i])
    # f += sum(abses)
    f += m_ratio * mass_weight + t_ratio * time_weight + sum(penalty)

    # Penalize going too close to central body
    if tc.rp_penalty:
        f += tc.rp_penalty_multiplier * max((1 - rp1 / tc.min_allowed_rp), 0)

    # Penalize for not leaving initial orbit
    if tc.is_no_thrust_penalty:
        kepl_scales = [ra2 - ra0, rp2 - rp0]
        dy0 = np.sum(np.abs(np.array([ra1 - ra0, rp1 - rp0]) / np.array(kepl_scales)))
        dy0 = 0. if dy0 > 0.2 else tc.no_thrust_penalty  # TODO set these values in traj_config
        f += dy0

    # Dimensionalize error values
    dr *= c.au_to_km
    dv *= yf_mag

    # If it's broken, be obnoxious
    if np.isnan(f):
        print('\n\n' + '*' * 50)
        print('FITNESS IS NAN')
        print('y = ')
        print(y)
        print('yf = ')
        print(yf)
        print('y0 = ')
        print(y0)
        print('m_ratio = ')
        print(m_ratio)
        print('t_ratio = ')
        print(t_ratio)
        print('*' * 50 + '\n\n')

    return f, dr, dv


def integrate_func_missed_thrust(thrust_fcn, y0: np.ndarray, yf: np.ndarray, times: list, config: neat.config.Config,
                                 save_full_traj: bool = False, fixed_step: bool = tc.fixed_step) \
                                -> (np.ndarray, np.ndarray, np.ndarray, list):
    """
    Integrate a trajectory using boost_2bp with each leg having fixed thrust. Updates the thrust vector between each
    leg. Optionally includes missed thrust events during the integration. Optionally computes an impulsive capture at
    the end of integration. Returns states at each thrust update (y), indices of y which experience missed thrust
    (miss_ind), states between thrust updates for smooth plotting (full_traj), and a list of capture maneuvers.

    If integration is successful, then: during training, return (y, miss_ind, full_traj, final_state_info) where
    final_state_info is a list that includes final mass after capture, final time after capture, and final state before
    capture; otherwise, during builder, return (y, miss_ind, full_traj, maneuvers) where maneuvers is a list with the
    delta V vectors and the times of flight betweeen the maneuvers. If integration is not successful, then return
    zeros for everything.
    :param thrust_fcn:
    :param y0:
    :param yf:
    :param config:
    :param save_full_traj:
    :param fixed_step:
    :return:
    """

    # Define lengths of vectors for C++
    state_size, time_size = len(y0), 2
    param_size = 17 if tc.variable_power else 5

    # Define flag for missed thrust
    tf = (times[1] - times[0]) / c.day_to_jc * c.day_to_sec
    if config.missed_thrust_allowed:
        outages = calc_mtes_v2(tf, 0, config.missed_thrust_tbe_factor, config.missed_thrust_rd_factor)
    else:
        outages = np.array([], dtype=np.float64).reshape(0, 2)

    # Create array of thrust segments based on outages

    _times = times
    times, is_outage = build_time_array(tf, outages)
    # Create placeholder matrix for trajectory - these are dimensionalized states
    y = np.empty((len(times), state_size), dtype=np.float64)

    # Assign initial condition
    y[0] = y0[:]

    # Choose type of integrator and equations of motion
    if fixed_step:
        integrator_type = 0  # 0 fixed step
    else:
        integrator_type = 1  # adaptive step
    eom_type = 0  # 0 constant power, 1 variable power, 2 state transition matrix

    mag = ou.mag2 if tc.n_dim == 2 else ou.mag3

    # Main loop
    if len(times) < 3:
        print('times = ')
        print(times)
        print('_times = ')
        print(_times)
        print('tf = %f' % tf)
    full_traj = np.empty(((len(times) - 3) * tc.n_steps, state_size + 1), dtype=np.float64)
    for i in range(len(times) - 3):
        # Check if orbital energy is within reasonable bounds - terminate integration if not
        r_mag = mag(y[i, :tc.n_dim])
        v_mag = mag(y[i, tc.n_dim:2 * tc.n_dim])
        eps = ou.energy_from_gm_r_v(tc.gm, r_mag, v_mag)
        if (eps > tc.max_energy or eps < tc.min_energy) and not save_full_traj:
            return np.array(0), 0, 0, np.array(0), [np.empty(3), np.empty(3), 0]

        # Fixed step integrator step size - take a small amount off so fixed steps land within bounds
        step_size = (times[i + 1] - times[i]) / tc.n_steps - 1e-15

        # Ratio of remaining propellant mass, and time ratio
        mass_ratio = (y[i, -1] - tc.m_dry) / (y[0, -1] - tc.m_dry)  # (curr - dry) / (init - dry) = curr_prp / init_prop
        time_ratio = times[i] / times[-1]  # curr time / final time

        # Check if i is supposed to miss thrust for this segment
        if is_outage[i] or y[i, -1] <= tc.m_dry:
            # Missed-thrust event, or no propellant remaining
            thrust = np.zeros(tc.n_dim, dtype=float)
        else:
            # Query NN to get next thrust vector
            nn_input = np.empty(tc.n_dim * 4 + 2, dtype=np.float64)
            nn_input[:tc.n_dim * 2] = y[i, :-1]
            nn_input[2 * tc.n_dim:4 * tc.n_dim] = yf
            nn_input[-2], nn_input[-1] = mass_ratio, time_ratio
            thrust = thrust_fcn(nn_input) * tc.T_max_kN / tc.fu

        # Create list of parameters to pass to integrator
        param = _update_params(thrust)

        # Propagate from the current state until the next time step
        traj = tbp.prop(list(y[i] / tc.state_scales), [times[i], times[i + 1]], param, state_size, time_size,
                        param_size, tc.rtol, tc.atol, step_size, integrator_type, eom_type, tc.n_dim)

        # Save full trajectory including intermediate states
        if save_full_traj:
            full_traj[i * tc.n_steps:(i + 1) * tc.n_steps] = traj[1:]

        # Save final state of current leg
        y[i + 1] = np.array(traj[-1])[1:] * tc.state_scales

    # Copy values from final state of integration to two capture states
    y[-1] = y[-2] = y[-3]

    # Compute maneuvers required to capture into a target orbit
    if config.do_terminal_lambert_arc:
        # TODO make this handle 2D
        # Compute relative state and position error
        if tc.n_dim == 2:
            state_f = np.empty(6)
            state_rel = np.empty(6)
            state_f[:2], state_f[2], state_f[3:5], state_f[5] = y[-3, :2], 0., y[-3, 2:4], 0.
            mf = y[-3, 4]
            state_rel[:2], state_rel[2] = state_f[:2] - yf[:2], 0.
            state_rel[3:5], state_rel[5] = state_f[3:5] - yf[2:4], 0.
        else:
            state_f = y[-3, :6]
            mf = y[-3, 6]
            state_rel = state_f - yf
        pos_error = ou.mag3(state_rel[:3])
        # Check if final position is "close" to target position
        if pos_error > tc.r_limit_soi * c.r_soi_mars:  # very far away
            # Far capture - Lambert arc
            maneuvers = ou.lambert_min_dv(tc.gm, state_f, _times[0] / c.day_to_jc * c.day_to_sec + times[-3] * tc.tu,
                                          tc.capture_time_low, tc.capture_time_high, max_count=10, short=tc.capture_short)
            gm_capture = tc.gm
            change_frame = False
            times[-1] += maneuvers[-1] * c.day_to_sec / tc.tu
        else:
            # Close capture - hyperbolic capture at Mars periapsis
            print('Performing close capture')
            state_f = state_rel
            gm_capture = tc.gm_target
            change_frame = True
            dv_capture = ou.hyperbolic_capture_from_infinity(ou.mag3(state_rel[3:]), tc.capture_periapsis_radius_km,
                                                             tc.capture_period_day * c.day_to_sec, gm_capture, False)
            mf = mf / np.exp(dv_capture * 1000 / c.g0_ms2 / tc.isp_chemical)
            maneuvers = [0., mf]
            y[-1, -1] = mf

        if save_full_traj:
            if not change_frame:
                # Compute the intermediate states of the capture
                y_capture, full_traj_capture = ou.propagate_capture(maneuvers, state_f, mf, du=ou.mag3(state_f[:3]),
                                                                    gm=gm_capture)
                # full_traj_capture[:, 0] += times[-2]  # add the time of the transfer to the time of capture
                full_traj_capture[:, 0] += tf  # add the time of the transfer to the time of capture
                # ti_capture = full_traj_capture[:, 0]

                # Append capture states to main array of states
                y[-2:] = y_capture
                full_traj = np.append(full_traj, full_traj_capture, axis=0)

            # Lambert arc is in heliocentric frame, other captures are in target body centric frame
            else:
                print('Changing frame')
                maneuvers = [np.array([dv_capture, 0, 0]), np.zeros((3))]
                # mag = ou.mag2 if tc.n_dim == 2 else ou.mag3
                # dv1_mag = mag(maneuvers[0])
                # dv2_mag = mag(maneuvers[1])
                # m1 = y[-3, -1] / np.exp(dv1_mag * 1000 / c.g0_ms2 / tc.isp_chemical)
                # m2 = m1 / np.exp(dv2_mag * 1000 / c.g0_ms2 / tc.isp_chemical)
                # y_capture = np.vstack((y[-3], y[-3]))
                # y_capture[:, -1] = m1, m2
                # full_traj_capture = full_traj[-1].reshape((1, -1))

        elif not change_frame:
            tof_capture, mf = ou.get_capture_final_values(maneuvers, mf)
            # If it's broken, be obnoxious
            if np.isnan(mf):
                print('\n\n' + '*' * 50)
                print('FINAL MASS IS NAN')
                if change_frame:
                    print('capture type = close')
                    print('other inputs = ')
                    print([tc.capture_periapsis_radius_km, tc.capture_period_day * c.day_to_sec,
                           tc.gm_target, tc.capture_low_not_high])
                else:
                    print('capture type = far')
                    print('other inputs = ')
                    print([tc.gm, times[-3] * tc.tu, tc.capture_time_low, tc.capture_time_high, 10, tc.capture_short])
                print('state_f = ')
                print(state_f)
                print('maneuvers = ')
                print(maneuvers)
                print('*' * 50 + '\n\n')
            maneuvers = tof_capture, mf
            y[-1, -1] = mf

    else:
        # No maneuver
        maneuvers = [np.zeros(3, float), np.zeros(3, float), 0]
        y[-1] = y[-2] = y[-3]
        times[-1] = times[-2] = times[-3]

    # Dimensionalize states
    if save_full_traj:
        full_traj[:, 1:] = full_traj[:, 1:] * tc.state_scales

    return y, times, is_outage, full_traj, maneuvers


@jit(nopython=True, cache=False)
def calc_mtes_v2(tf: float, t0: float = 0., tbe_factor: float = 1., rd_factor: float = 1.) -> np.ndarray:
    # Define Weibull distribution parameters
    k_tbe, lambda_tbe = 0.86737, 0.62394 * tbe_factor
    k_rd, lambda_rd = 1.144, 2.459 * rd_factor
    max_discovery_delay_days, op_recovery_days = 3, 0.5
    # Initialize list of outage start and stop times
    outages = list()
    time_between_events, recovery_duration, cascading = 0., 0., False
    prev_start = t0
    while True:
        time_between_events = np.random.weibull(k_tbe) * lambda_tbe * year_to_sec  # calculate time between events
        # Check that the previous outage has already finished.
        if len(outages) > 0:
            if time_between_events < (outages[-1][1] - outages[-1][0]):
                cascading = True
            else:
                cascading = False
        else:
            cascading = False
        # Check if the next event happens before the end of the time of flight
        if prev_start + time_between_events < tf:
            recovery_duration = np.random.weibull(k_rd) * lambda_rd * day_to_sec  # calculate the recovery duration
            recovery_duration = recovery_duration + (np.random.rand() * max_discovery_delay_days * day_to_sec) + \
                                 (op_recovery_days * day_to_sec)  # discovery delay and operational recovery
            if cascading:
                outages[-1][1] = outages[-1][0] + time_between_events + recovery_duration
            else:
                outages.append([prev_start + time_between_events, prev_start + time_between_events + recovery_duration])
            prev_start = outages[-1][0]
        # Return outages as an array if final time is reached
        else:
            if len(outages) > 0:
                return np.array(outages)
            else:
                return np.empty((0, 2), dtype=np.float64)


@jit(nopython=True, cache=False)
def build_time_array(tf, outages):
    outages_remaining = outages.shape[0]
    is_outage = [False]
    if outages_remaining == 0:
        times = np.empty(int(np.ceil(tf / tc.segment_duration_sec)) + 3)
        times[:-3] = np.arange(0, tf, tc.segment_duration_sec) / tc.tu
        times[-3:] = tf / tc.tu
        is_outage = [False] * (times.shape[0] - 3)
        return times, is_outage
    else:
        _times = [0.]
        is_outage.pop()
        while True:
            # Check if we should keep going
            if _times[-1] < tf:
                if _times[-1] + tc.segment_duration_sec < tf:
                    next_time = _times[-1] + tc.segment_duration_sec
                else:
                    next_time = tf
                # Check if we need to add a partial or a full segment
                if outages_remaining > 0 and outages[-outages_remaining, 0] < next_time:
                    _times.append(outages[-outages_remaining, 0])
                    _times.append(outages[-outages_remaining, 1])
                    outages_remaining -= 1
                    is_outage.append(False)
                    is_outage.append(True)
                else:
                    _times.append(next_time)
                    is_outage.append(False)
            else:
                _times.append(_times[-1])
                _times.append(_times[-1])
                return np.array(_times) / tc.tu, is_outage


def _update_params(thrust):
    if tc.variable_power:
        return [*thrust, float(tc.m_dry / tc.mu), tc.power_reference, tc.power_min, tc.power_max,
                float(tc.thrust_power_coef[0]), float(tc.thrust_power_coef[1]), float(tc.thrust_power_coef[2]),
                float(tc.thrust_power_coef[3]), float(tc.thrust_power_coef[4]),
                float(tc.isp_power_coef[0]), float(tc.isp_power_coef[1]), float(tc.isp_power_coef[2]),
                float(tc.isp_power_coef[3]), float(tc.isp_power_coef[4])]
    else:
        return [*thrust, float(c.g0_ms2 * tc.Isp / tc.du * tc.tu), float(tc.m_dry / tc.mu)]


@jit(nopython=True, cache=True)
def calculate_missed_thrust_events(ti: np.ndarray, tbe_factor=1.0, rd_factor=1.0) -> np.ndarray:
    """
    Return indices of random missed thrust events that occur according to the Weibull distributions given by Imken et al
    :param ti:
    :param tbe_factor:
    :param rd_factor:
    :return:
    """
    # Define Weibull distribution parameters
    k_tbe, lambda_tbe = 0.86737, 0.62394 * tbe_factor
    k_rd, lambda_rd = 1.144, 2.459 * rd_factor
    max_discovery_delay_days, op_recovery_days = 3, 0.5
    # initialize list of indices that experience missed thrust
    miss_indices = np.empty_like(ti, dtype=np.int64)
    t, tf = 0., ti[-1]
    ctr = 0
    while True:
        time_between_events = np.random.weibull(k_tbe) * lambda_tbe * year_to_sec  # calculate time between events
        t += time_between_events
        if t < tf:  # check if the next event happens before the end of the time of flight
            # while True:
            recovery_duration = np.random.weibull(k_rd) * lambda_rd * day_to_sec  # calculate the recovery duration
            recovery_duration = recovery_duration + (np.random.rand() * max_discovery_delay_days * day_to_sec) + \
                                 (op_recovery_days * day_to_sec)  # discovery delay and operational recovery
            start = find_nearest(ti, t)
            stop = max(start + 1, find_nearest(ti, t + recovery_duration))
            miss_indices[ctr:ctr + (stop - start)] = np.arange(start, stop)
            ctr += stop - start
            t += recovery_duration
        if t >= tf:
            return miss_indices[:ctr]  # TODO make sure this works if no outages occur


@jit(nopython=True, cache=True)
def find_nearest(array: np.ndarray, value: float) -> int:
    """
    Helper function that finds the closest index of an array to a specified value
    :param array:
    :param value:
    :return:
    """
    idx = (np.abs(array - value)).argmin()
    return idx


def calculate_prop_margins(genome: neat.genome.DefaultGenome, config: neat.config.Config) -> np.ndarray:
    """
    Evaluates a neural network's ability as a controller for the defined problem, using a NEAT-style network.
    :param genome:
    :param config:
    :return:
    """

    # warnings.warn('This function has not been tested since major code updates.')

    # Create network and get thrust vector
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    nc = Neurocontroller.init_from_neat_net(net, tc.scales_in, tc.scales_out)
    thrust_fcn = nc.get_thrust_vec_neat

    # Initialize score vector
    # f = np.empty(config.num_cases)
    # f[:] = np.infty

    # Main loop - evaluate network on "num_cases" number of random cases
    # Initialize score vector
    num_cases = 1000
    mf = np.empty(num_cases + 1, float)
    dr = np.empty_like(mf)
    dv = np.empty_like(mf)

    for i in range(num_cases + 1):
        if i == 0: # or not config.missed_thrust_allowed:
            missed_thrust_allowed = False
        else:
            missed_thrust_allowed = True
        config.missed_thrust_allowed = missed_thrust_allowed
        # Create a new case based on the given boundary condition boundary conditions
        _y0, yf, times = compute_bcs()

        # Append mass to initial state
        y0 = np.empty(len(_y0) + 1)
        y0[:-1], y0[-1] = _y0, tc.m0

        # Integrate trajectory
        y, times, is_outage, full_traj, maneuvers = integrate_func_missed_thrust(thrust_fcn, y0, yf, times, config)

        # Check if integration was stopped early - if so, assign a large penalty
        if len(y.shape) == 0:
            f = tc.big_penalty
            continue

        # # Get target final state
        yf_actual = y[-3, :-1]
        yf_target = yf[:]

        # Calculate final propellant mass ratio and final time ratio
        m_ratio = (y0[-1] - y[-1, -1]) / y0[-1]
        t_ratio = (times[-1] - times[-3]) / times[-3]

        # Get fitness
        f, _dr, _dv = traj_fit_func(yf_actual, yf_target, y0[:2 * tc.n_dim], m_ratio, t_ratio)

        # Save final mass
        # print(y[-1, -1])
        mf[i] = y[-1, -1]
        dr[i] = _dr
        dv[i] = _dv

    # Calculate propellant margin
    m_star = mf[0]
    m_prop_star = y0[-1] - m_star
    m_tilde = mf[1:]
    margin_mean = np.mean((m_star - m_tilde) / m_prop_star)
    margin_std = np.std((m_star - m_tilde) / m_prop_star)
    margin = margin_mean
    print('*********')
    print(margin_mean)
    print(margin_std)

    print(np.mean(m_tilde))
    print(np.std(m_tilde))

    dr_a = np.mean(dr)
    dr_s = np.std(dr)
    dv_a = np.mean(dv)
    dv_s = np.std(dv)

    print(dr_a)
    print(dr_s)
    print(dv_a)
    print(dv_s)

    return margin


def run_margins(fname: str = 'final'):
    """
    Runner function to calculate propellant margins for a given neural network.
    :return:
    """
    # warnings.warn('This function has not been tested since major code updates.')
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config', 'config_' + fname)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    with open('results//winner_' + fname, 'rb') as f:
        genome = pickle.load(f)
    margin = calculate_prop_margins(genome, config)


if __name__ == '__main__':
    test1 = True
    if test1:
        run_margins('final')

    test2 = False
    if test2:
        bc0, bcf = compute_bcs()
        print(bc0)
        print(bcf)

    test3 = False
    if test3:
        y = np.array([1.99306704e+08, -7.70086431e+07, 0, 9.75850417e+00, 2.36332437e+01, 0])
        yf = np.array([1.89499860e+08, -8.24217667e+07, 0, 1.06066182e+01, 2.42838341e+01, 0])
        y0 = np.array([1.49805074e+08, 7.10550646e+06, 0, -1.89741489e+00, 2.96493549e+01, 00])
        m_ratio = 0.7004071117860589
        t_ratio = 0.06
        print(traj_fit_func(y, yf, y0, m_ratio, t_ratio))


# TODO look into rendezvous with a moving target - e.g. Gateway, hyperbolic rendezvous,
#      GEO/LEO non-thrusting, GEO/LEO active avoidance

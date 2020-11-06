from nnet import Neurocontroller
import boost_tbp
import neatfast as neat
import numpy as np
import traj_config as tc
import orbit_util as ou
import constants as c
import os
import pickle
from numba import njit
from traj_config import gm, ind_dim
from constants import au_to_km


# Create 2-body integrator object
tbp = boost_tbp.TBP()
year_to_sec = c.year_to_sec
day_to_sec = c.day_to_sec


def eval_traj_neat(genome: neat.genome.DefaultGenome, config: neat.config.Config) -> float:
    """
    Evaluates a neural network's ability as a controller for the defined problem, using a NEAT-style network. Returns
    a scalar value representing the network's usefulness.
    :param genome:
    :param config:
    """
    # t_act_0 = time.time()
    # Create network and get thrust vector
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    nc = Neurocontroller.init_from_neat_net(net, tc.scales_in, tc.scales_out)
    thrust_fcn = nc.get_thrust_vec_neat

    if config.do_terminal_lambert_arc is None:
        _do_capture = tc.do_terminal_lambert_arc
    else:
        _do_capture = config.do_terminal_lambert_arc
    # Define times
    ti = np.empty(config.num_nodes + 1)
    # ti = np.power(np.linspace(0, 1, num_nodes), 3 / 2) * (tf - t0) + t0
    ti[:-1] = np.linspace(tc.t0, tc.tf, config.num_nodes)
    ti[-1] = ti[-2]
    ti /= tc.tu

    # Initialize score vector
    f = np.empty(config.num_cases)
    f[:] = np.infty

    # Main loop - evaluate network on "num_cases" number of random cases
    for i in range(config.num_cases):
        # Create a new case based on the given boundary condition boundary conditions
        _y0, yf = compute_bcs()  # formerly make_new_bcs()

        # Append mass to initial state
        y0 = np.empty(len(_y0) + 1)
        y0[:-1], y0[-1] = _y0, tc.m0

        # Integrate trajectory
        y, miss_ind, full_traj, maneuvers = integrate_func_missed_thrust(thrust_fcn, y0, ti, yf, config)
        # dv1, dv2 = maneuvers[:2]
        # mf, tf, yf_actual = maneuvers

        # Check if integration was stopped early - if so, assign a large penalty
        if len(y.shape) == 0:
            f[i] = tc.big_penalty
            continue

        # # Get target final state
        yf_actual = y[-2, tc.ind_dim]
        # yf_actual[tc.n_dim:] -= dv1[:tc.n_dim]
        yf_target = yf[tc.ind_dim[:-1]]

        # Calculate final propellant mass ratio and final time ratio
        m_ratio = (y0[-1] - y[-1, -1]) / y0[-1]
        t_ratio = ti[-1] / ti[-2]

        # Get fitness
        f[i], dr, dv = traj_fit_func(yf_actual, yf_target, y0[:6], m_ratio, t_ratio)

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

        # TODO Reliability-Based Design Optimization
        elif rbdo:
            raise NotImplementedError('RBDO has not been implemented yet - need to determine how to compute cases that'
                                      ' are "outside bounds".')
            # f_mean = np.mean(f)
            # f_constraint_violation = num_cases_outside_bounds / config.num_cases
            # c = 100.
            # f = f_mean + c * f_constraint_violation

        # Mean
        elif normal:
            f = np.mean(f)

    else:
        f = f[0]

    # t_act_f = time.time()
    # print('Time to compute activation: %.2e sec' % (t_act_f - t_act_0))

    return -f


# DEPRECATED
def make_new_bcs(true_final_f: bool = tc.true_final_f) -> (np.ndarray, np.ndarray):
    while True:
        a0 = np.random.rand() * (tc.a0_max - tc.a0_min) + tc.a0_min
        af = np.random.rand() * (tc.af_max - tc.af_min) + tc.af_min
        if np.abs((af - a0) / min(af, a0)) > 0.1:
            break
    e0 = np.random.rand() * (tc.e0_max - tc.e0_min) + tc.e0_min
    ef = np.random.rand() * (tc.ef_max - tc.ef_min) + tc.ef_min
    w0 = np.random.rand() * (tc.w0_max - tc.w0_min) + tc.w0_min
    wf = np.random.rand() * (tc.wf_max - tc.wf_min) + tc.wf_min
    f0 = np.random.rand() * (tc.f0_max - tc.f0_min) + tc.f0_min
    if true_final_f:
        assert tc.f0_ref is not None and tc.ff_ref is not None
        f_frac = (f0 - tc.f0_min) / (tc.f0_max - tc.f0_min)
        ff = f_frac * (tc.ff_max - tc.ff_min) + tc.ff_min
    else:
        ff = np.random.rand() * (tc.ff_max - tc.ff_min) + tc.ff_min
    
    if tc.n_dim == 2:
        y0 = ou.keplerian_to_inertial_2d(np.array([a0, e0, w0, f0]), gm=tc.gm)
        yf = ou.keplerian_to_inertial_2d(np.array([af, ef, wf, ff]), gm=tc.gm)
        y0 = np.insert(y0, 2, 0.)
        y0 = np.insert(y0, 5, 0.)
        yf = np.insert(yf, 2, 0.)
        yf = np.insert(yf, 5, 0.)
    else:
        i0 = np.random.rand() * (tc.i0_max - tc.i0_min) + tc.i0_min
        ifinal = np.random.rand() * (tc.if_max - tc.if_min) + tc.if_min
        om0 = np.random.rand() * (tc.om0_max - tc.om0_min) + tc.om0_min
        omf = np.random.rand() * (tc.omf_max - tc.omf_min) + tc.omf_min
        y0 = ou.keplerian_to_inertial_3d(np.array([a0, e0, i0, w0, om0, f0]), gm=tc.gm)
        yf = ou.keplerian_to_inertial_3d(np.array([af, ef, ifinal, wf, omf, ff]), gm=tc.gm)
    y0 = np.append(y0, tc.m0)
    yf = yf.ravel()
    return y0, yf


def compute_bcs() -> (np.ndarray, np.ndarray):
    """
    Computes the locations of the initial and target bodies at the initial and final times in Cartesian coordinates.
    The times used are actual dates, and the states are based on analytic ephemeris.
    :return:
    """
    elems = ['a', 'e', 'i', 'w', 'O', 'M']
    planets = [tc.init_body, tc.target_body]
    times = np.array([tc.times_jd1950_jc[0], tc.times_jd1950_jc[-1]])
    states_coe = c.ephem(elems, planets, times)
    states_coe[2, :, :] = 0.  # inclination
    # states_coe[4, :, :] = 0.  # longitude of ascending node
    state_0_i = ou.keplerian_to_inertial_3d(states_coe[:, 0, 0], mean_or_true='mean')
    state_f_i = ou.keplerian_to_inertial_3d(states_coe[:, 1, 1], mean_or_true='mean')
    return state_0_i, state_f_i


# DEPRECATED
"""
def eval_traj_ga(weights: np.ndarray, params: list) -> float:
    # Evaluates a NN that was trained with a genetic algorithm.
    # :param weights:
    # :param params:
    # :return:
    raise NotImplementedError()
    # # Extract parameters
    # n_in, n_hid, n_out, scales_in, scales_out, t0, tf, y0, yf, m_dry, T_max_kN, tol, num_nodes, num_cases,
    # num_outages = params
    # # Construct NN
    # nc = Neurocontroller(scales_in, scales_out, n_in, n_hid, n_out)
    # nc.setWeights(W)
    # thrust_fcn = nc.getThrustVec
    # # Define time vector
    # # ti = np.linspace(t0, tf, num_nodes)
    # ti = np.power(np.linspace(0, 1, num_nodes), 3 / 2) * (tf - t0) + t0
    # # Define scaling parameters
    # du = 6371.0
    # tu = np.sqrt(du**3 / gm)
    # mu = y0[-1]
    # fu = mu * du / tu / tu
    # # Initialize score vector
    # f = np.ones(num_cases) * np.inf
    # for i in range(num_cases):
    #     # Integrate trajectory
    #     y, miss_ind = integrate_func_missed_thrust(thrust_fcn, y0, ti, yf, m_dry, T_max_kN, du, tu, mu, fu)
    #     # Check if integration was stopped early
    #     if len(y.shape) == 0:
    #         if y == -1:
    #             return 1000
    #         else:
    #             frac = y / len(ti)
    #             return (1 - frac) * 500 + 500
    #     # Create logical list for indices of 2D components
    #     # ind_2d = [True, True, False, True, True, False, False]
    #     # Get final state
    #     yf_actual = y[-1, ind_dim]
    #     yf_target = yf[ind_dim[:-1]]
    #     # Calculate ratio of initial mass to final mass
    #     m_ratio = y0[-1] / y[-1, -1]
    #     f[i] = traj_fit_func(yf_actual, yf_target, m_ratio)
    # f = np.mean(f)
    # if f < 10:
    #     blah = 0
    # return f
"""


@njit
def traj_fit_func(y: np.ndarray, yf: np.ndarray, y0: np.ndarray, m_ratio: float, t_ratio: float = 0.) \
        -> (float, float, float):
    """
    Calculates a scalar fitness value for a trajectory based on weighted sum of final state error plus the final mass.
    """

    # Convert states to keplerian elements
    if tc.n_dim == 2:
        a0, e0, w0, f0 = ou.inertial_to_keplerian_2d(y0[ind_dim], gm=gm)
        a1, e1, w1, f1 = ou.inertial_to_keplerian_2d(y, gm=gm)
        a2, e2, w2, f2 = ou.inertial_to_keplerian_2d(yf, gm=gm)
    else:
        a0, e0, i0, w0, om0, f0 = ou.inertial_to_keplerian_3d(y0[:6], gm=gm)
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
    dr_tol_close = 0.00385 * au_to_km / tc.a0_max  # TODO where does 0.00385 come from?
    dr_tol_far = 0.3 * au_to_km / tc.a0_max  # TODO where does 0.3 come from?
    yf_mag = ou.mag3(yf[tc.n_dim:])
    # dv_tol = 0.241 / yf_mag
    dr = ou.mag3(yf[:tc.n_dim] - y[:tc.n_dim]) / tc.a0_max  # TODO if 2D, would need mag2 instead of mag3
    dv = ou.mag3(yf[tc.n_dim:] - y[tc.n_dim:]) / yf_mag
    drp = abs(rp2 - rp1) / tc.a0_min
    dra = abs(ra2 - ra1) / tc.a0_min
    dw = abs(ou.fix_angle(w2 - w1, np.pi, -np.pi) / np.pi)
    df = abs(ou.fix_angle(f2 - f1, np.pi, -np.pi) / np.pi)
    penalty = np.empty(4)
    if dr > dr_tol_far:
        # Far away
        penalty[:] = 100
        # penalty = np.array([100, 100, 80, 80])
        states = np.array([drp, dra, dw, df])
    else:
        states = np.array([drp, dra, dr, dv])
        if dr < dr_tol_close:
            # Within sphere-of-influence
            penalty[:] = 10
        else:
            # Intermediate
            penalty[:] = 50

    # Set cost function based on final trajectory type
    if tc.elliptical_final:
        state_weights = penalty
    else:
        state_weights = np.array([penalty[0], penalty[1], 0., 0.])
    mass_weight = 5
    time_weight = 20

    weighted = states * state_weights
    squares = np.square(weighted)
    abses = np.abs(weighted)
    f = 0.
    for i in range(4):
        f += max(squares[i], abses[i])
    f += m_ratio * mass_weight + (t_ratio - 1) * time_weight + penalty[0]

    # f = np.sum(np.max((np.square(states * state_weights),
    #                    np.abs(   states * state_weights)), axis=0))\
    #     + m_ratio * mass_weight + (t_ratio - 1) * time_weight + penalty[0]

    # Penalize going too close to central body
    if tc.rp_penalty:
        if rp1 < tc.min_allowed_rp:
            f += tc.rp_penalty_multiplier * (1 - rp1 / tc.min_allowed_rp)

    # Penalize for not leaving initial orbit
    if tc.no_thrust_penalty:
        kepl_scales = [ra2 - ra0, rp2 - rp0]
        dy0 = np.sum(np.abs(np.array([ra1 - ra0, rp1 - rp0]) / np.array(kepl_scales)))
        dy0 = 0. if dy0 > 0.2 else 1000.
        f += dy0

    # Dimensionalize error values
    dr *= tc.a0_max
    dv *= yf_mag

    return f, dr, dv


# TODO go back and look into redoing how time vector is saved - could probably adjust it to account for updated
#      maneuver calculations
def integrate_func_missed_thrust(thrust_fcn: Neurocontroller.get_thrust_vec_neat, y0: np.ndarray, ti: np.ndarray,
                                 yf: np.ndarray, config=neat.config.Config, save_full_traj: bool = False,
                                 fixed_step: bool = tc.fixed_step) -> \
                                (np.ndarray, np.ndarray, np.ndarray, list):
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
    :param ti:
    :param yf:
    :param config:
    :param save_full_traj:
    :param fixed_step:
    :return:
    """

    # Define lengths of vectors for C++
    state_size, time_size = len(y0), 2
    param_size = 17 if tc.variable_power else 5

    # Create placeholder matrix for trajectory - these are dimensionalized states
    y = np.empty((len(ti), state_size), dtype=np.float64)

    # Assign initial condition
    y[0] = y0[:]

    # Define flag for missed thrust
    if config.missed_thrust_allowed:
        miss_ind = calculate_missed_thrust_events(ti * tc.tu, config.missed_thrust_tbe_factor,
                                                  config.missed_thrust_rd_factor)
    else:
        miss_ind = np.array([], dtype=np.int64)

    # Main loop
    # Choose type of integrator and equations of motion
    if fixed_step:
        integrator_type = 0  # 0 fixed step
    else:
        integrator_type = 1  # adaptive step
    eom_type = 0  # 0 constant power, 1 variable power, 2 state transition matrix

    full_traj = np.empty((len(ti) * tc.n_steps, state_size + 1), dtype=np.float64)
    for i in range(len(ti) - 2):
        # t_act_0 = time.process_time()
        # Check if orbital energy is within reasonable bounds - terminate integration if not
        r_mag = ou.mag3(y[i, :3])
        v_mag = ou.mag3(y[i, 3:6])
        eps = ou.energy_from_gm_r_v(tc.gm, r_mag, v_mag)
        if (eps > tc.max_energy or eps < tc.min_energy) and not save_full_traj:
            return np.array(0), 0, 0, [np.empty(3), np.empty(3), 0]

        # Fixed step integrator step size
        step_size = (ti[i+1] - ti[i]) / tc.n_steps - 1e-16  # take a small amount off so fixed steps land within bounds

        # Ratio of remaining propellant mass, and time ratio
        mass_ratio = (y[i, -1] - tc.m_dry) / (y[0, -1] - tc.m_dry)  # (curr - dry) / (init - dry) = curr_prp / init_prp
        time_ratio = ti[i] / ti[-2]  # curr time / final time

        # Check if i is supposed to miss thrust for this segment
        if i in miss_ind or y[i, -1] <= tc.m_dry + 0.01:
            # Missed-thrust event, or no propellant remaining
            thrust = np.array([0., 0, 0])
        else:
            # Query NN to get next thrust vector
            # nn_input = np.hstack((y[i, tc.ind_dim], yf[tc.ind_dim[:-1]], mass_ratio, time_ratio))
            nn_input = np.empty(tc.n_dim * 4 + 2, dtype=np.float64)
            nn_input[:tc.n_dim * 2] = y[i, tc.ind_dim]
            nn_input[2 * tc.n_dim:4 * tc.n_dim] = yf[tc.ind_dim[:-1]]
            nn_input[-2], nn_input[-1] = mass_ratio, time_ratio
            thrust = thrust_fcn(nn_input) * tc.T_max_kN / tc.fu

        # Create list of parameters to pass to integrator
        if tc.variable_power:
            param = [float(tc.m_dry / tc.mu), float(thrust[0]), float(thrust[1]), float(thrust[2]), tc.power_reference,
                     tc.power_min, tc.power_max, float(tc.thrust_power_coef[0]), float(tc.thrust_power_coef[1]),
                     float(tc.thrust_power_coef[2]), float(tc.thrust_power_coef[3]), float(tc.thrust_power_coef[4]),
                     float(tc.isp_power_coef[0]), float(tc.isp_power_coef[1]), float(tc.isp_power_coef[2]),
                     float(tc.isp_power_coef[3]), float(tc.isp_power_coef[4])]
        else:
            param = [float(c.g0_ms2 * tc.Isp / tc.du * tc.tu), float(tc.m_dry / tc.mu), float(thrust[0]),
                     float(thrust[1]), float(thrust[2])]

        # Propagate from the current state until the next time step
        # traj = tbp.prop(list(y[i] / tc.state_scales), [ti[i], ti[i+1]], param, state_size, time_size, param_size,
        #                 tc.rtol, tc.atol, step_size, integrator_type, eom_type)
        asdf = (list(y[i] / tc.state_scales), [ti[i], ti[i + 1]], param, state_size, time_size, param_size,
                tc.rtol, tc.atol, step_size, integrator_type, eom_type)
        traj = asdfasdf(*asdf)

        # Save full trajectory including intermediate states
        if save_full_traj:
            try:
                full_traj[i * tc.n_steps:(i + 1) * tc.n_steps] = traj[1:]
            except ValueError as execption:
                full_traj[i * tc.n_steps:(i + 1) * tc.n_steps - 1] = traj[1:]
                n_skipped
                print('i = %i' % i)
        # Save final state of current leg
        y[i+1] = np.array(traj[-1])[1:] * tc.state_scales
        # t_act_f = time.process_time()
        # print('Time to propagate: %.2e sec' % (t_act_f - t_act_0))

    # Compute maneuvers required to capture into a target orbit
    if config.do_terminal_lambert_arc:
        state_f = y[-2, :6]
        mf = y[-2, 6]
        state_rel = state_f - yf
        pos_error = ou.mag3(state_rel[:3])
        # Check if final position is "close" to target position - if not, compute a Lambert arc to match target state
        if pos_error > tc.r_limit_soi * c.r_soi_mars:  # very far away
            # TODO convert ti[-1] to appropriate units
            maneuvers = ou.lambert_min_dv(tc.gm, state_f, ti[-1], tc.capture_time_low, tc.capture_time_high,
                                          short=tc.capture_short)
            # maneuvers = ou.lambert_min_dv(tc.gm, state_f, yf)
            gm_capture = tc.gm
            change_frame = False
        else:
            # dv1, dv2, tof = ou.min_dv_capture(y[-2, :6], yf, c.u_mars_km3s2, tc.capture_periapsis_radius_km)
            rp_target = c.r_mars_km + tc.capture_periapsis_alt_km
            maneuvers = ou.capture(state_rel, rp_target, tc.capture_period_day, tc.gm, c.r_soi_mars,
                                   tc.capture_low_not_high, tc.capture_current_not_optimal)
            state_f = state_rel
            gm_capture = tc.gm_target
            change_frame = True

        if save_full_traj:
            # Compute the intermediate states of the capture
            y_capture, full_traj_capture = ou.propagate_capture(maneuvers, state_f, mf, du=ou.mag3(state_f[:3]),
                                                                gm=gm_capture)
            full_traj_capture[:, 0] += ti[-1]  # add the time of the transfer to the time of capture
            ti_capture = full_traj_capture[:, 0]

            # Lambert arc is in heliocentric frame, other captures are in target body centric frame
            if change_frame:
                # Compute states of target body (Mars) relative to sun throughout the capture
                state_sun_mars = c.ephem(['a', 'e', 'i', 'w', 'O', 'M'], [tc.target_body],
                                         ti_capture + tc.times_jd1950_jc[-1])
                state_sun_mars = ou.keplerian_to_inertial_3d(state_sun_mars, gm=tc.gm, mean_or_true='mean')

                # Pull out indices of full_traj_capture corresponding to y_capture
                if full_traj_capture.shape[0] == 2 * tc.n_terminal_steps + 1:
                    state_sun_mars_y_ind = state_sun_mars[np.array([0, tc.n_terminal_steps,
                                                                    2 * tc.n_terminal_steps])].astype(int)
                else:
                    state_sun_mars_y_ind = state_sun_mars[np.array([0, tc.n_terminal_steps])].astype(int)

                # Add relative states to Mars' states
                y_capture[:, :6] += state_sun_mars_y_ind
                full_traj_capture[:, 1:-1] += state_sun_mars

            # Append capture states to main array of states
            # TODO can we allocate y properly ahead of time - we would know tc.do_terminal_lambert_arc initially
            y = np.append(y, y_capture, axis=0)
            full_traj = np.append(np.array(full_traj), full_traj_capture, axis=0)  # TODO same as above

        else:
            tof_capture, mf = ou.get_capture_final_values(maneuvers, mf)
            maneuvers = tof_capture, mf

        # # Add time of flight to time vector
        # ti[-1] = ti[-2] + tof / tc.tu

    else:
        # No maneuver
        maneuvers = [np.zeros(3, float), np.zeros(3, float), 0]
        y[-1] = y[-2]
        ti[-1] = ti[-2]  # this is used later since ti is passed by reference
        # full_traj = np.array(full_traj)
        full_traj = full_traj[:-(2 * tc.n_steps)]

    # Dimensionalize states
    if save_full_traj:
        full_traj[:, 1:] = full_traj[:, 1:] * tc.state_scales

    return y, miss_ind, full_traj, maneuvers


def asdfasdf(*args):
    return tbp.prop(*args)


@njit
def calculate_missed_thrust_events(ti: np.ndarray, tbe_factor=1.0, rd_factor=1.0) -> np.ndarray:
    """
    Return indices of random missed thrust events that occur according to the Weibull distributions given by Imken et al
    :param ti:
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
            recovery_duration += (np.random.rand() * max_discovery_delay_days * day_to_sec) + \
                                 (op_recovery_days * day_to_sec)  # discovery delay and operational recovery
            start = find_nearest(ti, t)
            stop = max(start + 1, find_nearest(ti, t + recovery_duration))
            miss_indices[ctr:ctr + (stop - start)] = np.arange(start, stop)
            ctr += stop - start
            t += recovery_duration
        if t >= tf:
            return miss_indices[:ctr]  # TODO make sure this works if no outages occur


@njit
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

    # Scales for network inputs
    # du = np.max((a0_max, af_max))
    # tu = np.sqrt(du ** 3 / gm)
    # mu = m0
    # fu = mu * du / tu / tu

    # Create network and get thrust vector
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    nc = Neurocontroller.init_from_neat_net(net, tc.scales_in, tc.scales_out)
    thrust_fcn = nc.get_thrust_vec_neat

    # Define times
    # ti = np.power(np.linspace(0, 1, num_nodes), 3 / 2) * (tf - t0) + t0
    ti = np.linspace(tc.t0, tc.tf, config.num_nodes)
    ti /= tc.tu

    # Create a new case based on the given boundary condition boundary conditions
    # y0, yf = make_new_bcs()
    y0, yf = compute_bcs()
    y0 = np.hstack((y0, tc.m0))
    print(y0)
    print(yf)

    # Initialize score vector
    num_cases = 1000
    mf = np.empty(num_cases + 1, np.float)
    dr = np.empty_like(mf)
    dv = np.empty_like(mf)

    for i in range(num_cases + 1):
        if i == 0 or not tc.missed_thrust_allowed:
            missed_thrust_allowed = False
        else:
            missed_thrust_allowed = True

        # Integrate trajectory
        y, miss_ind, full_traj, dv1, dv2 = integrate_func_missed_thrust(thrust_fcn, y0, ti, yf, missed_thrust_allowed)

        yf_actual = y[-2, tc.ind_dim]
        f, _dr, _dv = traj_fit_func(yf_actual, yf[tc.ind_dim[:-1]], y0, (y0[-1] - y[-1, -1]) / y0[-1], ti[-1] / ti[-2])

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


def run_margins():
    """
    Runner function to calculate propellant margins for a given neural network.
    :return:
    """
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config', 'config_default')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    with open('results//winner', 'rb') as f:
        genome = pickle.load(f)
    margin = calculate_prop_margins(genome, config)
    print(margin)


if __name__ == '__main__':
    test1 = False
    if test1:
        run_margins()

    test2 = False
    if test2:
        bc0, bcf = compute_bcs()
        print(bc0)
        print(bcf)


# TODO look into rendezvous with a moving target - e.g. Gateway, hyperbolic rendezvous,
#      GEO/LEO non-thrusting, GEO/LEO active avoidance

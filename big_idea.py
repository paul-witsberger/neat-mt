# Run the GA for:
# - first, one set of boundary conditions; each controller is used on several trajectories that each have a random single outage
# - next, each controller is used on several trajectories that have independent boundary conditions and random outages
# - next, include multiple outages

from nnet import Neurocontroller
import boost_tbp
import time
import neatfast as neat
from traj_config import *
from orbit_util import *
from copy import copy
import os
import pickle


def eval_traj_neat(genome, config):
    """
    Evaluates a neural network's ability as a controller for the defined problem, using a NEAT-style network.
    :param genome:
    :param config:
    :return:
    """

    # Scales for network inputs
    du = np.max((a0_max, af_max))
    tu = np.sqrt(du ** 3 / gm)
    mu = m0
    fu = mu * du / tu / tu

    # Create network and get thrust vector
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    nc = Neurocontroller.init_from_neat_net(net, scales_in, scales_out)
    thrust_fcn = nc.get_thrust_vec_neat

    # Define times
    # ti = np.power(np.linspace(0, 1, num_nodes), 3 / 2) * (tf - t0) + t0
    ti = np.linspace(t0, tf, num_nodes)
    ti = np.append(ti, ti[-1])
    ti /= tu

    # Initialize score vector
    f = np.ones(num_cases) * np.inf
    for i in range(num_cases):
        # Create a new case based on the given boundary condition boundary conditions
        y0, yf = make_new_bcs()

        # Integrate trajectory
        y, miss_ind, full_traj, dv1, dv2 = integrate_func_missed_thrust(thrust_fcn, y0, ti, yf, m_dry, T_max_kN, du, tu, mu, fu,
                                                              Isp, tol=tol)
        # Check if integration was stopped early
        if len(y.shape) == 0:
            f[i] = 10000
            continue
        # Create logical list for indices of 2D components
        # ind_2d = [True, True, False, True, True, False, False]
        # Get final state
        yf_actual = y[-2, ind_dim]
        yf_target = yf[ind_dim[:-1]]
        # Calculate ratio of initial mass to final mass
        # TODO fix mass ratio calculation
        m_ratio = (y0[-1] - y[-1, -1]) / y0[-1]
        t_ratio = ti[-1] / ti[-2]
        f[i] = traj_fit_func(yf_actual, yf_target, y0, m_ratio, t_ratio)

    # Calculate scalar fitness
    if num_cases > 1:
        # TODO test this out
        c = 1.  # weight to favor mean vs std
        f_mean = np.mean(f)
        f_std = np.std(f)
        f = c * f_mean + (1 - c) * f_std
    else:
        f = f[0]

    return -f


def make_new_bcs():
    while True:
        a0 = np.random.rand() * (a0_max - a0_min) + a0_min
        af = np.random.rand() * (af_max - af_min) + af_min
        if np.abs((af - a0) / np.min((af, a0))) > 0.1:
            break
    e0 = np.random.rand() * (e0_max - e0_min) + e0_min
    ef = np.random.rand() * (ef_max - ef_min) + ef_min
    w0 = np.random.rand() * (w0_max - w0_min) + w0_min
    wf = np.random.rand() * (wf_max - wf_min) + wf_min
    f0 = np.random.rand() * (f0_max - f0_min) + f0_min
    ff = np.random.rand() * (ff_max - ff_min) + ff_min
    
    if n_dim == 2:
        y0 = keplerian_to_inertial_2d(np.array([a0, e0, w0, f0]), gm=gm)
        yf = keplerian_to_inertial_2d(np.array([af, ef, wf, ff]), gm=gm)
        y0 = np.insert(y0, 2, 0.)
        y0 = np.insert(y0, 5, 0.)
        yf = np.insert(yf, 2, 0.)
        yf = np.insert(yf, 5, 0.)
    else:
        i0 = np.random.rand() * (i0_max - i0_min) + i0_min
        ifinal = np.random.rand() * (if_max - if_min) + if_min
        om0 = np.random.rand() * (om0_max - om0_min) + om0_min
        omf = np.random.rand() * (omf_max - omf_min) + omf_min
        y0 = keplerian_to_inertial_3d(np.array([a0, e0, i0, w0, om0, f0]), gm=gm)
        yf = keplerian_to_inertial_3d(np.array([af, ef, ifinal, wf, omf, ff]), gm=gm)
    y0 = np.append(y0, m0)
    return y0, yf


def evalTraj(W, params):
    # Extract parameters
    n_in, n_hid, n_out, scales_in, scales_out, t0, tf, y0, yf, m_dry, T_max_kN, tol, num_nodes, num_cases, num_outages = params
    # Construct NN
    nc = Neurocontroller(scales_in, scales_out, n_in, n_hid, n_out)
    nc.setWeights(W)
    thrust_fcn = nc.getThrustVec
    # Define time vector
    # ti = np.linspace(t0, tf, num_nodes)
    ti = np.power(np.linspace(0, 1, num_nodes), 3 / 2) * (tf - t0) + t0
    # Define scaling parameters
    du = 6371.0
    tu = np.sqrt(du**3 / gm)
    mu = y0[-1]
    fu = mu * du / tu / tu
    # Initialize score vector
    f = np.ones(num_cases) * np.inf
    for i in range(num_cases):
        # Integrate trajectory
        y, miss_ind = integrate_func_missed_thrust(thrust_fcn, y0, deepcopy(ti), yf, m_dry, T_max_kN, du, tu, mu, fu, tol=tol)
        # Check if integration was stopped early
        if len(y.shape) == 0:
            if y == -1:
                return 1000
            else:
                frac = y / len(ti)
                return (1 - frac) * 500 + 500
        # Create logical list for indices of 2D components
        # ind_2d = [True, True, False, True, True, False, False]
        # Get final state
        yf_actual = y[-1, ind_dim]
        yf_target = yf[ind_dim[:-1]]
        # Calculate ratio of initial mass to final mass
        m_ratio = y0[-1] / y[-1, -1]
        f[i] = traj_fit_func(yf_actual, yf_target, m_ratio)
    f = np.mean(f)
    if f < 10:
        blah = 0
    return f


def traj_fit_func(y, yf, y0, m_ratio, t_ratio=0.):
    """
    Calculates a scalar fitness value for a trajectory based on weighted sum of final state error plus the final mass.
    :param y:
    :param yf:
    :param m_ratio:
    :return:
    """
    if n_dim == 2:
        a0, e0, w0, f0 = inertial_to_keplerian_2d(y0[ind_dim], gm=gm)
        a1, e1, w1, f1 = inertial_to_keplerian_2d(y, gm=gm)
        a2, e2, w2, f2 = inertial_to_keplerian_2d(yf, gm=gm)
    else:
        a0, e0, i0, w0, om0, f0 = inertial_to_keplerian_3d(y0[:-1], gm=gm)
        a1, e1, i1, w1, om1, f1 = inertial_to_keplerian_3d(y, gm=gm)
        a2, e2, i2, w2, om2, f2 = inertial_to_keplerian_3d(yf, gm=gm)

    rp0 = a0 * (1 - e0)
    rp1 = a1 * (1 - e1)
    rp2 = a2 * (1 - e2)
    ra0 = a0 * (1 + e0)
    ra1 = a1 * (1 + e1)
    ra2 = a2 * (1 + e2)

    drp = np.abs(rp2 - rp1) / a0_min
    dra = np.abs(ra2 - ra1) / a0_min
    da = np.abs(a2 - a1) / a0_min
    dw = np.abs(fix_angle(w2 - w1, np.pi, -np.pi) / np.pi)
    df = np.abs(fix_angle(f2 - f1, np.pi, -np.pi) / np.pi)

    K = 0
    drp = 0 if drp < 0.001 else drp + K
    dra = 0 if dra < 0.001 else dra + K
    da = 0 if da < 0.001 else da + K
    dw = 0 if dw < 0.005 else dw + K
    df = 0 if df < 0.005 else df + K

    # Set cost function based on final trajectory type
    if elliptical_final:
        # Elliptic final
        weights = np.array([1, 1, 1, 0.5], np.float) * 10
        m_ratio *= 5
        # f = np.sum(np.max((np.square(np.array([drp, dra, dw, df]) * weights),
        #                    np.abs(   np.array([drp, dra, dw, df]) * weights)), axis=0)) + m_ratio * m_ratio
        f = np.sum(np.square(np.array([drp, dra, dw, df]) * weights)) + m_ratio + t_ratio - 1
    else:
        # Circular final
        weights = np.array([1, 1, 0, 0], np.float) * 20
        f = np.sum(np.max((np.square(np.array([drp, da, dw, df]) * weights),
                           np.abs(   np.array([drp, da, dw, df]) * weights)), axis=0)) + m_ratio * m_ratio

    # Penalize going too close to central body
    if rp_penalty:
        if rp1 < min_allowed_rp:
            f += rp_penalty_multiplier * (1 - rp1 / min_allowed_rp)

    # Penalize for not leaving initial orbit
    if no_thrust_penalty:
        kepl_scales = [ra2 - ra0, rp2 - rp0]
        dy0 = np.sum(np.abs(np.array([ra1 - ra0, rp1 - rp0]) / kepl_scales))
        dy0 = 0. if dy0 > 0.2 else 1000.
        f += dy0

    return f


def integrate_func_missed_thrust(thrust_fcn, y0, ti, yf, m_dry, T_max_kN, du, tu, mu, fu, Isp, tol=1e-8,
                                 save_full_traj=False):
    """
    Integrate a trajectory using boost_2bp with each leg having fixed thrust. Updates the thrust vector between each leg
    """

    # Define lengths of vectors for C++
    state_size, time_size = len(y0), 2
    param_size = 18 if variable_power else 6
    # param = [gm, mdry, thrust_vec x3, ref power, min power, max power, thrust coef x5, isp coef x5]

    # Create placeholder matrix for trajectory
    y = np.zeros((len(ti), state_size))

    # Assign initial condition
    y[0] = y0

    # Create 2-body integrator object
    tbp = boost_tbp.TBP()

    # Define flag for missed thrust
    global missed_thrust_allowed
    if missed_thrust_allowed:
        miss_ind = calculate_missed_thrust_events(ti * tu)
    else:
        miss_ind = np.array([]).astype(int)

    # Scales for nondim integration
    scales = np.array([du, du, du, du / tu, du / tu, du / tu, mu])

    # Main loop
    integrator_type = 0 # 0 fixed step, 1 adaptive step
    eom_type = 0 # 0 constant power, 1 variable power, 2 state transition matrix
    for i in range(len(ti) - 2):
        r = np.linalg.norm(y[i, :3])
        eps = (np.linalg.norm(y[i, 3:6])**2 / 2 - gm / r)
        if (eps > max_energy or eps < min_energy) and not save_full_traj:
            return np.array(0), 0, 0, 0, 0
        # Fixed step integrator step size
        step_size = (ti[i+1] - ti[i]) / 20
        # ratio of dry to wet mass for NN
        mass_ratio = m_dry / y[i, -1]
        time_ratio = ti[i] / ti[-2]
        # Check if i is supposed to miss thrust for this segment
        if i in miss_ind or y[i, -1] <= m_dry + 0.01:
            thrust = np.array([0, 0, 0])
        else:
            # query NN to get next thrust vector
            thrust = thrust_fcn(np.hstack((y[i, ind_dim], yf[ind_dim[:-1]], mass_ratio, time_ratio))) * T_max_kN / fu
        # create list of parameters to pass to integrator
        if variable_power:
            param = [float(gm), float(m_dry / mu), float(thrust[0]), float(thrust[1]), float(thrust[2]), power_reference,
                     power_min, power_max, float(thrust_power_coef[0]), float(thrust_power_coef[1]), float(thrust_power_coef[2]),
                     float(thrust_power_coef[3]), float(thrust_power_coef[4]), float(isp_power_coef[0]), float(isp_power_coef[1]),
                     float(isp_power_coef[2]), float(isp_power_coef[3]), float(isp_power_coef[4])]
        else:
            param = [float(gm), float(Isp / du * tu), float(m_dry / mu), float(thrust[0]), float(thrust[1]), float(thrust[2])]
        # propagate from the current state until the next time step
        traj = tbp.prop(list(y[i] / scales), [ti[i], ti[i+1]], param, state_size, time_size, param_size, tol, step_size,
                        int(integrator_type), int(eom_type))
        # save full trajectory
        if i == 0:
            full_traj = traj
        else:
            full_traj.extend(traj[1:])
        # save final state of current leg
        y[i+1] = np.array(traj[-1])[1:] * scales

    # Compute a minimum-energy Lambert arc to the target point

    final_tol = 0.05 # nondimensional position
    if np.linalg.norm((y[-2, :3] - yf[:3]) / scales[:3]) > final_tol:
        v1, v2, tof = min_energy_lambert(y[-2, :3], yf[:3])
        dv1 = v1 - y[-2, 3:6]
        dv2 = yf[3:6] - v2
        dv1_mag = np.linalg.norm(dv1)
        dv2_mag = np.linalg.norm(dv2)
        ti[-1] = ti[-2] + tof / tu
        m_penultimate = y[-2, -1] / np.exp(dv1_mag * 1000 / g0_ms2 / Isp_chemical) / mu
        m_final = m_penultimate / np.exp(dv2_mag * 1000 / g0_ms2 / Isp_chemical)

        # y[-2, -1] = m_penultimate # TODO add an integrator that only does six orbital elements - then mass can be zero
        integrator_type = 0  # constant
        eom_type = 0  # constant
        state_size = len(y[-2])
        step_size = (ti[-1] - ti[-2]) / 20
        param = [float(gm), float(Isp / du * tu), float(m_dry / mu), float(0), float(0), float(0)]
        traj = tbp.prop(list(y[-2] / scales), [ti[-2], ti[-1]], param, state_size, time_size, param_size,
                        tol, step_size, int(integrator_type), int(eom_type))

        # Add last leg to the trajectory
        last_leg = np.array(traj[1:])
        full_traj = np.vstack((full_traj, last_leg))
        full_traj[-1, -1] = m_final

        # save final state of current leg
        y[-1] = np.hstack((traj[-1][1:7], m_final)) * scales

    else:
        dv1, dv2 = np.zeros(3, float), np.zeros(3, float)
        y[-1] = y[-2]
        ti[-1] = ti[-2]
        full_traj = np.array(full_traj)
        # m_penultimate, m_final = y[-2, -1], y[-2, -1]
        # print()
        # print('else statement')
        # print()

    # rescale states
    full_traj[:, 1:] = full_traj[:, 1:] * scales
    return y, miss_ind, full_traj, dv1, dv2


def calculate_missed_thrust_events(ti):
    """
    Return indices of random missed thrust events that occur according to the Weibull distributions given by Imken et al
    :param ti:
    :return:
    """

    # Define Weibull distribution parameters
    lambda_tbe = 0.62394 * missed_thrust_tbe_factor
    k_tbe = 0.86737
    lambda_rd = 2.459 * missed_thrust_rd_factor
    k_rd = 1.144
    max_discovery_delay_days = 3
    op_recovery_days = 0.5
    # initialize list of indices that experience missed thrust
    miss_indices = list()
    t = 0.
    while True:
        time_between_events = np.random.weibull(k_tbe) * lambda_tbe * year_to_sec  # calculate time between events
        t += time_between_events
        if t < ti[-1]:  # check if the next event happens before the end of the time of flight
            # while True:
                recovery_duration = np.random.weibull(k_rd) * lambda_rd * day_to_sec  # calculate the recovery duration
                recovery_duration += (np.random.rand() * max_discovery_delay_days * day_to_sec) + \
                                     (op_recovery_days * day_to_sec)  # account for discovery delay and operational recovery
                miss_indices.append(np.arange(find_nearest(ti, t), find_nearest(ti, t + recovery_duration)))
                t += recovery_duration
        if t >= ti[-1]:
            if len(miss_indices) > 0:
                return np.hstack(miss_indices).astype(int)
            else:
                return np.array([]).astype(int)


def find_nearest(array, value):
    """
    Helper function that finds the closest index of an array to a specified value
    :param array:
    :param value:
    :return:
    """
    idx = (np.abs(array - value)).argmin()
    return idx


def calculate_prop_margins(genome, config):
    """
    Evaluates a neural network's ability as a controller for the defined problem, using a NEAT-style network.
    :param genome:
    :param config:
    :return:
    """

    # Scales for network inputs
    du = np.max((a0_max, af_max))
    tu = np.sqrt(du ** 3 / gm)
    mu = m0
    fu = mu * du / tu / tu

    # Create network and get thrust vector
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    nc = Neurocontroller.init_from_neat_net(net, scales_in, scales_out)
    thrust_fcn = nc.get_thrust_vec_neat

    # Define times
    # ti = np.power(np.linspace(0, 1, num_nodes), 3 / 2) * (tf - t0) + t0
    ti = np.linspace(t0, tf, num_nodes)
    ti /= tu

    # Create a new case based on the given boundary condition boundary conditions
    y0, yf = make_new_bcs()
    print(y0)
    print(yf)

    # Initialize score vector
    num_cases = 10
    mf = np.ones(num_cases + 1) * np.inf
    global missed_thrust_allowed, missed_thrust_tbe_factor, missed_thrust_rd_factor
    for i in range(num_cases + 1):
        if i == 0:
            missed_thrust_allowed = False
        else:
            missed_thrust_allowed = True
            missed_thrust_tbe_factor = 0.2
            missed_thrust_rd_factor = 3

        # Integrate trajectory
        y, miss_ind = integrate_func_missed_thrust(thrust_fcn, y0, copy(ti), yf, m_dry, T_max_kN, du, tu, mu, fu,
                                                   Isp, tol=tol)

        # Save final mass
        print(y[-1, -1])
        mf[i] = y[-1, -1]

    # Calculate propellant margin
    m_star = mf[0]
    m_prop_star = y0[-1] - m_star
    m_tilde = np.mean(mf[1:])
    M = (m_star - m_tilde) / m_prop_star

    return M


def run_margins():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    with open('winner-feedforward', 'rb') as f:
        genome = pickle.load(f)
    M = calculate_prop_margins(genome, config)
    print(M)


if __name__ == '__main__':
    run_margins()

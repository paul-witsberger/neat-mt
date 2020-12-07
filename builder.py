import pickle
import os
import numpy as np
import h5py
import multiprocessing as mp
from itertools import repeat
import neatfast as neat
from neatfast import visualize
from nnet import Neurocontroller
import constants as c
import traj_config as tc
from missed_thrust import integrate_func_missed_thrust, traj_fit_func, compute_bcs
from plotter import plot_traj_2d, plot_mass_history, plot_thrust_history, final_point, point_size
from orbit_util import period_from_inertial, rotate_vnc_to_inertial_3d, mag3, keplerian_to_inertial_3d
import boost_tbp


def recreate_traj_from_pkl(fname: str, neat_net: bool = True, print_mass: bool = False, save_traj: bool = False,
                           traj_fname: str = "traj_data_", config_name='default'):
    # Load best generation from pickle file
    with open(fname, 'rb') as f:
        xopt = pickle.load(f)

    # Recreate neurocontroller from file
    if neat_net:
        local_dir = os.path.dirname(__file__)
        config_name = 'default' if config_name is None else config_name
        config_path = os.path.join(local_dir, 'config', 'config_' + config_name)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                             neat.DefaultStagnation, config_path)
        net = neat.nn.FeedForwardNetwork.create(xopt, config)
        nc = Neurocontroller.init_from_neat_net(net, tc.scales_in, tc.scales_out)
        thrust_fcn = nc.get_thrust_vec_neat
    else:
        # nc = Neurocontroller(tc.scales_in, tc.scales_out, tc.n_in, tc.n_hid, tc.n_out)
        # nc.setWeights(xopt)
        # thrust_fcn = nc.getThrustVec
        raise NotImplementedError('Only NEAT is supported in recreate_traj_from_pkl()')

    # Generate a problem
    y0, yf = compute_bcs()
    y0 = np.hstack((y0, tc.m0))

    # Create time vector
    ti = np.empty(config.num_nodes + 2)
    # ti = np.power(np.linspace(0, 1, num_nodes), 3 / 2) * (tf - t0) + t0
    ti[:-2] = np.linspace(tc.t0, tc.tf, config.num_nodes)
    ti[-2:] = ti[-3]
    ti /= tc.tu

    # Get the period of initial, final and target orbits, then integrate each of the trajectories
    tbp = boost_tbp.TBP()
    step_type, eom_type = 0, 3
    yinit_tf = period_from_inertial(y0[:-1], gm=tc.gm, max_time_sec=tc.max_final_time)
    ytarg_tf = period_from_inertial(yf, gm=tc.gm, max_time_sec=tc.max_final_time)
    yinit = tbp.prop(list(y0[tc.ind_dim] / tc.state_scales[:-1]), [tc.t0 / tc.tu, yinit_tf / tc.tu], [], 6, 2, 0,
                     tc.rtol, tc.atol, (yinit_tf - tc.t0) / (tc.n_terminal_steps + 1) / tc.tu, step_type, eom_type)
    yinit = (np.array(yinit)[:, 1:] * tc.state_scales[:-1]).T
    y, miss_ind, full_traj, maneuvers = integrate_func_missed_thrust(thrust_fcn, y0, ti, yf, config,
                                                                     save_full_traj=True, fixed_step=True)
    yfinal_tf = period_from_inertial(y[-1, :-1], gm=tc.gm, max_time_sec=tc.max_final_time)
    yfinal = tbp.prop(list(y[-1, tc.ind_dim] / tc.state_scales[:-1]), [tc.t0 / tc.tu, yfinal_tf / tc.tu], [], 6, 2, 0,
                      tc.rtol, tc.atol, (yfinal_tf - tc.t0) / (tc.n_terminal_steps + 1) / tc.tu, step_type, eom_type)
    yfinal = (np.array(yfinal)[:, 1:] * tc.state_scales[:-1]).T
    ytarg = tbp.prop(list(yf[tc.ind_dim[:-1]] / tc.state_scales[:-1]), [tc.t0 / tc.tu, ytarg_tf / tc.tu], [], 6, 2, 0,
                     tc.rtol, tc.atol, (ytarg_tf - tc.t0) / (tc.n_terminal_steps + 1) / tc.tu, step_type, eom_type)
    ytarg = (np.array(ytarg)[:, 1:] * tc.state_scales[:-1]).T

    # Calculate thrust vectors throughout transfer trajectory
    thrust_vec_body = get_thrust_history(ti, y, yf, thrust_fcn)[:, :tc.n_dim]
    thrust_vec_body[miss_ind] = 0.
    if tc.n_dim == 2:
        thrust_vec_inertial = rotate_thrust(thrust_vec_body, y)
    else:
        thrust_vec_inertial = [rotate_vnc_to_inertial_3d(tv, yi) for tv, yi in zip(thrust_vec_body, y)]
    dv1, dv2, *args = maneuvers  # TODO make a plot_capture_maneuvers() function to handle 2- or 3- impulse captures
    thrust_vec_inertial = np.vstack((thrust_vec_inertial,
                                     dv1[:tc.n_dim] / tc.du * tc.tu,
                                     dv2[:tc.n_dim] / tc.du * tc.tu))

    # Get indices where thrust occurs (for plotting)
    thrust_mag = np.sqrt(np.sum(np.square(thrust_vec_body), 1))
    thrust_ind = np.argwhere(thrust_mag > 0).ravel()

    # Define colors for different sections
    arrow_color = 'chocolate'
    missed_color = 'red'
    thrust_color = 'green'

    # Plot transfer, final, and initial orbits
    fig, ax = plot_traj_2d(yinit, False, False, label='Initial', show_legend=False)
    fig, ax = plot_traj_2d(full_traj[:, 1:4].T, False, False, fig_ax=(fig, ax), label='Transfer', start=True,
                           end=False, show_legend=False)
    fig, ax = plot_traj_2d(yfinal, False, False, fig_ax=(fig, ax), label='Final', show_legend=False)

    # Add colors to the missed-thrust and thrusting segments
    for mi in miss_ind:
        ax.plot(full_traj[tc.n_steps * mi:tc.n_steps * (mi + 1), 1] / c.au_to_km,
                full_traj[tc.n_steps * mi:tc.n_steps * (mi + 1), 2] / c.au_to_km, c=missed_color, zorder=7)
    for thi in thrust_ind:
        ax.plot(full_traj[tc.n_steps * thi:tc.n_steps * (thi + 1), 1] / c.au_to_km,
                full_traj[tc.n_steps * thi:tc.n_steps * (thi + 1), 2] / c.au_to_km, c=thrust_color, zorder=7)

    # Plot arrows with heads
    q_scale_thrust = np.max(np.linalg.norm(thrust_vec_body, axis=1)) * 20
    q_scale_capture = max(mag3(dv1), mag3(dv2)) / 5
    quiver_opts = {'angles': 'xy', 'zorder': 8, 'width': 0.0025, 'units': 'width', 'scale_units': 'width',
                   'minlength': 0.1, 'headaxislength': 3, 'headlength': 6, 'headwidth': 6, 'color': arrow_color}
    ax.quiver(y[:-2, 0] / c.au_to_km, y[:-2, 1] / c.au_to_km, thrust_vec_inertial[:-2, 0], thrust_vec_inertial[:-2, 1],
              scale=q_scale_thrust, **quiver_opts)
    ax.quiver(y[-2:, 0] / c.au_to_km, y[-2:, 1] / c.au_to_km, thrust_vec_inertial[-2:, 0], thrust_vec_inertial[-2:, 1],
              scale=q_scale_capture, **quiver_opts)

    # Plot arrows without heads
    # q_scale = np.max(np.linalg.norm(thrust_vec_body, axis=1)) * 20
    # quiver_opts = {'angles': 'xy', 'zorder': 8, 'width': 0.004, 'units': 'width', 'scale': q_scale,
    #                'scale_units': 'width', 'minlength': 0.1, 'headaxislength': 0, 'headlength': 0, 'headwidth': 0,
    #                'color': arrow_color}
    # ax.quiver(y[:-2, 0] / c.au_to_km, y[:-2, 1] / c.au_to_km, thrust_vec_inertial[:-2, 0],
    #           thrust_vec_inertial[:-2, 1], **quiver_opts)
    # ax.quiver(y[-2:, 0] / c.au_to_km, y[-2:, 1] / c.au_to_km, thrust_vec_inertial[-2:, 0] * tc.T_max_kN * 20,
    #           thrust_vec_inertial[-2:, 1] * tc.T_max_kN * 20, **quiver_opts)

    # Plot Mars after the capture
    tf_jc = (ti[-1] - ti[-3]) * tc.tu * c.sec_to_day * c.day_to_jc + tc.times_jd1950_jc[-1]
    mars_post_capture = c.ephem(['a', 'e', 'i', 'w', 'O', 'M'], [tc.target_body], np.array([tf_jc]))
    mars_post_capture[2] = 0.
    mars_post_capture = keplerian_to_inertial_3d(mars_post_capture, tc.gm, 'mean')[:3] / c.au_to_km
    ax.scatter(mars_post_capture[0], mars_post_capture[1], edgecolor='red', facecolor='none', s=point_size*1.5)
    ax.scatter(y[-3, 0] / c.au_to_km, y[-3, 1] / c.au_to_km, edgecolor='none', facecolor=final_point, s=point_size)
    ax.scatter(y[-1, 0] / c.au_to_km, y[-1, 1] / c.au_to_km, edgecolor=final_point, facecolor='none', s=point_size)

    # Plot the target orbit - also, save the figure since this is the last element of the plot
    plot_traj_2d(ytarg, False, True, fig_ax=(fig, ax), label='Target', end=True, show_legend=False,
                 config_name=config_name)

    # Plot mass and thrust
    plot_mass_history(ti * tc.tu * c.sec_to_day, y[:, -1], mt_ind=miss_ind, config_name=config_name)
    plot_thrust_history(ti[:-2] * tc.tu * c.sec_to_day, thrust_vec_body, mt_ind=miss_ind, config_name=config_name)

    # Save trajectory to file (states, times, controls)
    if save_traj:
        with h5py.File(os.path.join('results', traj_fname + config_name + '.hdf5'), "w") as f:
            f.create_dataset('t', data=ti)
            f.create_dataset('x', data=y)
            f.create_dataset('u', data=thrust_vec_inertial)

    # Print results
    if print_mass:
        print('Final mass = {0:.3f} kg'.format(y[-1, -1]))

    # Remove Lambert arc delta V from final NN-controlled state for fitness calculation
    yf_actual = y[-2, tc.ind_dim]
    yf_actual[tc.n_dim:] -= dv1[:tc.n_dim]

    # TODO figure out why the fitness is so crazy when terminal Lambert arc is performed or not
    #      ----> switching the first delta v with the frame change made it mostly better - but why is the final plot so
    #               garbage?
    #      ----> compare states at end of integration with no terminal maneuver case
    # Calculate and print fitness
    m_ratio = (y0[-1] - y[-1, -1]) / y0[-1]
    t_ratio = (ti[-1] - ti[-3]) / ti[-3]
    f, dr, dv = traj_fit_func(yf_actual, yf[tc.ind_dim[:-1]], y0, m_ratio, t_ratio)
    print('Final fitness = %f\n' % -f)


def make_last_traj(print_mass: bool = True, save_traj: bool = True, neat_net: bool = True, config_name=None):

    # Choose file to load
    if neat_net:
        ext = 'default' if config_name is None else config_name
        fname = 'winner_' + ext
        fname = 'winner_tmp'
    else:
        fname = 'lgen.pkl'
    # Run
    recreate_traj_from_pkl(os.path.join('results', fname), neat_net, print_mass=print_mass,
                           save_traj=save_traj, config_name=config_name)
    make_neat_network_diagram(config_name=config_name)


def get_thrust_history(ti: np.ndarray, y: np.ndarray, yf: np.ndarray, thrust_fcn: Neurocontroller.get_thrust_vec_neat)\
        -> np.ndarray:
    # Calculate thrust vector at each time step
    thrust_vec = np.zeros((len(ti) - 2, 3), float)
    for i in range(len(ti) - 3):
        # Check if there is any remaining propellant mass
        if y[i, -1] > tc.m_dry + 0.01:
            # Compute mass and time ratios
            mass_ratio = (y[i, -1] - tc.m_dry) / (y[0, -1] - tc.m_dry)
            time_ratio = ti[i] / ti[-2]  # TODO ti[-2] or ti[-1]?
            # Query NN to get thrust vector
            thrust_vec[i, :] = thrust_fcn(
                np.hstack((y[i, tc.ind_dim], yf[tc.ind_dim[:-1]], mass_ratio, time_ratio))) * tc.T_max_kN
        else:
            # No thrust
            thrust_vec[i, :] = np.array([0, 0, 0])
    return thrust_vec


def rotate_thrust(thrust_vec_body: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Velocity angle with respect to inertial X
    rot_ang = np.arctan2(y[:, 4], y[:, 3])
    # Direction cosine matrix
    dcm = np.array([[np.cos(rot_ang), -np.sin(rot_ang)], [np.sin(rot_ang), np.cos(rot_ang)]])
    # Apply DCM to each state
    thrust_vec_inertial = np.array([np.matmul(dcm[:, :, i], tvb) for i, tvb in enumerate(thrust_vec_body)])
    return thrust_vec_inertial


def make_neat_network_diagram(config_name=None):
    """
    Creates a network diagram in .svg format showing the nodes and connections.
    """
    from platform import system as sys
    if sys() == 'Linux':
        return
    # Load network
    ext = 'tmp' if config_name is None else config_name
    with open('results//winner_' + ext, 'rb') as f:
        winner = pickle.load(f)

    # Load configuration
    config_name = 'default' if config_name is None else config_name
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config', 'config_' + config_name)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    _node_input_names_2d = {-1: '<<i>a<i><SUB>c</SUB>>',
                            -2: '<<i>e</i><SUB>c</SUB>>',
                            -3: '<&#969;<SUB>c</SUB>>', # omega
                            -4: '<<i>f</i><SUB>c</SUB>>',
                            -5: '<<i>a</i><SUB>t</SUB>>',
                            -6: '<<i>e</i><SUB>t</SUB>>',
                            -7: '<&#969;<SUB>t</SUB>>', # omega
                            -8: '<<i>f</i><SUB>t</SUB>>',
                            -9: '<&#956;>',  # mu, mass ratio
                            -10: '<&#964;>'}  # tau, time ratio

    _node_input_names_3d = {-1: '<<i>a</i><SUB>current</SUB>>',
                            -2: '<<i>e</i><SUB>current</SUB>>',
                            -3: '<<i>i</i><SUB>current</SUB>>',
                            -4: '<&#969;<SUB>current</SUB>>',  # omega
                            -5: '<&#937;<SUB>current</SUB>>',  # Omega
                            -6: '<<i>f</i><SUB>current</SUB>>',
                            -7: '<<i>a</i><SUB>target</SUB>>',
                            -8: '<<i>e</i><SUB>target</SUB>>',
                            -9: '<<i>i</i><SUB>target</SUB>>',
                            -10: '<&#969;<SUB>target</SUB>>',  # omega
                            -11: '<&#937;<SUB>target</SUB>>',  # Omega
                            -12: '<<i>f</i><SUB>target</SUB>>',
                            -13: '<&#956;>',  # mu, mass ratio
                            -14: '<&#964;>'}  # tau, time ratio

    _node_output_names_2d = {0: '<&#952;>',  # theta, thrust angle
                             1: '<&#932;>'}  # Tau, throttle

    _node_output_names_3d = {0: '<&#945;>',  # alpah, thrust angle (long)
                             1: '<&#946;>',  # beta, thrust angle (lat)
                             2: '<&#932;>'}  # Tau, throttle

    # Define node names
    node_names = _node_input_names_2d if tc.n_dim == 2 else _node_input_names_3d
    node_names = {-i-1: node_names[-key-1] for i, key in enumerate(tc.input_indices)}
    node_names.update(_node_output_names_2d if tc.n_outputs == 2 else _node_output_names_3d)

    # Draw network (remove disabled and unused nodes/connections)
    visualize.draw_net(config, winner, node_names=node_names, filename="results//winner-diagram.gv",
                       show_disabled=False, prune_unused=True)


def load_traj(traj_fname: str = 'traj_data.hdf5') -> (np.ndarray, np.ndarray, np.ndarray):
    # Load saved trajectory data
    with h5py.File(os.path.join('results', traj_fname), 'r') as f:
        t = f['t'][()]
        x = f['x'][()]
        u = f['u'][()]
    return t, x, u


def sensitivity(t: np.ndarray, x: np.ndarray, m: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Calculate sensitivty matrix.
    :param t:
    :param x:
    :param m:
    :param u:
    :return:
    """
    # Number of rows = number of timesteps
    n = len(t)

    # Get the partial derivative of each row
    # Serial
    # ddvdx = np.empty((4*n, 2*(n-1)))
    # for i in range(4*n):
    #     ddvdx[i] = get_ddvdx([x, m, u, i])
    # Parallel
    with mp.Pool(os.cpu_count() - 3) as p:
        ddvdx = p.map_async(get_ddvdx, zip(repeat(x, 4*n), repeat(m, 4*n), repeat(u, 4*n), range(4*n))).get()

    # OLD - Calculate L2 norm (Frobenius/matrix norm) of the matrix and return
    # Calculate the L2 norm of each state
    s = np.sqrt(np.sum(np.square(ddvdx), 1)).reshape((-1, 1))
    # print('Sensitivity = %f' % s)
    return s


def get_ddvdx(args: list) -> np.ndarray:
    """
    Calculate partial derivative of delta v with respect to state for ith element of state
    """
    x, m, u, i = args
    # Choose step size (are you perturbing position or velocity)
    x_step, v_step = 1, 0.001
    step = x_step if np.mod(i, 4) < 2 else v_step
    # Copy the vector of states
    x_perturbed = x.copy()
    # Perturb the ith element
    x_perturbed[i] += step
    # Assign dx to be the step size
    dx = step
    # Calculate the new trajectory after the perturbation
    u_of_x_perturbed = evaluate_perturbed_trajectory(x_perturbed, m, i)
    # Append original control before perturbed point to the new control
    u_of_x_perturbed = np.hstack((u[:i // (2 * tc.n_dim) * 2], u_of_x_perturbed))
    # Calculate the difference between the new and old trajectories
    du = u_of_x_perturbed - u
    # Return the difference in trajectories divided by the step size
    return du / dx


def evaluate_perturbed_trajectory(x_perturbed: np.ndarray, m: np.ndarray, i: int) -> np.ndarray:
    # Calculate the first point from which to propagate
    first_point = i // (tc.n_dim * 2)

    # Load best generation from pickle file and get thrust function
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_default')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, config_path)
    fname = 'winner'
    with open(fname, 'rb') as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    nc = Neurocontroller.init_from_neat_net(net, tc.scales_in, tc.scales_out)
    thrust_fcn = nc.get_thrust_vec_neat
    del local_dir, config_path, config, fname, genome, net, nc

    # Propagate for number of steps
    y0, yf = x_perturbed[first_point * tc.n_dim * 2:(first_point + 1) * tc.n_dim * 2], compute_bcs()[-1]
    if tc.n_dim == 2:
        y0 = np.insert(y0, 2, 0.)
        y0 = np.insert(y0, 5, 0.)
    y0 = np.append(y0, m[first_point])
    # du = np.max((tc.a0_max, tc.af_max))
    # tu = np.sqrt(du ** 3 / tc.gm)
    # mu = tc.m0
    # fu = mu * du / tu / tu
    ti = np.linspace(tc.t0, tc.tf, tc.num_nodes)
    ti /= tc.tu
    ti = ti[first_point:]
    y, miss_ind = integrate_func_missed_thrust(thrust_fcn, y0, ti, yf)

    # Need to make a state matrix with mass
    if first_point > 0:
        y_start = np.reshape(x_perturbed[:first_point * tc.n_dim * 2], [-1, 2 * tc.n_dim])
        if tc.n_dim == 2:
            yzeros = np.zeros((y_start.shape[0], 1))
            y_start = np.hstack((y_start[:, :2], yzeros, y_start[:, 2:], yzeros))
        y_start = np.hstack((y_start, m[:first_point]))
    else:
        y_start = np.array([]).reshape((0, 7))
    y_matrix = np.vstack((y_start, y))
    thrust_vec_body = get_thrust_history(ti, y_matrix, yf, thrust_fcn)[:, :tc.n_dim]
    thrust_vec_body[miss_ind] = 0.
    thrust_vec_inertial = rotate_thrust(thrust_vec_body, y_matrix)
    # if not np.mod(i+1, 100):
    #     print(i+1)
    return thrust_vec_inertial.ravel()


def save_traj_data(t: np.ndarray, x: np.ndarray, m: np.ndarray, u: np.ndarray,
                   traj_fname: str = 'traj_data_desensitized.hdf5'):
    x = np.reshape(x, (len(t), -1))
    x = np.hstack((x, m))
    with h5py.File(os.path.join('results', traj_fname), 'w') as f:
        f.create_dataset('t', data=t)
        f.create_dataset('x', data=x)
        f.create_dataset('u', data=u)


def desensitize():
    # Load trajectory
    t, x, u = load_traj()

    # Reshape to vectors
    m = x[:, -1].ravel().reshape((-1, 1))
    x = np.hstack((x[:, :2], x[:, 3:5])).ravel()
    u = u.ravel()

    # Calculate sensitivity of original trajectory
    s = sensitivity(t, x, m, u)
    np.save('results//traj_data_s.npy', s)
    # print('Original sensitivity: %f' % s)

    # Calculate sensitivity of perturbed trajectories
    dsdx = np.empty((len(x), len(x)))  # , n_dim*(len(t)-1)))
    for i in range(len(x)):
        x_step, v_step = 1, 0.001
        step = x_step if np.mod(i, 4) < 2 else v_step
        x_perturbed = x.copy()
        x_perturbed[i] += step
        s_perturbed = sensitivity(t, x_perturbed, m, u)
        dsdx[i] = ((s_perturbed - s) / step).T
        print('Finished %i / %i' % (i + 1, len(x)))
    np.save('results//traj_data_dsdx.npy', dsdx)

    # Calculate update to states
    # delta_x = s / dsdx / len(x)
    delta_x = np.matmul(np.linalg.inv(dsdx), s)
    np.save('results//traj_data_delta_x.npy', delta_x)

    # Update states
    x += delta_x
    np.save('results//traj_data_x.npy', x)

    # Save new trajectory to file
    save_traj_data(t, x, m, u)

    # Calculate new sensitivity
    # s = sensitivity(t, x, m, u)
    # print('New sensitivity: %f' % s)

    return x, delta_x, dsdx

# REGARDING SENSITIVITY:
# TODO - see if there could be a benefit from computing a step from minimizing delta v and minimizing s,
#        and then add them together
# TODO - look into effect of learning rate
# TODO - see if the above two TODO's could be combined into one; use learning rate as a Pareto coefficient
# TODO plot the magnitude of sensitivity for both position and velocity as a color along the trajectory


if __name__ == '__main__':
    desensitize()

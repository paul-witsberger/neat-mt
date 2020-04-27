import pickle
import os
import neatfast as neat
from neatfast import visualize
from nnet import Neurocontroller
from traj_config import *
from scipy import integrate
from eom import eom2BP
from big_idea import integrate_func_missed_thrust, make_new_bcs, traj_fit_func
from plotter import *
from orbit_util import period_from_inertial
import h5py
import multiprocessing as mp
from itertools import repeat


def recreate_traj_from_pkl(fname: str, neat_net: bool = False, print_mass: bool = False, save_traj: bool = False,
                           traj_fname: str = "traj_data.hdf5"):
    # Load best generation from pickle file
    with open(fname, 'rb') as f:
        xopt = pickle.load(f)

    # Recreate neurocontroller from file
    if neat_net:
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-feedforward')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                             neat.DefaultStagnation, config_path)
        net = neat.nn.FeedForwardNetwork.create(xopt, config)
        nc = Neurocontroller.init_from_neat_net(net, scales_in, scales_out)
        thrust_fcn = nc.get_thrust_vec_neat
    else:
        nc = Neurocontroller(scales_in, scales_out, n_in, n_hid, n_out)
        nc.setWeights(xopt)
        thrust_fcn = nc.getThrustVec

    # Define scaling parameters
    du = np.max((a0_max, af_max))
    tu = np.sqrt(du ** 3 / gm)
    mu = m0
    fu = mu * du / tu / tu

    # Generate a problem
    y0, yf = make_new_bcs()

    # Create time vector
    # ti = np.power(np.linspace(0, 1, num_nodes), 3 / 2) * (tf - t0) + t0
    ti = np.linspace(t0, tf, num_nodes)
    ti = np.append(ti, ti[-1])
    ti /= tu

    # Get the period of initial, final and target orbits, then integrate each of the trajectories
    yinit_tf = period_from_inertial(y0[:-1], gm=gm)
    ytarg_tf = period_from_inertial(yf, gm=gm)
    yinit = integrate.solve_ivp(eom2BP, [t0, yinit_tf], y0[ind_dim], rtol=rtol, atol=atol)
    y, miss_ind, full_traj, dv1, dv2 = integrate_func_missed_thrust(thrust_fcn, y0, ti, yf, m_dry, T_max_kN, du, tu, mu,
                                                                    fu, Isp, save_full_traj=True, fixed_step=True)
    yfinal_tf = period_from_inertial(y[-1, :-1], gm=gm)
    yfinal = integrate.solve_ivp(eom2BP, [t0, yfinal_tf], y[-1, ind_dim], rtol=rtol, atol=atol)
    ytarg = integrate.solve_ivp(eom2BP, [t0, ytarg_tf], yf[ind_dim[:-1]], rtol=rtol, atol=atol)

    # Calculate thrust vectors throughout transfer trajectory
    thrust_vec_body = get_thrust_history(ti, y, yf, m_dry, T_max_kN, thrust_fcn)[:, :n_dim]
    thrust_vec_body[miss_ind] = 0.
    thrust_vec_inertial = rotate_thrust(thrust_vec_body, y)
    thrust_vec_inertial = np.vstack((thrust_vec_inertial, dv1[:n_dim] / du * tu, dv2[:n_dim] / du * tu))

    # Get indices where thrust occurs (for plotting)
    thrust_mag = np.sqrt(np.sum(np.square(thrust_vec_body), 1))
    thrust_ind = np.argwhere(thrust_mag > 0).ravel()

    # Define colors for different sections
    arrow_color = 'chocolate'
    missed_color = 'red'
    thrust_color = 'green'

    # Plot transfer, final, and initial orbits
    fig, ax = plotTraj2D(full_traj[:, 1:-1], False, False, label='Transfer', start=True, end=True, show_legend=False)
    fig, ax = plotTraj2DStruct(yfinal, False, False, fig_ax=(fig, ax), label='Final', show_legend=False)
    fig, ax = plotTraj2DStruct(yinit, False, False, fig_ax=(fig, ax), label='Initial', show_legend=False)

    # Add colors to the missed-thrust and thrusting segments
    for mi in miss_ind:
        ax.plot(full_traj[n_steps * mi:n_steps * mi + n_steps, 1] / au_to_km,
                full_traj[n_steps * mi:n_steps * mi + n_steps, 2] / au_to_km, c=missed_color, zorder=7)
    for thi in thrust_ind:
        ax.plot(full_traj[n_steps * thi:n_steps * thi + n_steps, 1] / au_to_km,
                full_traj[n_steps * thi:n_steps * thi + n_steps, 2] / au_to_km, c=thrust_color, zorder=7)

    # Plot arrows with heads
    # q_scale = np.max(np.linalg.norm(thrust_vec_body, axis=1)) * 20
    # ax.quiver(y[:-1, 0] / au_to_km, y[:-1, 1] / au_to_km, thrust_vec_inertial[:, 0], thrust_vec_inertial[:, 1],
    #           angles='xy', zorder=8, width=0.0025, units='width', scale=q_scale, scale_units='width', minlength=0.1,
    #           headaxislength=5, headlength=6, headwidth=5, color=arrow_color)

    # Plot arrows without heads
    q_scale = np.max(np.linalg.norm(thrust_vec_body, axis=1)) * 20
    quiver_opts = {'angles': 'xy', 'zorder': 8, 'width': 0.004, 'units': 'width', 'scale': q_scale,
                   'scale_units': 'width', 'minlength': 0.1, 'headaxislength': 0, 'headlength': 0, 'headwidth': 0,
                   'color': arrow_color}
    ax.quiver(y[:-2, 0] / au_to_km, y[:-2, 1] / au_to_km,
              thrust_vec_inertial[:-2, 0], thrust_vec_inertial[:-2, 1], **quiver_opts)
    ax.quiver(y[-2:, 0] / au_to_km, y[-2:, 1] / au_to_km,
              thrust_vec_inertial[-2:, 0] * T_max_kN * 20, thrust_vec_inertial[-2:, 1] * T_max_kN * 20, **quiver_opts)

    # Plot the target orbit - also, save the figure since this is the last element of the plot
    plotTraj2DStruct(ytarg, False, True, fig_ax=(fig, ax), label='Target', end=True, show_legend=False)

    # Plot mass and thrust
    plotMassHistory(ti * tu * sec_to_day, y[:, -1], mt_ind=miss_ind)
    plotThrustHistory(ti[:-2] * tu * sec_to_day, thrust_vec_body, T_max_kN, mt_ind=miss_ind)

    # Save trajectory to file (states, times, controls)
    if save_traj:
        with h5py.File(traj_fname, "w") as f:
            f.create_dataset('t', data=ti)
            f.create_dataset('x', data=y)
            f.create_dataset('u', data=thrust_vec_inertial)

    # Print results
    if print_mass:
        print('Final mass = {0:.3f} kg'.format(y[-1, -1]))

    # Remove Lambert arc delta V from final NN-controlled state for fitness calculation
    yf_actual = y[-2, ind_dim]
    yf_actual[n_dim:] -= dv1[:n_dim]

    # Calculate and print fitness
    f, dr, dv = traj_fit_func(yf_actual, yf[ind_dim[:-1]], y0, (y0[-1] - y[-1, -1]) / y0[-1], ti[-1] / ti[-2])
    print('Final fitness = %f' % -f)


def make_last_traj(print_mass: bool = True, save_traj: bool = True):
    # Choose file to load
    neat_net = True
    if neat_net:
        fname = 'winner-feedforward'
    else:
        fname = 'lgen.pkl'
    # Run
    recreate_traj_from_pkl(fname, neat_net, print_mass=print_mass, save_traj=save_traj)


def get_thrust_history(ti: np.ndarray, y: np.ndarray, yf: np.ndarray, m_dry: float, T_max_kN: float,
                       thrust_fcn: Neurocontroller.get_thrust_vec_neat) -> np.ndarray:
    # Calculate thrust vector at each time step
    thrust_vec = np.zeros((len(ti) - 2, 3), float)
    for i in range(len(ti) - 2):
        # Check if there is any remaining propellant mass
        if y[i, -1] > m_dry + 0.01:
            # Compute mass and time ratios
            mass_ratio = (y[i, -1] - m_dry) / (y[0, -1] - m_dry)
            time_ratio = ti[i] / ti[-1]
            # Query NN to get thrust vector
            thrust_vec[i, :] = thrust_fcn(np.hstack((y[i, ind_dim], yf[ind_dim[:-1]], mass_ratio, time_ratio))) * T_max_kN
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


def make_neat_network_diagram():
    """
    Creates a network diagram in .svg format showing the nodes and connections.
    """
    # Load network
    with open('winner-feedforward', 'rb') as f:
        winner = pickle.load(f)

    # Load configuration
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Define node names
    if n_dim == 2:
        node_names = {-1: 'a_c', -2: 'e_c', -3: 'w_c', -4: 'f_c', -5: 'a_t', -6: 'e_t', -7: 'w_t', -8: 'f_t',
                      -9: 'mass', -10: 'time', 0: 'Angle', 1: 'Throttle'}
    else:
        node_names = {-1: 'a_c', -2: 'e_c', -3: 'i_c', -4: 'w_c', -5: 'om_c', -6: 'f_c', -7: 'a_t', -8: 'e_t',
                      -9: 'i_t', -10: 'w_t', -11: 'om_t', -12: 'f_t', 0: 'Alpha', 1: 'Beta', 2: 'Throttle'}

    # Draw network (remove disabled and unused nodes/connections)
    visualize.draw_net(config, winner, node_names=node_names, filename="winner-feedforward-diagram.gv",
                       show_disabled=False, prune_unused=True)


def load_traj(traj_fname: str = 'traj_data.hdf5') -> (np.ndarray, np.ndarray, np.ndarray):
    # Load saved trajectory data
    with h5py.File(traj_fname, 'r') as f:
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
    u_of_x_perturbed = np.hstack((u[:i // (2 * n_dim) * 2], u_of_x_perturbed))
    # Calculate the difference between the new and old trajectories
    du = u_of_x_perturbed - u
    # Return the difference in trajectories divided by the step size
    return du / dx


def evaluate_perturbed_trajectory(x_perturbed: np.ndarray, m: np.ndarray, i: int) -> np.ndarray:
    # Calculate the first point from which to propagate
    first_point = i // (n_dim * 2)

    # Load best generation from pickle file and get thrust function
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, config_path)
    fname = 'winner-feedforward'
    with open(fname, 'rb') as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    nc = Neurocontroller.init_from_neat_net(net, scales_in, scales_out)
    thrust_fcn = nc.get_thrust_vec_neat
    del local_dir, config_path, config, fname, genome, net, nc

    # Propagate for number of steps
    y0, yf = x_perturbed[first_point*n_dim*2:(first_point+1)*n_dim*2], make_new_bcs()[-1]
    if n_dim == 2:
        y0 = np.insert(y0, 2, 0.)
        y0 = np.insert(y0, 5, 0.)
    y0 = np.append(y0, m[first_point])
    du = np.max((a0_max, af_max))
    tu = np.sqrt(du ** 3 / gm)
    mu = m0
    fu = mu * du / tu / tu
    ti = np.linspace(t0, tf, num_nodes)
    ti /= tu
    ti = ti[first_point:]
    y, miss_ind = integrate_func_missed_thrust(thrust_fcn, y0, ti, yf, m_dry, T_max_kN, du, tu, mu, fu, Isp)

    # Need to make a state matrix with mass
    if first_point > 0:
        y_start = np.reshape(x_perturbed[:first_point*n_dim*2], [-1, 2*n_dim])
        if n_dim == 2:
            yzeros = np.zeros((y_start.shape[0], 1))
            y_start = np.hstack((y_start[:, :2], yzeros, y_start[:, 2:], yzeros))
        y_start = np.hstack((y_start, m[:first_point]))
    else:
        y_start = np.array([]).reshape((0, 7))
    y_matrix = np.vstack((y_start, y))
    thrust_vec_body = get_thrust_history(ti, y_matrix, yf, m_dry, T_max_kN, thrust_fcn)[:, :n_dim]
    thrust_vec_body[miss_ind] = 0.
    thrust_vec_inertial = rotate_thrust(thrust_vec_body, y_matrix)
    # if not np.mod(i+1, 100):
    #     print(i+1)
    return thrust_vec_inertial.ravel()


def save_traj(t: np.ndarray, x: np.ndarray, m: np.ndarray, u: np.ndarray,
              traj_fname: str = 'traj_data_desensitized.hdf5'):
    x = np.reshape(x, (len(t), -1))
    x = np.hstack((x, m))
    with h5py.File(traj_fname, 'w') as f:
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
    np.save('traj_data_s.npy', s)
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
    np.save('traj_data_dsdx.npy', dsdx)

    # Calculate update to states
    # delta_x = s / dsdx / len(x)
    delta_x = np.matmul(np.linalg.inv(dsdx), s)
    np.save('traj_data_delta_x.npy', delta_x)

    # Update states
    x += delta_x
    np.save('traj_data_x.npy', x)

    # Save new trajectory to file
    save_traj(t, x, m, u)

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

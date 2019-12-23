import pickle
import os
import neatfast as neat
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


def recreate_traj_from_pkl(fname, neat_net=False, print_mass=False, save_traj=False, traj_fname="traj_data.hdf5"):
    # Load best generation from pickle file
    with open(fname, 'rb') as f:
        xopt = pickle.load(f)

    # Recreate neurocontroller
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

    y0, yf = make_new_bcs()

    # Integrate transfer, final, and target trajectories
    # ti = np.power(np.linspace(0, 1, num_nodes), 3 / 2) * (tf - t0) + t0
    ti = np.linspace(t0, tf, num_nodes)
    ti /= tu

    # Integrate each of the trajectories
    tol_analytic = tol
    yinit_tf = period_from_inertial(y0[:-1], gm=gm)
    ytarg_tf = period_from_inertial(yf, gm=gm)
    yinit = integrate.solve_ivp(eom2BP, [t0, yinit_tf], y0[ind_dim], rtol=tol_analytic)
    y, miss_ind, full_traj = integrate_func_missed_thrust(thrust_fcn, y0, ti, yf, m_dry, T_max_kN, du, tu, mu, fu, Isp,
                                                          tol=tol, save_full_traj=True)
    yfinal_tf = period_from_inertial(y[-1, :-1], gm=gm)
    yfinal = integrate.solve_ivp(eom2BP, [t0, yfinal_tf], y[-1, ind_dim], rtol=tol_analytic)
    ytarg = integrate.solve_ivp(eom2BP, [t0, ytarg_tf], yf[ind_dim[:-1]], rtol=tol_analytic)

    # Calculate thrust vectors throughout transfer trajectory
    thrust_vec_body = get_thrust_history(ti, y, yf, m_dry, T_max_kN, thrust_fcn)[:, :n_dim]
    thrust_vec_body[miss_ind] = 0.
    thrust_vec_inertial = rotate_thrust(thrust_vec_body, y)

    # Plot the transfer
    arrow_color = 'chocolate'
    missed_color = 'red'
    fig, ax = plotTraj2D(full_traj[:, 1:-1], False, False, label='Transfer', start=True, end=True, show_legend=False)
    for mi in miss_ind:
        ax.plot(full_traj[20*mi:20*mi+21, 1] / au_to_km, full_traj[20*mi:20*mi+21, 2] / au_to_km, c=missed_color, zorder=7)
    fig, ax = plotTraj2DStruct(yfinal, False, False, fig_ax=(fig, ax), label='Final', show_legend=False)
    fig, ax = plotTraj2DStruct(yinit, False, False, fig_ax=(fig, ax), label='Initial', show_legend=False)
    q_scale = np.max(np.linalg.norm(thrust_vec_body, axis=1)) * 20

    # Arrows with heads
    # ax.quiver(y[:-1, 0] / au_to_km, y[:-1, 1] / au_to_km, thrust_vec_inertial[:, 0], thrust_vec_inertial[:, 1],
    #           angles='xy', zorder=8, width=0.0025, units='width', scale=q_scale, scale_units='width', minlength=0.1,
    #           headaxislength=5, headlength=6, headwidth=5, color=arrow_color)

    # Arrows without heads
    ax.quiver(y[:-1, 0] / au_to_km, y[:-1, 1] / au_to_km, thrust_vec_inertial[:, 0], thrust_vec_inertial[:, 1],
              angles='xy', zorder=8, width=0.004, units='width', scale=q_scale, scale_units='width', minlength=0.1,
              headaxislength=0, headlength=0, headwidth=0, color=arrow_color)
    plotTraj2DStruct(ytarg, False, True, fig_ax=(fig,ax), label='Target', end=True, show_legend=False)

    # Plot mass and thrust
    plotMassHistory(ti * tu * sec_to_day, y[:, -1], mt_ind=miss_ind)
    plotThrustHistory(ti[:-1] * tu * sec_to_day, thrust_vec_body, T_max_kN, mt_ind=miss_ind)

    # Save trajectory to file (states, times, controls)
    if save_traj:
        with h5py.File(traj_fname, "w") as f:
            f.create_dataset('t', data=ti)
            f.create_dataset('x', data=y)
            f.create_dataset('u', data=thrust_vec_inertial)

    # Print results
    if print_mass:
        print('Final mass = {0:.3f} kg'.format(y[-1, -1]))
    print('Final fitness = %f' % -traj_fit_func(y[-1, ind_dim], yf[ind_dim[:-1]], y0, y0[-1] / y[-1, -1]))


def make_last_traj(print_mass=False, save_traj=True):
    neat_net = True
    if neat_net:
        fname = 'winner-feedforward'
    else:
        fname = 'lgen.pkl'
    recreate_traj_from_pkl(fname, neat_net, print_mass=print_mass, save_traj=save_traj)


def get_thrust_history(ti, y, yf, m_dry, T_max_kN, thrust_fcn):
    # Calculate thrust vector at each time step
    thrust_vec = np.zeros((len(ti) - 1, 3))
    for i in range(len(ti) - 1):
        if y[i, -1] > m_dry + 0.01:
            mass_ratio = m_dry / y[i, -1]
            time_ratio = ti[i] / ti[-1]
            thrust_vec[i, :] = thrust_fcn(np.hstack((y[i, ind_dim], yf[ind_dim[:-1]], mass_ratio, time_ratio))) * T_max_kN
        else:
            thrust_vec[i, :] = np.array([0, 0, 0])
    return thrust_vec


def rotate_thrust(thrust_vec_body, y):
    fpa = np.arctan2(y[:, 4], y[:, 3])
    rot_ang = fpa
    dcm = np.array([[np.cos(rot_ang), -np.sin(rot_ang)], [np.sin(rot_ang), np.cos(rot_ang)]])
    thrust_vec_inertial = np.array([np.matmul(dcm[:, :, i], tvb) for i, tvb in enumerate(thrust_vec_body)])
    return thrust_vec_inertial


def make_neat_network_diagram():
    """
    Creates a network diagram in .svg format showing the nodes and connections.
    :return:
    """
    with open('winner-feedforward', 'rb') as f:
        winner = pickle.load(f)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    if n_dim == 2:
        node_names = {-1: 'a_c', -2: 'e_c', -3: 'w_c', -4: 'f_c', -5: 'a_t', -6: 'e_t', -7: 'w_t', -8: 'f_t',
                      -9: 'mass', -10: 'time', 0: 'Angle', 1: 'Throttle'}
    else:
        node_names = {-1: 'a_c', -2: 'e_c', -3: 'i_c', -4: 'w_c', -5: 'om_c', -6: 'f_c', -7: 'a_t', -8: 'e_t',
                      -9: 'i_t', -10: 'w_t', -11: 'om_t', -12: 'f_t', 0: 'Alpha', 1: 'Beta', 2: 'Throttle'}
    neat.visualize.draw_net(config, winner, node_names=node_names,
                            filename="winner-feedforward.gv", show_disabled=True, prune_unused=False)
    neat.visualize.draw_net(config, winner, node_names=node_names,
                            filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)


def load_traj(traj_fname='traj_data.hdf5'):
    # Load saved trajectory data
    with h5py.File(traj_fname, 'r') as f:
        t = f['t'][()]
        x = f['x'][()]
        u = f['u'][()]
    return t, x, u


def sensitivity(t, x, u):
    '''
    Calculate sensitivity matrix
    :return:
    '''
    # Reshape to a vector
    m = x[:, -1].ravel().reshape((-1, 1))
    x = np.hstack((x[:, :2], x[:, 3:5])).ravel()
    u = u.ravel()

    # Number of rows = number of timesteps
    n = len(t)

    # Get the partial derivative of each row
    # Serial
    # ddvdx = np.empty((4*n, 2*(n-1)))
    # for i in range(4*n):
    #     ddvdx[i] = get_ddvdx([x, m, u, i])
    # Parallel
    with mp.Pool(os.cpu_count() - 1) as p:
        ddvdx = p.map_async(get_ddvdx, zip(repeat(x, 4*n), repeat(m, 4*n), repeat(u, 4*n), range(4*n))).get()

    # Calculate L2 norm (Frobenius/matrix norm) of the matrix and return
    sensitivity = np.sqrt(np.sum(np.square(ddvdx)))
    return sensitivity


def get_ddvdx(args):
    """
    Calculate partial derivative of delta v with respect to state for ith element of state
    :param x:
    :param u:
    :param i:
    :return:
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


def evaluate_perturbed_trajectory(x_perturbed, m, i):
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
    y, miss_ind = integrate_func_missed_thrust(thrust_fcn, y0, ti, yf, m_dry, T_max_kN, du, tu, mu, fu, Isp, tol)

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
    if not np.mod(i+1, 100):
        print(i+1)
    return thrust_vec_inertial.ravel()


def desensitize():
    # Load trajectory
    t, x, u = load_traj()

    # Calculate sensitivity of original trajectory
    s = sensitivity(t, x, u)
    print('Original sensitivity: %f' % s)

    # Calculate sensitivity of perturbed trajectories
    dsdx = np.empty(len(x))
    for i in range(len(x)):
        x_step, v_step = 1, 0.001
        step = x_step if np.mod(i, 4) < 2 else v_step
        x_perturbed = x.copy()
        x_perturbed[i] += step
        s_perturbed = sensitivity(t, x_perturbed, u)
        dsdx[i] = (s_perturbed - s) / step

    # Calculate update to states
    delta_x = s / dsdx

    # Update states
    x += delta_x

    # Calculate new sensitivity
    s = sensitivity(x, u ,t)
    print('New sensitivity: %f' % s)

    return x

# TODO - there is a memory leak issue; after each sensitivity analysis runs, the memory use jumps up

# TODO - see if there could be a benefit from computing a step from minimizing delta v and minimizing s, and then add them together
# TODO - look into effect of learning rate
# TODO - see if the above two TODO's could be combined into one; use learning rate as a Pareto coefficient


if __name__ == '__main__':
    desensitize()

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


def recreate_traj_from_pkl(fname, neat_net=False, print_mass=False):
    # Load best generation from pickle file
    with open(fname, 'rb') as f:
        xopt = pickle.load(f)
    # ind_2d = [True, True, False, True, True, False, False]

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
    # thrust_vec_body = thrust_vec_inertial.copy()

    # Plot everything
    fig, ax = plotTraj2D(full_traj[:, 1:-1], False, False, label='Transfer', start=True, end=True, show_legend=False)
    ax.scatter(y[miss_ind, 0], y[miss_ind, 1], c='m', s=10, zorder=6)
    fig, ax = plotTraj2DStruct(yfinal, False, False, fig_ax=(fig, ax), label='Final', show_legend=False)
    fig, ax = plotTraj2DStruct(yinit, False, False, fig_ax=(fig, ax), label='Initial', show_legend=False)
    # q_scale = 1 / (np.linalg.norm(yf[:3]) * 3e2)
    q_scale = np.max(np.linalg.norm(thrust_vec_body, axis=1)) * 20
    ax.quiver(y[:-1, 0], y[:-1, 1], thrust_vec_inertial[:, 0], thrust_vec_inertial[:, 1], angles='xy', zorder=5,
              width=0.002, units='width', scale=q_scale, scale_units='width', minlength=0.1, headaxislength=3, headlength=6, headwidth=5)
    plotTraj2DStruct(ytarg, False, True, fig_ax=(fig,ax), label='Target', end=True, show_legend=False)
    plotMassHistory(ti * tu * sec_to_day, y[:, -1], mt_ind=miss_ind)
    plotThrustHistory(ti[:-1] * tu * sec_to_day, thrust_vec_body, T_max_kN, mt_ind=miss_ind)

    if print_mass:
        print('Final mass = {0:.3f} kg'.format(y[-1, -1]))

    print('Final fitness = %f' % -traj_fit_func(y[-1, ind_dim], yf[ind_dim[:-1]], y0, y0[-1] / y[-1, -1]))


def make_last_traj(print_mass=False):
    neat_net = True
    if neat_net:
        fname = 'winner-feedforward'
    else:
        fname = 'lgen.pkl'
    recreate_traj_from_pkl(fname, neat_net, print_mass=print_mass)


def get_thrust_history(ti, y, yf, m_dry, T_max_kN, thrust_fcn):
    # ind_2d = [True, True, False, True, True, False, False]
    # Calculate thrust vector at each time step
    thrust_vec = np.zeros((len(ti) - 1, 3))
    for i in range(len(ti) - 1):
        if y[i, -1] > m_dry + 0.01:
            mass_ratio = m_dry / y[i, -1]
            time_ratio = ti[i] / ti[-1]
            thrust_vec[i, :] = thrust_fcn(np.hstack((y[i, ind_dim], yf[ind_dim[:-1]], mass_ratio, time_ratio))) * T_max_kN
            # vel_angle = np.arctan2(y[i, n_dim+1], y[i, n_dim])
            # dcm = np.array([[np.cos(vel_angle), -np.sin(vel_angle)], [np.sin(vel_angle), np.cos(vel_angle)]])
            # thrust_vec[i, :] = np.matmul(dcm, thrust)
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

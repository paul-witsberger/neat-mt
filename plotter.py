import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

def plotTraj3D(y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(y[:, 0], y[:, 1], y[:, 2])
    ax.scatter3D(0, 0, 0, c='k')
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.set_title('Trajectory')
    set_axes_equal(ax)
    plt.show()

def plotTraj2D(yout, show_plot=True, save_plot=False, fname='tmp_traj_plot', title='', fig_ax=None, label='',
               start=False, end=False, show_legend=True, mt_ind=None):
    """
    Used for the transfer trajectory.
    :param yout:
    :param show_plot:
    :param save_plot:
    :param fname:
    :param title:
    :param fig_ax:
    :param label:
    :param start:
    :param end:
    :return:
    """
    if fig_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.grid(True)
    else:
        fig, ax = fig_ax
    ax.scatter(0, 0, c='k')
    ax.plot(yout[:, 0], yout[:, 1], label=label, zorder = 4)
    if start:
        ax.scatter(yout[0, 0], yout[0, 1], c='g', label=label+' - Start')
    if end:
        ax.scatter(yout[-1,0], yout[-1,1], c='#ff7f0e', label=label+' - End')
    if mt_ind is not None:
        ax.scatter(yout[mt_ind, 0], yout[mt_ind, 1], c='m', s=10)
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_title(title)
    ax.axis('equal')
    if show_legend:
        plt.legend()
    if show_plot:
        plt.show()
    if save_plot:
        fig.savefig(fname, dpi=600)
    return fig, ax

def plotTraj2DStruct(yout, show_plot=True, save_plot=False, fname='tmp_traj_plot', title='', fig_ax=None, label='',
                     start=False, end=False, show_legend=True):
    """
    Used for the initial and final orbits.
    :param yout:
    :param show_plot:
    :param save_plot:
    :param fname:
    :param title:
    :param fig_ax:
    :param label:
    :param start:
    :param end:
    :return:
    """
    if fig_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.grid(True)
    else:
        fig, ax = fig_ax
    ax.scatter(0, 0, c='k')
    if label == 'Final':
        ax.plot(yout.y[0], yout.y[1], label=label, zorder=5, dashes=[4, 4])
    else:
        ax.plot(yout.y[0], yout.y[1], label=label, zorder=5, dashes=[4, 4])
    if start:
        ax.scatter(yout.y[0,0], yout.y[1,0], c='g', label=label)
    if end:
        ax.scatter(yout.y[0,-1], yout.y[1,-1], c='r', label=label)
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_title(title)
    ax.axis('equal')
    if show_legend:
        plt.legend()
    if show_plot:
        plt.show()
    if save_plot:
        fig.savefig(fname, dpi=300)
    return fig, ax

def plotMassHistory(t, m, show_plot=False, save_plot=True, fname='tmp_mass_hist', mt_ind=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(True)
    ax.plot(t, m)
    if mt_ind is not None:
        ax.scatter(t[mt_ind], m[mt_ind], c='m', s=10)
    ax.set_title('Mass History')
    ax.set_xlabel('Time (day)')
    ax.set_ylabel('Mass (kg)')
    if show_plot:
        plt.show()
    if save_plot:
        fig.savefig(fname, dpi=300)

def plotThrustHistory(t, thrust_vec, T_max_kN, show_plot=False, save_plot=True, fname='tmp_thrust_hist', mt_ind=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    thrust_mag = np.linalg.norm(thrust_vec, axis=1)
    throttle = thrust_mag / T_max_kN
    angle = np.rad2deg(np.arctan2(thrust_vec[:, 1], thrust_vec[:, 0]))
    ax1.plot(t, throttle)
    ax1.set_title('Throttle')
    ax1.set_xlabel('Time (day)')
    ax1.set_ylabel('Throttle')
    ax1.grid(True)
    ax2.plot(t, angle)
    ax2.set_title('Thrust Angle')
    ax2.set_xlabel('Time (day)')
    ax2.set_ylabel('Angle (deg)')
    if mt_ind is not None:
        ax1.scatter(t[mt_ind], throttle[mt_ind], c='m', s=10)
        ax2.scatter(t[mt_ind], throttle[mt_ind], c='m', s=10)
    plt.grid(True)
    plt.tight_layout()
    if show_plot:
        plt.show()
    if save_plot:
        fig.savefig(fname, dpi=300)

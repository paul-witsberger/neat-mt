import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from constants import au_to_km

matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "serif"

# Plot colors
initial_color = 'grey'
initial_point = 'g'
initial_style = '-.'
initial_weight = 1
final_color = 'grey'
final_point = 'blue'
final_style = ':'
final_weight = 1.5
target_color = 'grey'
target_point = 'm'
target_style = '--'
target_weight = 1
transfer_color = 'k'
transfer_style = '-'
transfer_weight = 1.5
missed_color = 'r'
missed_style = '-'
missed_weight = 1
point_size = 15
sun_size = 20
sun_color = 'orange'
plot_points = True


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
               start=False, end=False, show_legend=True, scale_distance=True):
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
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        plt.grid(False)
    else:
        fig, ax = fig_ax
    if scale_distance:
        yout_scaled = yout / au_to_km
    else:
        yout_scaled = yout.copy()
    ax.scatter(0, 0, c=sun_color, s=sun_size)
    ax.plot(yout_scaled[:, 0], yout_scaled[:, 1], label=label, zorder=6, c=transfer_color, linewidth=transfer_weight)
    if start and plot_points:
        ax.scatter(yout_scaled[0, 0], yout_scaled[0, 1], c=initial_point, label=label+' - Start', s=point_size)
    if end and plot_points:
        ax.scatter(yout_scaled[-1, 0], yout_scaled[-1, 1], c=final_point, label=label+' - End', s=point_size)
    if scale_distance:
        ax.set_xlabel('X [AU]')
        ax.set_ylabel('Y [AU]')
    else:
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
                     start=False, end=False, show_legend=True, scale_distance=True):
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
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)
        plt.grid(False)
    else:
        fig, ax = fig_ax
    if label == 'Final':
        ax.plot(yout.y[0] / au_to_km, yout.y[1] / au_to_km, label=label, zorder=5, c=final_color, linewidth=final_weight, linestyle=final_style)
    elif label == 'Initial':
        ax.plot(yout.y[0] / au_to_km, yout.y[1] / au_to_km, label=label, zorder=5, c=initial_color, linewidth=initial_weight, linestyle=initial_style)
    else: # Target
        ax.plot(yout.y[0] / au_to_km, yout.y[1] / au_to_km, label=label, zorder=5, c=target_color, linewidth=target_weight, linestyle=target_style)
    if end and plot_points: # Target
        ax.scatter(yout.y[0,-1] / au_to_km, yout.y[1,-1] / au_to_km, c=target_point, label=label, s=point_size)

    if scale_distance:
        ax.set_xlabel('X [AU]', fontname='Times New Roman')
        ax.set_ylabel('Y [AU]', fontname='Times New Roman')
    else:
        ax.set_xlabel('X [km]', fontname='Times New Roman')
        ax.set_ylabel('Y [km]', fontname='Times New Roman')
    ax.set_title(title, fontname='Times New Roman')
    ax.axis('equal')
    if show_legend:
        plt.legend()
    if show_plot:
        plt.show()
    if save_plot:
        fig.savefig(fname, dpi=600)
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

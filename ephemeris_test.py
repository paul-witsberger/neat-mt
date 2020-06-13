import constants as c
import jplephem
from jplephem.spk import SPK
from time import time
from datetime import datetime
import numpy as np
from orbit_util import euler313, keplerian_to_inertial_3d, mean_to_true_anomaly
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ae = c.ephem
je = SPK.open('de438.bsp')

init_body = 'earth'
target_body = 'mars'
central_body = 'sun'
t0_str = '2025 Jan 01 00:00:00'
tf_str = '2026 Jan 01 00:00:00'
fmt = '%Y %b %d %H:%M:%S'
ordinal_to_julian = 1721424.5
t0 = datetime.strptime(t0_str, fmt).toordinal() + ordinal_to_julian
tf = datetime.strptime(tf_str, fmt).toordinal() + ordinal_to_julian
times = np.linspace(t0, tf, 365)
dt = times - c.reference_date_jd1950
obliquity = 23.437 * np.pi / 180

do_speed_test = False
max_iter = 1000
if do_speed_test:
    ts = time()
    for i in range(max_iter):
        sb2eb_p, sb2eb_v = je[0, 3].compute_and_differentiate(times)
        sb2s_p, sb2s_v = je[0, 10].compute_and_differentiate(times)
        s2eb_p, s2eb_v = sb2eb_p - sb2s_p, sb2eb_v - sb2s_v
        s2eb_v /= 86400
        s2eb_p = euler313(s2eb_p, 0, -obliquity, 0)
        s2eb_v = euler313(s2eb_v, 0, -obliquity, 0)
    print('Elapsed time (JPL): %f' % (time() - ts))

    ts = time()
    for i in range(max_iter):
        states_coe = ae(['a', 'e', 'i', 'w', 'O', 'M'], [init_body], dt / 36525).T
        # states_coe[:, 2] = 0.0
        states_coe[:, 5] = np.array([mean_to_true_anomaly(m, e) for m, e, in zip(states_coe[:, 5], states_coe[:, 2])])
        # states_i = keplerian_to_inertial_3d(states_coe.T, mean_or_true='mean')
        # states_i = np.array([keplerian_to_inertial_3d(state, mean_or_true='mean') for state in states_coe])
        # pos_i, vel_i = states_i[:, :3].T, states_i[:, 3:6].T
    print('Elapsed time (analytic): %f' % (time() - ts))

show_plots = True
if show_plots:
    # Make data - reference
    a, e, i, w, lan = c.a_earth_km, c.e_earth, c.i_earth_rad, c.w_earth_rad, c.lan_earth_rad
    dt = times - c.reference_date_jd1950
    per = c.per_earth_day
    f = (dt % per) / per * 2 * np.pi + c.f_jd1950
    ref = keplerian_to_inertial_3d(np.vstack((np.ones_like(f) * a, np.ones_like(f) * e, np.ones_like(f) * i, np.ones_like(f) * w,
                                    np.ones_like(f) * lan, f)), gm=c.u_sun_km3s2).T

    # Make data - analytic ephemeris
    states_coe = ae(['a', 'e', 'i', 'w', 'O', 'M'], [init_body], dt / 36525).T
    # states_coe[:, 2] = 0.0
    states_i = keplerian_to_inertial_3d(states_coe.T, mean_or_true='mean')
    analytic = states_i[:, :3].T

    # Make data - JPL ephemeris
    sb2eb_p, sb2eb_v = je[0, 3].compute_and_differentiate(times)
    sb2s_p, sb2s_v = je[0, 10].compute_and_differentiate(times)
    s2eb_p, s2eb_v = sb2eb_p - sb2s_p, sb2eb_v - sb2s_v
    s2eb_v /= 86400
    ephem = euler313(s2eb_p, 0, -obliquity, 0)

    # Make figure
    fig = plt.figure()
    ax = fig.add_subplot('111', projection='3d')
    ax.scatter(0, 0, 0, c='y', s=100, label='Sun')
    ax.scatter(*ref[:, 0], label='ref_0')
    ax.scatter(*analytic[:, 0], label='analytic_0')
    ax.scatter(*ephem[:, 0], label='ephem_0')
    ax.plot(ref[0], ref[1], ref[2], label='ref')
    ax.plot(analytic[0], analytic[1], analytic[2], label='analytic')
    ax.plot(ephem[0], ephem[1], ephem[2], label='ephem')

    # Create cubic bounding box to simulate equal aspect ratio
    xmin, xmax = min(analytic[0]), max(analytic[0])
    ymin, ymax = min(analytic[1]), max(analytic[1])
    zmin, zmax = min(analytic[2]), max(analytic[2])
    max_range = np.array([xmax - xmin, ymax - ymin, zmax - zmin]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (xmax + xmin)
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (ymax + ymin)
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (zmax + zmin)
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.show()

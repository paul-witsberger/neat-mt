import constants as c
import jplephem
from jplephem.spk import SPK
from time import time
from datetime import datetime
import numpy as np
from orbit_util import euler313, keplerian_to_inertial_3d, inertial_to_keplerian_3d, mag3, cross,\
    keplerian_to_perifocal_3d, mean_to_true_anomaly, fix_angle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

keplerian_to_perifocal_3d(np.array([1e8, 0, 0, 0, 0, 0]))
keplerian_to_inertial_3d(np.array([1e8, 0, 0, 0, 0, 0]))
inertial_to_keplerian_3d(np.random.rand(6))
euler313(np.array([0, 0, 0]), 0, 0, 0)
mag3(np.array([1, 0, 0]))
cross(np.array([1, 0, 0]), np.array([0, 0, 1]))
mean_to_true_anomaly(0, 0.1)
fix_angle(0)

ae = c.ephem
je = SPK.open('de438.bsp')

init_body = 'earth'
target_body = 'mars'
central_body = 'sun'
t0_str = '2005 Jan 01 00:00:00'
tf_str = '2048 Jan 01 00:00:00'
fmt = '%Y %b %d %H:%M:%S'
ordinal_to_julian = 1721424.5
t0 = datetime.strptime(t0_str, fmt).toordinal() + ordinal_to_julian
tf = datetime.strptime(tf_str, fmt).toordinal() + ordinal_to_julian
num_points = 10000
times = np.linspace(t0, tf, num_points)
dt = times - c.reference_date_jd1950
obliquity = 23.437 * np.pi / 180

do_speed_test = True
if do_speed_test:
    max_iter = 1
    print('Testing %i iterations:' % max_iter)
    ts = time()
    for i in range(max_iter):
        sb2eb_p, sb2eb_v = je[0, 3].compute_and_differentiate(times)
        sb2mb_p, sb2mb_v = je[0, 4].compute_and_differentiate(times)
        sb2s_p, sb2s_v = je[0, 10].compute_and_differentiate(times)
        s2eb_p, s2eb_v = sb2eb_p - sb2s_p, sb2eb_v - sb2s_v
        s2eb_v /= 86400
        s2mb_p, s2mb_v = sb2mb_p - sb2s_p, sb2mb_v - sb2s_v
        s2mb_v /= 86400
        s2eb_p = euler313(s2eb_p, 0, -obliquity, 0)
        s2eb_v = euler313(s2eb_v, 0, -obliquity, 0)
        s2mb_p = euler313(s2mb_p, 0, -obliquity, 0)
        s2mb_v = euler313(s2mb_v, 0, -obliquity, 0)
        s2eb = np.hstack((s2eb_p, s2eb_v))
        s2mb = np.hstack((s2mb_p, s2mb_v))
        s2eb_coe = np.array([inertial_to_keplerian_3d(s) for s in s2eb])
        s2mb_coe = np.array([inertial_to_keplerian_3d(s) for s in s2mb])
    print('Elapsed time [sec] - (JPL, Cart->COE):\t\t%f' % (time() - ts))

    ts = time()
    for i in range(max_iter):
        states_coe = ae(['a', 'e', 'i', 'w', 'O', 'M'], [init_body, target_body], dt / 36525).T
        states_e_coe, states_m_coe = states_coe[:, 0, :], states_coe[:, 1, :]
        states_e_i = keplerian_to_inertial_3d(states_e_coe.T, mean_or_true='mean')
        states_m_i = keplerian_to_inertial_3d(states_m_coe.T, mean_or_true='mean')
        pos_e_i, vel_e_i = states_e_i[:, :3].T, states_e_i[:, 3:6].T
        pos_m_i, vel_m_i = states_m_i[:, :3].T, states_m_i[:, 3:6].T
    print('Elapsed time [sec] - (analytic, COE->Cart):\t%f' % (time() - ts))

show_plots = True
if show_plots:
    plot_actual = True
    plot_difference = True
    # Make data - reference
    a, e, i, w, lan = c.a_earth_km, c.e_earth, c.i_earth_rad, c.w_earth_rad, c.lan_earth_rad
    dt = times - c.reference_date_jd1950
    per = c.per_earth_day
    f = (dt % per) / per * 2 * np.pi + c.f_jd1950
    ref = keplerian_to_inertial_3d(np.vstack((np.ones_like(f) * a, np.ones_like(f) * e, np.ones_like(f) * i,
                                              np.ones_like(f) * w, np.ones_like(f) * lan, f)), gm=c.u_sun_km3s2).T

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
    ephem = euler313(s2eb_p, 0, -obliquity, 0).T

    if plot_actual:
        # Make figure
        fig = plt.figure(1)
        ax = fig.add_subplot('111', projection='3d')
        ax.scatter(0, 0, 0, c='y', s=100, label='Sun')
        ax.scatter(*ref[:3, 0], label='ref_0')
        ax.scatter(*analytic[:, 0], label='analytic_0')
        ax.scatter(*ephem[:, 0], label='ephem_0')
        ax.plot(ref[0], ref[1], ref[2], label='ref')
        ax.plot(analytic[0], analytic[1], analytic[2], label='analytic')
        ax.plot(ephem[0], ephem[1], ephem[2], label='ephem')
        plt.legend()

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

    if plot_difference:
        # Make figure
        fig = plt.figure(2)
        ax = fig.add_subplot('111', projection='3d')
        ax.plot(ref[0] - analytic[0], ref[1] - analytic[1], ref[2] - analytic[2], label='Ref-Anl')
        ax.plot(ref[0] - ephem[0], ref[1] - ephem[1], ref[2] - ephem[2], label='Ref-JPL')
        ax.plot(analytic[0] - ephem[0], analytic[1] - ephem[1], analytic[2] - ephem[2], label='Anl-JPL')
        plt.legend()

    plt.show()

'''
Which system should be used?

Reference values are convenient because they are constant.
Analytic ephemerides provide slightly more accuracy than reference values and can be assumed constant at a time.
JPL ephemerides are most accurate, but are therefore not periodic.
 
Reference values are fastest in COE, but slow in Cartesian coordinates.
Analytic ephemerides are very fast in COE, but slow-ish in Cartesian coordinates.
JPL ephemerides are fast in Cartesian, but slow in COE. When called few times, is fast, but can be quite slow when
    called 1000's of times.
    
After speed improvements, analytic ephemerides are sufficiently close in speed in JPL ephemerides. Ref and analytic are
effectively the same speed in COE and therefore basically the same in Cartesian. If millions of calls are need ref may
be faster, but if that many values are needed its probably over a long period and the accuracy of analytic may be nice.

It looks like analytic ephemerides is the current best choice. It will at least have the illusion of accuracy to bolster
the legitimacy of the code, provides the relative location of the planets at different times, is fairly straightforward
to use, does not require external packages/files, and I can say I implemented myself.
'''

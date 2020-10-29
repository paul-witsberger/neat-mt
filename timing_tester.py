import numpy as np
from time import time
import timeit
from numba import njit, vectorize, float64, int64
from missed_thrust import find_nearest
import constants as c
import traj_config as tc
import orbit_util as ou
from orbit_util import mag3, cross
from traj_config import ind_dim
from constants import au_to_km
import boost_tbp
from builder import make_last_traj


def timeit_auto(stmt="pass", setup="pass", repeat=3):
    """
    http://stackoverflow.com/q/19062202/190597 (endolith)
    Imitate default behavior when timeit is run as a script.

    Runs enough loops so that total execution time is greater than 0.2 sec,
    and then repeats that 3 times and keeps the lowest value.

    Returns the number of loops and the time for each loop in microseconds
    """
    t = timeit.Timer(stmt, setup)

    # determine number so that 0.1 <= total time < 1.0
    number = 1
    for i in range(0, 10):
        number = 10**i
        x = t.timeit(number) # seconds
        if x >= 0.1:
            break
    r = t.repeat(repeat, number)
    best = min(r)
    usec = best * 1e6 / number
    return number, usec

# num_iter = int(1e4)

# Set up
tbp = boost_tbp.TBP()
y = np.array([10000, 0, 0, 0, 6.5, 0, 1000])
ti = np.arange(0, 10000, 100)
i = 0
param = [float(c.g0_ms2 * tc.Isp / tc.du * tc.tu), float(tc.m_dry / tc.mu), 0., 0., 0.]
state_size = 7
time_size = 2
param_size = 5
rtol1, atol1 = 1e-7, 1e-7
rtol2, atol2 = 1e-8, 1e-8
step_size = float((ti[i+1] - ti[i]) / tc.n_steps)
integrator_type_1 = int(0)  # 0 fixed step
integrator_type_2 = int(1)  # adaptive step
eom_type = int(3)
args = [list(y[i] / tc.state_scales), [float(ti[i]), float(ti[i + 1])], param, state_size, time_size, param_size]

def version1(inputs=args, rtol=rtol1, atol=atol1, ss=step_size, it=integrator_type_2, et=eom_type):
    make_last_traj()

def version2(inputs=args, rtol=rtol2, atol=atol2, ss=step_size, it=integrator_type_2, et=eom_type):
    traj = tbp.prop(*inputs, rtol, atol, ss, it, et)


# Version 1
version1()
# ts = time()
# for i in range(num_iter):
#     version1()
# tf = time()
# total_1 = tf - ts
# print('\n\nVersion 1\nTotal time = %e sec' % total_1)
# print('Avg time = %e sec' % (total_1 / num_iter))
print('Version 1')
num, timing = timeit_auto(setup='from __main__ import version1', stmt='version1()')
print('%i loops, best of 3: %.4f usec per loop' % (num, timing))

# Version 2
version2()
# ts = time()
# for i in range(num_iter):
#     version2(inputs)
# tf = time()
# total_2 = tf - ts
# print('\n\nVersion 2\nTotal time = %e sec' % total_2)
# print('Avg time = %e sec\n\n' % (total_2 / num_iter))
print('Version 2')
num, timing = timeit_auto(setup='from __main__ import version2', stmt='version2()')
print('%i loops, best of 3: %.4f usec per loop' % (num, timing))


# Print results and winner
# if total_1 < total_2:
#     print('Version 1 is faster by %.2f%% (%.2e sec)' % ((total_2 - total_1) / total_1 * 100, total_2 / num_iter - total_1 / num_iter))
# elif total_2 < total_1:
#     print('Version 2 is faster by %.2f%% (%.2e sec)' % ((total_1 - total_2) / total_2 * 100, total_1 / num_iter - total_2 / num_iter))
# else:
#     print('It''s a tie!')
# print()

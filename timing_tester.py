import numpy as np
from time import time
import timeit
import os
from operator import itemgetter
import traj_config as tc
import orbit_util as ou


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
        x = t.timeit(number)  # seconds
        if x >= 0.1:
            break
    r = t.repeat(repeat, number)
    best = min(r)
    usec = best * 1e6 / number
    return number, usec


# num_iter = int(1e4)

# Set up
state_f = np.array([-1.32427491e+08, -9.62901286e+07,  0.,  1.59757739e+01, -2.32281637e+01, 0.])
tf = 9.146361329085716
short = tc.capture_short

def version1():
    num_iter = 1
    ou.lambert_min_dv(tc.gm, state_f, tf, tc.capture_time_low, tc.capture_time_high, num_iter, short=short)


def version2():
    num_iter = 10
    ou.lambert_min_dv(tc.gm, state_f, tf, tc.capture_time_low, tc.capture_time_high, num_iter, short=short)


# Version 1
version1()
print('Version 1')
num, timing = timeit_auto(setup='from __main__ import version1', stmt='version1()')
print('%i loops, best of 3: %.4f usec per loop' % (num, timing))

# Version 2
version2()
print('Version 2')
num, timing = timeit_auto(setup='from __main__ import version2', stmt='version2()')
print('%i loops, best of 3: %.4f usec per loop' % (num, timing))


# Print results and winner
# if total_1 < total_2:
#     print('Version 1 is faster by %.2f%% (%.2e sec)' % ((total_2 - total_1) / total_1 * 100,
#     total_2 / num_iter - total_1 / num_iter))
# elif total_2 < total_1:
#     print('Version 2 is faster by %.2f%% (%.2e sec)' % ((total_1 - total_2) / total_2 * 100,
#     total_1 / num_iter - total_2 / num_iter))
# else:
#     print('It''s a tie!')
# print()

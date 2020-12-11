import numpy as np
from time import time
import timeit
import os
from operator import itemgetter
import traj_config as tc
import orbit_util as ou
import constants as c
from numba import njit, jit


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
    number = 3
    for i in range(0, 10):
        number = 10**i
        x = t.timeit(number)  # seconds
        if x >= 0.1:
            break
    r = t.repeat(repeat, number)
    best = min(r)
    usec = best * 1e6 / number
    return number, usec


# Set up


@jit(nopython=True, cache=True)
def version1():
    tf = tc.tf
    t0 = 0.
    tbe_factor = 1.
    rd_factor = 1.
    # Define Weibull distribution parameters
    k_tbe, lambda_tbe = 0.86737, 0.62394 * tbe_factor
    k_rd, lambda_rd = 1.144, 2.459 * rd_factor
    max_discovery_delay_days, op_recovery_days = 3, 0.5
    # Initialize list of outage start and stop times
    outages = list()
    time_between_events, recovery_duration, cascading = 0., 0., False
    prev_start = t0
    while True:
        time_between_events = np.random.weibull(k_tbe) * lambda_tbe * c.year_to_sec  # calculate time between events
        # Check that the previous outage has already finished.
        if len(outages) > 0:
            # try:
            if time_between_events < (outages[-1][1] - outages[-1][0]):
                cascading = True
            # except IndexError as err:
            #     print('spot 2')
            #     raise err
        else:
            cascading = False
        # check if the next event happens before the end of the time of flight
        if (time_between_events < tf) and (prev_start + time_between_events < tf):
            recovery_duration = np.random.weibull(k_rd) * lambda_rd * c.day_to_sec  # calculate the recovery duration
            recovery_duration += (np.random.rand() * max_discovery_delay_days * c.day_to_sec) + \
                                 (op_recovery_days * c.day_to_sec)  # discovery delay and operational recovery
            if cascading:
                outages[-1][1] = outages[-1][0] + time_between_events + recovery_duration

            else:
                outages.append([prev_start + time_between_events, prev_start + time_between_events + recovery_duration])
            prev_start = outages[-1][0]
        else:
            if len(outages) > 0:
                return np.array(outages)
            else:
                return np.empty((0, 0), dtype=np.float64)


def version2():
    tf = tc.tf
    t0 = 0.
    tbe_factor = 1.
    rd_factor = 1.
    # Define Weibull distribution parameters
    k_tbe, lambda_tbe = 0.86737, 0.62394 * tbe_factor
    k_rd, lambda_rd = 1.144, 2.459 * rd_factor
    max_discovery_delay_days, op_recovery_days = 3, 0.5
    # Initialize list of outage start and stop times
    outages = list()
    time_between_events, recovery_duration, cascading = 0., 0., False
    prev_start = t0
    while True:
        time_between_events = np.random.weibull(k_tbe) * lambda_tbe * c.year_to_sec  # calculate time between events
        # Check that the previous outage has already finished.
        if len(outages) > 0:
            # try:
            if time_between_events < (outages[-1][1] - outages[-1][0]):
                cascading = True
            # except IndexError as err:
            #     print('spot 2')
            #     raise err
        else:
            cascading = False
        # check if the next event happens before the end of the time of flight
        if (time_between_events < tf) and (prev_start + time_between_events < tf):
            recovery_duration = np.random.weibull(k_rd) * lambda_rd * c.day_to_sec  # calculate the recovery duration
            recovery_duration += (np.random.rand() * max_discovery_delay_days * c.day_to_sec) + \
                                 (op_recovery_days * c.day_to_sec)  # discovery delay and operational recovery
            if cascading:
                outages[-1][1] = outages[-1][0] + time_between_events + recovery_duration

            else:
                outages.append([prev_start + time_between_events, prev_start + time_between_events + recovery_duration])
            prev_start = outages[-1][0]
        else:
            if len(outages) > 0:
                return np.array(outages)
            else:
                return np.empty((0, 0), dtype=np.float64)


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

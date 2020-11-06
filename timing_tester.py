import numpy as np
from time import time
import timeit
import os
import neatfast as neat
from operator import itemgetter


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
config_name = 'default'
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config_' + config_name)
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)
pop = neat.Population(config)


def version1():
    itemgetter(*[k for k in pop.population.keys()])(pop.population)


def version2():
    list(map(pop.population.get, [k for k in pop.population.keys()]))


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
#     print('Version 1 is faster by %.2f%% (%.2e sec)' % ((total_2 - total_1) / total_1 * 100,
#     total_2 / num_iter - total_1 / num_iter))
# elif total_2 < total_1:
#     print('Version 2 is faster by %.2f%% (%.2e sec)' % ((total_1 - total_2) / total_2 * 100,
#     total_1 / num_iter - total_2 / num_iter))
# else:
#     print('It''s a tie!')
# print()

from __future__ import print_function

import os
import pickle

import neatfast as neat
# import visualize
from big_idea import eval_traj_neat
from builder import make_last_traj, make_neat_network_diagram


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Initialize population
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    # Run
    pe = neat.ParallelEvaluator(os.cpu_count() - 1, eval_traj_neat)
    max_generations = 1000
    winner = pop.run(pe.evaluate, n=max_generations)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    # Make plots
    neat.visualize.plot_stats(stats, ylog=True, filename="feedforward-fitness.svg")
    neat.visualize.plot_species(stats, filename="feedforward-speciation.svg")
    make_neat_network_diagram()
    make_last_traj(print_mass=True)


if __name__ == '__main__':
    run()

# TODO figure out what weight and bias mutate rate, mutate power, etc mean
# ...and then tailor values for this problem - I think there needs to be more "fine" updates of the weights/biases

# TODO double check how the best fitness is saved - make sure the genome that had the best fitness ever is saved, not
# just the genome with the most recent best fitness

# TODO experiment with changing the probability of adding nodes/connections to be slightly higher than removing them

# NOTE: changed bias initial std to 0.0 from 1.0, weight initial std to 0.1 from 1.0, bias and weight max/min values to
#       5/-5 from 10/-10

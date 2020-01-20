from __future__ import print_function

import os
import pickle

import neatfast as neat
from neatfast import visualize
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
    num_workers = os.cpu_count() - 1
    # num_workers = 1  # for debugging purposes
    pe = neat.ParallelEvaluator(num_workers, eval_traj_neat)
    max_generations = 100
    winner = pop.run(pe.evaluate, n=max_generations)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    # Make plots
    visualize.plot_stats(stats, ylog=True, filename="tmp_fitness_history.svg")
    visualize.plot_species(stats, filename="tmp_speciation_history.svg")
    make_neat_network_diagram()
    make_last_traj(print_mass=True)


if __name__ == '__main__':
    run()

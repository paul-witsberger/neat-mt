import os
import pickle
import neatfast as neat
from neatfast import visualize
from missed_thrust import eval_traj_neat
from builder import make_last_traj, make_neat_network_diagram
import traj_config
import cProfile
import time


def run(config_name='default', init_state=None, parallel=True, max_gens=traj_config.max_generations):
    # Load the config file, which is assumed to live in the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config', 'config_' + config_name)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Initialize population
    pop = neat.Population(config, initial_state=init_state)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    # Run
    if parallel:
        num_workers = os.cpu_count() - 1
        pe = neat.ParallelEvaluator(num_workers, eval_traj_neat)
        winner, best_pop = pop.run(pe.evaluate, n=max_gens)
    else:
        se = neat.SerialEvaluator(eval_traj_neat)
        winner, best_pop = pop.run(se.evaluate, n=max_gens)

    print('\n\n' + '*' * 60 + '\n' + '*' * 24 + '  FINISHED  ' + '*' * 24 + '\n' + '*' * 60 + '\n\n')

    # Save the winner.
    with open('results//winner_' + config_name, 'wb') as f:
        pickle.dump(winner, f)

    # Save the winner in plain text.
    with open('results//network_' + config_name + '.txt', 'w') as f:
        f.write(str(winner))

    pop.population = best_pop

    # Make plots
    visualize.plot_stats(stats, ylog=True, filename="results//fitness_history_" + config_name + ".svg")
    visualize.plot_species(stats, filename="results//speciation_history_" + config_name + ".svg")
    # make_neat_network_diagram(config_name=config_name)
    make_last_traj(print_mass=True, config_name=config_name)

    return pop


def prepare_population(_pop):
    # Reset some of the population statistics between runs
    for k in _pop.species.species.keys():
        _pop.species.species[k].last_improved = 1
        _pop.species.species[k].created = 1
    _pop.best_genome = None
    _pop.generation = 1
    return [_pop.population, _pop.species, _pop.generation]


if __name__ == '__main__':
    # NOTE add -OO to configuration when running for slight speed improvement
    _get_timing = False
    _parallel = True
    _max_gens = [100, 50, 20]

    t_start = time.time()
    if _get_timing:
        # Run serially to capture all function calls
        cProfile.run('run(parallel=False)', 'results//neat_mt_timing_info')

    else:
        # Step 1 of 3: run with a coarse distribution of nodes to get general steering law
        coarse_str = 'coarse'
        pop = run(config_name=coarse_str,
                  parallel=_parallel,
                  max_gens=_max_gens[0])

        # Step 2 of 3: run with a finer distribution of nodes and some missed thrust to reasonably close solution
        intermediate_str = 'intermediate'
        intermediate_state_1 = prepare_population(pop)
        pop = run(config_name=intermediate_str,
                  init_state=intermediate_state_1,
                  parallel=_parallel,
                  max_gens=_max_gens[1])

        # Step 3 of 3: run with little mutation and many cases of missed thrust to perform RBDO
        final_str = 'final'
        intermediate_state_2 = prepare_population(pop)
        run(config_name=final_str,
            init_state=intermediate_state_2,
            parallel=_parallel,
            max_gens=_max_gens[2])

    t_end = time.time()
    print('\nTotal runtime = %.1f sec' % (t_end - t_start))

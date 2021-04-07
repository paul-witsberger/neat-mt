import os
import pickle
import neatfast as neat
from neatfast import visualize
from missed_thrust import eval_traj_neat
from builder import make_last_traj, make_neat_network_diagram
import cProfile
import time
import numpy as np


def dummy_fitness_func(genome, config):
    return -np.random.rand() * 100


def run(config_name: str = 'default', init_state: list = None, parallel: bool = True,
        max_gens: int = None, save_population: bool = False) -> neat.population.Population:
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
        num_workers = os.cpu_count() - 16
        num_workers = 60
        timeout = None
        pe = neat.ParallelEvaluator(num_workers, eval_traj_neat, timeout=timeout)
        winner, best_pop = pop.run(pe.evaluate, n=max_gens, num_workers=24)
    else:
        se = neat.SerialEvaluator(dummy_fitness_func)
        winner, best_pop = pop.run(se.evaluate, n=max_gens)

    print('\n\n' + '*' * 60 + '\n' + '*' * 24 + '  FINISHED  ' + '*' * 24 + '\n' + '*' * 60 + '\n\n')

    # Save the winner.
    with open('results//winner_' + config_name, 'wb') as f:
        pickle.dump(winner, f)

    # Save the winner in plain text.
    with open('results//network_' + config_name + '.txt', 'w') as f:
        f.write(str(winner))

    pop.population = best_pop

    if save_population:
        with open('results//population_' + config_name, 'wb') as f:
            pickle.dump(pop, f)

    # Make plots
    visualize.plot_stats(stats, ylog=True, filename="results//fitness_history_" + config_name + ".svg")
    visualize.plot_species(stats, filename="results//speciation_history_" + config_name + ".svg")
    make_neat_network_diagram(config_name=config_name)
    try:
        make_last_traj(print_mass=True, config_name=config_name)
    except NotImplementedError:
        pass

    return pop


def prepare_population(_pop: neat.population.Population) -> list:
    # Reset some of the population statistics between runs
    for k in _pop.species.species.keys():
        _pop.species.species[k].last_improved = 1
        _pop.species.species[k].created = 1
    _pop.best_genome = None
    _pop.generation = 1
    return [_pop.population, _pop.species, _pop.generation]


def load_population(config_name: str) -> neat.population.Population:
    with open('results//population_' + config_name, 'rb') as f:
        pop = pickle.load(f)
    return pop


if __name__ == '__main__':
    # NOTE add -OO to configuration when running for slight speed improvement
    _get_timing = False
    _parallel = True
    _max_gens = [0, 200, 0]
    _save_population = [True, True, True]
    _load_population = [False, False, True]
    pop = None

    t_start = time.time()
    if _get_timing:
        # Run serially to capture all function calls
        cProfile.run('run(parallel=False)', 'results//neat_mt_timing_info')
    else:
        phase_strs = ['coarse', 'intermediate', 'final']
        num_phases = 3
        for phase in range(num_phases):
            if _max_gens[phase] > 0:
                phase_str = phase_strs[phase]
                pop = load_population(phase_str) if _load_population[phase] else pop
                phase_init_state = prepare_population(pop) if pop is not None else None
                pop = run(config_name=phase_str,
                          init_state=phase_init_state,
                          parallel=_parallel,
                          max_gens=_max_gens[phase],
                          save_population=_save_population[0])

    t_end = time.time()
    print('\nTotal runtime = %.1f sec' % (t_end - t_start))


# TODO continuing after extinction fails with KeyError
# TODO sometimes fails with KeyError after stagnation
# TODO best individual can get passed between species
# TODO sometimes fails with KeyError randomly
#   -> I bet all of these are actually related somehow - if a genome moves to another species while being the best,
#      it can't be found during post_evalate (...actually, they are separate)
#   -> actually, it may be partially tied to my edit for "same child" if I wasn't adding (...actually, not this)
#   -> also, this can happen with just one species
#   -> somehow, {genome.key: species.key} is not being set correctly - this occurs in speciate() [species.py, 131]
#       -> I don't see how the loop assigning self.genome_to_species could cause this error - the issue probably
#          occurs above - need to check new_representatives and new_members (error is probably with new_members)

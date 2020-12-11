"""Implements the core evolution algorithm."""
from __future__ import print_function
import pickle
import time
from neatfast.reporting import ReporterSet
from neatfast.math_util import mean
from neatfast.six_util import iteritems, itervalues
from neatfast import visualize
import numpy as np
from operator import itemgetter
import copy


class CompleteExtinctionException(Exception):
    pass


class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config, self.reporters, stagnation)
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type, config.genome_config, config.pop_size)
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 1
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state
            if initial_state[0] is None:
                self.population = self.reproduction.create_new(config.genome_type, config.genome_config, config.pop_size)
            if initial_state[1] is None:
                self.species = config.species_set_type(config.species_set_config, self.reporters)
            if initial_state[2] is None:
                self.generation = 1
            self.species.speciate(config, self.population, self.generation)

        self.best_genome = None

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def run(self, fitness_function, n=None, num_workers=None):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        best_fitness = -1e10
        best = None

        # TODO - PAUL EDIT
        # Make a list/array of the best individuals of this run. Keep as many individuals as the population. Sort by
        # fitness, and disregard if individuals are in the same or different species. This "overall best" population
        # will be used to seed future runs. Save and return at the end of the run. By saving, we can restart/retry
        # training later.
        fitness_library = np.empty((len(self.population), 2), np.float64)

        while n is None or self.generation <= n:

            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided function.
            # PAUL EDIT #
            # only evaluate new individuals
            # TODO I think it's possible that new individuals can slip through without being evaluated, thus giving
            #   them a fitness of None and causing line 118 to break (if best is None or g.fitness > best.fitness:)
            # individuals_to_evaluate = [i for i in list(iteritems(self.population)) if (i[0] not in fitness_library[:, 0])]
            # print('Evaluating %i individuals.' % len(individuals_to_evaluate))
            individuals_to_evaluate = [i for i in list(iteritems(self.population))]
            if self.generation > 1:
                fitness_function(individuals_to_evaluate, self.config)
            else:
                if num_workers is not None:
                    fitness_function(individuals_to_evaluate[:num_workers], self.config, no_timeout=True)
                    fitness_function(individuals_to_evaluate[num_workers:], self.config)
                else:
                    fitness_function(individuals_to_evaluate, self.config, no_timeout=True)

            # # Store new entries
            # if len(individuals_to_evaluate) > 0:
            #     fitness_library = np.append(fitness_library, [[i[0], i[1].fitness] for i in individuals_to_evaluate
            #                                                   if i[0] not in fitness_library[:, 0]], axis=0)
            # END PAUL EDIT #

            # Get best individual
            curr_fit, curr_pop_keys, curr_pop_values = None, None, None
            curr_fit = np.array([v.fitness for v in itervalues(self.population)])
            # curr_pop = np.array([v for v in itervalues(self.population)])
            curr_pop_keys = np.array(list(self.population.keys()))
            curr_pop_values = np.array(list(self.population.values()))
            # pop_fit = np.array([[v.fitness, v] for v in itervalues(self.population)])
            # pop_fit = pop_fit[pop_fit[:, 0].argsort()]  # sort by column 0 (fitness)
            sorted_ind = curr_fit.argsort()
            curr_fit, curr_pop_keys, curr_pop_values = curr_fit[sorted_ind], curr_pop_keys[sorted_ind],\
                                                       curr_pop_values[sorted_ind]
            if best is None or curr_fit[-1] > best.fitness:
                best = curr_pop_values[-1]

            # Save any better individuals
            if self.generation <= 1:
                # fitness_library = copy.deepcopy(pop_fit)  # or pop_fit.copy() ?
                best_fit = curr_fit
                best_pop_keys = curr_pop_keys
                best_pop_values = curr_pop_values
                # fitness_library = {p: f for p, f in zip(curr_pop, curr_fit)}
            else:
                combined_fit, combined_pop_keys, combined_pop_values, indices_fit = None, None, None, None
                combined_fit = np.hstack((curr_fit, best_fit))
                combined_fit, indices_fit = np.unique(combined_fit, return_index=True)
                best_fit = combined_fit[-len(self.population):]
                combined_pop_keys = np.hstack((curr_pop_keys, best_pop_keys))
                combined_pop_values = np.hstack((curr_pop_values, best_pop_values))
                best_pop_keys = combined_pop_keys[indices_fit][-len(self.population):]
                best_pop_values = combined_pop_values[indices_fit][-len(self.population):]

            # best = None
            # for g in itervalues(self.population):
            #     if g.fitness is None:
            #         raise RuntimeError('An individual has not been evaluated in neatfast.population.run()')
            #     if best is None or g.fitness > best.fitness:
            #         best = g

            # Report statistics
            # try:
            self.reporters.post_evaluate(self.config, self.population, self.species, curr_pop_values[-1])
            # except KeyError:
            #     print('KeyError found...skipping reporting...')

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > best_fitness:
                best_fitness = best.fitness
                self.best_genome = best
                success = False
                for i in range(5):
                    try:
                        with open('results//winner_tmp', 'wb') as f:
                            pickle.dump(best, f)
                        success = True
                    except PermissionError:
                        time.sleep(0.001)
                if not success:
                    print('winner was not saved' + '*' * 50)

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                    best = None  # necessary to reset for next generation
                else:
                    print('Complete Extinction has occured')
                    break
                    # raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)

            ### Paul's custom code
            # Create plots occasionally during the run - assume StatisticsReporter is first/only reporter
            if not int(self.generation % self.config.fitness_plot_frequency):
                visualize.plot_stats(self.reporters.reporters[0], ylog=True,
                                     filename="results//tmp_fitness_history.svg")
                visualize.plot_species(self.reporters.reporters[0],
                                       filename="results//tmp_speciation_history.svg")

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        best_pop = {k: v for k, v in zip(best_pop_keys, best_pop_values)}
        return self.best_genome, best_pop

from trajectory import Trajectory
import neatfast as neat
import os
import pickle
import visualize
from builder import make_neat_network_diagram, make_last_traj
from traj_config import max_generations


def run_eval():
    t = Trajectory()
    f = t.evaluate()
    print(f)
    print(t.unscaled_states[-1, -1])


def eval_func(genome, config):
    traj = Trajectory(skip_controller=True)
    return traj.evaluate(genome=genome, config=config)


def train():
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
    # te = neat.ThreadedEvaluator(num_workers, eval_func)
    pe = neat.ParallelEvaluator(num_workers, eval_func)
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
    run_eval()
    # train()


    # DONE figure out why fitness from makeLastTraj doesn't match that from traj_runner
    # ---- DONE> they do not start with same ICs - need to link compute_bcs() and make_new_bcs()
    # ---- ---- DONE> there was also an issue with the last segment of the main leg

    # TODO implement/fix post-capture maneuver
    # ---- DONE> get framework set up in Trajectory
    # ---- TASK> fix ou.min_dv_capture
    # ---- ---- DONE> first impulse
    # ---- ---- TASK> second impulse
    # ---- TASK> fix ou.lambert_min_dv

    # DONE verify that traj_runner and neat_main provide the same output/do the same thing
    # ---- DONE> issue with NEAT breaking during runtime
    # ---- DONE> issue with different numbers of nodes

    # TODO do actual timing test between traj_runner and neat_main
    # ---- if within ~10-20%, go with traj_runner because code quality is better and because pride

    # TODO make plots for 3D - trajectory and thrust history

    # TODO make sure traj_runner still works for 2D cases

    # TODO do a full test case for 2D and a full test case for 3D
    # ---- at this point code will be fully verified and I will be caught up to where I was; now, move forward

    # TODO look at NEAT optimization - max vs average fitness of species with one or multiple species

    # TODO look at NEAT options - what are other styles/types installed like CRNN or IZNN or whatever

    # TODO take another stab at hyperparamter tuning
    # ---- at this point code will be tuned

    # TODO enable variable time of flight (VTOF)
    # ---- once VTOF is enabled, can do RBDO and RDO

    # TODO do test each with RBDO and RDO

    # TODO improve statistical analysis - what metrics should I use to present results?

    # TODO make sure I have made all improvements/suggestions from prelims

    # TODO repeat everything above for Venus - 2D and 3D with RBDO and RDO
    # ---- enough for journal paper?

    # TODO ...oh buddy... go back and fix vanilla GA

    # TODO do comparison between NEAT and GA

    # TODO ...wtf... learn reinforcement learning

    # TODO do comparison between RL and NEAT
    # ---- enough for conference/journal paper?
    # ---- enough for thesis?

    # TODO talk with advisory committee - get feedback on direction to move forward

    # TODO ???

    # TODO profit

from trajectory import Trajectory
import neatfast as neat
import os
import pickle

# fname = 'winner-feedforward'
# with open(fname, 'rb') as f:
#     genome = pickle.load(f)
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
pop = neat.Population(config)
genome = pop.population[1]
t = Trajectory(genome=genome)
f = t.evaluate()
print(f)

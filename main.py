from GeneticEncoding import *
from GeneticAlgorithm import *
from collections import deque
import gym

env_name = "CartPole-v1"


env = gym.make(env_name)
no_of_inputs = env.observation_space.shape[0]
no_of_outputs = env.action_space.n
env.close()

print("no_of_inputs: " + str(no_of_inputs))
print("no_of_outputs: " + str(no_of_outputs))

num_population = 10
num_generations = 50
mutation_rate = 0.1
crossover_rate = 0.8
R = (40 * num_population)//100
no_of_rollouts = 20

algo = NEATAlgorithm(num_population, num_generations, mutation_rate, crossover_rate, R,  no_of_inputs, no_of_outputs, env_name, no_of_rollouts)
best_genome = algo.run()

# render the best performing genome
print("\n")
print("\n")
print("rendering the best performing genome")

fitness = best_genome.render(10)

# print the fitness of the best genome
print("fitness of the best performing genome: " + str(fitness))
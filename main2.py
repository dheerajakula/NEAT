from GeneticEncoding import *
from GeneticAlgorithm import *
from collections import deque
import gym

#CartPole-v0
#MountainCar-v0
#Acrobot-v1
env_name = "CartPole-v0"
# env_name = "MountainCar-v0"
env_name = "Acrobot-v1"

env = gym.make(env_name)
no_of_inputs = env.observation_space.shape[0]
no_of_outputs = env.action_space.n
env.close()

print("no_of_inputs: " + str(no_of_inputs))
print("no_of_outputs: " + str(no_of_outputs))

num_population = 20
num_generations = 10
mutation_rate = 0.1
crossover_rate = 0.8
R = (40 * num_population)//100
no_of_rollouts = 5

algo = NEATAlgorithm(num_population, num_generations, mutation_rate, crossover_rate, R,  no_of_inputs, no_of_outputs, env_name, no_of_rollouts)
best_genome = algo.run()

# render the best performing genome
best_genome.render()
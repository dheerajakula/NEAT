from GeneticEncoding import *
from GeneticAlgorithm import *
from collections import deque
import gym

env_name = "CartPole-v1"
# env_name = "MountainCar-v0"
# env_name = "Acrobot-v1"

env = gym.make(env_name)
no_of_inputs = env.observation_space.shape[0]
no_of_outputs = env.action_space.n
env.close()

print("no_of_inputs: " + str(no_of_inputs))
print("no_of_outputs: " + str(no_of_outputs))

num_population = 20
num_generations = 50
mutation_rate = 0.1
crossover_rate = 0.8
R = (40 * num_population)//100
no_of_rollouts = 20

# parameters for mountain car
# num_population = 100
# num_generations = 10
# mutation_rate = 0.2
# crossover_rate = 0.8
# R = (40 * num_population)//100
# no_of_rollouts = 10

# parameters for acrobat
# num_population = 20
# num_generations = 10
# mutation_rate = 0.1
# crossover_rate = 0.8
# R = (40 * num_population)//100
# no_of_rollouts = 10

algo = NEATAlgorithm(num_population, num_generations, mutation_rate, crossover_rate, R,  no_of_inputs, no_of_outputs, env_name, no_of_rollouts)
best_genome, bestFitnessPerGeneration, worstFitnessPerGeneration, averageFitnessPerGeneration = algo.run()

# render the best performing genome
print("\n")
print("\n")
print("rendering the best performing genome")

fitness = best_genome.render(10)

# print the fitness of the best genome
print("fitness of the best performing genome: " + str(fitness))

import matplotlib.pyplot as plt
plt.plot(bestFitnessPerGeneration, 'b')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.show()

plt.plot(worstFitnessPerGeneration, 'r')
plt.xlabel('Generation')
plt.ylabel('Worst Fitness')
plt.show()

plt.plot(averageFitnessPerGeneration, 'g')
plt.xlabel('Generation')
plt.ylabel('Average Fitness')
plt.show()

# combined plot
plt.plot(bestFitnessPerGeneration, 'b')
plt.plot(worstFitnessPerGeneration, 'r')
plt.plot(averageFitnessPerGeneration, 'g')
plt.xlabel('Generation')
plt.ylabel('combined plot')
plt.show()
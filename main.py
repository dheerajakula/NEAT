from GeneticEncoding import *
import gym

no_of_inputs = 4
no_of_outputs = 1
member_of_population = Genome(no_of_inputs, no_of_outputs)

member_of_population.evaluate()

member_of_population.getFitness()
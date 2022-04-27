from GeneticEncoding import *
from collections import deque

class NEATAlgorithm:

    def __init__(self, num_population, num_generations, mutation_rate, crossover_rate, R, no_of_inputs, no_of_outputs, env_name, no_of_rollouts):
        self.num_population = num_population
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.R = R
        self.no_of_inputs = no_of_inputs
        self.no_of_outputs = no_of_outputs
        self.env_name = env_name
        self.no_of_rollouts = no_of_rollouts

        self.innovation_number = no_of_inputs * no_of_outputs + 1
        self.population = []

        self.bestFitnessPerGeneration = []
        self.worstFitnessPerGeneration = []
        self.averageFitnessPerGeneration = []


    # run the NEAT algorithm
    def run(self):
        
        # create the initial population
        for i in range(self.num_population):
            genome = Genome(self, self.no_of_inputs, self.no_of_outputs, self.env_name, self.no_of_rollouts)
            self.population.append(genome)

        elite = None
        # run the algorithm
        for i in range(self.num_generations):
            print("\n")
            print("\n")
            print("******************************************************************************************")
            # print the generation number
            print("Generation: " + str(i))
            print("******************************************************************************************")

            
            # evaluate each genome
            for genome in self.population:
          
                genome.evaluate()
                

            # sort the population by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            if elite is None:
                elite = self.population[0]
            else:
                if elite.fitness < self.population[0].fitness:
                    elite = self.population[0]

            # render the best performing genome
            self.population[0].render(1)
            

            # print the best performing genome
            print("Generation: " + str(i) + " Best fitness: " + str(self.population[0].fitness))

            # print the best performing genome length and the worst performing genome length
            print("Best genome length: " + str(len(self.population[0].genes)))
            print("Worst genome length: " + str(len(self.population[-1].genes)))

            # # print the best performing genome
            # print(self.population[0])
            # get fitness of the population
            fitness = [x.fitness for x in self.population]

            # print average fitness
            print("Average fitness: " + str(sum(fitness)/len(fitness)))

            self.bestFitnessPerGeneration.append(fitness[0])
            self.worstFitnessPerGeneration.append(fitness[-1])
            self.averageFitnessPerGeneration.append(sum(fitness)/len(fitness))

            # select first R parents
            R_parents = self.population[:self.R]
            
            # create the next generation
            next_generation = []

            for parent in R_parents:
                next_generation.append(parent)

            while len(next_generation) < self.num_population:
                parent1 = random.choice(R_parents)
                parent2 = random.choice(R_parents)
                child = parent1.crossover(parent2, self.crossover_rate)
                child.mutate(self.mutation_rate)
                next_generation.append(child)

            self.population = next_generation

        # return random elite
        return elite, self.bestFitnessPerGeneration, self.worstFitnessPerGeneration, self.averageFitnessPerGeneration
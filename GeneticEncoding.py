import random
import gym
import numpy as np
import copy

from NeuralNetwork import *
        
# Each Gene contains In node, Out node, Weight, isEnabled, and an innovation number
class Gene:
    def __init__(self, in_node_id, out_node_id, weight, innovation_number):
        self.in_node_id = in_node_id
        self.out_node_id = out_node_id
        self.weight = weight
        self.innovation_number = innovation_number
        self.isEnabled = True

    def disable(self):
        self.isEnabled = False

    def __str__(self):
        return str(self.in_node_id) + " -> " + str(self.out_node_id) + " : " + str(self.weight) + " : " + str(self.isEnabled) + " : " + str(self.innovation_number)

# genome in NEAT algorithm is a collection of genes
class Genome:
    
    def __init__(self, NEATAlgorithm, no_of_inputs, no_of_outputs, env_name, no_of_rollouts):
        
        self.NEATAlgorithm = NEATAlgorithm
        self.no_of_inputs = no_of_inputs
        self.no_of_outputs = no_of_outputs
        self.env_name = env_name
        self.no_of_rollouts = no_of_rollouts

        self.genes = []
        self.fitness = 0
        self.node_id = 0
        
        # now connect every input node to every output node
        count = 1

        for in_node_id in range(1, no_of_inputs + 1):
            for out_node_id in range(no_of_inputs + 1, no_of_inputs + no_of_outputs + 1):
                weight = random.uniform(-1, 1)
                gene = Gene(in_node_id, out_node_id, weight, count)
                self.genes.append(gene)
                count += 1
        self.node_id = no_of_inputs + no_of_outputs
    
    def mutate_weights(self, mutation_rate):
        for gene in self.genes:
            if random.random() < mutation_rate:
                # add little random noise to the weight
                gene.weight += random.uniform(-0.1, 0.1)
                # clip the weight
                if gene.weight > 1:
                    gene.weight = 1
                elif gene.weight < -1:
                    gene.weight = -1

    def mutate_add_connection(self, mutation_rate):
        pass

    def mutate_add_node(self, mutation_rate):
        for i in range(len(self.genes)):

            gene = self.genes[i]
            if random.random() < mutation_rate:
                # disable the previous gene
                gene.disable()

                self.node_id = max(self.node_id, gene.out_node_id, gene.in_node_id)
                
                # increment the node id
                self.node_id += 1

                # create two new genes and add them to the genome
                new_gene1 = Gene(gene.in_node_id, self.node_id, 1, self.NEATAlgorithm.innovation_number)
                self.NEATAlgorithm.innovation_number += 1
                self.genes.append(new_gene1)

                new_gene2 = Gene(self.node_id, gene.out_node_id, gene.weight, self.NEATAlgorithm.innovation_number)
                self.NEATAlgorithm.innovation_number += 1
                self.genes.append(new_gene2)
                
                

                

    def mutate(self, mutation_rate):
        mutation_rate = mutation_rate/len(self.genes)

        self.mutate_weights(mutation_rate)

        self.mutate_add_connection(mutation_rate)

        self.mutate_add_node(mutation_rate/self.no_of_outputs)

    def crossover(self, parent2, crossover_rate):
        
        parent1 = self
        parent2 = parent2

        if(random.random() < crossover_rate):
            # get all the possible innovation numbers
            map_innovation_number_to_gene = {}

            # map innovation number to gene
            for gene in parent1.genes:
                map_innovation_number_to_gene[gene.innovation_number] = gene

            for gene in parent2.genes:
                if gene.innovation_number not in map_innovation_number_to_gene:
                    map_innovation_number_to_gene[gene.innovation_number] = gene
                else:
                    if parent2.fitness > parent1.fitness:
                        map_innovation_number_to_gene[gene.innovation_number] = gene

            # copy the genes from the map
            child_genes = []
            for key in map_innovation_number_to_gene:
                gene = copy.copy(map_innovation_number_to_gene[key])
                child_genes.append(gene)

            # create a new genome
            child = Genome(self.NEATAlgorithm, self.no_of_inputs, self.no_of_outputs, self.env_name, self.no_of_rollouts)

            child.genes = child_genes
            
            return child
        else:
            return self
            

    def construct_network(self):

        network = NeuralNetwork(self.no_of_inputs, self.no_of_outputs)

        # remove genes that form loops
        for i in range(len(self.genes)):
            for j in range(len(self.genes)):
                if i != j:
                    incoming_node_id_a = self.genes[i].in_node_id
                    incoming_node_id_b = self.genes[j].in_node_id
                    outgoing_node_id_a = self.genes[i].out_node_id
                    outgoing_node_id_b = self.genes[j].out_node_id
                    if incoming_node_id_a == outgoing_node_id_b and incoming_node_id_b == outgoing_node_id_a:
                        self.genes[i].disable()
                        
        for gene in self.genes:
            if gene.isEnabled:
                network.add_node(gene.in_node_id)
                network.add_node(gene.out_node_id)
                network.add_connection(gene.in_node_id, gene.out_node_id, gene.weight)
        
        return network
            
    def render(self, episodes):
        network = self.construct_network()
        env = gym.make(self.env_name)
        total_reward = 0

        for i_episode in range(episodes):
            observation = env.reset()
            while(1):
                env.render()
                action = network.compute_output(observation)
                # find the max action from the action space
                max_val = max(action)
                max_index = np.argmax(action)
                action = max_index
            
                observation, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
        env.close()
        return total_reward/episodes

    def evaluate(self):
        network = self.construct_network()
        env = gym.make(self.env_name)
        total_reward = 0
        
        for i_episode in range(self.no_of_rollouts):
            observation = env.reset()
            while(1):
                action = network.compute_output(observation)

                # find the max action from the action space
                max_val = max(action)
                max_index = np.argmax(action)
                action = max_index
            
                observation, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
        env.close()
        self.fitness = total_reward/self.no_of_rollouts

    def __str__(self):
        # print each gene in genome pretty
        print("length of genome: ", len(self.genes))
        s = "**********************************************************\n"
        for gene in self.genes:
            s += str(gene) + "\n"
        s += "**********************************************************\n"
        return s

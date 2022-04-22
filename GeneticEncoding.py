import random
import gym
import math

# Each Node has a innovation number incoming nodes and outgoing nodes
class Node:
    def __init__(self):
        self.incoming_nodes = {}
        self.outgoing_nodes = {}
        self.val = 0
        self.computed = False
    
    def add_incoming_node(self, node, weight):
        self.incoming_nodes[node] = weight
    
    def add_outgoing_node(self, node, weight):
        self.outgoing_nodes[node] = weight

# define the neural network class
class NeuralNetwork:
    def __init__(self, no_of_inputs, no_of_outputs):
        self.input_nodes = []
        self.output_nodes = []
        self.hidden_nodes = []

         # first create the input nodes
        for i in range(no_of_inputs):
            node = Node()
            self.input_nodes.append(node)
        
        # then create the output nodes
        for i in range(no_of_outputs):
            node = Node()
            self.output_nodes.append(node)

    # compute the output of the neural network given the input
    def compute_output(self, input_values):
        # set the input values
        for i in range(len(input_values)):
            self.input_nodes[i].val = input_values[i]
            self.input_nodes[i].computed = True

        # compute the output values for the output nodes
        for node in self.output_nodes:
            self.recurse_nodes(node)
        
        # return the output values
        output_values = []
        for node in self.output_nodes:
            output_values.append(node.val)
        
        # reset the computed values
        for node in self.input_nodes:
            node.computed = False
            node.val = 0
        for node in self.hidden_nodes:
            node.computed = False
            node.val = 0
        for node in self.output_nodes:
            node.computed = False
            node.val = 0

        return output_values
        
    # backtrack to get the value of output nodes
    def recurse_nodes(self, node):
        if node.computed:
            return node.val
        else:
            for n in node.incoming_nodes:
                if n.computed:
                    node.val += n.val * node.incoming_nodes[n]
                else:
                    node.val += self.recurse_nodes(n) * node.incoming_nodes[n]

            node.computed = True

            # if output node, apply sigmoid activation function
            if node in self.output_nodes:
                node.val = 1 / (1 + math.exp(-node.val))

            # apply relu activation function
            elif node.val < 0:
                node.val = 0

            return node.val
        
# Each Gene contains In node, Out node, Weight, isEnabled, and an innovation number
class Gene:
    
    def __init__(self, in_node, out_node, weight, innovation_number):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.innovation_number = innovation_number
        self.isEnabled = True

# genome in NEAT algorithm is a collection of genes
class Genome:
    
    def __init__(self, no_of_inputs, no_of_outputs):
        self.innovation_number = 1
        self.network = NeuralNetwork(no_of_inputs, no_of_outputs)
        self.genes = []
        self.env = gym.make('CartPole-v0')
        # now connect every input node to every output node
        for in_node in self.network.input_nodes:
            for out_node in self.network.output_nodes:
                weight = random.uniform(-1, 1)
                gene = Gene(in_node, out_node, weight, self.innovation_number)
                self.genes.append(gene)
                self.innovation_number += 1
                in_node.add_outgoing_node(out_node, weight)
                out_node.add_incoming_node(in_node, weight)
    
    def mutate_add_connection(self):
        
    
    def getFitness(self):
        total_reward = 0
        observation = self.env.reset()
        while(1):
            action = self.network.compute_output(observation)
            # map sigmoid output to action
            if action[0] > 0.5:
                action = 1
            else:
                action = 0
           
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        self.env.close()
        return total_reward

    def evaluate(self):
        for i_episode in range(10):
            total_reward = 0
            observation = self.env.reset()
            while(1):
                self.env.render()
                action = self.network.compute_output(observation)
                # map sigmoid output to action
                if action[0] > 0.5:
                    action = 1
                else:
                    action = 0
                observation, reward, done, info = self.env.step(action)
                total_reward += reward

                if done:
                    print("done")
                    break

            print(total_reward)

        self.env.close()
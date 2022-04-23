import numpy as np
# Each Node has a innovation number incoming nodes and outgoing nodes
class Node:
    def __init__(self, node_identifier):
        self.incoming_nodes = {}
        self.outgoing_nodes = {}
        self.val = 0
        self.computed = False
        self.node_identifier = node_identifier

    def add_incoming_connection(self, node, weight):
        self.incoming_nodes[node] = weight
    
    def add_outgoing_connection(self, node, weight):
        self.outgoing_nodes[node] = weight

    def remove_incoming_node(self, node):
        self.incoming_nodes.pop(node)
    
    def remove_outgoing_node(self, node):
        self.outgoing_nodes.pop(node)
        
    def get_connection_weight(self, node):
        if node in self.incoming_nodes:
            return self.incoming_nodes[node]
        elif node in self.outgoing_nodes:
            return self.outgoing_nodes[node]
        else:
            print("Please check your code")

    def __str__(self):
        return str(self.node_identifier)
    
        
# define the neural network class
class NeuralNetwork:
    def __init__(self, no_of_inputs, no_of_outputs):
        self.no_of_inputs = no_of_inputs
        self.no_of_outputs = no_of_outputs
        self.nodes = {}

    def add_connection(self, in_node_id, out_node_id, weight):
        in_node = self.nodes[in_node_id]
        out_node = self.nodes[out_node_id]

        in_node.add_outgoing_connection(out_node, weight)
        out_node.add_incoming_connection(in_node, weight)

    def add_node(self, node_id):
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id)
        return self.nodes[node_id]

    def get_input_nodes(self):
        input_nodes = []
        for i in range(1, self.no_of_inputs + 1):
            input_nodes.append(self.nodes[i])
        return input_nodes
    
    def get_output_nodes(self):
        output_nodes = []
        for i in range(self.no_of_inputs + 1, self.no_of_inputs + self.no_of_outputs + 1):
            output_nodes.append(self.nodes[i])
        return output_nodes

    # compute the output of the neural network given the input
    def compute_output(self, input_values):
        # set the input values
        input_nodes = self.get_input_nodes()
        output_nodes = self.get_output_nodes()

        for i in range(len(input_values)):
            input_nodes[i].val = input_values[i]
            input_nodes[i].computed = True

        # compute the output values for the output nodes
        for node in output_nodes:
            self.recurse_nodes(node)
        
        # return the output values
        output_values = []
        for node in output_nodes:
            output_values.append(node.val)
        
        # reset the computed values
        for node in self.nodes.values():
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
                    node.val += n.val * node.get_connection_weight(n)
                else:
                    node.val += self.recurse_nodes(n) * node.get_connection_weight(n)

            node.computed = True

            if node in self.get_output_nodes():
                node.val = 1 / (1 + np.exp(-node.val))
                return node.val
            # apply relu activation function
            elif node.val < 0:
                node.val = 0

            return node.val
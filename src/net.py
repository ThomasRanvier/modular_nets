import numpy as np
from src.initialisers.he import He

class Net(object):
    """
    Class that implements a modular neural network.

    You give it a list of layers that are all layers Objects initialised as you 
    want.
    You also have to give the network input and output sizes.
    You can set the initialiser with an initialiser object, the default one is
    He.
    """
    def __init__(self, layers, input_size, output_size, initialiser = He()
            , reg = 0.0):
        """
        Instantiates a neural network with the layers and initialiser that you 
        want.
        :param layers: A list containing all the layers that you want in the 
        network.
        :type layers: A list of layers objects.
        :param input_size: The size of the input of the network.
        :type input_size: integer.
        :param output_size: The size of the output of the network.
        :type output_size: integer.
        :param initialiser: The initialiser for the weights, the default one is 
        He.
        :type initialiser: initialiser Object.
        """
        self.layers_sizes = [input_size]
        for layer in layers:
            if layer.layer_mode == 'connected':
                self.layers_sizes.append(layer.size)
        self.layers_sizes.append(output_size)
        #Initialise the weights and biases
        self.weights, self. biases = initialiser.initialise(self.layers_sizes)
        self.reg = reg

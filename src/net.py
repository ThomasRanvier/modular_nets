import numpy as np
from src.initialiser import Initialiser

class Net(object):
    """
    Class that implements a modular neural network.

    You give it a list of layers that are all layers Objects initialised as you 
    want.
    You also have to give the network input and output sizes.
    You can set the initialiser with an initialiser object, the default one is
    He.
    """
    def __init__(self, layers, input_size, output_size, 
            initialiser = Initialiser(), reg = 0.0):
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
        self.layers = layers
        self.layers_sizes = [input_size]
        for layer in layers:
            if layer.layer_type == 'connected':
                self.layers_sizes.append(layer.size)
        self.layers_sizes.append(output_size)
        #Initialise the weights and biases.
        initialiser.initialise(self.layers, self.layers_sizes)
        self.reg = reg

    def loss(self, X, y = None):
        """
        Computes the loss and gradients of a minibatch.
        :param X: The input datas.
        :type X: A numpy array of shape (N, d_1, ..., d_k).
        :param y: The labels.
        :type y: A numpy array of shape (N,)
        """
        mode = 'test' if y == None else 'train'
        out = X
        #For each layer call forward.
        for layer in self.layers:
            if layer.layer_type == 'connected' or layer.layer_type == 'activation':
                out = layer.forward(out)
            elif layer.layer_type == 'normalisation':
                out = layer.forward(out, mode)
        scores = out
        if mode == 'test':
            return scores
        #Compute loss.
        #For each layer call backward.

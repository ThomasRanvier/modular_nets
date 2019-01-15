import numpy as np

class He():
    """
    Class that implements a He initialiser.
    """
    def __init__(self, seed = None):
        """
        Instantiates a He initialiser.
        :param seed: The seed to initialise the numpy random with.
        :type seed: integer.
        """
        self.seed = seed

    def initialise(self, layers_sizes):
        """
        Performs an initialisation of the weights and biases for the network.
        :param layers_sizes: The sizes of the layers, input and output layers 
                            included.
        :type layers_sizes: A list of integers.
        """
        np.random.seed(self.seed)
        weights = []
        biases = []
        for i in range(len(layers_sizes) - 1):
            #Define the 2 dimensions of the weights.
            d_1 = layers_sizes[i]
            d_2 = layers_sizes[i + 1]
            #Set the weights.
            weights.append(np.random.randn(d_1, d_2) / np.sqrt(2.0 / d_1))
            #Set the biases at a numpy array of zeros.
            biases.append(np.zeros(d_2))
        return (weights, biases)

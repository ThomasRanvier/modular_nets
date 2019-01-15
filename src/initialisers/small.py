import numpy as np

class Small():
    """
    Class that implements an initialiser that sets the weights to small values.
    """
    def __init__(self, seed = None, alpha = 0.01):
        """
        Instantiates a Small initialiser.
        :param seed: The seed to initialise the numpy random with.
        :type seed: integer.
        :param alpha: The scale to apply to the weights.
        :type alpha: float.
        """
        self.seed = seed
        self.alpha = alpha

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
            weights.append(np.random.randn(d_1, d_2) * self.alpha)
            #Set the biases at a numpy array of zeros.
            biases.append(np.zeros(d_2))
        return (weights, biases)

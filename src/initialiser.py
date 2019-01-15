import numpy as np

class Initialiser(object):
    """
    Class that implements an initialiser.
    Used to initialise the weights of a neural network.
    The biases are initialised at zeros.

    The possible functions that can be used are (default: he_normal):
    - he_normal
    - random_normal
    - normal        config key to add: 'std_dev', default is 1e-2
    - small         config key to add: 'alpha', default is 0.01
    - xavier
    """
    def __init__(self, seed = None, config = {'method': 'he_normal'}):
        """
        Instantiates an initialiser.
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
            #Get the initialisation method to use.
            if not hasattr(self, config['method']):
                raise ValueError('Invalid initialisation method: ' + \
                        config[method])
            initialisation_method = getattr(self, config['method'])
            #Set the weights.
            weights.append(initialisation_method(d_1, d_2, config))
            #Set the biases at a numpy array of zeros.
            biases.append(np.zeros(d_2))
        return (weights, biases)

    def he_normal(self, d_1, d_2, config):
        """
        He normal initialisation.
        :param d_1: Dimension of the previous layer.
        :type d_1: integer.
        :param d_2: Dimension of the next layer.
        :type d_2: integer.
        :param config: Dictionary containing keys to use for some methods.
        :type config: dictionary.
        """
        return np.random.randn(d_1, d_2) / np.sqrt(2.0 / d_1)

    def normal(self, d_1, d_2, config):
        """
        Normal initialisation.
        :param d_1: Dimension of the previous layer.
        :type d_1: integer.
        :param d_2: Dimension of the next layer.
        :type d_2: integer.
        :param config: Dictionary containing keys to use for some methods.
        :type config: dictionary.
        """
        std_dev = config.get('std_dev', 1e-2)
        return np.random.normal(0, self.std_dev, size=(d_1, d_2))

    def random_normal(self, d_1, d_2, config):
        """
        Random normal initialisation.
        :param d_1: Dimension of the previous layer.
        :type d_1: integer.
        :param d_2: Dimension of the next layer.
        :type d_2: integer.
        :param config: Dictionary containing keys to use for some methods.
        :type config: dictionary.
        """
        return np.random.randn(d_1, d_2)

    def small(self, d_1, d_2, config):
        """
        Small initialisation.
        :param d_1: Dimension of the previous layer.
        :type d_1: integer.
        :param d_2: Dimension of the next layer.
        :type d_2: integer.
        :param config: Dictionary containing keys to use for some methods.
        :type config: dictionary.
        """
        alpha = config.get('alpha', 0.01)
        return np.random.randn(d_1, d_2) * alpha

    def xavier(self, d_1, d_2, config):
        """
        Xavier initialisation.
        :param d_1: Dimension of the previous layer.
        :type d_1: integer.
        :param d_2: Dimension of the next layer.
        :type d_2: integer.
        :param config: Dictionary containing keys to use for some methods.
        :type config: dictionary.
        """
        return np.random.randn(d_1, d_2) / np.sqrt(d_1)

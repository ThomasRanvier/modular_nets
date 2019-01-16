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
        self.config = config

    def initialise(self, layers, layers_sizes):
        """
        Performs an initialisation of the weights and biases for the network.
        :param layers: The layers.
        :type layers: A list of layer Objects.
        :param layers_sizes: The layers sizes.
        :type layers_sizes: A list of integers.
        """
        np.random.seed(self.seed)
        index = 0
        for layer in layers:
            if layer.layer_type == 'connected':
                #Define the 2 dimensions of the weights.
                d_1 = layers_sizes[index]
                d_2 = layers_sizes[index + 1]
                #Get the initialisation method to use.
                if not hasattr(self, self.config['method']):
                    raise ValueError('Invalid initialisation method: ' + \
                            self.config[method])
                initialisation_method = getattr(self, self.config['method'])
                #Set the weights.
                layer.weights = []
                layer.weights.append(initialisation_method(d_1, d_2))
                #Set the biases at a numpy array of zeros.
                layer.biases = []
                layer.biases.append(np.zeros(d_2))
                index += 1

    def he_normal(self, d_1, d_2):
        """
        He normal initialisation.
        :param d_1: Dimension of the previous layer.
        :type d_1: integer.
        :param d_2: Dimension of the next layer.
        :type d_2: integer.
        """
        return np.random.randn(d_1, d_2) / np.sqrt(2.0 / d_1)

    def normal(self, d_1, d_2):
        """
        Normal initialisation.
        :param d_1: Dimension of the previous layer.
        :type d_1: integer.
        :param d_2: Dimension of the next layer.
        :type d_2: integer.
        """
        std_dev = self.config.get('std_dev', 1e-2)
        return np.random.normal(0, std_dev, size=(d_1, d_2))

    def random_normal(self, d_1, d_2):
        """
        Random normal initialisation.
        :param d_1: Dimension of the previous layer.
        :type d_1: integer.
        :param d_2: Dimension of the next layer.
        :type d_2: integer.
        """
        return np.random.randn(d_1, d_2)

    def small(self, d_1, d_2):
        """
        Small initialisation.
        :param d_1: Dimension of the previous layer.
        :type d_1: integer.
        :param d_2: Dimension of the next layer.
        :type d_2: integer.
        """
        alpha = self.config.get('alpha', 0.01)
        return np.random.randn(d_1, d_2) * alpha

    def xavier(self, d_1, d_2):
        """
        Xavier initialisation.
        :param d_1: Dimension of the previous layer.
        :type d_1: integer.
        :param d_2: Dimension of the next layer.
        :type d_2: integer.
        """
        return np.random.randn(d_1, d_2) / np.sqrt(d_1)

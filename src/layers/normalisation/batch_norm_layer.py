import numpy as np

class Batch_norm_layer():
    """
    Class that implements a batch normalisation layer.

    A recently developed technique by Ioffe and Szegedy called Batch 
    Normalization alleviates a lot of headaches with properly initializing 
    neural networks by explicitly forcing the activations throughout a network 
    to take on a unit gaussian distribution at the beginning of the training. 
    The core observation is that this is possible because normalization is a 
    simple differentiable operation. 
    In the implementation, applying this technique usually amounts to insert 
    the BatchNorm layer immediately after fully connected layers or 
    convolutional layers, and before non-linearities.

    Link to the course: http://cs231n.github.io/neural-networks-2/#batchnorm
    """
    def __init__(self, epsilon = 1e-5, momentum = 0.9):
        """
        Instantiates a batch normalisation layer.
        :param epsilon: The epsilon parameter.
        :type epsilon: float.
        :param momentum: The momentum parameter.
        :type momentum: float.
        """
        self.cache = None
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None
        self.epsilon = epsilon
        self.momentum = momentum
        self.layer_type = 'normalisation'

    def forward(self, x, mode):
        """
        Performs a forward pass in the batch normalisation layer.
        :param x: The datas to normalise.
        :type x: A numpy array.
        :param mode: Either 'train' or 'test' depending on what you are doing.
        :type mode: str.
        :return out: The normalised datas.
        :rtype out: A numpy array of the same dimensions as x.
        """
        N, D = x.shape
        #Initialise the variables if they are not already.
        if self.gamma is None:
            self.gamma = np.ones(D)
            self.beta = np.zeros(D)
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
        out = None
        self.cache = None
        if mode == 'train':
            mu = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            x_norm = (x - mu) / np.sqrt(var + self.epsilon)
            out = x_norm * self.gamma + self.beta
            mom = self.momentum
            self.running_mean = mom * self.running_mean + (1.0 - mom) * mu
            self.running_var = mom * self.running_var + (1.0 - mom) * var
            self.cache = (var, x_norm)
        elif mode == 'test':
            eps = self.epsilon
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + eps)
            out = x_norm * self.gamma + self.beta
        else:
            raise ValueError('Invalid forward batch norm mode "%s"' % mode)
        return out

    def backward(self, dout):
        """
        Performs a backward pass in the batch normalisation layer.
        This implementation comes from:
        https://kevinzakka.github.io/2016/09/14/batch_normalization/
        :param dout: The upstream gradients.
        :type dout: A numpy array.
        :return dx: The computed gradients.
        :rtype dx: A numpy array of the same dimensions as dout.
        """
        var, x_norm = self.cache
        N, D = dout.shape
        std_inv = 1.0 / np.sqrt(var + self.epsilon)
        dx_hat = dout * self.gamma
        dx = (1.0 / N) * std_inv * (N * dx_hat - np.sum(dx_hat, axis=0) - \
                x_norm * np.sum(dx_hat * x_norm, axis=0))
        self.beta = np.sum(dout, axis=0)
        self.gamma = np.sum(x_norm * dout, axis=0)
        return dx

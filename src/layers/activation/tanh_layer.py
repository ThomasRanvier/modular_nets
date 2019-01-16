import numpy as np

class Tanh_layer():
    """
    Class that implement a Tanh layer, it uses the tanh function.

    It is in practice not advised to use tanh since it is basically a scaled 
    Sigmoid function.
    However, unlike the Sigmoid this function is zero centered which is 
    preferable.
    """
    def __init__(self):
        """
        Instantiates a Tanh layer.
        """
        self.layer_type = 'activation'

    def forward(self, x):
        """
        Computes a forward pass by applying the tanh function to the input.
        :param x: The input data.
        :type x: A numpy array.
        :return out: The computed output of the layer.
        :rtype out: A numpy array of the same shape as x.
        """
        #Apply tanh function.
        out = np.tanh(x)
        return out

    def backward(self, dout):
        """
        Computes a backward pass by applying the derivative of the tanh 
        function to the upstream gradient.
        :param dout: Upstream derivatives.
        :type dout: A numpy array.
        :return dx: Gradient with respect of x.
        :rtype dx: A numpy array of same shape as dout.
        """
        #Apply the derivative of the tanh function.
        dx = 1.0 - np.tanh(dout)**2
        return dx

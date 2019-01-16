import numpy as np

class Leaky_relu_layer():
    """
    This class implements a Leaky ReLU layer, it uses a modified version 
    of the Rectified Linear Unit function.

    Should be considered as an alternative to the classic Relu layer, it can 
    eventually give better results but not necessarily.
    """
    def __init__(self, alpha = 0.01):
        """
        Instantiates a Leaky ReLU layer.
        """
        self.alpha = alpha
        self.layer_type = 'activation'
        
    def forward(self, x):
        """
        Computes a forward pass by applying the Leaky ReLU function to the 
        input.
        :param x: The input data.
        :type x: A numpy array.
        :return out: The computed output of the layer.
        :rtype out: A numpy array of the same shape as x.
        """
        #Apply Leaky ReLU function.
        out = np.maximum(self.alpha * x, x)
        return out

    def backward(self, dout):
        """
        Computes a backward pass by applying the derivative of the Leaky ReLU 
        function to the upstream gradient.
        :param dout: Upstream derivatives.
        :type dout: A numpy array.
        :return dx: Gradient with respect of x.
        :rtype dx: A numpy array of same shape as dout.
        """
        #Apply the derivative of the Leaky ReLU function.
        dx = 1.0 * (dout > 0)
        dx[dx == 0] = self.alpha
        return dx

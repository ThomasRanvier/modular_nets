import numpy as np

class Sgd():
    """
    Class that implements a vanilla Stochastic Gradient Descent layer.

    This is the simplest form of update, it changes the parameters along the 
    negative gradient direction.
    """
    def __init__(self, learning_rate = 1e-2):
        """
        Instantiates a Sgd optimisation.
        :param learning_rate: The learning rate to apply.
        :type learning_rate: float.
        """
        self.learning_rate = learning_rate

    def update(self, w, dw):
        """
        Performs one Sgd update.
        :param w: The weights.
        :type w: A numpy array.
        :param dw: The gradients of the weights.
        :type dw: A numpy array of the same shape.
        :return w: The updated weights.
        :rtype w: A numpy array of the same shape as w.
        """
        w += -self.learning_rate * dw
        return w

import numpy as np

class Nesterov_momentum_sgd():
    """
    Class that implements a Nesterov momentum Stochastic Gradient Descent layer.

    Nesterov Momentum is a slightly different version of the momentum update 
    that has recently been gaining popularity.
    It enjoys stronger theoretical converge guarantees for convex functions and
    in practice it also consistently works slightly better than standard 
    momentum.

    The core idea behind Nesterov momentum is that when the current parameter 
    vector is at some position x, then looking at the momentum update above, we
    know that the momentum term alone (i.e. ignoring the second term with the 
    gradient) is about to nudge the parameter vector by mu * v.
    Therefore, if we are about to compute the gradient, we can treat the future
    approximate position x + mu * v as a “lookahead” - this is a point in the 
    vicinity of where we are soon going to end up. Hence, it makes sense to 
    compute the gradient at x + mu * v instead of at the “old/stale” position x.

    Link to the course: http://cs231n.github.io/neural-networks-3/#sgd
    """
    def __init__(self, learning_rate = 1e-2, momentum = 0.9):
        """
        Instantiates a Nesterov momentum SGD.
        :param learning_rate: The learning rate to apply.
        :type learning_rate: float.
        :param momentum: The momentum to apply.
        :type momentum: float.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        #Velocity parameter that is updated at each update.
        self.velocity = None

    def update(self, w, dw):
        """
        Performs an udate of Nesterov momentum SGD.
        :param w: The weights.
        :type w: A numpy array.
        :param dw: The gradients of the weights.
        :type dw: A numpy array of the shape as w.
        :return w: The updated weights.
        :rtype w: A numpy array of the same shape as w.
        """
        #If not initialised set the velocity to a numpy array full of zeros of 
        #the same shape as w.
        if self.velocity == None:
            self.velocity = np.zeros_like(w)
        #Backup the velocity.
        v_prev = self.velocity
        #Update the velocity.
        self.velocity = self.momentum * self.velocity - self.learning_rate * dw
        w += -self.momentum * v_prev + (1 + self.momentum) * self.velocity
        return w

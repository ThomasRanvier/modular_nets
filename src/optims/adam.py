import numpy as np

class Adam():
    """
    Class that implements an Adam optimisation.

    It is currently the recommended optimisation methods to update the weights.
    It is often worth trying out the Nesterov momentum SGD as an alternative.
    """
    def __init__(self, learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999, 
            epsilon = 1e-8):
        """
        Instantiates an Adam optimisation.
        :param learning_rate: The learning rate to apply.
        :type learning_rate: float.
        :param beta_1: Beta 1.
        :type beta_1: float.
        :param beta_2: Beta 2.
        :type beta_2: float.
        :param epsilon: The epsilon hyperparameter.
        :type epsilon: float.
        """
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = None

    def update(self, w, dw):
        """
        Performs an update of the Adam optimisation.
        :param w: The weights.
        :type w: A numpy array.
        :param dw: The gradients of the weights.
        :type dw: A numpy array of the same shape as w.
        :return w: The updated weights.
        :rtype w: A numpy array of the same shape as w.
        """
        #If not initialised set the m and v variable to zero arrays, set t to 1.
        if self.m == None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
            self.t = 1
        #Increment t by 1 at each iteration of this update.
        self.t += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * dw
        mt = self.m / (1 - self.beta_1**self.t)
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (dw**2)
        vt = self.v / (1 - self.beta_2**self.t)
        w += -self.learning_rate * mt / (np.sqrt(vt) + self.epsilon)
        return w

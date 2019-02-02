import numpy as np

class Rms_prop():
    """
    Class that implements a RMSProp optimisation.

    RMSprop is a very effective, but currently unpublished adaptive learning 
    rate method.
    Amusingly, everyone who uses this method in their work currently cites slide
    29 of Lecture 6 of Geoff Hinton's Coursera class.
    The RMSProp update adjusts the Adagrad method in a very simple way in an 
    attempt to reduce its aggressive, monotonically decreasing learning rate.

    Link to the course: http://cs231n.github.io/neural-networks-3/#sgd
    """
    def __init__(self, learning_rate = 1e-2, decay_rate = 0.99, epsilon = 1e-8):
        """
        Instantiates a RMSProp optimisation.
        :param learning_rate: The learning rate to apply.
        :type learning_rate: float.
        :param decay_rate: The decay rate.
        :type decay_rate: float.
        :param epsilon: The epsilon hyperparameter.
        :type epsilon: float.
        """
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None

    def update(self, w, dw):
        """
        Performs an update of RMSProp.
        :param w: The weights.
        :type w: A numpy array.
        :param dw: The gradients of the weights.
        :type dw: A numpy array of the same shape as w.
        :return w: The updated weights.
        :rtype w: A numpy array of the same shape as w.
        """
        #If cache is not initialised it is set at a numpy array full of zeros of
        #the same shape as w.
        if self.cache is None:
            self.cache = np.zeros_like(w)
        dr = self.decay_rate
        self.cache = dr * self.cache + (1 - dr) * dw**2
        w += -self.learning_rate * dw / (np.sqrt(self.cache) + self.epsilon)
        return w

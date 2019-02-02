import numpy as np

class Momentum_sgd():
    """
    Class that implements a momentum Stochastic Gradient Descent optimisation.

    Momentum update is another SGD approach that almost always enjoys better 
    converge rates than the vanilla SGD on deep networks.
    This update can be motivated from a physical perspective of the optimization
    problem.
    In particular, the loss can be interpreted as the height of a hilly terrain 
    (and therefore also to the potential energy since U=mgh and therefore Uah).
    Initializing the parameters with random numbers is equivalent to setting a 
    particle with zero initial velocity at some location. 
    The optimization process can then be seen as equivalent to the process of 
    simulating the parameter vector (i.e. a particle) as rolling on the 
    landscape.
    Since the force on the particle is related to the gradient of potential 
    energy (i.e. F=-dU), the force felt by the particle is precisely the
    (negative) gradient of the loss function.
    Moreover, F=ma so the (negative) gradient is in this view proportional to 
    the acceleration of the particle.

    Link to the course: http://cs231n.github.io/neural-networks-3/#sgd
    """
    def __init__(self, learning_rate = 1e-2, momentum = 0.9):
        """
        Instantiates a momentum SGD.
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
        Performs an udate of momentum SGD.
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
        #Update the velocity
        self.velocity = self.velocity * self.momentum - self.learning_rate * dw
        w += self.velocity
        return w

import numpy as np

class Relu_layer():
    """
    Class that implement a ReLU layer, it uses the Rectified Linear Unit 
    function.

    Pros:
        - It was found to greatly accelerate (e.g. a factor of 6 in Krizhevsky 
        et al.) the convergence of stochastic gradient descent compared to the 
        sigmoid/tanh functions.
        - Compared to tanh/sigmoid neurons that involve expensive operations 
        (exponentials, etc.), the ReLU can be implemented by simply thresholding
        a matrix of activations at zero.
    
    Cons:
        - Unfortunately, ReLU units can be fragile during training and can 
        “die”. For example, a large gradient flowing through a ReLU neuron could
        cause the weights to update in such a way that the neuron will never 
        activate on any datapoint again. If this happens, then the gradient 
        flowing through the unit will forever be zero from that point on. 
        That is, the ReLU units can irreversibly die during training since they 
        can get knocked off the data manifold. For example, you may find that as
        much as 40% of your network can be “dead” (i.e. neurons that never 
        activate across the entire training dataset) if the learning rate is set
        too high. With a proper setting of the learning rate this is less 
        frequently an issue.
    
    Link to the course: http://cs231n.github.io/neural-networks-1/
    """
    def __init__(self):
        """
        Instantiates a ReLU layer.
        """
        self.cache = None
        self.layer_mode = 'activation'

    def forward(self, x):
        """
        Computes a forward pass by applying the ReLU function to the input.
        :param x: The input data.
        :type x: A numpy array.
        :return out: The computed output of the layer.
        :rtype out: A numpy array of the same shape as x.
        """
        #Apply ReLU function.
        out = np.maximum(0, x)
        #Save the value in the cache.
        self.cache = x
        return out

    def backward(self, dout):
        """
        Computes a backward pass by applying the derivative of the ReLU function
        to the upstream gradient.
        :param dout: Upstream derivatives.
        :type dout: A numpy array.
        :return dx: Gradient with respect of x.
        :rtype dx: A numpy array of same shape as dout.
        """
        #Extract the value from the cache.
        x = self.cache
        #Apply the derivative of the ReLU function.
        dx = dout
        dx[x <= 0] = 0
        return dx

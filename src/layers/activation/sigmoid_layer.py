import numpy as np

class Sigmoid_layer():
    """
    Class that implement a Sigmoid layer, it uses the Sigmoid function.

    It is in practice not advised to use Sigmoid anymore.

    Cons:
    - Sigmoids saturate and kill gradients. A very undesirable property of the 
    sigmoid neuron is that when the neuron's activation saturates at either
    tail of 0 or 1, the gradient at these regions is almost zero. Recall that 
    during backpropagation, this (local) gradient will be multiplied to the 
    gradient of this gate's output for the whole objective. Therefore, if the 
    local gradient is very small, it will effectively "kill" the gradient and 
    almost no signal will flow through the neuron to its weights and recursively
    to its data. Additionally, one must pay extra caution when initializing the
    weights of sigmoid neurons to prevent saturation. For example, if the 
    initial weights are too large then most neurons would become saturated and 
    the network will barely learn.
    - Sigmoid outputs are not zero-centered. This is undesirable since neurons 
    in later layers of processing in a Neural Network (more on this soon) would 
    be receiving data that is not zero-centered. This has implications on the 
    dynamics during gradient descent, because if the data coming into a neuron 
    is always positive (e.g. x > 0 elementwise in f=w.T * x + b)), then the 
    gradient on the weights w will during backpropagation become either all be 
    positive, or all negative (depending on the gradient of the whole expression
    f). This could introduce undesirable zig-zagging dynamics in the gradient 
    updates for the weights. However, notice that once these gradients are added
    up across a batch of data the final update for the weights can have variable
    signs, somewhat mitigating this issue. Therefore, this is an inconvenience 
    but it has less severe consequences compared to the saturated activation 
    problem above.
    
    Link to the course: http://cs231n.github.io/neural-networks-1/
    """
    def __init__(self):
        """
        Instantiates a Sigmoid layer.
        """
        self.layer_type = 'activation'

    def forward(self, x):
        """
        Computes a forward pass by applying the Sigmoid function to the input.
        :param x: The input data.
        :type x: A numpy array.
        :return out: The computed output of the layer.
        :rtype out: A numpy array of the same shape as x.
        """
        #Apply Sigmoid function.
        out = 1.0 / (1.0 +np.exp(-x))
        return out

    def backward(self, dout):
        """
        Computes a backward pass by applying the derivative of the Sigmoid 
        function to the upstream gradient.
        :param dout: Upstream derivatives.
        :type dout: A numpy array.
        :return dx: Gradient with respect of x.
        :rtype dx: A numpy array of same shape as dout.
        """
        #Apply the derivative of the Sigmoid function.
        dx = (1.0 - dout) * dout
        return dx

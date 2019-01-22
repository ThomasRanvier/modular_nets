import numpy as np

class Affine_layer():
    """
    This class implements an affine layer, also called fully-connected layer.
    """
    def __init__(self, size):
        """
        Instantiates an affine layer.
        :param size: The size of the layer.
        :type size: integer.
        """
        self.cache = None
        self.size = size
        self.layer_type = 'connected'
        self.weights = None
        self.biases = None

    def forward(self, x):
        """
        Computes the forward pass of the affine layer.
        :param x: The N input data.
        :type x: A numpy array of shape (N, d_1, ..., d_k).
        :return out: The computed output.
        :rtype out: A numpy array of shape (N, M).
        """
        #Reshape the input into rows.
        new_x_shape = (x.shape[0], -1)
        reshaped_x = x.reshape(new_x_shape)
        #Compute the forward pass of this layer as a dot product between the 
        #input and the weights, then add the biases.
        out = np.dot(reshaped_x, self.weights) + self.biases
        #Save the values in the cache.
        self.cache = x
        return out

    def backward(self, dout):
        """
        Computes the backward pass of the affine layer.
        :param dout: The upstram derivative.
        :type dout: A numpy array of shape (N, M)
        :return dx: The gradient with respect of x.
        :rtype dx: A numpy array of shape (N, d_1, ..., d_k).
        :return dw: The gradient with respect of w.
        :rtype dw: A numpy array of shape (D, M).
        :return db: The gradient with respect of b.
        :rtype db: A numpy array of shape (M,).
        """
        #Extract the values from the cache.
        x = self.cache
        #Compute the gradient with respect of x 
        #and reshape it to the same dims as x.
        dx = np.dot(dout, self.weights.T).reshape(x.shape)
        #Reshape x into rows.
        new_x_shape = (x.shape[0], -1)
        reshaped_x = x.reshape(new_x_shape)
        #Compute the gradient with respect of the weights.
        dw = np.dot(reshaped_x.T, dout)
        #Compute the gradient with respect of the biases.
        db = np.sum(dout, axis=0)
        return dx, dw, db

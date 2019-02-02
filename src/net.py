import numpy as np
from src.initialiser import Initialiser
from src.loss_computer import Loss_computer
from src.layers.affine_layer import Affine_layer

class Net(object):
    """
    Class that implements a modular neural network.

    You give it a list of layers that are all layers Objects initialised as you 
    want.
    You also have to give the network input and output sizes.
    You can set the initialiser with an initialiser object, the default one is
    He.
    """
    def __init__(self, layers, input_size, output_size, initialiser = Initialiser(),
            loss_computer = Loss_computer(), reg = 0.0):
        """
        Instantiates a neural network with the layers and initialiser that you 
        want.
        :param layers: A list containing all the layers that you want in the 
        network. /!\ It should not include an output layer!
        :type layers: A list of layers objects.
        :param input_size: The size of the input of the network.
        :type input_size: integer.
        :param output_size: The size of the output of the network.
        :type output_size: integer.
        :param initialiser: The initialiser for the weights, the default one is 
        He.
        :type initialiser: initialiser Object.
        """
        self.layers = layers
        self.loss_computer = loss_computer
        #Recuperate the sizes of each connected layer.
        self.layers_sizes = [input_size]
        for layer in layers:
            if layer.layer_type == 'connected':
                self.layers_sizes.append(layer.size)
        self.layers.append(Affine_layer(output_size))
        self.layers_sizes.append(output_size)
        #Initialise the weights and biases.
        initialiser.initialise(self.layers, self.layers_sizes)
        self.reg = reg

    def loss(self, X, y = None):
        """
        Computes the loss and gradients of a minibatch.
        :param X: The input datas.
        :type X: A numpy array of shape (N, d_1, ..., d_k).
        :param y: The labels.
        :type y: A numpy array of shape (N,)
        :return loss: The computed loss.
        :rtype loss: float.
        :return grads: The computed gradients, from the first layer to the last.
        :rtype grads: list.
        """
        mode = 'test' if y is None else 'train'
        out = X
        #For each layer call forward.
        for layer in self.layers:
            if layer.layer_type == 'connected' or layer.layer_type == 'activation':
                out = layer.forward(out)
            elif layer.layer_type == 'normalisation':
                out = layer.forward(out, mode)
        scores = out
        if mode == 'test':
            return scores
        #Compute the loss.
        loss, dx = self.loss_computer.compute_loss(scores, y)
        #For each layer call backward, and recuperate dw and db for connected ones.
        grads = []
        for layer in self.layers[::-1]:
            if layer.layer_type == 'connected':
                #Apply the regularisation to the computed loss for each connected layer.
                loss += 0.5 * self.reg * np.sum(layer.weights**2)
                #Backward pass in the layer
                dx, dw, db = layer.backward(dx)
                #Stock dw and db in grads and apply the regularisation to dw.
                grads.append({'w': dw + (self.reg * layer.weights), 'b': db})
            else:
                dx = layer.backward(dx)
        return loss, grads[::-1]

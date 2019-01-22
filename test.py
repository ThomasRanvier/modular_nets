import numpy as np
from src import *

input_size = 5
output_size = 2

hidden_layers = []

hidden_layers.append(Affine_layer(4))
hidden_layers.append(Batch_norm_layer())
hidden_layers.append(Relu_layer())
hidden_layers.append(Affine_layer(3))
hidden_layers.append(Batch_norm_layer())
hidden_layers.append(Leaky_relu_layer())

initialiser = Initialiser(config = {'method': 'normal', 'std_dev': 5e-2})
net = Net(hidden_layers, input_size, output_size, initialiser = initialiser)

X_train = np.array([[1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6],
                    [3, 4, 5, 6, 7],
                    [4, 5, 6, 7, 8]])
y_train = np.array([0, 1, 0, 1])
X_val = np.array([[1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6],
                    [3, 4, 5, 6, 7],
                    [4, 5, 6, 7, 8]])
y_val = np.array([0, 1, 0, 1]) 

datas = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val}

solver = Solver(net, datas, verbose = True, num_epochs = 50)
solver.train()

from src.layers.affine_layer import Affine_layer
from src.layers.normalisation.batch_norm_layer import Batch_norm_layer
from src.layers.activation.relu_layer import Relu_layer
from src.layers.activation.leaky_relu_layer import Leaky_relu_layer
from src.layers.activation.sigmoid_layer import Sigmoid_layer
from src.layers.activation.tanh_layer import Tanh_layer
from src.net import Net

input_size = 1200
output_size = 10

layers = []

layers.append(Affine_layer(260))
layers.append(Batch_norm_layer())
layers.append(Relu_layer())
layers.append(Affine_layer(220))
layers.append(Batch_norm_layer())
layers.append(Leaky_relu_layer())
layers.append(Affine_layer(180))
layers.append(Batch_norm_layer())
layers.append(Tanh_layer())
layers.append(Affine_layer(140))
layers.append(Batch_norm_layer())
layers.append(Sigmoid_layer())
layers.append(Affine_layer(100))

net = Net(layers, input_size, output_size)

#TODO: Put initialisers in one file and just select what method we want through 
#a parameter

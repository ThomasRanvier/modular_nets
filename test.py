from src import Affine_layer
from src import Batch_norm_layer
from src import Relu_layer
from src import Leaky_relu_layer
from src import Sigmoid_layer
from src import Tanh_layer
from src import Net
from src import Initialiser

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

initialiser = Initialiser(config = {'method': 'normal', 'std_dev': 5e-2})
net = Net(layers, input_size, output_size, initialiser = initialiser)

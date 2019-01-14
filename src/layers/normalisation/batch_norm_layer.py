import numpy as np

class Batch_norm_layer():
    def __init__(self, epsilon = 1e-5, momentum = 0.9):
        self.cache = None
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None
        self.epsilon = epsilon
        self.momentum = momentum
        self.layer_mode = 'normalisation'

    def forward(self, x, mode):
        N, D = x.shape
        if self.gamma == None:
            self.gamma = np.ones(D)
            self.beta = np.zeros(D)
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        out = None
        self.cache = None
        if mode == 'train':
            mu = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            x_norm = (x - mu) / np.sqrt(var + self.epsilon)
            out = x_norm * self.gamma + self.beta
            mom = self.momentum
            self.running_mean = mom * self.running_mean + (1.0 - mom) * mu
            self.running_var = mom * self.running_var + (1.0 - mom) * var
            self.cache = (var, x_norm)
        elif mode == 'test':
            x_norm = (x - self.running_mean) / 
                np.sqrt(self.running_var + self.epsilon)
            out = x_norm * self.gamma + self.beta
        else:
            raise ValueError('Invalid forward batch norm mode "%s"' % mode)

        return out

    def backward(self, dout):
        #https://kevinzakka.github.io/2016/09/14/batch_normalization/
        var, x_norm = self.cache
        N, D = dout.shape

        std_inv = 1.0 / np.sqrt(var + self.epsilon)
        dx_hat = dout * self.gamma
        dx = (1.0 / N) * std_inv * (N * dx_hat - np.sum(dx_hat, axis=0) - x_norm
                * np.sum(dx_hat * x_norm, axis=0))
        self.beta = np.sum(dout, axis=0)
        self.gamma = np.sum(x_norm * dout, axis=0)
        return dx

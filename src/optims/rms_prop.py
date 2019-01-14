import numpy as np

class Rms_prop():
    def __init__(self, learning_rate = 1e-2, decay_rate = 0.99, epsilon = 1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None

    def update(self, w, dw):
        if self.cache == None:
            self.cache = np.zeros_like(w)
        dr = self.decay_rate
        self.cache = dr * self.cache + (1 - dr) * dw**2
        w += -self.learning_rate * dw / (np.sqrt(self.cache) + self.epsilon)
        return w

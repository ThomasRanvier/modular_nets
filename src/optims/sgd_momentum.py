import numpy as np

class Sgd_momentum():
    def __init__(self, learning_rate = 1e-2, momentum = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, w, dw):
        if self.velocity == None:
            self.velocity = np.ones_like(w)
        self.velocity = self.velocity * self.momentum - self.learning_rate * dw
        w += self.velocity
        return w

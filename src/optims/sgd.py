import numpy as np

class Sgd():
    def __init__(self, learning_rate = 1e-2):
        self.learning_rate = learning_rate

    def update(self, w, dw):
        w += -self.learning_rate * dw
        return w

import numpy as np

class Rms_prop():
    def __init__(self, learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999, 
            epsilon = 1e-8):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = None

    def update(self, w, dw):
        if self.m == None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
            self.t = 1
        self.t += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * dw
        mt = self.m / (1 - self.beta_1**self.t)
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (dw**2)
        vt = self.v / (1 - self.beta_2**self.t)
        w += -self.learning_rate * mt / (np.sqrt(vt) + self.epsilon)
        return w

import numpy as np


class SGD:
    def __init__(self, lr=0.05):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            for name, value in layer.params.items():
                value -= self.lr * layer.grads[name]


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, layers):
        self.t += 1
        for li, layer in enumerate(layers):
            for name, value in layer.params.items():
                key = (li, name)
                grad = layer.grads[name]
                if key not in self.m:
                    self.m[key] = np.zeros_like(value)
                    self.v[key] = np.zeros_like(value)
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad ** 2
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                value -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

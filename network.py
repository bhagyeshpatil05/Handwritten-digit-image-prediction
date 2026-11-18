import numpy as np
from layers import Dense, ReLU, Flatten, Conv2D, MaxPool2D, SoftmaxCrossEntropy


class Network:
    def __init__(self, layers, name):
        self.layers = layers
        self.name = name
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def loss_and_backward(self, x, y):
        logits = self.forward(x)
        loss = self.loss_fn.forward(logits, y)
        self.backward(self.loss_fn.backward())
        return loss

    def predict_proba(self, x):
        logits = self.forward(x)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)

    def accuracy(self, x, y):
        return float((self.predict(x) == y).mean())

    def count_parameters(self):
        return int(sum(v.size for layer in self.layers for v in layer.params.values()))


def build_ann(seed=0):
    rng = np.random.default_rng(seed)
    layers = [
        Dense(64, 64, rng), ReLU(),
        Dense(64, 32, rng), ReLU(),
        Dense(32, 10, rng),
    ]
    return Network(layers, "ANN")


def build_cnn(seed=0):
    rng = np.random.default_rng(seed)
    layers = [
        Conv2D(1, 8, rng), ReLU(),
        MaxPool2D(), Flatten(),
        Dense(128, 64, rng), ReLU(),
        Dense(64, 10, rng),
    ]
    return Network(layers, "CNN")

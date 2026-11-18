import numpy as np


class Dense:
    def __init__(self, n_in, n_out, rng):
        self.params = {
            "W": rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in),
            "b": np.zeros(n_out),
        }
        self.grads = {"W": np.zeros((n_in, n_out)), "b": np.zeros(n_out)}
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout):
        self.grads["W"] = self.x.T @ dout
        self.grads["b"] = dout.sum(axis=0)
        return dout @ self.params["W"].T


class ReLU:
    params = {}
    grads = {}

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class Flatten:
    params = {}
    grads = {}

    def __init__(self):
        self.shape = None

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)


class Conv2D:
    def __init__(self, in_channels, n_filters, rng, kernel=3, pad=1):
        self.k = kernel
        self.pad = pad
        fan_in = in_channels * kernel * kernel
        self.params = {
            "W": rng.standard_normal((n_filters, in_channels, kernel, kernel)) * np.sqrt(2.0 / fan_in),
            "b": np.zeros(n_filters),
        }
        self.grads = {"W": np.zeros_like(self.params["W"]), "b": np.zeros(n_filters)}
        self.cache = None

    def forward(self, x):
        n, c, h, w = x.shape
        k, p = self.k, self.pad
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        cols = np.zeros((n, c, k, k, h, w))
        for i in range(k):
            for j in range(k):
                cols[:, :, i, j] = xp[:, :, i:i + h, j:j + w]
        cols = cols.reshape(n, c * k * k, h * w)
        wm = self.params["W"].reshape(self.params["W"].shape[0], -1)
        out = np.einsum("fk,nkp->nfp", wm, cols) + self.params["b"][None, :, None]
        self.cache = (x.shape, cols)
        return out.reshape(n, -1, h, w)

    def backward(self, dout):
        (n, c, h, w), cols = self.cache
        k, p = self.k, self.pad
        f = self.params["W"].shape[0]
        d2 = dout.reshape(n, f, h * w)
        wm = self.params["W"].reshape(f, -1)
        self.grads["W"] = np.einsum("nfp,nkp->fk", d2, cols).reshape(self.params["W"].shape)
        self.grads["b"] = d2.sum(axis=(0, 2))
        dcols = np.einsum("fk,nfp->nkp", wm, d2).reshape(n, c, k, k, h, w)
        dxp = np.zeros((n, c, h + 2 * p, w + 2 * p))
        for i in range(k):
            for j in range(k):
                dxp[:, :, i:i + h, j:j + w] += dcols[:, :, i, j]
        return dxp[:, :, p:p + h, p:p + w]


class MaxPool2D:
    params = {}
    grads = {}

    def __init__(self):
        self.idx = None
        self.shape = None

    def forward(self, x):
        n, c, h, w = x.shape
        h2, w2 = h // 2, w // 2
        blocks = x.reshape(n, c, h2, 2, w2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h2, w2, 4)
        self.idx = blocks.argmax(axis=-1)
        self.shape = x.shape
        return blocks.max(axis=-1)

    def backward(self, dout):
        n, c, h, w = self.shape
        h2, w2 = h // 2, w // 2
        blocks = np.zeros((n, c, h2, w2, 4))
        np.put_along_axis(blocks, self.idx[..., None], dout[..., None], axis=-1)
        return blocks.reshape(n, c, h2, w2, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h, w)


class SoftmaxCrossEntropy:
    def __init__(self):
        self.probs = None
        self.labels = None

    def forward(self, logits, labels):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probs = exp / exp.sum(axis=1, keepdims=True)
        self.labels = labels
        rows = np.arange(len(labels))
        return -np.log(self.probs[rows, labels] + 1e-12).mean()

    def backward(self):
        grad = self.probs.copy()
        grad[np.arange(len(self.labels)), self.labels] -= 1
        return grad / len(self.labels)

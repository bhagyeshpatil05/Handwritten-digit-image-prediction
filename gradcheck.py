import numpy as np
from network import build_cnn


def run_gradient_check(samples=6, checks_per_tensor=6, eps=1e-5, seed=3):
    rng = np.random.default_rng(seed)
    net = build_cnn(seed=1)
    x = rng.random((samples, 1, 8, 8))
    y = rng.integers(0, 10, samples)
    net.loss_and_backward(x, y)
    worst = 0.0
    for layer in net.layers:
        for name, value in layer.params.items():
            analytic = layer.grads[name].copy()
            flat = value.reshape(-1)
            picks = rng.choice(flat.size, size=min(checks_per_tensor, flat.size), replace=False)
            for idx in picks:
                original = flat[idx]
                flat[idx] = original + eps
                plus = net.loss_fn.forward(net.forward(x), y)
                flat[idx] = original - eps
                minus = net.loss_fn.forward(net.forward(x), y)
                flat[idx] = original
                numeric = (plus - minus) / (2 * eps)
                a = analytic.reshape(-1)[idx]
                rel = abs(a - numeric) / max(abs(a) + abs(numeric), 1e-12)
                worst = max(worst, rel)
    return worst

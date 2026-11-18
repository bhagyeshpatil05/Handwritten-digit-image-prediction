import time
import numpy as np


def train_model(net, optimizer, x_train, y_train, x_test, y_test, epochs=30, batch_size=32, seed=0, log_every=(1, 5, 10, 15, 20, 25, 30)):
    rng = np.random.default_rng(seed)
    history = {"loss": [], "train_acc": [], "test_acc": []}
    start = time.time()
    n = len(x_train)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        running = 0.0
        batches = 0
        for i in range(0, n, batch_size):
            pick = order[i:i + batch_size]
            running += net.loss_and_backward(x_train[pick], y_train[pick])
            optimizer.step(net.layers)
            batches += 1
        train_acc = net.accuracy(x_train, y_train)
        test_acc = net.accuracy(x_test, y_test)
        history["loss"].append(running / batches)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        if epoch in log_every:
            print(f"[{net.name} + {type(optimizer).__name__}] epoch {epoch:2d}/{epochs}  "
                  f"loss={history['loss'][-1]:.4f}  train_acc={train_acc * 100:.2f}%  test_acc={test_acc * 100:.2f}%")
    history["time"] = time.time() - start
    return history

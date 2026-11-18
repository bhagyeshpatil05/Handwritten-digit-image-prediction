import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_data(seed=42):
    digits = load_digits()
    images = (digits.images / 16.0).astype(np.float64)
    labels = digits.target.astype(np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    return x_train, x_test, y_train, y_test


def as_flat(images):
    return images.reshape(len(images), -1)


def as_image(images):
    return images[:, None, :, :]


def add_noise(images, sd, seed=7):
    rng = np.random.default_rng(seed)
    return np.clip(images + rng.normal(0.0, sd, images.shape), 0.0, 1.0)

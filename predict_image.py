import sys
import numpy as np
from PIL import Image, ImageOps

from data import load_data, as_flat, as_image
from network import build_ann, build_cnn
from optimizers import Adam
from train import train_model


def prepare_image(path):
    img = ImageOps.grayscale(Image.open(path))
    arr = np.asarray(img, dtype=np.float64) / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    rows = np.where(arr.max(axis=1) > 0.3)[0]
    cols = np.where(arr.max(axis=0) > 0.3)[0]
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("No digit found in the image. Use dark ink on a plain background.")
    arr = arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    size = max(arr.shape)
    side = int(size * 1.15)
    canvas = np.zeros((side, side))
    top = (side - arr.shape[0]) // 2
    left = (side - arr.shape[1]) // 2
    canvas[top:top + arr.shape[0], left:left + arr.shape[1]] = arr
    small = Image.fromarray((canvas * 255).astype(np.uint8)).resize((8, 8), Image.LANCZOS)
    out = np.asarray(small, dtype=np.float64)
    return out / out.max()


def train_models():
    x_train, x_test, y_train, y_test = load_data()
    ann, cnn = build_ann(), build_cnn()
    print("Training the models first (a few seconds)...")
    train_model(ann, Adam(lr=0.001), as_flat(x_train), y_train, as_flat(x_test), y_test, log_every=())
    train_model(cnn, Adam(lr=0.001), as_image(x_train), y_train, as_image(x_test), y_test, log_every=())
    return ann, cnn


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py your_digit.png [another.png ...]")
        return
    ann, cnn = train_models()
    for path in sys.argv[1:]:
        digit = prepare_image(path)
        ann_probs = ann.predict_proba(digit.reshape(1, -1))[0]
        cnn_probs = cnn.predict_proba(digit.reshape(1, 1, 8, 8))[0]
        print(f"\n{path}")
        print(f"  ANN says {ann_probs.argmax()}  ({ann_probs.max() * 100:.1f}% sure)")
        print(f"  CNN says {cnn_probs.argmax()}  ({cnn_probs.max() * 100:.1f}% sure)")


if __name__ == "__main__":
    main()

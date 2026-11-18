import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from data import load_data, as_flat, as_image, add_noise
from network import build_ann, build_cnn
from optimizers import SGD, Adam
from train import train_model
from gradcheck import run_gradient_check
import plots

EPOCHS = 30
BATCH_SIZE = 32
OUT = "results"
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]


def main():
    os.makedirs(OUT, exist_ok=True)
    x_train, x_test, y_train, y_test = load_data()

    error = run_gradient_check()
    status = "PASSED" if error < 1e-6 else "FAILED"
    print(f"Gradient check (CNN backprop): max relative error = {error:.2e} -> {status}")
    print(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}\n")

    plots.save_samples(x_train[:40] * 16, y_train[:40], os.path.join(OUT, "fig1_samples.png"))

    xtr_flat, xte_flat = as_flat(x_train), as_flat(x_test)
    xtr_img, xte_img = as_image(x_train), as_image(x_test)

    runs = [
        ("ANN + SGD", build_ann(), SGD(lr=0.05), xtr_flat, xte_flat),
        ("ANN + Adam", build_ann(), Adam(lr=0.001), xtr_flat, xte_flat),
        ("CNN + Adam", build_cnn(), Adam(lr=0.001), xtr_img, xte_img),
    ]

    histories = {}
    models = {}
    for name, net, opt, xtr, xte in runs:
        print(f"--- Training {name} (parameters: {net.count_parameters():,}) ---")
        histories[name] = train_model(net, opt, xtr, y_train, xte, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE)
        models[name] = (net, xtr, xte)
        print(f"Final test accuracy: {histories[name]['test_acc'][-1] * 100:.2f}%  (time {histories[name]['time']:.1f}s)\n")

    plots.save_training_curves(histories, os.path.join(OUT, "fig3_training_curves.png"))

    print("Model comparison on the 360 test images")
    print(f"{'Model':<12}{'Params':>9}{'Train acc':>12}{'Test acc':>11}{'Time (s)':>10}{'Errors':>10}")
    for name, (net, xtr, xte) in models.items():
        errors = int((net.predict(xte) != y_test).sum())
        print(f"{name:<12}{net.count_parameters():>9,}{net.accuracy(xtr, y_train) * 100:>11.2f}%"
              f"{net.accuracy(xte, y_test) * 100:>10.2f}%{histories[name]['time']:>10.1f}{errors:>6d} / {len(y_test)}")

    cnn, _, cnn_test = models["CNN + Adam"]
    probs = cnn.predict_proba(cnn_test)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    print("\nClassification report (CNN + Adam):")
    print(classification_report(y_test, pred, digits=3))

    plots.save_confusion(confusion_matrix(y_test, pred), os.path.join(OUT, "fig4_confusion_matrix.png"))
    plots.save_predictions(x_test, y_test, pred, os.path.join(OUT, "fig5_predictions.png"))
    plots.save_mistakes(x_test, y_test, pred, conf, os.path.join(OUT, "fig6_mistakes.png"))

    ann = models["ANN + Adam"][0]
    ann_scores, cnn_scores = [], []
    print("Robustness test (Gaussian noise added to test images):")
    for sd in NOISE_LEVELS:
        noisy = add_noise(x_test, sd) if sd > 0 else x_test
        a = ann.accuracy(as_flat(noisy), y_test)
        c = cnn.accuracy(as_image(noisy), y_test)
        ann_scores.append(a)
        cnn_scores.append(c)
        print(f"  noise sd={sd:.1f}  ANN={a * 100:.2f}%  CNN={c * 100:.2f}%")
    plots.save_noise_plot(NOISE_LEVELS, ann_scores, cnn_scores, os.path.join(OUT, "fig7_noise_robustness.png"))

    print(f"\nAll figures saved in the '{OUT}' folder.")


if __name__ == "__main__":
    main()

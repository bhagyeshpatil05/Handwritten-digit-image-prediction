import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"ANN + SGD": "#d9692b", "ANN + Adam": "#2a6fb0", "CNN + Adam": "#1f8a70"}


def save_samples(images, labels, path, rows=4, cols=10):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.9, rows * 1.0))
    for ax, img, lab in zip(axes.ravel(), images, labels):
        ax.imshow(img, cmap="gray_r")
        ax.set_title(str(lab), fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_training_curves(histories, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, h in histories.items():
        epochs = np.arange(1, len(h["loss"]) + 1)
        axes[0].plot(epochs, h["loss"], label=name, color=COLORS[name])
        axes[1].plot(epochs, np.array(h["train_acc"]) * 100, label=name, color=COLORS[name])
        axes[2].plot(epochs, np.array(h["test_acc"]) * 100, label=name, color=COLORS[name])
    axes[0].set_yscale("log")
    axes[0].set_title("Training loss")
    axes[1].set_title("Training accuracy (%)")
    axes[2].set_title("Test accuracy (%)")
    axes[1].set_ylim(60, 100.5)
    axes[2].set_ylim(60, 100.5)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("Actual digit")
    ax.set_title("Confusion matrix (CNN + Adam)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_predictions(images, truth, pred, path, count=30):
    cols = 10
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.0, rows * 1.2))
    for ax, img, t, p in zip(axes.ravel(), images[:count], truth[:count], pred[:count]):
        ax.imshow(img, cmap="gray_r")
        ax.set_title(f"P:{p} A:{t}", fontsize=8, color="green" if p == t else "red")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_mistakes(images, truth, pred, conf, path):
    wrong = np.where(truth != pred)[0]
    if len(wrong) == 0:
        return
    fig, axes = plt.subplots(1, len(wrong), figsize=(len(wrong) * 1.3, 1.8), squeeze=False)
    for ax, i in zip(axes[0], wrong):
        ax.imshow(images[i], cmap="gray_r")
        ax.set_title(f"P:{pred[i]} A:{truth[i]}\n{conf[i] * 100:.0f}% sure", fontsize=7, color="red")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_noise_plot(levels, ann_scores, cnn_scores, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(levels, np.array(ann_scores) * 100, "o-", label="ANN + Adam", color=COLORS["ANN + Adam"])
    ax.plot(levels, np.array(cnn_scores) * 100, "s-", label="CNN + Adam", color=COLORS["CNN + Adam"])
    ax.set_xlabel("Noise level (std. dev.)")
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

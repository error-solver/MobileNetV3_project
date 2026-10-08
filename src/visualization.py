"""Reusable plots saved to disk without opening interactive windows."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import ensure_dir


def _save(path):
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_class_distribution(counts, path):
    plt.figure(figsize=(8, 4))
    sns.barplot(x=counts.index, y=counts.values)
    plt.xlabel("Class")
    plt.ylabel("Images")
    plt.title("HAM10000 class distribution")
    _save(path)


def plot_sample_grid(samples, path, columns=4):
    """samples: sequence of (PIL image, label) pairs."""
    rows = (len(samples) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (image, label) in zip(axes.flat, samples):
        axis.imshow(image)
        axis.set_title(label)
    _save(path)


def plot_history(history, path, metric):
    plt.figure(figsize=(7, 4))
    plt.plot(history[f"train_{metric}"], label="Train")
    plt.plot(history[f"val_{metric}"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(metric.title())
    plt.legend()
    _save(path)


def plot_loss(history, path):
    plot_history(history, path, "loss")


def plot_accuracy(history, path):
    plot_history(history, path, "accuracy")


def plot_confusion_matrix(matrix, class_names, path):
    plt.figure(figsize=(8, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    _save(path)


def plot_model_comparison(comparison, path):
    columns = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1",
               "Macro Sensitivity", "Macro Specificity"]
    comparison.set_index("Model")[columns].T.plot.bar(figsize=(10, 5), ylim=(0, 1))
    plt.ylabel("Score")
    _save(path)

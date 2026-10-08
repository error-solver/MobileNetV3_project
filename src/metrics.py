"""Overall and one-versus-rest classification metrics."""
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def calculate_class_metrics(y_true, y_pred, class_names):
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    rows = []
    for index, name in enumerate(class_names):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        rows.append({"Class": name, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                     "Sensitivity": sensitivity, "Specificity": specificity,
                     "Precision": precision, "Recall": sensitivity,
                     "F1": 2 * precision * sensitivity / (precision + sensitivity)
                     if precision + sensitivity else 0.0})
    return pd.DataFrame(rows)


def calculate_overall_metrics(y_true, y_pred, class_names):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), average="macro", zero_division=0)
    per_class = calculate_class_metrics(y_true, y_pred, class_names)
    return {"Accuracy": accuracy_score(y_true, y_pred), "Macro Precision": precision,
            "Macro Recall": recall, "Macro F1": f1,
            "Macro Sensitivity": float(per_class["Sensitivity"].mean()),
            "Macro Specificity": float(per_class["Specificity"].mean())}

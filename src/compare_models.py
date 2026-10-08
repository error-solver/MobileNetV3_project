"""Compare saved evaluation outputs from both trained models."""
import pandas as pd

from src.config import CLASS_NAMES, RESULTS_DIR
from src.models import create_model
from src.utils import count_parameters, ensure_dir
from src.visualization import plot_model_comparison


def main():
    rows = []
    for name in ("baseline", "frequency_attention"):
        directory = RESULTS_DIR / name
        overall = pd.read_csv(directory / "overall_metrics.csv").iloc[0].to_dict()
        per_class = pd.read_csv(directory / "class_metrics.csv").set_index("Class")
        if set(per_class.index) != set(CLASS_NAMES):
            raise ValueError(f"Incomplete class metrics for {name}")
        row = {"Model": name, **overall,
               "Parameter Count": count_parameters(create_model(name, pretrained=False))}
        for class_name in CLASS_NAMES:
            row[f"{class_name} Sensitivity"] = per_class.loc[class_name, "Sensitivity"]
            row[f"{class_name} Specificity"] = per_class.loc[class_name, "Specificity"]
        rows.append(row)
    comparison = pd.DataFrame(rows)
    output = ensure_dir(RESULTS_DIR) / "model_comparison.csv"
    comparison.to_csv(output, index=False)
    plot_model_comparison(comparison, RESULTS_DIR / "model_comparison.png")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

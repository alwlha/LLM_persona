import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制人格分类混淆矩阵热力图")
    parser.add_argument(
        "--input",
        default="results/persona_confusion_matrix.csv",
        help="输入 CSV 路径",
    )
    parser.add_argument(
        "--output",
        default="results/persona_confusion_matrix.png",
        help="输出图片路径",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="按行归一化后绘制比例热力图",
    )
    return parser.parse_args()


def load_confusion_matrix(csv_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        pred_labels = header[1:]

        true_labels = []
        rows = []
        for row in reader:
            true_labels.append(row[0])
            rows.append([int(value) for value in row[1:]])

    matrix = np.asarray(rows, dtype=float)
    return true_labels, pred_labels, matrix


def plot_confusion_matrix(
    true_labels: list[str],
    pred_labels: list[str],
    matrix: np.ndarray,
    output_path: Path,
    normalize: bool,
) -> None:
    display_matrix = matrix.copy()
    value_format = ".2f" if normalize else ".0f"

    if normalize:
        row_sums = display_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        display_matrix = display_matrix / row_sums

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(display_matrix, cmap="Blues", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Proportion" if normalize else "Count")

    ax.set_xticks(np.arange(len(pred_labels)))
    ax.set_yticks(np.arange(len(true_labels)))
    ax.set_xticklabels(pred_labels, rotation=30, ha="right")
    ax.set_yticklabels(true_labels)
    ax.set_xlabel("Predicted Personality")
    ax.set_ylabel("True Personality")
    ax.set_title("Persona Confusion Matrix")

    threshold = display_matrix.max() / 2 if display_matrix.size else 0
    for i in range(display_matrix.shape[0]):
        for j in range(display_matrix.shape[1]):
            ax.text(
                j,
                i,
                format(display_matrix[i, j], value_format),
                ha="center",
                va="center",
                color="white" if display_matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = repo_root / input_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    true_labels, pred_labels, matrix = load_confusion_matrix(input_path)
    plot_confusion_matrix(true_labels, pred_labels, matrix, output_path, args.normalize)
    print(f"Saved confusion matrix plot to: {output_path}")


if __name__ == "__main__":
    main()

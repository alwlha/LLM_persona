import argparse
import csv
from pathlib import Path
from typing import Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 results 下每个模型目录中的所有 run summary")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="结果根目录，默认 results",
    )
    parser.add_argument(
        "--output-name",
        default="all_runs_summary.csv",
        help="每个模型目录下输出的汇总文件名",
    )
    return parser.parse_args()


def aggregate_model_dir(model_dir: Path, output_name: str) -> Optional[Tuple[Path, int]]:
    rows = []
    fieldnames = []

    for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        summary_path = run_dir / "summary_results.csv"
        if not summary_path.exists():
            continue

        with summary_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for field in reader.fieldnames:
                    if field not in fieldnames:
                        fieldnames.append(field)
            for row in reader:
                rows.append(row)

    if not rows:
        return None

    output_path = model_dir / output_name
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path, len(rows)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        aggregated = aggregate_model_dir(model_dir, args.output_name)
        if not aggregated:
            continue
        output_path, row_count = aggregated
        print(f"{model_dir.name}: {row_count} rows -> {output_path}")


if __name__ == "__main__":
    main()

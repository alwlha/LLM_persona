import argparse
import csv
from datetime import datetime
from pathlib import Path

from src.activation import load_activations
from src.models import CLOSED_MODELS_REGISTRY, build_closed_model
from src.runner import ACTIVATION_TYPES, TASK_TYPES, build_tasks, resolve_activation, run_experiment
from src.scoring import LLMJudge
from src.utils import get_logger, load_config

logger = get_logger("main_closed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="闭源模型人格实验入口")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(CLOSED_MODELS_REGISTRY),
        help="闭源模型简称",
    )
    parser.add_argument(
        "--activation-method",
        default="prompt",
        choices=["prompt"],
        help="激活方法（闭源入口仅支持 prompt）",
    )
    parser.add_argument(
        "--activation-type",
        default="base",
        choices=ACTIVATION_TYPES,
        help="激活类型",
    )
    parser.add_argument("--task", default="bfi", choices=TASK_TYPES, help="任务类型")
    parser.add_argument(
        "--judge",
        default=None,
        choices=list(CLOSED_MODELS_REGISTRY),
        help="生成任务可选评分模型",
    )
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--api_key", default=None, help="覆盖 config 中 API Key")
    parser.add_argument("--run-id", default=None, help="运行 ID（默认自动生成）")
    return parser.parse_args()


def _update_all_summary(model_run_dir: Path) -> Path | None:
    model_dir = model_run_dir.parent
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
            rows.extend(reader)

    if not rows:
        return None

    output_path = model_dir / "all_runs_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.api_key:
        cfg["api"]["api_key"] = args.api_key

    if not cfg["api"].get("api_key"):
        raise ValueError("未检测到 API Key，请在 .env 或 --api_key 中提供")

    activation_file = Path(cfg["paths"]["activations_dir"]) / f"{args.activation_method}.json"
    all_activations = load_activations(activation_file)
    activation = resolve_activation(
        all_activations=all_activations,
        activation_method=args.activation_method,
        activation_type=args.activation_type,
    )
    tasks = build_tasks([args.task], cfg)

    model = build_closed_model(args.model, cfg)

    judge = None
    if args.judge:
        judge = LLMJudge(build_closed_model(args.judge, cfg))

    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = run_experiment(
        model_key=args.model,
        model=model,
        tasks=tasks,
        activation=activation,
        results_root=cfg["paths"]["results_dir"],
        run_id=run_id,
        judge=judge,
    )
    all_summary_path = _update_all_summary(output_dir)
    if all_summary_path:
        logger.info(f"总汇总已更新：{all_summary_path}")
    logger.info(f"实验完成，结果目录：{output_dir}")


if __name__ == "__main__":
    main()

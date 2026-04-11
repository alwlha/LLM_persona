import argparse
import csv
from datetime import datetime
from pathlib import Path

from src.activation import load_activations
from src.models import CLOSED_MODELS_REGISTRY, build_closed_model
from src.runner import ACTIVATION_TYPES, TASK_TYPES, build_tasks, resolve_activation, run_experiment
from src.scoring import BraggingJudge, LLMJudge
from src.utils import get_logger, load_config

logger = get_logger("main_closed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="闭源模型人格实验入口")
    parser.add_argument("--model", required=True, choices=list(CLOSED_MODELS_REGISTRY), help="闭源模型简称")
    parser.add_argument("--activation-method", default="prompt", choices=["prompt"], help="激活方法（闭源入口仅支持 prompt）")
    parser.add_argument(
        "--activation-type",
        nargs="+",
        action="append",
        default=None,
        choices=ACTIVATION_TYPES,
        help="激活类型，支持一次传多个值或重复传参",
    )
    parser.add_argument("--task", default="bfi", choices=TASK_TYPES, help="任务类型")
    parser.add_argument("--judge", default=None, choices=list(CLOSED_MODELS_REGISTRY), help="生成任务可选评分模型")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--api_key", default=None, help="覆盖 config 中 API Key")
    parser.add_argument("--base_url", default=None, help="覆盖 config 中 Base URL")
    parser.add_argument("--run-id", default=None, help="运行 ID（默认自动生成）")
    parser.add_argument("--api-workers", type=int, default=None, help="生成阶段并发 worker 数")
    parser.add_argument("--judge-workers", type=int, default=None, help="评分阶段并发 worker 数")
    return parser.parse_args()


def _normalize_activation_types(raw_activation_types: list[list[str]] | None) -> list[str]:
    if not raw_activation_types:
        return ["base"]

    activation_types = []
    seen = set()
    for group in raw_activation_types:
        for activation_type in group:
            if activation_type in seen:
                continue
            seen.add(activation_type)
            activation_types.append(activation_type)
    return activation_types


def _update_all_summary(model_run_dir: Path) -> Path | None:
    return _merge_csv_across_runs(model_run_dir, "summary_results.csv", "all_runs_summary.csv")


def _update_all_metrics(model_run_dir: Path) -> Path | None:
    return _merge_csv_across_runs(model_run_dir, "metrics_long.csv", "all_runs_metrics_long.csv")


def _merge_csv_across_runs(model_run_dir: Path, input_name: str, output_name: str) -> Path | None:
    model_dir = model_run_dir.parent
    rows = []
    fieldnames = []

    for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        summary_path = run_dir / input_name
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

    output_path = model_dir / output_name
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
    if args.base_url:
        cfg["api"]["base_url"] = args.base_url

    if not cfg["api"].get("api_key"):
        raise ValueError("未检测到 API Key，请在 .env 或 --api_key 中提供")

    activation_file = Path(cfg["paths"]["activations_dir"]) / f"{args.activation_method}.json"
    all_activations = load_activations(activation_file)
    tasks = build_tasks([args.task], cfg)
    activation_types = _normalize_activation_types(args.activation_type)

    model = build_closed_model(args.model, cfg)

    judge = None
    if args.judge:
        judge_model = build_closed_model(args.judge, cfg)
        judge = BraggingJudge(judge_model) if args.task == "bragging_generation" else LLMJudge(judge_model)

    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = None
    for index, activation_type in enumerate(activation_types, start=1):
        activation = resolve_activation(
            all_activations=all_activations,
            activation_method=args.activation_method,
            activation_type=activation_type,
        )
        logger.info(
            "开始运行 activation-type=%s (%d/%d)",
            activation_type,
            index,
            len(activation_types),
        )
        output_dir = run_experiment(
            model_key=args.model,
            model=model,
            tasks=tasks,
            activation=activation,
            results_root=cfg["paths"]["results_dir"],
            run_id=run_id,
            judge=judge,
            api_workers=args.api_workers or cfg.get("experiments", {}).get("api_workers", 8),
            judge_workers=args.judge_workers or cfg.get("experiments", {}).get("judge_workers", 8),
        )
    all_summary_path = _update_all_summary(output_dir)
    all_metrics_path = _update_all_metrics(output_dir)
    if all_summary_path:
        logger.info(f"总汇总已更新：{all_summary_path}")
    if all_metrics_path:
        logger.info(f"长表总汇总已更新：{all_metrics_path}")
    logger.info(f"实验完成，结果目录：{output_dir}")


if __name__ == "__main__":
    main()

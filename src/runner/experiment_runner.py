import json
import asyncio
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import pandas as pd

from src.activation import ActivationConfig
from src.tasks import BFITask, BraggingGenerationTask, GenerationTask
from src.utils import ensure_dir, get_logger

logger = get_logger("runner")

TASK_TYPES = ["bfi", "social_scenario", "bragging_generation"]
ACTIVATION_TYPES = [
    "base",
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "neuroticism",
    "openness",
]

PROMPT_ACTIVATION_MAP = {
    "base": "base",
    "extraversion": "high_extraversion",
    "agreeableness": "high_agreeableness",
    "conscientiousness": "high_conscientiousness",
    "neuroticism": "high_neuroticism",
    "openness": "high_openness",
}

VECTOR_ACTIVATION_MAP = {
    "base": "base",
    "extraversion": "vector_extraversion",
    "agreeableness": "vector_agreeableness",
    "conscientiousness": "vector_conscientiousness",
    "neuroticism": "vector_neuroticism",
    "openness": "vector_openness",
}


def resolve_activation(
    all_activations: list[ActivationConfig],
    activation_method: str,
    activation_type: str,
) -> ActivationConfig:
    by_name = {a.name: a for a in all_activations}

    if activation_method == "prompt":
        target_name = PROMPT_ACTIVATION_MAP[activation_type]
    elif activation_method == "vector":
        target_name = VECTOR_ACTIVATION_MAP[activation_type]
    else:
        raise ValueError(f"Unsupported activation method: {activation_method}")

    if target_name not in by_name:
        raise ValueError(
            f"Activation '{target_name}' not found in selected activation file. "
            "Please check data/activation/*.json."
        )

    activation = by_name[target_name]
    if activation_type != "base" and activation.method != activation_method:
        raise ValueError(
            f"Activation '{activation.name}' method mismatch: expected '{activation_method}', "
            f"got '{activation.method}'"
        )
    return activation


def build_tasks(task_keys: list[str], cfg: dict) -> list:
    tasks = []
    for key in task_keys:
        if key == "bfi":
            tasks.append(BFITask(cfg["paths"]["bfi_file"]))
            continue
        if key == "bragging_generation":
            experiments_cfg = cfg.get("experiments", {})
            task_file = Path(cfg["paths"]["tasks_dir"]) / "Bragging_data.json"
            tasks.append(
                BraggingGenerationTask(
                    task_file=task_file,
                    num_samples=experiments_cfg.get("bragging_num_samples", 1),
                    max_samples=experiments_cfg.get("bragging_max_samples"),
                    random_seed=experiments_cfg.get("bragging_random_seed", 42),
                )
            )
            continue

        task_file = Path(cfg["paths"]["tasks_dir"]) / f"{key}.json"
        if not task_file.exists():
            raise ValueError(f"Task file not found: {task_file}")
        tasks.append(GenerationTask(task_file))
    return tasks


def _save_raw_result(result: dict, raw_dir: Path):
    fname = f"{result['activation']}_{result['task']}_raw.json"
    out_path = raw_dir / fname
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"  Raw result saved: {out_path}")


def _aggregate_task_scores(raw_responses: list[dict]) -> dict:
    metric_scores = defaultdict(list)

    for item in raw_responses:
        judge_scores = item.get("judge_scores") or {}
        for metric, value in judge_scores.items():
            if metric in {"judge_rationale", "judge_raw_output"}:
                continue
            if isinstance(value, (int, float)):
                metric_scores[metric].append(float(value))

        if item.get("score") is not None:
            metric_name = item.get("dimension") or "score"
            metric_scores[metric_name].append(float(item["score"]))

    return {
        metric: round(sum(values) / len(values), 2)
        for metric, values in metric_scores.items()
        if values
    }


def _build_metric_rows(
    raw_responses: list[dict],
    *,
    run_id: str,
    timestamp: str,
    model_key: str,
    model_type: str,
    activation: ActivationConfig,
    task_name: str,
) -> list[dict]:
    metric_rows = []

    for item in raw_responses:
        judge_scores = item.get("judge_scores") or {}
        for metric, value in judge_scores.items():
            if metric in {"judge_rationale", "judge_raw_output"}:
                continue
            if not isinstance(value, (int, float)):
                continue
            metric_rows.append(
                {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "model": model_key,
                    "model_type": model_type,
                    "activation": activation.name,
                    "activation_method": activation.method,
                    "vector_strength": activation.meta.get("vector_strength"),
                    "task": task_name,
                    "source_id": item.get("source_id"),
                    "item_id": item.get("id"),
                    "sample_idx": item.get("sample_idx"),
                    "metric_name": metric,
                    "metric_value": value,
                }
            )

        if item.get("score") is not None:
            metric_rows.append(
                {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "model": model_key,
                    "model_type": model_type,
                    "activation": activation.name,
                    "activation_method": activation.method,
                    "vector_strength": activation.meta.get("vector_strength"),
                    "task": task_name,
                    "source_id": item.get("source_id"),
                    "item_id": item.get("id"),
                    "sample_idx": item.get("sample_idx"),
                    "metric_name": item.get("dimension") or "score",
                    "metric_value": item.get("score"),
                }
            )

    return metric_rows


def run_experiment(
    model_key: str,
    model,
    tasks: list,
    activation: ActivationConfig,
    results_root: str | Path,
    run_id: str,
    judge: object | None = None,
    api_workers: int = 8,
    judge_workers: int = 8,
) -> Path:
    model_run_dir = ensure_dir(Path(results_root) / model_key / run_id)
    raw_dir = ensure_dir(model_run_dir / "raw")
    timestamp = datetime.now().isoformat(timespec="seconds")

    all_summary_rows = []
    all_metric_rows = []

    logger.info(f"Model: {model_key}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Activation: [{activation.name}] ({activation.method})")

    for task in tasks:
        logger.info(f"Task: {task.name}")
        task_result = asyncio.run(
            _run_task_with_optional_async(
                task=task,
                model=model,
                activation=activation,
                judge=judge,
                raw_dir=raw_dir,
                api_workers=api_workers,
                judge_workers=judge_workers,
            )
        )

        full_result = {
            "run_id": run_id,
            "timestamp": timestamp,
            "model_key": model_key,
            "model_id": getattr(model, "name", model_key),
            "model_type": "API" if model.__class__.__name__ == "APIModel" else "Local",
            "activation": activation.name,
            "activation_method": activation.method,
            "vector_strength": activation.meta.get("vector_strength"),
            "task": task.name,
            **task_result,
        }
        _save_raw_result(full_result, raw_dir)
        all_metric_rows.extend(
            _build_metric_rows(
                task_result.get("raw_responses", []),
                run_id=run_id,
                timestamp=timestamp,
                model_key=model_key,
                model_type=full_result["model_type"],
                activation=activation,
                task_name=task.name,
            )
        )

        summary_row = {
            "run_id": run_id,
            "timestamp": timestamp,
            "model": model_key,
            "model_type": full_result["model_type"],
            "activation": activation.name,
            "activation_method": activation.method,
            "vector_strength": activation.meta.get("vector_strength"),
            "task": task.name,
            **task_result.get("scores", {}),
        }
        all_summary_rows.append(summary_row)

    summary_path = model_run_dir / "summary_results.csv"
    new_summary_df = pd.DataFrame(all_summary_rows)
    if summary_path.exists():
        existing_summary_df = pd.read_csv(summary_path)
        new_summary_df = pd.concat([existing_summary_df, new_summary_df], ignore_index=True)
    new_summary_df.to_csv(summary_path, index=False)
    logger.info(f"Done. Summary saved to: {summary_path}")

    if all_metric_rows:
        metrics_path = model_run_dir / "metrics_long.csv"
        new_metrics_df = pd.DataFrame(all_metric_rows)
        if metrics_path.exists():
            existing_metrics_df = pd.read_csv(metrics_path)
            new_metrics_df = pd.concat([existing_metrics_df, new_metrics_df], ignore_index=True)
        new_metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Metric rows saved to: {metrics_path}")
    return model_run_dir


async def _run_task_with_optional_async(
    *,
    task,
    model,
    activation: ActivationConfig,
    judge: object | None,
    raw_dir: Path,
    api_workers: int,
    judge_workers: int,
) -> dict:
    responses_checkpoint = raw_dir / f"{activation.name}_{task.name}_responses.jsonl"
    judged_checkpoint = raw_dir / f"{activation.name}_{task.name}_judged.jsonl"

    if hasattr(task, "run_async"):
        task_result = await task.run_async(
            model=model,
            activation=activation,
            checkpoint_path=responses_checkpoint,
            max_concurrency=api_workers,
        )
    else:
        task_result = await asyncio.to_thread(task.run, model, activation)

    if task.name != "bfi" and judge and task_result.get("raw_responses"):
        logger.info("Running LLM Judge...")
        if hasattr(judge, "score_batch_async"):
            task_result["raw_responses"] = await judge.score_batch_async(
                task_result["raw_responses"],
                checkpoint_path=judged_checkpoint,
                max_concurrency=judge_workers,
            )
        else:
            task_result["raw_responses"] = await asyncio.to_thread(
                judge.score_batch,
                task_result["raw_responses"],
            )
        task_result["scores"] = _aggregate_task_scores(task_result["raw_responses"])

    return task_result

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.activation import ActivationConfig
from src.scoring import LLMJudge
from src.tasks import BFITask, GenerationTask
from src.utils import ensure_dir, get_logger

logger = get_logger("runner")

TASK_TYPES = ["bfi", "social_scenario"]
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


def run_experiment(
    model_key: str,
    model,
    tasks: list,
    activation: ActivationConfig,
    results_root: str | Path,
    run_id: str,
    judge: LLMJudge | None = None,
) -> Path:
    model_run_dir = ensure_dir(Path(results_root) / model_key / run_id)
    raw_dir = ensure_dir(model_run_dir / "raw")
    timestamp = datetime.now().isoformat(timespec="seconds")

    all_summary_rows = []

    logger.info(f"Model: {model_key}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Activation: [{activation.name}] ({activation.method})")

    for task in tasks:
        logger.info(f"Task: {task.name}")
        task_result = task.run(model, activation=activation)

        if task.name != "bfi" and judge and task_result.get("raw_responses"):
            logger.info("Running LLM Judge...")
            task_result["raw_responses"] = judge.score_batch(task_result["raw_responses"])

            from collections import defaultdict

            dim_scores = defaultdict(list)
            for item in task_result["raw_responses"]:
                if item["score"] is not None:
                    dim_scores[item["dimension"]].append(item["score"])
            task_result["scores"] = {
                dim: round(sum(s) / len(s), 2) for dim, s in dim_scores.items()
            }

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
    return model_run_dir

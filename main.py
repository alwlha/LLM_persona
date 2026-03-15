"""
LLM Personality Experiment Runner
==================================
统一实验入口，支持通过命令行参数灵活配置：
  - 测试模型（API / 本地）
  - 激活方法（prompt，未来支持 finetune / weight）
  - 测试任务（bfi / 生成任务）
  - 评分方式（BFI 规则计分 / LLM-as-Judge）

用法示例：
  # 用 BFI 测试 GPT 和 Claude，使用所有激活方法
  python main.py --models gpt-5.2 claude-4.5 --tasks bfi

  # 同时运行 BFI 和生成任务，只测试 baseline 和高外向激发
  python main.py --models gpt-5.2 --tasks bfi social_scenario --activations baseline high_extraversion

  # 对生成任务的输出用 GPT 打分
  python main.py --models claude-4.5 --tasks social_scenario --judge gpt-5.2

  # 测试所有模型和任务
  python main.py --all
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from src.activation import load_activations
from src.models import APIModel, LocalModel
from src.scoring import LLMJudge
from src.tasks import BFITask, GenerationTask
from src.utils import ensure_dir, get_logger, load_config

logger = get_logger("main")

# ======================================================================
# 模型注册表（简称 -> 实际模型名/路径）
# 修改此处即可增删模型，无需改动其他代码
# ======================================================================
API_MODELS_REGISTRY = {
    "gpt-5.2": "gpt-5.2-2025-12-11",
    "gemini-3.0": "gemini-3-flash-preview-thinking",
    "claude-4.5": "claude-opus-4-5-20251101",
    "deepseek": "deepseek-v3.1"
}

LOCAL_MODELS_REGISTRY = {
    "Llama-3-8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen3-8B": "Qwen/Qwen2.5-7B-Instruct",
}

ALL_MODEL_KEYS = list(API_MODELS_REGISTRY) + list(LOCAL_MODELS_REGISTRY)


def build_model(model_key: str, cfg: dict):
    """根据 key 实例化对应模型"""
    if model_key in API_MODELS_REGISTRY:
        return APIModel(
            model_name=API_MODELS_REGISTRY[model_key],
            api_key=cfg["api"]["api_key"],
            base_url=cfg["api"]["base_url"],
        )
    if model_key in LOCAL_MODELS_REGISTRY:
        return LocalModel(
            model_path=LOCAL_MODELS_REGISTRY[model_key],
            model_name=model_key,
        )
    raise ValueError(f"Unknown model key: '{model_key}'. Available: {ALL_MODEL_KEYS}")


def build_tasks(task_keys: list[str], cfg: dict) -> list:
    """根据 task key 列表实例化任务对象"""
    tasks = []
    for key in task_keys:
        if key == "bfi":
            tasks.append(BFITask(cfg["paths"]["bfi_file"]))
        else:
            # 尝试从 data/tasks/{key}.json 加载生成任务
            task_file = Path(cfg["paths"]["tasks_dir"]) / f"{key}.json"
            if task_file.exists():
                tasks.append(GenerationTask(task_file))
            else:
                logger.warning(f"Task file not found: {task_file}, skipping '{key}'")
    return tasks


def save_result(result: dict, results_dir: Path):
    """保存单条实验结果（含原始响应）的 JSON"""
    fname = f"{result['model_key']}_{result['activation']}_{result['task']}_raw.json"
    out_path = results_dir / fname
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"  Raw result saved: {out_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Personality Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help=f"Models to test. Available: {ALL_MODEL_KEYS}",
    )
    parser.add_argument("--all", action="store_true", help="Test all registered models")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["bfi"],
        metavar="TASK",
        help="Tasks to run. 'bfi' is built-in; others load from data/tasks/<name>.json",
    )
    parser.add_argument(
        "--activations",
        nargs="+",
        metavar="ACTIVATION",
        help="Activation method names to use (default: all in prompts.json)",
    )
    parser.add_argument(
        "--judge",
        metavar="MODEL",
        help="Model key to use as LLM judge for generation tasks (e.g., gpt-5.2)",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="FILE",
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument("--api_key", default=None, help="Override API key from config")

    args = parser.parse_args()

    # ── 加载配置 ──────────────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.api_key:
        cfg["api"]["api_key"] = args.api_key
    print('cfg:\n',cfg)
    results_dir = ensure_dir(cfg["paths"]["results_dir"])

    # ── 确定要测试的模型 ──────────────────────────────────────────────
    if args.all:
        model_keys = ALL_MODEL_KEYS
    elif args.models:
        model_keys = args.models
    else:
        parser.error("Please specify --models <key...> or use --all")

    # ── 加载激活配置 ──────────────────────────────────────────────────
    all_activations = load_activations(cfg["paths"]["prompts_file"])
    if args.activations:
        activations = [a for a in all_activations if a.name in args.activations]
        missing = set(args.activations) - {a.name for a in activations}
        if missing:
            logger.warning(f"Activations not found in prompts.json: {missing}")
    else:
        activations = all_activations

    # ── 实例化任务 ────────────────────────────────────────────────────
    tasks = build_tasks(args.tasks, cfg)
    if not tasks:
        logger.error("No valid tasks found. Exiting.")
        return

    # ── 准备 LLM Judge（可选）────────────────────────────────────────
    judge = None
    if args.judge:
        judge_model = build_model(args.judge, cfg)
        if isinstance(judge_model, APIModel):
            judge = LLMJudge(judge_model)
        else:
            logger.warning("LLM Judge should be an API model. Ignoring --judge.")

    # ── 主循环：模型 × 激活 × 任务 ──────────────────────────────────
    all_summary_rows = []

    for model_key in model_keys:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Model: {model_key}")
        logger.info(f"{'=' * 60}")

        try:
            model = build_model(model_key, cfg)
        except Exception as e:
            logger.error(f"  Failed to load model '{model_key}': {e}")
            continue

        for activation in activations:
            logger.info(f"  Activation: [{activation.name}] ({activation.method})")

            for task in tasks:
                logger.info(f"    Task: {task.name}")
                task_result = task.run(
                    model, activation_system=activation.system_prompt
                )

                # 对生成任务做 LLM 打分
                if task.name != "bfi" and judge and task_result.get("raw_responses"):
                    logger.info(f"    Running LLM Judge...")
                    task_result["raw_responses"] = judge.score_batch(
                        task_result["raw_responses"]
                    )
                    # 计算均分作为 dimension-level 得分
                    from collections import defaultdict

                    dim_scores = defaultdict(list)
                    for item in task_result["raw_responses"]:
                        if item["score"] is not None:
                            dim_scores[item["dimension"]].append(item["score"])
                    task_result["scores"] = {
                        dim: round(sum(s) / len(s), 2) for dim, s in dim_scores.items()
                    }

                # 构造完整结果记录
                full_result = {
                    "model_key": model_key,
                    "model_id": getattr(model, "name", model_key),
                    "model_type": "API"
                    if model_key in API_MODELS_REGISTRY
                    else "Local",
                    "activation": activation.name,
                    "activation_method": activation.method,
                    "task": task.name,
                    **task_result,
                }
                save_result(full_result, results_dir)

                # 汇总行（展平得分用于 CSV）
                summary_row = {
                    "model": model_key,
                    "model_type": full_result["model_type"],
                    "activation": activation.name,
                    "task": task.name,
                    **task_result.get("scores", {}),
                }
                all_summary_rows.append(summary_row)

    # ── 写入汇总 CSV ─────────────────────────────────────────────────
    if all_summary_rows:
        summary_path = results_dir / "summary_results.csv"
        new_df = pd.DataFrame(all_summary_rows)

        if summary_path.exists():
            existing_df = pd.read_csv(summary_path)
            new_df = pd.concat([existing_df, new_df], ignore_index=True)

        new_df.to_csv(summary_path, index=False)
        logger.info(f"\nDone. Summary saved to: {summary_path}")
        print(new_df.to_string(index=False))
    else:
        logger.warning("No results generated.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
uv run scripts/judge_bragging_persona.py  --input results/Qwen3-8B-Instruct/run_20260411_214046-copy/raw  --judge gpt-5.2    --workers 16
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from openai import AsyncOpenAI

from src.utils import (
    PROJECT_ROOT,
    append_jsonl,
    ensure_dir,
    extract_json_object,
    get_logger,
    load_config,
    load_jsonl,
)

LOGGER = get_logger("JudgeBraggingPersona")
PERSONALITIES = [
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "neuroticism",
    "openness",
]
LABEL_SET = set(PERSONALITIES)
TASK_SUFFIX = "_bragging_generation_responses"
CLOSED_MODELS_REGISTRY = {
    "gpt-5.2": "gpt-5.2",
    "gemini-3.0": "gemini-3-flash-preview-thinking",
    "claude-4.5": "claude-opus-4-5-20251101",
    "deepseek": "deepseek-v3.1",
}

PERSONA_JUDGE_SYSTEM = (
    "You are an expert evaluator of Big Five personality signals in short bragging statements. "
    "Infer which single Big Five trait the text most strongly reflects. "
    "Choose only from: extraversion, agreeableness, conscientiousness, neuroticism, openness. "
    "Return valid JSON only."
)

PERSONA_JUDGE_TEMPLATE = """You are evaluating a bragging sentence produced under a personality activation experiment.

Potential Social Context:
{social_context}

Speaker's Intent:
{speaker_intent}

Generated Bragging Sentence:
{response}

Task:
Infer which single Big Five personality trait the sentence most strongly reflects overall.
You must choose exactly one label from:
- extraversion
- agreeableness
- conscientiousness
- neuroticism
- openness

Scoring guidance:
- extraversion: socially bold, energetic, attention-seeking, high social confidence
- agreeableness: warm, affiliative, considerate, harmony-seeking, gentle prosocial framing
- conscientiousness: disciplined, planned, reliable, achievement-through-effort, orderly self-presentation
- neuroticism: insecurity, worry, sensitivity, emotional volatility, defensiveness, stress-laden self-focus
- openness: novelty, imagination, aesthetics, unconventionality, reflective or creative self-expression

Return JSON only in this exact schema:
{{
  "predicted_personality": "one label from the allowed list",
  "confidence": 0,
  "judge_rationale": "one short sentence"
}}
"""


class AsyncJudgeModel:
    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.model_name = model_name
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60.0,
        )

    async def async_query(
        self,
        prompt: str,
        system: str | None = None,
        max_retries: int = 3,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                )
                content = response.choices[0].message.content
                return content.strip() if content else ""
            except Exception as exc:
                if attempt == max_retries - 1:
                    return f"[API_ERROR] {exc}"
                await asyncio.sleep((attempt + 1) * 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 bragging generation 的 response 文件做 Big Five 人格可辨识度判别，并统计正确率。"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入路径：可传单个 *_responses.jsonl 文件，或包含这些文件的 raw/ 目录",
    )
    parser.add_argument(
        "--judge",
        default="deepseek",
        help="闭源 judge 模型 key，默认 deepseek",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径，默认项目根目录 config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录，默认写到输入目录下的 persona_discernment/",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并发请求数，默认 8",
    )
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="包含 base 样本。由于 judge 只能在五个人格中强制选择，默认不统计 base。",
    )
    return parser.parse_args()


def normalize_true_label(activation_name: str) -> str | None:
    raw = activation_name.strip().lower()
    if raw == "base":
        return "base"

    matches = [
        label
        for label in PERSONALITIES
        if re.search(rf"(^|_){re.escape(label)}($|_)", raw)
    ]
    if len(matches) == 1:
        return matches[0]

    for prefix in ("high_", "vector_"):
        if raw.startswith(prefix):
            candidate = raw[len(prefix) :].split("_", 1)[0]
            return candidate if candidate in LABEL_SET else None

    return raw if raw in LABEL_SET else None


def infer_activation_name_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith(TASK_SUFFIX):
        return stem[: -len(TASK_SUFFIX)]
    return stem


def discover_response_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    return sorted(input_path.glob(f"*{TASK_SUFFIX}.jsonl"))


def build_jobs(files: list[Path], include_base: bool) -> tuple[list[dict], list[str]]:
    jobs = []
    skipped_files = 0
    skipped_rows = 0
    unknown_files = []

    for path in files:
        activation_name = infer_activation_name_from_path(path)
        true_label = normalize_true_label(activation_name)
        if true_label is None:
            skipped_files += 1
            unknown_files.append(path.name)
            LOGGER.warning("Skipping file with unknown activation label: %s", path.name)
            continue
        if true_label == "base" and not include_base:
            skipped_files += 1
            LOGGER.info("Skipping base file by default: %s", path.name)
            continue

        rows = load_jsonl(path)
        for row in rows:
            response = str(row.get("response", "")).strip()
            if not response:
                skipped_rows += 1
                continue
            jobs.append(
                {
                    **row,
                    "source_file": str(path),
                    "activation_name": activation_name,
                    "true_personality": true_label,
                }
            )

    LOGGER.info(
        "Prepared %d rows from %d files (skipped_files=%d, skipped_rows=%d)",
        len(jobs),
        len(files),
        skipped_files,
        skipped_rows,
    )
    return jobs, unknown_files


def build_prompt(item: dict) -> str:
    return PERSONA_JUDGE_TEMPLATE.format(
        social_context=item.get("social_context", ""),
        speaker_intent=item.get("speaker_intent", ""),
        response=item.get("response", ""),
    )


def parse_prediction(raw_output: str) -> tuple[str | None, int | None, str]:
    parsed = extract_json_object(raw_output) or {}
    predicted = str(parsed.get("predicted_personality", "")).strip().lower()
    if predicted not in LABEL_SET:
        predicted = None

    confidence = parsed.get("confidence")
    try:
        confidence = int(confidence)
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None:
        confidence = max(0, min(10, confidence))

    rationale = str(parsed.get("judge_rationale", "")).strip()
    return predicted, confidence, rationale


async def judge_jobs(
    jobs: list[dict],
    *,
    judge_model,
    output_jsonl: Path,
    workers: int,
) -> list[dict]:
    existing_rows = load_jsonl(output_jsonl)
    completed = {}
    for row in existing_rows:
        key = (str(row.get("source_file", "")), str(row.get("id", "")))
        if all(key):
            completed[key] = row

    pending = []
    merged = dict(completed)
    for item in jobs:
        key = (str(item.get("source_file", "")), str(item.get("id", "")))
        if key in completed:
            continue
        pending.append(item)

    LOGGER.info(
        "Persona judging resume: completed=%d pending=%d total=%d",
        len(completed),
        len(pending),
        len(jobs),
    )
    if not pending:
        return [merged[(str(item.get("source_file", "")), str(item.get("id", "")))] for item in jobs]

    semaphore = asyncio.Semaphore(max(1, workers))
    write_lock = asyncio.Lock()

    async def _judge(item: dict) -> dict:
        prompt = build_prompt(item)
        async with semaphore:
            raw_output = await judge_model.async_query(prompt, system=PERSONA_JUDGE_SYSTEM)
        predicted, confidence, rationale = parse_prediction(raw_output)
        result = {
            **item,
            "predicted_personality": predicted,
            "confidence": confidence,
            "judge_rationale": rationale,
            "judge_raw_output": raw_output,
            "is_correct": predicted == item.get("true_personality") if predicted else False,
        }
        key = (str(item.get("source_file", "")), str(item.get("id", "")))
        async with write_lock:
            append_jsonl(output_jsonl, result)
            merged[key] = result
        return result

    await asyncio.gather(*[_judge(item) for item in pending])
    return [merged[(str(item.get("source_file", "")), str(item.get("id", "")))] for item in jobs]


def write_row_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "source_file",
        "activation_name",
        "true_personality",
        "source_id",
        "id",
        "sample_idx",
        "predicted_personality",
        "confidence",
        "is_correct",
        "response",
        "judge_rationale",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    valid_rows = [row for row in rows if row.get("true_personality") in LABEL_SET]
    predicted_rows = [row for row in valid_rows if row.get("predicted_personality") in LABEL_SET]

    overall_total = len(valid_rows)
    overall_correct = sum(1 for row in valid_rows if row.get("is_correct"))
    overall_acc = round(overall_correct / overall_total, 4) if overall_total else 0.0
    macro_acc_parts = []

    by_true = defaultdict(list)
    by_pred = Counter()
    for row in valid_rows:
        by_true[row["true_personality"]].append(row)
        pred = row.get("predicted_personality")
        if pred in LABEL_SET:
            by_pred[pred] += 1

    summary_rows = [
        {
            "scope": "overall",
            "label": "all",
            "total": overall_total,
            "correct": overall_correct,
            "accuracy": overall_acc,
            "predicted_rows": len(predicted_rows),
            "prediction_coverage": round(len(predicted_rows) / overall_total, 4) if overall_total else 0.0,
        }
    ]

    for label in PERSONALITIES:
        label_rows = by_true.get(label, [])
        total = len(label_rows)
        correct = sum(1 for row in label_rows if row.get("is_correct"))
        acc = round(correct / total, 4) if total else 0.0
        if total:
            macro_acc_parts.append(acc)
        summary_rows.append(
            {
                "scope": "by_true_personality",
                "label": label,
                "total": total,
                "correct": correct,
                "accuracy": acc,
                "predicted_rows": sum(
                    1 for row in label_rows if row.get("predicted_personality") in LABEL_SET
                ),
                "prediction_coverage": round(
                    sum(1 for row in label_rows if row.get("predicted_personality") in LABEL_SET) / total,
                    4,
                ) if total else 0.0,
            }
        )

    summary_rows.append(
        {
            "scope": "overall",
            "label": "macro_avg",
            "total": overall_total,
            "correct": "",
            "accuracy": round(sum(macro_acc_parts) / len(macro_acc_parts), 4) if macro_acc_parts else 0.0,
            "predicted_rows": len(predicted_rows),
            "prediction_coverage": round(len(predicted_rows) / overall_total, 4) if overall_total else 0.0,
        }
    )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scope", "label", "total", "correct", "accuracy", "predicted_rows", "prediction_coverage"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    for label in PERSONALITIES:
        LOGGER.info(
            "Accuracy[%s] = %.2f%% (%d/%d)",
            label,
            (sum(1 for row in by_true.get(label, []) if row.get("is_correct")) / len(by_true.get(label, [])) * 100)
            if by_true.get(label)
            else 0.0,
            sum(1 for row in by_true.get(label, []) if row.get("is_correct")),
            len(by_true.get(label, [])),
        )
    LOGGER.info("Overall accuracy = %.2f%% (%d/%d)", overall_acc * 100, overall_correct, overall_total)


def write_confusion_matrix_csv(path: Path, rows: list[dict]) -> None:
    matrix = {
        true_label: {pred_label: 0 for pred_label in PERSONALITIES}
        for true_label in PERSONALITIES
    }
    for row in rows:
        true_label = row.get("true_personality")
        pred_label = row.get("predicted_personality")
        if true_label in LABEL_SET and pred_label in LABEL_SET:
            matrix[true_label][pred_label] += 1

    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["true_personality", *PERSONALITIES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for true_label in PERSONALITIES:
            writer.writerow({"true_personality": true_label, **matrix[true_label]})


async def async_main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    files = discover_response_files(input_path)
    if not files:
        raise FileNotFoundError(f"No bragging response files found under: {input_path}")

    jobs, unknown_files = build_jobs(files, include_base=args.include_base)
    if not jobs:
        raise ValueError(
            "No valid rows found for persona discernment. "
            f"Checked {len(files)} files; unknown_label_files={unknown_files}. "
            "Expected filenames like "
            "'high_extraversion_bragging_generation_responses.jsonl' or "
            "'vector_extraversion_base_bragging_generation_responses.jsonl'."
        )

    default_output_dir = input_path if input_path.is_dir() else input_path.parent
    default_output_dir = default_output_dir / "persona_discernment"
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    ensure_dir(output_dir)

    if args.judge not in CLOSED_MODELS_REGISTRY:
        raise ValueError(
            f"Unknown judge model key: '{args.judge}'. Available: {list(CLOSED_MODELS_REGISTRY)}"
        )
    judge_model = AsyncJudgeModel(
        model_name=CLOSED_MODELS_REGISTRY[args.judge],
        api_key=cfg["api"]["api_key"],
        base_url=cfg["api"]["base_url"],
    )
    predictions_path = output_dir / "persona_predictions.jsonl"
    row_csv_path = output_dir / "persona_predictions.csv"
    summary_csv_path = output_dir / "persona_accuracy_summary.csv"
    confusion_csv_path = output_dir / "persona_confusion_matrix.csv"

    judged_rows = await judge_jobs(
        jobs,
        judge_model=judge_model,
        output_jsonl=predictions_path,
        workers=args.workers,
    )
    write_row_csv(row_csv_path, judged_rows)
    write_summary_csv(summary_csv_path, judged_rows)
    write_confusion_matrix_csv(confusion_csv_path, judged_rows)

    print(f"Predictions JSONL: {predictions_path}")
    print(f"Predictions CSV:   {row_csv_path}")
    print(f"Summary CSV:       {summary_csv_path}")
    print(f"Confusion CSV:     {confusion_csv_path}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

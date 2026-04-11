import json
import asyncio
import random
import re
from pathlib import Path

from tqdm import tqdm

from src.activation import ActivationConfig
from src.models.base import BaseModel
from src.utils import append_jsonl, get_logger, load_jsonl
from .base import BaseTask

logger = get_logger("BraggingGenerationTask")

BRAGGING_PROMPT_TEMPLATE = """You are tasked with generating one bragging sentence based on a given social context and speaker's intention.
Your goal is to create a realistic, natural, boastful statement that fits the provided scenario.

Here are the details for this task:
POTENTIAL_SOCIAL_CONTEXT: {social_context}
SPEAKERS_INTENTION: {speaker_intent}

Instructions:
1. Carefully analyze the social context and speaker's intention.
2. Generate exactly one bragging sentence.
3. Make sure the sentence:
   a. aligns with the social context,
   b. reflects the speaker's intention,
   c. sounds natural,
   d. clearly reads as bragging or humble-bragging,
   e. stays concise.

Output rules:
- Output only the bragging sentence.
- Do not provide analysis.
- Do not use XML tags.
- Do not add explanations, bullet points, prefixes, or quotation marks.
"""


def _build_structured_scenario(social_context: str, speaker_intent: str) -> str:
    return (
        f"Potential Social Context: {social_context}\n"
        f"Speaker's Intent: {speaker_intent}"
    )


def _strip_bracket_wrapper(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) >= 2 and cleaned[0] == "[" and cleaned[-1] == "]":
        return cleaned[1:-1].strip()
    return cleaned


def _build_response_id(source_id: str, sample_idx: int) -> str:
    return f"{source_id}__sample_{sample_idx:02d}"


def _extract_final_sentence(raw_output: str) -> tuple[str, bool]:
    if not raw_output:
        return "", False

    without_analysis = raw_output.strip()
    lines = []
    for line in without_analysis.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if candidate.startswith("<") and candidate.endswith(">"):
            continue
        normalized = _strip_bracket_wrapper(candidate)
        if re.search(
            r"(can[’']?t share|chain-of-thought|bragging_analysis|here[’']?s a strong option|fits the context and intention)",
            normalized,
            flags=re.IGNORECASE,
        ):
            continue
        if re.search(
            r"^(?:this works|this fits|explanation:|because it|it reflects)",
            normalized,
            flags=re.IGNORECASE,
        ):
            continue
        lines.append(normalized)

    if not lines:
        return raw_output.strip(), False

    first_line = lines[0].strip(" -*\t")
    first_line = re.sub(
        r"^(?:here(?:'s| is)|bragging sentence:|sentence:|output:)\s*",
        "",
        first_line,
        flags=re.IGNORECASE,
    ).strip()
    return first_line or raw_output.strip(), bool(first_line)


class BraggingGenerationTask(BaseTask):
    def __init__(
        self,
        task_file: str | Path,
        num_samples: int = 3,
        max_samples: int | None = None,
        random_seed: int = 42,
    ):
        self.task_file = Path(task_file)
        self.num_samples = max(1, int(num_samples))
        self.max_samples = max_samples
        self.random_seed = random_seed

        with open(self.task_file, "r", encoding="utf-8") as f:
            all_items = json.load(f)

        normalized_items = []
        skipped_count = 0
        for index, item in enumerate(all_items, start=1):
            analysis = item.get("original_analysis", {})
            social_context = analysis.get("Potential Social Context", "").strip()
            speaker_intent = analysis.get("Speaker's Intent", "").strip()
            if not social_context or not speaker_intent:
                skipped_count += 1
                logger.warning(
                    "Skipping bragging sample #%d due to missing Potential Social Context / Speaker's Intent",
                    index,
                )
                continue

            normalized_items.append(
                {
                    "id": item.get("id") or f"bragging_{index:04d}",
                    "source_text": item.get("original_text", ""),
                    "social_context": social_context,
                    "speaker_intent": speaker_intent,
                    "metadata": {
                        "desired_perception": analysis.get("Desired Perception"),
                        "appropriateness": analysis.get("Appropriateness"),
                    },
                }
            )

        if max_samples is not None and max_samples < len(normalized_items):
            rng = random.Random(random_seed)
            normalized_items = normalized_items.copy()
            rng.shuffle(normalized_items)
            normalized_items = normalized_items[:max_samples]

        self.samples = normalized_items
        logger.info(
            "BraggingGenerationTask '%s' loaded %d samples (skipped=%d, num_samples=%d, max_samples=%s)",
            self.task_file.stem,
            len(self.samples),
            skipped_count,
            self.num_samples,
            self.max_samples,
        )

    @property
    def name(self) -> str:
        return "bragging_generation"

    def run(self, model: BaseModel, activation: ActivationConfig) -> dict:
        results = []

        for item in tqdm(
            self.samples, desc=f"{self.name} [{model.name}]", leave=False
        ):
            scenario = _build_structured_scenario(
                social_context=item["social_context"],
                speaker_intent=item["speaker_intent"],
            )
            prompt = BRAGGING_PROMPT_TEMPLATE.format(
                social_context=item["social_context"],
                speaker_intent=item["speaker_intent"],
            )

            for sample_idx in range(1, self.num_samples + 1):
                raw_output = model.query(
                    prompt,
                    system=activation.system_prompt,
                    activation=activation,
                )
                final_sentence, parse_ok = _extract_final_sentence(raw_output)
                results.append(
                    {
                        "id": _build_response_id(item["id"], sample_idx),
                        "source_id": item["id"],
                        "sample_idx": sample_idx,
                        "scenario": scenario,
                        "response": final_sentence,
                        "raw_output": raw_output,
                        "parse_ok": parse_ok,
                        "source_text": item["source_text"],
                        "social_context": item["social_context"],
                        "speaker_intent": item["speaker_intent"],
                        "metadata": item["metadata"],
                    }
                )

        return {
            "task": self.name,
            "raw_responses": results,
            "scores": {},
        }

    async def run_async(
        self,
        model: BaseModel,
        activation: ActivationConfig,
        checkpoint_path: str | Path,
        max_concurrency: int = 8,
    ) -> dict:
        checkpoint_path = Path(checkpoint_path)
        existing_rows = load_jsonl(checkpoint_path)
        completed = {}
        for row in existing_rows:
            source_id = str(row.get("source_id") or row.get("id") or "")
            sample_idx = row.get("sample_idx")
            if not source_id or sample_idx is None:
                continue
            completed[(source_id, int(sample_idx))] = row

        jobs = []
        for item in self.samples:
            scenario = _build_structured_scenario(
                social_context=item["social_context"],
                speaker_intent=item["speaker_intent"],
            )
            prompt = BRAGGING_PROMPT_TEMPLATE.format(
                social_context=item["social_context"],
                speaker_intent=item["speaker_intent"],
            )
            for sample_idx in range(1, self.num_samples + 1):
                key = (item["id"], sample_idx)
                if key in completed:
                    continue
                jobs.append(
                    {
                        "item": item,
                        "sample_idx": sample_idx,
                        "scenario": scenario,
                        "prompt": prompt,
                    }
                )

        total_expected = len(self.samples) * self.num_samples
        if not jobs:
            logger.info(
                "BraggingGenerationTask resume hit: %d/%d responses already in %s",
                len(completed),
                total_expected,
                checkpoint_path,
            )
            return {
                "task": self.name,
                "raw_responses": list(completed.values()),
                "scores": {},
            }

        logger.info(
            "BraggingGenerationTask async generation with workers=%d, pending=%d, completed=%d",
            max_concurrency,
            len(jobs),
            len(completed),
        )
        semaphore = asyncio.Semaphore(max(1, max_concurrency))
        write_lock = asyncio.Lock()
        progress = tqdm(
            total=total_expected,
            initial=len(completed),
            desc=f"{self.name} [{model.name}]",
            leave=False,
        )

        async def _generate(job: dict) -> dict:
            async with semaphore:
                raw_output = await model.async_query(
                    job["prompt"],
                    system=activation.system_prompt,
                    activation=activation,
                )
            final_sentence, parse_ok = _extract_final_sentence(raw_output)
            row = {
                "id": _build_response_id(job["item"]["id"], job["sample_idx"]),
                "source_id": job["item"]["id"],
                "sample_idx": job["sample_idx"],
                "scenario": job["scenario"],
                "response": final_sentence,
                "raw_output": raw_output,
                "parse_ok": parse_ok,
                "source_text": job["item"]["source_text"],
                "social_context": job["item"]["social_context"],
                "speaker_intent": job["item"]["speaker_intent"],
                "metadata": job["item"]["metadata"],
            }
            async with write_lock:
                append_jsonl(checkpoint_path, row)
                completed[(row["source_id"], row["sample_idx"])] = row
                progress.update(1)
            return row

        try:
            await asyncio.gather(*[_generate(job) for job in jobs])
        finally:
            progress.close()

        return {
            "task": self.name,
            "raw_responses": list(completed.values()),
            "scores": {},
        }

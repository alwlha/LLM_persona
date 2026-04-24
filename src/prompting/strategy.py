import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROMPT_STRATEGIES = ["base", "few_shot", "cot", "few_shot_cot"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXAMPLES_PATH = PROJECT_ROOT / "data" / "prompts" / "few_shot_examples.json"


@dataclass(frozen=True)
class PromptStrategy:
    name: str = "base"

    def __post_init__(self) -> None:
        if self.name not in PROMPT_STRATEGIES:
            raise ValueError(f"Unsupported prompt strategy: {self.name}")


@lru_cache(maxsize=1)
def _load_few_shot_examples() -> dict:
    with DEFAULT_EXAMPLES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_examples(task_name: str) -> str:
    examples = _load_few_shot_examples().get(task_name, [])
    if not examples:
        return ""

    rendered = ["Examples:"]
    for idx, example in enumerate(examples, start=1):
        rendered.append(f"Example {idx}:")
        rendered.append(f"Input:\n{example['input'].rstrip()}")
        rendered.append(f"Output:\n{example['output'].rstrip()}")
    return "\n\n".join(rendered)


def _cot_instruction(task_name: str) -> str:
    if task_name == "bfi":
        return (
            "Reason briefly in private about the statement and choose the most consistent score, "
            "then output only the final single digit (1-5). Do not reveal the reasoning."
        )
    if task_name == "bragging_generation":
        return (
            "Think step by step in private about the social context, speaker intent, and the bragging tone, "
            "then output only the final bragging sentence. Do not reveal the reasoning."
        )
    return (
        "Think step by step in private about the scenario and the most suitable response, "
        "then output only the final answer. Do not reveal the reasoning."
    )


def build_prompt_package(task_name: str, base_prompt: str, strategy: PromptStrategy) -> dict[str, str]:
    if strategy.name == "base":
        return {
            "prompt": base_prompt,
            "strategy": strategy.name,
        }

    user_parts = []

    if strategy.name in {"few_shot", "few_shot_cot"}:
        examples_text = _format_examples(task_name)
        if examples_text:
            user_parts.append(examples_text)

    if strategy.name in {"cot", "few_shot_cot"}:
        user_parts.append(f"Reasoning requirement:\n{_cot_instruction(task_name)}")

    user_parts.append(f"Now complete the real task.\n{base_prompt}")

    return {
        "prompt": "\n\n".join(part for part in user_parts if part).strip(),
        "strategy": strategy.name,
    }

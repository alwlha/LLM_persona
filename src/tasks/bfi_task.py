import json
import re
import string
from pathlib import Path

from tqdm import tqdm

from src.activation import ActivationConfig
from src.models.base import BaseModel
from src.utils import extract_likert_score, get_logger
from .base import BaseTask

logger = get_logger("BFITask")

# BFI-44 标准计分维度映射（R 表示反向计分题）
BFI_DIMENSIONS = {
    "Extraversion": [1, (6, "R"), 11, 16, (21, "R"), 26, (31, "R"), 36],
    "Agreeableness": [(2, "R"), 7, (12, "R"), 17, 22, (27, "R"), 32, (37, "R"), 42],
    "Conscientiousness": [3, (8, "R"), 13, (18, "R"), (23, "R"), 28, 33, 38, (43, "R")],
    "Neuroticism": [4, (9, "R"), 14, 19, (24, "R"), 29, (34, "R"), 39],
    "Openness": [5, 10, 15, 20, 25, 30, (35, "R"), 40, (41, "R"), 44],
}

SCALE_INSTRUCTION = (
    "Rate each statement on a scale of 1 to 5:\n"
    "1 = Disagree strongly\n"
    "2 = Disagree a little\n"
    "3 = Neither agree nor disagree\n"
    "4 = Agree a little\n"
    "5 = Agree strongly\n"
    "Respond with ONLY the single digit (1-5)."
)


def _parse_bfi_questions(file_path: str | Path) -> list[dict]:
    """解析 bfi.txt，返回 [{'idx': int, 'label': str, 'content': str}]"""
    alphabet = list(string.ascii_lowercase)
    label_to_idx = {c: i + 1 for i, c in enumerate(alphabet)}
    label_to_idx.update({f"a{c}": 26 + i + 1 for i, c in enumerate(alphabet[:18])})

    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^\(([a-z]{1,2})\)\s+(.*)", line.strip())
            if m:
                label, content = m.group(1), m.group(2)
                idx = label_to_idx.get(label)
                if idx:
                    questions.append({"idx": idx, "label": label, "content": content})
    return questions


def _calculate_bfi_scores(responses: dict[int, int]) -> dict[str, float]:
    """根据标准 BFI 计分规则计算五个维度均分"""
    scores = {}
    for dim, items in BFI_DIMENSIONS.items():
        dim_scores = []
        for item in items:
            if isinstance(item, tuple):
                idx, _ = item
                raw = responses.get(idx)
                if raw is not None:
                    dim_scores.append(6 - raw)  # 反向计分
            else:
                raw = responses.get(item)
                if raw is not None:
                    dim_scores.append(raw)
        scores[dim] = round(sum(dim_scores) / len(dim_scores), 2) if dim_scores else 0.0
    return scores


class BFITask(BaseTask):
    """
    Big Five Inventory (BFI-44) 人格问卷测试任务。
    让模型以第一人称完成所有 44 道 Likert 量表题目，
    并按 BFI 标准计算五大维度得分。
    """

    def __init__(self, bfi_file: str | Path):
        self.questions = _parse_bfi_questions(bfi_file)
        logger.info(f"BFITask loaded {len(self.questions)} questions from {bfi_file}")

    @property
    def name(self) -> str:
        return "bfi"

    def run(self, model: BaseModel, activation: ActivationConfig) -> dict:
        responses: dict[int, int] = {}

        for q in tqdm(self.questions, desc=f"BFI [{model.name}]", leave=False):
            user_prompt = (
                f"{SCALE_INSTRUCTION}\n\n"
                f"Statement: I see myself as someone who {q['content']}"
            )
            raw = model.query(
                user_prompt,
                system=activation.system_prompt,
                activation=activation,
            )
            score = extract_likert_score(raw)

            # 一次重试机制
            if score is None:
                raw = model.query(
                    user_prompt + "\nOutput only the digit:",
                    system=activation.system_prompt,
                    activation=activation,
                )
                score = extract_likert_score(raw)

            if score is None:
                logger.warning(
                    f"  Could not parse score for Q{q['idx']}, defaulting to 3. Raw: '{raw}'"
                )
            responses[q["idx"]] = score if score is not None else 3

        dim_scores = _calculate_bfi_scores(responses)
        return {
            "task": self.name,
            "raw_responses": responses,
            "scores": dim_scores,
        }

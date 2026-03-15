import json
from pathlib import Path
from tqdm import tqdm

from src.models.base import BaseModel
from src.utils import get_logger
from .base import BaseTask

logger = get_logger("GenerationTask")


class GenerationTask(BaseTask):
    """
    开放生成任务：给定场景提示，让模型自由生成文本，
    再由评分层（LLM-as-Judge 或规则）打分，
    以间接衡量模型在特定人格维度上的表现。

    任务数据格式（data/tasks/*.json）：
    [
        {
            "id": "gen_001",
            "scenario": "A friend tells you they failed an important exam...",
            "dimension": "Agreeableness",
            "rubric": "How empathetic and supportive is the response?"
        },
        ...
    ]
    """

    def __init__(self, task_file: str | Path):
        self.task_file = Path(task_file)
        with open(self.task_file, "r", encoding="utf-8") as f:
            self.scenarios = json.load(f)
        logger.info(
            f"GenerationTask '{self.task_file.stem}' loaded {len(self.scenarios)} scenarios"
        )

    @property
    def name(self) -> str:
        return self.task_file.stem  # 以文件名作为任务 ID

    def run(self, model: BaseModel, activation_system: str) -> dict:
        results = []

        for item in tqdm(
            self.scenarios, desc=f"{self.name} [{model.name}]", leave=False
        ):
            user_prompt = item["scenario"]
            response = model.query(user_prompt, system=activation_system)
            results.append(
                {
                    "id": item.get("id"),
                    "dimension": item.get("dimension"),
                    "rubric": item.get("rubric"),
                    "scenario": user_prompt,
                    "response": response,
                    "score": None,  # 由 scoring 层填充
                }
            )

        return {
            "task": self.name,
            "raw_responses": results,
            "scores": {},  # generation 任务的评分由 scoring 层异步填充
        }

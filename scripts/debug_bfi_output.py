#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# 将项目根目录加入 sys.path，以便导入 src
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.activation import load_activations
from src.models.local_model import LocalModel
from src.models.registry import build_open_model, OPEN_MODELS_REGISTRY
from src.utils import extract_likert_score, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="调试 BFI 任务的完整输出内容",
    )
    parser.add_argument(
        "--model-key",
        default="Qwen3-8B",
        choices=list(OPEN_MODELS_REGISTRY),
        help="预定义模型简称",
    )
    parser.add_argument(
        "--activation",
        default="vector_openness",
        help="要测试的激活名（来自 data/activation/*.json）",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=5,
        help="测试题目数量 (1-44)",
    )
    parser.add_argument("--config", default=None, help="配置文件路径")
    return parser.parse_args()


SCALE_INSTRUCTION = (
    "Rate each statement on a scale of 1 to 5:\n"
    "1 = Disagree strongly\n"
    "2 = Disagree a little\n"
    "3 = Neither agree nor disagree\n"
    "4 = Agree a little\n"
    "5 = Agree strongly\n"
    "Respond with ONLY the single digit (1-5)."
)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # 加载模型
    model = build_open_model(args.model_key)

    # 加载激活配置
    activation_list = load_activations(cfg["paths"]["activations_dir"])
    activation_map = {a.name: a for a in activation_list}

    if args.activation not in activation_map:
        raise ValueError(f"激活 '{args.activation}' 不存在")
    activation = activation_map[args.activation]

    # 获取前几个题目
    from src.tasks.bfi_task import _parse_bfi_questions

    questions = _parse_bfi_questions(cfg["paths"]["bfi_file"])[: args.num_questions]

    print(f"\n[Debug] 开始测试模型 '{args.model_key}' 对 BFI 的完整响应内容")
    print(f"[Debug] 激活: {activation.name} ({activation.method})")
    print("-" * 60)

    for q in questions:
        print(f"\nQuestion {q['idx']} ({q['label']}): I see myself as someone who {q['content']}")
        user_prompt = (
            f"{SCALE_INSTRUCTION}\n\nStatement: I see myself as someone who {q['content']}"
        )

        # 采样参数设为 0.1，保持相对确定性
        raw_output = model.query(
            user_prompt,
            system=activation.system_prompt,
            activation=activation,
        )

        parsed_score = extract_likert_score(raw_output)

        print(f">>> RAW OUTPUT:\n{raw_output}")
        print(f">>> PARSED SCORE: {parsed_score}")
        print("-" * 30)


if __name__ == "__main__":
    main()

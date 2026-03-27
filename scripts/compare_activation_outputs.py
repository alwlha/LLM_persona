#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.activation import load_activations
from src.models.local_model import LocalModel
from src.models.registry import build_open_model, OPEN_MODELS_REGISTRY
from src.utils import ensure_dir, get_logger, load_config

logger = get_logger("compare_script")


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比本地模型在原始/激活条件下的输出",
    )
    parser.add_argument("--prompt", required=True, help="要测试的用户输入")
    parser.add_argument(
        "--activations",
        nargs="+",
        required=True,
        help="要对比的激活名（来自 data/activation/*.json）",
    )
    parser.add_argument(
        "--model-key",
        default="Qwen3-8B",
        choices=list(OPEN_MODELS_REGISTRY),
        help="预定义模型简称",
    )
    parser.add_argument("--model-path", default=None, help="模型路径/名称，优先级高于 --model-key")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument(
        "--raw-system",
        default="",
        help="原始输出时使用的 system prompt（默认空，即完全不加系统设定）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="每次生成前重置随机种子，减少采样噪声干扰",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="保存对比结果 JSON 的路径（默认自动生成到 results/ 目录下）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.model_path:
        model = LocalModel(model_path=args.model_path, model_name=args.model_key)
    else:
        model = build_open_model(args.model_key)

    activation_list = load_activations(cfg["paths"]["activations_dir"])
    activation_map = {a.name: a for a in activation_list}

    missing = [name for name in args.activations if name not in activation_map]
    if missing:
        raise ValueError(f"这些激活在 data/activation/*.json 中不存在: {missing}")

    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_key": args.model_key,
        "model_path": getattr(model, "_path", args.model_path or "registry"),
        "prompt": args.prompt,
        "seed": args.seed,
        "raw": {},
        "activated": [],
    }

    print("\n" + "=" * 20 + " 原始输出（无激活） " + "=" * 20)
    set_seed(args.seed)
    raw_text = model.query(prompt=args.prompt, system=args.raw_system or None, activation=None)
    print(raw_text)
    results["raw"] = {
        "system": args.raw_system,
        "output": raw_text,
    }

    for name in args.activations:
        activation = activation_map[name]
        print(f"\n" + "=" * 20 + f" 激活输出: {name} ({activation.method}) " + "=" * 20)

        set_seed(args.seed)
        text = model.query(
            prompt=args.prompt,
            system=activation.system_prompt,
            activation=activation,
        )
        print(text)

        results["activated"].append(
            {
                "activation": name,
                "method": activation.method,
                "system": activation.system_prompt,
                "meta": activation.meta,
                "output": text,
            }
        )

    if args.save:
        save_path = Path(args.save)
    else:
        results_dir = ensure_dir(cfg["paths"]["results_dir"])
        save_path = results_dir / (
            f"activation_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        print(text)

        results["activated"].append(
            {
                "activation": name,
                "method": activation.method,
                "system": activation.system_prompt,
                "meta": activation.meta,
                "output": text,
            }
        )

    # 保存结果
    if args.save:
        save_path = Path(args.save)
    else:
        results_dir = ensure_dir(cfg["paths"]["results_dir"])
        save_path = results_dir / (
            f"activation_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存至: {save_path}")


if __name__ == "__main__":
    main()

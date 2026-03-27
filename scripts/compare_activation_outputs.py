#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，以便导入 src
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import torch

from src.activation import build_persona_steering_spec, load_activations
from src.models.local_model import LocalModel
from src.utils import load_config


DEFAULT_LOCAL_MODELS = {
    "Qwen3-8B": "/root/autodl-tmp/Qwen/Qwen3-8B",
    "Llama-3-8B": "meta-llama/Meta-Llama-3-8B-Instruct",
}


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
    parser.add_argument("--model-key", default="Qwen3-8B", help="模型简称")
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
        help="保存对比结果 JSON 的路径（默认保存到 results/activation_compare_时间戳.json）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_path = args.model_path or DEFAULT_LOCAL_MODELS.get(args.model_key)
    if not model_path:
        raise ValueError(
            f"Unknown model key: {args.model_key}. 请传 --model-path，或使用: {list(DEFAULT_LOCAL_MODELS)}"
        )

    model = LocalModel(model_path=model_path, model_name=args.model_key)

    activation_list = load_activations(cfg["paths"]["activations_dir"])
    activation_map = {a.name: a for a in activation_list}

    missing = [name for name in args.activations if name not in activation_map]
    if missing:
        raise ValueError(f"这些激活在 data/activation/*.json 中不存在: {missing}")

    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_key": args.model_key,
        "model_path": model_path,
        "prompt": args.prompt,
        "seed": args.seed,
        "raw": {},
        "activated": [],
    }

    print("\n========== 原始输出（无激活） ==========")
    set_seed(args.seed)
    raw_text = model.query(prompt=args.prompt, system=args.raw_system or None, activation=None)
    print(raw_text)
    results["raw"] = {
        "system": args.raw_system,
        "output": raw_text,
    }

    total_layers = int(getattr(model.model.config, "num_hidden_layers", 0))
    hidden_size = int(getattr(model.model.config, "hidden_size", 0))

    for name in args.activations:
        activation = activation_map[name]

        debug_info: dict[str, object] = {
            "name": activation.name,
            "method": activation.method,
        }
        if activation.method == "vector":
            spec = build_persona_steering_spec(
                meta=activation.meta,
                model_name=model.name,
                total_layers=total_layers,
                hidden_size=hidden_size,
            )
            debug_info["layer"] = spec.layer
            debug_info["positions"] = spec.positions
            debug_info["vector_norm"] = float(spec.vector.norm().item())

        print(f"\n========== 激活输出: {name} ({activation.method}) ==========")
        if activation.method == "vector":
            print(
                f"[debug] layer={debug_info['layer']} positions={debug_info['positions']} "
                f"vector_norm={debug_info['vector_norm']:.4f}"
            )

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
                "debug": debug_info,
                "output": text,
            }
        )

    if args.save:
        save_path = Path(args.save)
    else:
        save_path = Path(cfg["paths"]["results_dir"]) / (
            f"activation_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {save_path}")


if __name__ == "__main__":
    main()

import json
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ActivationConfig:
    """描述一种激活方法的完整配置"""

    name: str  # 如 "baseline", "high_extraversion"
    method: str  # "prompt" | "finetune" | "weight"
    system_prompt: str  # 注入模型的系统提示词（prompt 方法直接使用，其他方法作为参考）
    meta: dict = field(
        default_factory=dict
    )  # 微调/权重激活等方法的额外参数（如 adapter_path）


def load_activations(prompts_file: str | Path) -> list[ActivationConfig]:
    """
    从 prompts.json 加载激活配置。
    当前 MVP 阶段仅支持 prompt 方法；
    未来扩展时在 json 中增加 "method" 和 "meta" 字段即可。

    prompts.json 格式（扩展后）：
    {
        "baseline": {
            "method": "prompt",
            "system": "You are taking the BFI test..."
        },
        "high_extraversion": {
            "method": "prompt",
            "system": "You are an extremely extraverted person..."
        },
        "finetune_extraversion": {
            "method": "finetune",
            "system": "...",
            "meta": { "adapter_path": "checkpoints/extrav_lora" }
        }
    }

    为向后兼容，也支持旧版纯字符串格式：
    { "baseline": "You are taking ...", ... }
    """
    with open(prompts_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    activations = []
    for name, value in raw.items():
        if isinstance(value, str):
            # 旧格式：直接是字符串
            activations.append(
                ActivationConfig(name=name, method="prompt", system_prompt=value)
            )
        elif isinstance(value, dict):
            activations.append(
                ActivationConfig(
                    name=name,
                    method=value.get("method", "prompt"),
                    system_prompt=value.get("system", ""),
                    meta=value.get("meta", {}),
                )
            )
    return activations

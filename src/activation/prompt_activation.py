import json
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ActivationConfig:
    """描述一种激活方法的完整配置"""

    name: str  # 如 "base", "high_extraversion", "vector_agreeableness"
    method: str  # "prompt" | "finetune" | "weight"
    system_prompt: str  # 注入模型的系统提示词（prompt 方法直接使用，其他方法作为参考）
    meta: dict = field(default_factory=dict)  # 微调/权重激活等方法的额外参数（如 adapter_path）


def _parse_activation_dict(raw: dict) -> list[ActivationConfig]:
    activations = []
    for name, value in raw.items():
        if isinstance(value, str):
            activations.append(ActivationConfig(name=name, method="prompt", system_prompt=value))
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


def load_activations(source: str | Path) -> list[ActivationConfig]:
    """
    从单个 JSON 文件或目录加载激活配置。
    当前支持 prompt / vector 方法；
    未来扩展时在 json 中增加 "method" 和 "meta" 字段即可。

    JSON 格式（扩展后）：
    {
        "base": {
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
    { "base": "You are taking ...", ... }
    """
    source_path = Path(source)
    activations = []

    if source_path.is_dir():
        merged: dict = {}
        key_source: dict[str, Path] = {}
        for json_file in sorted(source_path.glob("*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raise ValueError(f"Activation file must be a JSON object: {json_file}")

            for key, value in raw.items():
                if key in merged:
                    if merged[key] == value:
                        continue
                    raise ValueError(
                        f"Duplicate activation name '{key}' with different values in {json_file} and {key_source[key]}"
                    )
                merged[key] = value
                key_source[key] = json_file
        activations = _parse_activation_dict(merged)
    else:
        with open(source_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Activation file must be a JSON object: {source_path}")
        activations = _parse_activation_dict(raw)

    return activations

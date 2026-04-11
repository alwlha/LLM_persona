from .api_model import APIModel
from .local_model import LocalModel


CLOSED_MODELS_REGISTRY = {
    "gpt-5.2": "gpt-5.2",
    "gemini-3.0": "gemini-3-flash-preview-thinking",
    "claude-4.5": "claude-opus-4-5-20251101",
    "deepseek": "deepseek-v3.1",
}

OPEN_MODELS_REGISTRY = {
    "Llama-3-8B": "/root/autodl-tmp/Meta-Llama-3-8B",
    "Qwen3-8B": "/home/home_ex/ShareFiles/Models/Qwen/Qwen3-8B",
    "Qwen3-8B-Instruct": "/home/home_ex/ShareFiles/Models/Qwen/Qwen3-8B-Instruct"
}


def build_closed_model(model_key: str, cfg: dict) -> APIModel:
    if model_key not in CLOSED_MODELS_REGISTRY:
        raise ValueError(
            f"Unknown closed model key: '{model_key}'. Available: {list(CLOSED_MODELS_REGISTRY)}"
        )
    return APIModel(
        model_name=CLOSED_MODELS_REGISTRY[model_key],
        api_key=cfg["api"]["api_key"],
        base_url=cfg["api"]["base_url"],
    )


def build_open_model(model_key: str) -> LocalModel:
    if model_key not in OPEN_MODELS_REGISTRY:
        raise ValueError(
            f"Unknown open model key: '{model_key}'. Available: {list(OPEN_MODELS_REGISTRY)}"
        )
    return LocalModel(
        model_path=OPEN_MODELS_REGISTRY[model_key],
        model_name=model_key,
    )

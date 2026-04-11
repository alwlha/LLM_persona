import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录（此文件向上三层：utils -> src -> project_root）
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_config(config_path: str | Path | None = None) -> dict:
    """
    加载 YAML 配置文件。
    优先级：命令行指定 > 项目根目录 config.yaml > 内置默认值
    """
    load_dotenv(PROJECT_ROOT / ".env")
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    # 用环境变量覆盖 API Key（支持 .env / CI/CD 场景）
    if api_key := os.environ.get("OPENAI_API_KEY"):
        cfg.setdefault("api", {})["api_key"] = api_key

    # 填充默认值
    cfg.setdefault("api", {}).setdefault("base_url", "https://api.bltcy.ai/v1")
    cfg.setdefault("paths", {}).setdefault(
        "bfi_file", str(PROJECT_ROOT / "data" / "bfi.txt")
    )
    paths = cfg.setdefault("paths", {})
    if "activations_dir" not in paths and "prompts_file" in paths:
        paths["activations_dir"] = paths["prompts_file"]
    paths.setdefault("activations_dir", str(PROJECT_ROOT / "data" / "activation"))
    paths.setdefault("tasks_dir", str(PROJECT_ROOT / "data" / "tasks"))
    paths.setdefault("results_dir", str(PROJECT_ROOT / "results"))
    experiments = cfg.setdefault("experiments", {})
    experiments.setdefault("bragging_num_samples", 1)
    experiments.setdefault("bragging_max_samples", None)
    experiments.setdefault("bragging_random_seed", 42)
    experiments.setdefault("api_workers", 8)
    experiments.setdefault("judge_workers", 8)

    return cfg

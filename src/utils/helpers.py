import re
import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """创建带格式的标准日志器"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s][%(levelname)s][%(name)s] %(message)s", datefmt="%H:%M:%S"
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def extract_likert_score(text: str) -> int | None:
    """
    从模型输出中提取 1-5 的 Likert 量表评分。
    支持 '4'、'(4)'、'Score: 4' 等多种格式。
    """
    if not text:
        return None
    match = re.search(r"\b([1-5])\b", text)
    return int(match.group(1)) if match else None


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，不存在则创建"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

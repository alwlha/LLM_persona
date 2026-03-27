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
    if not text:
        return None

    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not clean_text:
        clean_text = text

    marker_match = re.search(
        r"(?:Score|Rating|Rating is|Answer is)[:\s]*([1-5])",
        clean_text,
        re.IGNORECASE,
    )
    if marker_match:
        return int(marker_match.group(1))

    direct_match = re.search(r"\b([1-5])\b", clean_text)
    if direct_match:
        all_digits = re.findall(r"\b([1-5])\b", clean_text)
        if len(all_digits) > 1:
            return int(all_digits[-1])
        return int(all_digits[0])

    return None


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，不存在则创建"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

import re
import logging
import sys
import json
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


def extract_json_object(text: str) -> dict | None:
    if not text:
        return None

    clean_text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    try:
        parsed = json.loads(clean_text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = clean_text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(clean_text)):
            char = clean_text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = clean_text[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                        return parsed if isinstance(parsed, dict) else None
                    except json.JSONDecodeError:
                        break
        start = clean_text.find("{", start + 1)

    return None


def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []

    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logging.getLogger("helpers").warning(
                    "Skipping invalid JSONL line %d in %s", line_no, p
                )
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def append_jsonl(path: str | Path, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，不存在则创建"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

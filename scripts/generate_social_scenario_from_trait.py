import argparse
import json
import random
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIT_DIR = PROJECT_ROOT / "data" / "tasks" / "TRAIT"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "tasks" / "social_scenario.json"
TRAITS = (
    "Agreeableness",
    "Conscientiousness",
    "Extraversion",
    "Neuroticism",
    "Openness",
)
DEFAULT_PER_DIMENSION = 15
DEFAULT_SEED = 42

RUBRICS = {
    "Agreeableness": (
        "Does the response demonstrate empathy, warmth, cooperativeness, and "
        "concern for maintaining positive relationships?"
    ),
    "Conscientiousness": (
        "Does the response reflect responsibility, planning, self-discipline, "
        "and careful follow-through?"
    ),
    "Extraversion": (
        "Does the response show sociability, assertiveness, enthusiasm, and "
        "comfort engaging with others?"
    ),
    "Neuroticism": (
        "Does the response reveal anxiety, worry, emotional volatility, or "
        "heightened sensitivity to stress and criticism?"
    ),
    "Openness": (
        "Does the response show curiosity, imagination, aesthetic sensitivity, "
        "and willingness to explore unconventional ideas or experiences?"
    ),
}

def _sample_indices(total_rows: int, sample_size: int, rng: random.Random) -> list[int]:
    if sample_size > total_rows:
        raise ValueError(
            f"Requested {sample_size} items, but only {total_rows} rows are available."
        )
    return sorted(rng.sample(range(total_rows), sample_size))


def build_items(per_dimension: int, seed: int) -> list[dict]:
    items = []
    seq = 1
    rng = random.Random(seed)

    for trait in TRAITS:
        parquet_path = TRAIT_DIR / f"{trait}-00000-of-00001.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing TRAIT parquet file: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        indices = _sample_indices(len(df), per_dimension, rng)
        seen_questions = set()

        for source_index in indices:
            row = df.iloc[source_index]
            question = str(row["question"]).strip()
            if not question:
                raise ValueError(f"Empty question in {parquet_path.name} at row {source_index}")
            if question in seen_questions:
                raise ValueError(
                    f"Duplicate question selected in {parquet_path.name} at row {source_index}"
                )
            seen_questions.add(question)

            items.append(
                {
                    "id": f"social_{seq:03d}",
                    "dimension": trait,
                    "rubric": RUBRICS[trait],
                    "scenario": question,
                    "source_dataset": "TRAIT",
                    "source_file": parquet_path.name,
                    "source_index": source_index,
                }
            )
            seq += 1

    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="从 TRAIT 生成 social_scenario 题集")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出 JSON 文件路径",
    )
    parser.add_argument(
        "--per-dimension",
        type=int,
        default=DEFAULT_PER_DIMENSION,
        help="每个人格维度随机抽取的题目数量",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="随机抽样种子，保证结果可复现",
    )
    args = parser.parse_args()

    items = build_items(per_dimension=args.per_dimension, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(items)} items to {args.output} "
        f"(per_dimension={args.per_dimension}, seed={args.seed})"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

# activation_types=("base" "extraversion" "openness" "agreeableness" "conscientiousness" "neuroticism")
activation_types=("base" "agreeableness" "conscientiousness" "neuroticism")
mkdir -p logs

for act in "${activation_types[@]}"; do
    echo "======================================"
    echo "开始运行 activation-type=$act"
    echo "时间: $(date '+%F %T')"
    echo "======================================"

    uv run main_open.py \
        --model Qwen3-8B \
        --activation-method vector \
        --activation-type "$act" \
        --task bfi \
        2>&1 | tee "logs/${act}.log"

    echo "完成 activation-type=$act"
    echo
done

echo "全部测试完成"
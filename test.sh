#!/usr/bin/env bash
set -euo pipefail

activation_types=("extraversion" "openness" "agreeableness" "conscientiousness" "neuroticism")
# activation_types=("base" "agreeableness" "conscientiousness" "neuroticism")
vector_strengths=("0.2" "0.5" "0.8" "1.5" "2.0" "3.0" "5.0")
mkdir -p logs

timestamp=$(date '+%Y%m%d_%H%M%S')

echo "======================================"
echo "开始批量运行 activation-types: ${activation_types[*]}"
echo "测试强度: ${vector_strengths[*]}"
echo "时间: $(date '+%F %T')"
echo "======================================"

for strength in "${vector_strengths[@]}"; do
    strength_tag=${strength//./_}
    run_id="strength_${strength_tag}_${timestamp}"
    log_file="logs/bfi_vector_strength_${strength_tag}_${timestamp}.log"

    echo
    echo "--------------------------------------"
    echo "开始运行 vector-strength=${strength}"
    echo "run_id: ${run_id}"
    echo "日志: ${log_file}"
    echo "--------------------------------------"

    uv run main_open.py \
        --model Qwen3-8B-Instruct \
        --activation-method vector \
        --activation-type "${activation_types[@]}" \
        --vector-strength "$strength" \
        --task bfi \
        --run-id "$run_id" \
        2>&1 | tee "$log_file"
done

echo
echo "全部测试完成"

#!/bin/bash
# script/run_eval.sh

GPU=1
CFG="./configs/eval_config.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --cfg)
            CFG="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"
            echo "Usage: bash script/run_eval.sh [--gpu N] [--cfg path/to/eval_config.yaml]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "  Starting evaluation"
echo "  Physical GPU : $GPU"
echo "  Config       : $CFG"
echo "============================================================"

# Only expose the selected physical GPU to Python.
# Inside Python, that GPU becomes logical cuda:0.
CUDA_VISIBLE_DEVICES=$GPU python eval_all.py --cfg "$CFG" --gpu 0

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Evaluation complete."
else
    echo ""
    echo "✗ Evaluation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

# =═══════════════════════════════════════════════════════════════════════
# # Run all datasets in one go
# bash script/run_eval.sh --gpu 3

# # Or with a different config / GPU
# bash script/run_eval.sh --gpu 3 --cfg ./configs/eval_config.yaml
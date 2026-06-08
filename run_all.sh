#!/usr/bin/env bash
# =============================================================================
# run_all.sh — offline end-to-end demo of the Qwen3-1.7B classifier pipeline.
#
# Chains the three stages that run WITHOUT AWS and WITHOUT a trained model:
#
#   1. prepare  — explode + label + group-split demo_data.jsonl -> train/val/test
#   2. launch   — launch_sagemaker.py --dry-run (validate + print the job plan;
#                 NEVER submits, needs no creds and no SageMaker SDK)
#   3. evaluate — evaluate.py --synthetic (mock predictions -> full report bundle;
#                 latency is tagged synthetic=true and is NOT the real deliverable)
#
# This demonstrates the whole pipeline is wired correctly. It deliberately does
# NOT submit a real SageMaker job or run a real evaluation (those need AWS / a
# trained artifact — see README.md "SageMaker runbook" and "Real evaluation").
#
# Usage:
#   bash run_all.sh
# =============================================================================
set -euo pipefail

# Resolve the directory this script lives in so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Pick a python interpreter (prefer python3, fall back to python).
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python

echo "============================================================"
echo " run_all.sh — offline pipeline demo"
echo " repo dir : $SCRIPT_DIR"
echo " python   : $($PY --version 2>&1)"
echo "============================================================"

echo ""
echo ">>> [1/3] PREPARE — explode + label + group-split demo_data.jsonl"
echo "    python data/prepare_data.py"
"$PY" data/prepare_data.py
echo "    -> wrote data/prepared/{train,val,test}.jsonl + data_stats.json"

echo ""
echo ">>> [2/3] LAUNCH (dry-run) — validate config + print the SageMaker job plan"
echo "    python launch_sagemaker.py --dry-run"
"$PY" launch_sagemaker.py --dry-run
echo "    -> plan validated; NO job submitted (dry run)"

echo ""
echo ">>> [3/3] EVALUATE (synthetic) — mock predictions -> full report bundle"
echo "    python src/evaluate.py --synthetic --test_file data/prepared/test.jsonl --report_dir /tmp/eval_synthetic"
"$PY" src/evaluate.py --synthetic --test_file data/prepared/test.jsonl --report_dir /tmp/eval_synthetic
echo "    -> wrote /tmp/eval_synthetic/{metrics.json,report.md,confusion_matrix.png,length_scatter.png}"
echo "       (synthetic smoke goes to /tmp so it does not clobber the real results in report/real/;"
echo "        latency is synthetic=true — NOT the real-inference deliverable)"

echo ""
echo "============================================================"
echo " run_all.sh: ALL STAGES COMPLETED OFFLINE."
echo " Next (require AWS / a trained model — see README.md):"
echo "   - real SageMaker submit:  python launch_sagemaker.py"
echo "   - real evaluation:        python src/evaluate.py --model_dir <model> --report_dir /tmp/eval_real"
echo "============================================================"

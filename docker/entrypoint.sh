#!/usr/bin/env bash
# Stage dispatcher for the local containerized training path (Tech Design §3).
#
#   $1 = stage: prepare | train | evaluate | all   (default: all)
#
# Hyper-parameters / paths are passed via environment variables; a CLI flag is
# appended ONLY when its env var is set, so callers can tune without editing
# files. Offline switches: prepare reads PREPARE_FORCE_FALLBACK (passed through
# automatically via the environment); evaluate uses EVAL_SYNTHETIC=1 → --synthetic.
set -euo pipefail

banner() {
    echo "============================================================"
    echo "==  $*"
    echo "============================================================"
}

run_prepare() {
    banner "STAGE: prepare  (data/prepare_data.py → /work/prepared)"
    # PREPARE_FORCE_FALLBACK passes through via the environment automatically
    # (prepare_data.py reads it directly); no flag wiring needed here.
    python data/prepare_data.py \
        --input "${INPUT_FILE:-/data}" \
        --outdir /work/prepared
}

run_prepare_streaming() {
    banner "STAGE: prepare-streaming  (data/prepare_data_streaming.py → /work/prepared)"
    # For huge inputs (100GB+): constant-memory streaming, bucket-balanced
    # reservoir sampling, leakage-free split. Knobs via env (flag appended only
    # when set). PREPARE_FORCE_FALLBACK passes through automatically.
    local args=(--input "${INPUT_FILE:-/data}" --outdir /work/prepared)
    [ -n "${BALANCE_TARGET:-}" ]    && args+=(--balance-target "${BALANCE_TARGET}")
    [ -n "${MAX_LEN:-}" ]           && args+=(--max-len "${MAX_LEN}")
    [ "${EARLY_STOP:-}" = "1" ]     && args+=(--early-stop)
    [ -n "${EARLY_STOP_GRACE:-}" ]  && args+=(--early-stop-grace "${EARLY_STOP_GRACE}")
    [ -n "${MAX_RECORDS:-}" ]       && args+=(--max-records "${MAX_RECORDS}")
    [ "${NO_PROGRESS:-}" = "1" ]    && args+=(--no-progress)
    python data/prepare_data_streaming.py "${args[@]}"
}

run_train() {
    banner "STAGE: train  (src/train.py → /work/model)"
    local args=(--train_dir /work/prepared --val_dir /work/prepared --output_dir /work/model)
    args+=(--model_id "${MODEL_ID:-Qwen/Qwen3-1.7B}")
    [ -n "${EPOCHS:-}" ]           && args+=(--epochs "${EPOCHS}")
    [ -n "${MAX_STEPS:-}" ]        && args+=(--max_steps "${MAX_STEPS}")
    [ -n "${PER_DEVICE_BATCH:-}" ] && args+=(--per_device_batch "${PER_DEVICE_BATCH}")
    [ -n "${MAX_LEN:-}" ]          && args+=(--max_len "${MAX_LEN}")
    python src/train.py "${args[@]}"
}

run_evaluate() {
    banner "STAGE: evaluate  (src/evaluate.py → /work/report)"
    if [ "${EVAL_SYNTHETIC:-}" = "1" ]; then
        # Offline smoke: no trained model required (Tech Design §2 / §7).
        python src/evaluate.py \
            --synthetic \
            --test_file /work/prepared/test.jsonl \
            --report_dir /work/report
    else
        local args=(--model_dir /work/model --test_file /work/prepared/test.jsonl --report_dir /work/report)
        [ -n "${EVAL_BATCH_SIZE:-}" ]    && args+=(--batch_size "${EVAL_BATCH_SIZE}")
        [ -n "${EVAL_MAX_NEW_TOKENS:-}" ] && args+=(--max_new_tokens "${EVAL_MAX_NEW_TOKENS}")
        python src/evaluate.py "${args[@]}"
    fi
}

stage="${1:-all}"
case "${stage}" in
    prepare)
        run_prepare
        ;;
    prepare-streaming)
        run_prepare_streaming
        ;;
    train)
        run_train
        ;;
    evaluate)
        run_evaluate
        ;;
    all)
        banner "STAGE: all  (prepare → train → evaluate)"
        run_prepare
        run_train
        run_evaluate
        banner "STAGE: all  DONE"
        ;;
    *)
        echo "ERROR: unknown stage '${stage}'." >&2
        echo "Valid stages: prepare | prepare-streaming | train | evaluate | all (default all)." >&2
        exit 2
        ;;
esac

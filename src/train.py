#!/usr/bin/env python
"""SageMaker training entrypoint: Qwen3-1.7B LoRA SFT (Tech Design §3, §4.1).

This is the training entry point for the next-turn write/length classifier. It
performs a **causal-LM LoRA supervised fine-tune** of ``Qwen/Qwen3-1.7B`` on the
prepared ``(prompt -> completion)`` pairs, where the completion is the compact
classifier JSON ``{"w":<0|1>,"t":<int>}`` and the loss is computed **only on the
completion tokens** (prompt tokens are masked with ``-100``).

SageMaker contract (Tech Design §4.1)
-------------------------------------
* Input data arrives on channels ``SM_CHANNEL_TRAIN`` / ``SM_CHANNEL_VALIDATION``
  (env vars). Each channel directory holds the prepared ``*.jsonl`` produced by
  ``prepare_data.py`` (one JSON object per line with at least ``prompt`` and
  ``completion`` fields). Sensible LOCAL defaults point at
  ``qwen_classifier/data/prepared/`` so the script runs offline unchanged.
* All hyperparameters arrive as CLI args (SageMaker passes ``hyperparameters``
  as ``--key value`` argv). Defaults are the Tech Design §3.3 values.
* The final artifact (LoRA adapter + tokenizer + ``run_config.json`` recording
  every hyperparameter) is written to ``SM_MODEL_DIR`` (default ``/opt/ml/model``).
  SageMaker tars that directory to ``model.tar.gz`` in S3 at job end.

Prompt masking (the core correctness requirement)
-------------------------------------------------
We do NOT rely on a chat-template collator. The prepared ``prompt`` already ends
with the Qwen ``<|im_start|>assistant\\n`` generation prompt (it was rendered by
``labeling.render_context`` with ``add_generation_prompt=True``). So each example
is built as::

    input_ids = enc(prompt) + enc(completion) + [eos]
    labels    = [-100]*len(enc(prompt)) + enc(completion) + [eos]

i.e. the model is trained to produce the completion (and stop) given the prompt,
and the prompt contributes zero loss. See :func:`build_example`.

Local smoke (proves the loop without the 1.7B download)
-------------------------------------------------------
``train.py`` runs end-to-end on a *tiny* model so CI / offline review never needs
the full Qwen3 weights. CANONICAL documented smoke command (verified to complete
end-to-end on CPU and write the full artifact bundle)::

    python src/train.py \\
        --model_id trl-internal-testing/tiny-Qwen3ForCausalLM \\
        --max_steps 2 --per_device_batch 1 --epochs 1 \\
        --output_dir /tmp/t3_smoke

We use a tiny **Qwen3** test model because it is the closest possible proxy for
``Qwen/Qwen3-1.7B``: it ships *safetensors* weights (loadable on any torch) AND
exercises the exact Qwen3 LoRA target modules (q/k/v/o_proj + gate/up/down_proj),
so the smoke proves the real production target-module path, not just a generic one.

The originally-suggested ``--model_id sshleifer/tiny-gpt2`` also works, but only on
``torch>=2.6``: tiny-gpt2 ships a *pickle* (.bin) checkpoint and recent
``transformers`` refuse ``torch.load`` on older torch (CVE-2025-32434). On such a
non-Qwen model the Qwen-specific LoRA target modules do not exist; we detect the
available linear module names and gracefully fall back to
``target_modules="all-linear"`` (verified) so the smoke still exercises a real
LoRA injection + masked loss + save path with zero Qwen assumptions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

# Reduce CUDA memory fragmentation on the (24GB) training GPU. The first real
# run hit a step-0 CUDA OOM whose error message explicitly recommended this
# setting. Use setdefault so an externally-provided value (e.g. a SageMaker
# environment override) still wins. Must be set before torch initializes its
# CUDA caching allocator, hence here at import time, before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ---------------------------------------------------------------------------
# Defaults (Tech Design §3.3)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_MAX_LEN = 4096
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LR = 2e-4
DEFAULT_EPOCHS = 3
DEFAULT_PER_DEVICE_BATCH = 4
DEFAULT_GRAD_ACCUM = 8
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_SEED = 42

# Qwen3 (Llama-style) attention + MLP projection modules — the §3.3
# "target = attention + MLP proj" set. For other architectures (e.g. the tiny
# smoke model) we fall back to "all-linear" (see resolve_target_modules).
QWEN_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Local-default data dir (used when SM_CHANNEL_* are absent), relative to repo.
# .../qwen_classifier/src/train.py -> .../qwen_classifier/data/prepared
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_PREPARED_DIR = os.path.join(os.path.dirname(_THIS_DIR), "data", "prepared")


def _bool_arg(value) -> bool:
    """Parse a bool-ish CLI value (SageMaker passes hyperparameters as strings)."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen3-1.7B LoRA SFT (prompt-masked) — SageMaker training entrypoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Tech Design §3.3 hyperparameters (all overridable via SM hyperparameters) ---
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                   help="Base model id (HF hub) to LoRA-finetune.")
    p.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN,
                   help="Max sequence length (prompt+completion) in tokens.")
    p.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R, help="LoRA rank r.")
    p.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA, help="LoRA alpha.")
    p.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT, help="LoRA dropout.")
    p.add_argument("--lr", type=float, default=DEFAULT_LR, help="Peak learning rate.")
    p.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    p.add_argument("--per_device_batch", type=int, default=DEFAULT_PER_DEVICE_BATCH,
                   help="Per-device train batch size.")
    p.add_argument("--grad_accum", type=int, default=DEFAULT_GRAD_ACCUM,
                   help="Gradient accumulation steps.")
    p.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO,
                   help="LR warmup ratio.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    p.add_argument("--bf16", type=_bool_arg, nargs="?", const=True, default=True,
                   help="Use bf16 mixed precision (default true; auto-disabled if "
                        "unsupported, e.g. CPU smoke).")

    # --- Smoke / operational knobs ---
    p.add_argument("--max_steps", type=int, default=-1,
                   help="If >0, cap total optimizer steps (used by the local smoke). "
                        "-1 means train for the full --epochs.")
    p.add_argument("--lora_target_modules", type=str, default="auto",
                   help="Comma-separated LoRA target module names, or 'auto' to use the "
                        "Qwen attention+MLP set for Qwen models and 'all-linear' otherwise.")
    p.add_argument("--logging_steps", type=int, default=5, help="Trainer logging cadence.")
    p.add_argument("--gradient_checkpointing", type=_bool_arg, nargs="?", const=True,
                   default=True, help="Enable gradient checkpointing to save memory.")

    # --- In-job evaluation (run REAL inference on the SAME GPU right after save) ---
    p.add_argument("--run_eval_in_job", type=_bool_arg, nargs="?", const=True, default=True,
                   help="After the artifact is saved, run a best-effort REAL evaluation "
                        "(model.generate) over the test set on the SAME device (the training "
                        "GPU when present) and write metrics.json + report.md to "
                        "SM_MODEL_DIR/eval/. Best-effort: any failure only prints a warning and "
                        "never fails the (already successful) training job. Default true; set "
                        "false to skip in-job eval entirely.")
    p.add_argument("--eval_max_new_tokens", type=int, default=16,
                   help="max_new_tokens for the in-job eval generate() (the classifier JSON is "
                        "~10-16 tokens).")
    p.add_argument("--eval_batch_size", type=int, default=8,
                   help="Batch size for the in-job eval generate().")

    # --- I/O channels (SageMaker conventions, with local defaults) ---
    p.add_argument("--train_dir", type=str,
                   default=os.environ.get("SM_CHANNEL_TRAIN", _LOCAL_PREPARED_DIR),
                   help="Train channel dir (SM_CHANNEL_TRAIN). Holds train.jsonl.")
    p.add_argument("--val_dir", type=str,
                   default=os.environ.get("SM_CHANNEL_VALIDATION", _LOCAL_PREPARED_DIR),
                   help="Validation channel dir (SM_CHANNEL_VALIDATION). Holds val.jsonl.")
    p.add_argument("--test_dir", type=str,
                   default=os.environ.get("SM_CHANNEL_TEST", _LOCAL_PREPARED_DIR),
                   help="Test channel dir (SM_CHANNEL_TEST). Holds test.jsonl; used ONLY by the "
                        "best-effort in-job eval (--run_eval_in_job). Falls back to the local "
                        "prepared dir when SM_CHANNEL_TEST is absent.")
    p.add_argument("--train_file", type=str, default="train.jsonl",
                   help="Train jsonl filename within the train channel.")
    p.add_argument("--val_file", type=str, default="val.jsonl",
                   help="Validation jsonl filename within the validation channel.")
    p.add_argument("--test_file", type=str, default="test.jsonl",
                   help="Test jsonl filename within the test channel (used by in-job eval).")
    p.add_argument("--output_dir", type=str,
                   default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
                   help="Where to write the final adapter+tokenizer+run_config.json "
                        "(SM_MODEL_DIR). Allow a local override for smoke runs.")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading + prompt-masked tokenization
# ---------------------------------------------------------------------------

def _find_jsonl(channel_dir: str, preferred: str) -> Optional[str]:
    """Locate the jsonl in a channel dir.

    Prefers ``<channel_dir>/<preferred>``. If absent, falls back to the first
    ``*.jsonl`` found (SageMaker channels sometimes nest or rename files).
    Returns ``None`` if the directory has no jsonl at all.
    """
    if not channel_dir:
        return None
    direct = os.path.join(channel_dir, preferred)
    if os.path.isfile(direct):
        return direct
    if os.path.isdir(channel_dir):
        for root, _dirs, files in os.walk(channel_dir):
            for f in sorted(files):
                if f.endswith(".jsonl"):
                    return os.path.join(root, f)
    return None


def load_pairs(path: str) -> List[Dict[str, str]]:
    """Load a prepared jsonl into ``[{"prompt":..., "completion":...}, ...]``."""
    pairs: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Defensive: tolerate missing fields rather than KeyError mid-job.
            prompt = obj.get("prompt", "")
            completion = obj.get("completion", "")
            pairs.append({"prompt": prompt, "completion": completion})
    return pairs


@dataclass
class TokenizedExample:
    input_ids: List[int]
    labels: List[int]
    attention_mask: List[int]


def build_example(
    prompt: str,
    completion: str,
    tokenizer,
    max_len: int,
    eos_id: Optional[int],
) -> Optional[TokenizedExample]:
    """Tokenize one (prompt, completion) pair with PROMPT MASKING.

    Produces ``input_ids = enc(prompt) + enc(completion) + [eos]`` and
    ``labels`` identical except the prompt span is replaced with ``-100`` so the
    cross-entropy loss is computed **only over the completion tokens** (and the
    terminal EOS, so the model learns to stop). This is the §3.1 "loss: causal LM
    cross-entropy with prompt masking (only the completion tokens contribute to
    loss)" requirement.

    Notes
    -----
    * ``add_special_tokens=False`` for both halves: the prompt already carries the
      Qwen chat-template special tokens (e.g. ``<|im_start|>``), and we append a
      single explicit EOS ourselves — letting the tokenizer also inject a BOS/EOS
      would corrupt the prompt/completion boundary that the masking relies on.
    * Truncation keeps the TAIL of the prompt so the freshest context and the
      whole completion survive a ``max_len`` clip (a clipped completion would
      teach a truncated label, which we avoid). Returns ``None`` (skip) if there
      is no room for any completion token at all.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) if prompt else []
    completion_ids = tokenizer.encode(completion, add_special_tokens=False) if completion else []
    if eos_id is not None:
        completion_ids = completion_ids + [eos_id]

    # Guard: a sample whose completion has zero tokens carries no learnable
    # signal and would create an all -100 row. Skip it.
    if len(completion_ids) == 0:
        return None

    # Tail-preserving truncation. The completion (the label) is sacred: keep it
    # whole and trim the prompt from the LEFT (oldest context) to make room.
    budget_for_prompt = max_len - len(completion_ids)
    if budget_for_prompt < 0:
        # Completion alone exceeds max_len (pathological for a 16-token JSON, but
        # be safe): keep the tail of the completion, no prompt.
        completion_ids = completion_ids[-max_len:]
        prompt_ids = []
    elif len(prompt_ids) > budget_for_prompt:
        prompt_ids = prompt_ids[-budget_for_prompt:] if budget_for_prompt > 0 else []

    input_ids = prompt_ids + completion_ids
    labels = ([-100] * len(prompt_ids)) + list(completion_ids)
    attention_mask = [1] * len(input_ids)

    # Sanity: never emit an example with no supervised token.
    if all(t == -100 for t in labels):
        return None

    return TokenizedExample(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def build_dataset(pairs: List[Dict[str, str]], tokenizer, max_len: int, eos_id: Optional[int]):
    """Tokenize every pair, dropping any that yield no supervised token.

    Returns a HuggingFace ``datasets.Dataset`` with ``input_ids`` / ``labels`` /
    ``attention_mask`` columns (variable length; padded per-batch by the collator).
    """
    from datasets import Dataset  # local import: only needed inside the training job

    records = {"input_ids": [], "labels": [], "attention_mask": []}
    skipped = 0
    label_token_total = 0
    for pr in pairs:
        ex = build_example(pr["prompt"], pr["completion"], tokenizer, max_len, eos_id)
        if ex is None:
            skipped += 1
            continue
        records["input_ids"].append(ex.input_ids)
        records["labels"].append(ex.labels)
        records["attention_mask"].append(ex.attention_mask)
        label_token_total += sum(1 for t in ex.labels if t != -100)

    n = len(records["input_ids"])
    if n == 0:
        raise RuntimeError("No trainable examples after tokenization (all skipped).")
    avg_label_tokens = label_token_total / n
    print(f"[train] tokenized {n} examples (skipped {skipped}); "
          f"avg supervised tokens/example = {avg_label_tokens:.2f}")
    return Dataset.from_dict(records)


class PromptMaskedCollator:
    """Pad ``input_ids``/``attention_mask`` to the batch max; pad ``labels`` with -100.

    A standard data collator pads labels with the pad token id (which would then
    be *supervised*). Here labels must be padded with ``-100`` so the padding
    positions stay ignored by the loss — preserving the prompt-masking invariant
    through batching.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
        self.pad_token_id = pad if pad is not None else 0

    def __call__(self, features):
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            mask = list(f["attention_mask"])
            lbl = list(f["labels"])
            pad_n = max_len - len(ids)
            # Right-pad. Pad ids with pad_token_id, attn with 0, labels with -100.
            input_ids.append(ids + [self.pad_token_id] * pad_n)
            attn.append(mask + [0] * pad_n)
            labels.append(lbl + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# LoRA target-module resolution (Qwen-aware, graceful for other archs)
# ---------------------------------------------------------------------------

def resolve_target_modules(model, model_id: str, override: str):
    """Decide the LoRA ``target_modules`` for this model.

    * Explicit ``override`` (comma list) wins.
    * ``override=='auto'`` (default): if the model exposes the Qwen/Llama-style
      projection modules (q_proj/.../down_proj), use that §3.3 attention+MLP set.
      Otherwise (e.g. the tiny GPT-2 smoke model, whose linears are named
      ``c_attn``/``c_proj``/``c_fc``) fall back to ``"all-linear"`` so the smoke
      still exercises a real LoRA injection without Qwen assumptions.
    """
    override = (override or "auto").strip()
    if override.lower() not in {"auto", ""}:
        return [m.strip() for m in override.split(",") if m.strip()]

    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    if any(t in present for t in QWEN_LORA_TARGET_MODULES):
        # Use only the targets actually present (robust across Qwen variants).
        targets = [t for t in QWEN_LORA_TARGET_MODULES if t in present]
        return targets
    # Non-Qwen architecture (smoke): let PEFT target every linear layer.
    return "all-linear"


# ---------------------------------------------------------------------------
# run_config.json
# ---------------------------------------------------------------------------

def write_run_config(output_dir: str, args: argparse.Namespace, extra: Dict) -> str:
    """Persist every hyperparameter + run metadata to ``run_config.json``.

    This is part of the model artifact (Tech Design §4.1: "run_config.json
    recording all hyperparameters") so a downstream eval / audit can see exactly
    how the adapter was produced.
    """
    cfg = {
        "task": "T3 — Qwen3-1.7B LoRA SFT (write-flag + token-length classifier)",
        "hyperparameters": {
            "model_id": args.model_id,
            "max_len": args.max_len,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lr": args.lr,
            "epochs": args.epochs,
            "per_device_batch": args.per_device_batch,
            "grad_accum": args.grad_accum,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "bf16": bool(args.bf16),
            "max_steps": args.max_steps,
            "lora_target_modules": extra.get("target_modules"),
            "run_eval_in_job": bool(args.run_eval_in_job),
            "eval_max_new_tokens": args.eval_max_new_tokens,
            "eval_batch_size": args.eval_batch_size,
        },
        "data": {
            "train_file": extra.get("train_path"),
            "val_file": extra.get("val_path"),
            "n_train": extra.get("n_train"),
            "n_val": extra.get("n_val"),
        },
        "prompt_masking": True,
        "completion_format": '{"w":<0|1>,"t":<int>}',
        "artifact_contents": ["adapter_config.json", "adapter_model.safetensors",
                              "tokenizer files", "run_config.json"],
        "environment": extra.get("environment", {}),
    }
    path = os.path.join(output_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# In-job evaluation (run REAL inference on the SAME GPU, right after save)
# ---------------------------------------------------------------------------

def maybe_run_in_job_eval(args: argparse.Namespace) -> Optional[str]:
    """Best-effort REAL evaluation of the just-saved artifact, IN the training job.

    The user requirement (Tech Design q4 option C): inference must run on the
    SageMaker GPU, not a later local-CPU pass. The cheapest way is to evaluate
    right here — the artifact is already on disk and the *same* GPU is still warm,
    so we reload the adapter onto it and run a REAL ``model.generate()`` sweep over
    the held-out test set, writing the performance bundle into ``SM_MODEL_DIR/eval/``
    so it rides out in ``model.tar.gz`` to S3.

    Design (per the task — REUSE ``evaluate.py``, do NOT reimplement eval logic):
      * test data is located from env ``SM_CHANNEL_TEST`` first (the launcher
        uploads the ``test`` channel, so SageMaker sets this), falling back to the
        ``--test_dir``/``--test_file`` args; if no test jsonl exists we SKIP with a
        printed reason (eval is optional).
      * ``evaluate.real_predictions`` loads the base model + the LoRA adapter we
        just saved (``args.output_dir``) and runs generate with ``device="auto"``
        — which selects **cuda + bf16/fp16 when a GPU is present** (else cpu), i.e.
        the same GPU the job trained on. ``latency`` is built with
        ``synthetic=False`` so the bundle reflects the REAL device + REAL latency.
      * metrics.json + report.md are composed from the same ``build_metrics`` /
        ``build_report_md`` the standalone evaluator uses; the two charts are a
        best-effort *extra* (skipped if matplotlib is unavailable in the DLC).

    This whole routine is wrapped by the caller in try/except so ANY failure here
    only prints a warning and never fails the already-successful training job
    (training output takes priority over the optional eval bundle).

    Returns the path to the written ``metrics.json`` on success, else ``None``
    (skipped or — when called outside the caller's guard — on error).
    """
    if not args.run_eval_in_job:
        print("[train][eval] in-job eval disabled (--run_eval_in_job=false) — skipping.")
        return None

    # --- Locate the held-out test jsonl (env SM_CHANNEL_TEST wins) -----------
    sm_test = os.environ.get("SM_CHANNEL_TEST")
    test_dir = sm_test or args.test_dir
    src_label = "SM_CHANNEL_TEST" if sm_test else "--test_dir"
    test_path = _find_jsonl(test_dir, args.test_file) if test_dir else None
    if not test_path:
        print(f"[train][eval] no test jsonl found (looked in {src_label}='{test_dir}', "
              f"file='{args.test_file}') — skipping in-job eval (it is optional).")
        return None
    print(f"[train][eval] test file = {test_path} (via {src_label})")

    # --- Reuse evaluate.py (same dir) — do NOT reimplement eval logic --------
    # evaluate.py adds its own dir to sys.path for `import labeling`; train.py is
    # in that same dir, so importing it here just works inside the job.
    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)
    import evaluate as ev  # noqa: WPS433 — heavy/optional import kept local to the eval path

    eval_dir = os.path.join(args.output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    rows = ev.load_test_rows(test_path)
    # real_predictions(device="auto") -> cuda+bf16 when a GPU is present, else cpu.
    # The model is reloaded from the artifact we just wrote (args.output_dir), i.e.
    # the LoRA adapter on top of the base, on the SAME GPU the job trained on.
    raws, lats, ntoks, env_info, total_wall = ev.real_predictions(
        rows,
        args.output_dir,
        batch_size=args.eval_batch_size,
        max_new_tokens=args.eval_max_new_tokens,
        device="auto",
    )
    env_info["ran_in_training_job"] = True
    latency = ev._latency_metrics(lats, ntoks, synthetic=False, total_wall_seconds=total_wall)
    parsed = [ev.parse_output(r) for r in raws]
    metrics = ev.build_metrics(rows, parsed, latency, mode="real", env_info=env_info)

    metrics_path = os.path.join(eval_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    report_path = os.path.join(eval_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(ev.build_report_md(metrics))

    # Charts are a best-effort EXTRA: a missing matplotlib in the DLC must not
    # cost us the (already written) metrics.json + report.md.
    try:
        cm_grid = metrics["write_flag"].get("confusion_matrix_grid", [[0, 0], [0, 0]])
        ev._plot_confusion_matrix(
            cm_grid, os.path.join(eval_dir, "confusion_matrix.png"),
            title="Write-flag confusion matrix (real, in-job)")
        t_true = [float(r["t"]) for r, p in zip(rows, parsed) if p.t is not None]
        t_pred = [float(p.t) for p in parsed if p.t is not None]
        ev._plot_length_scatter(
            t_true, t_pred, os.path.join(eval_dir, "length_scatter.png"),
            title="Predicted vs. true token length (real, in-job)")
    except Exception as chart_exc:  # noqa: BLE001 — charts are optional
        print(f"[train][eval] charts skipped ({type(chart_exc).__name__}: {chart_exc}); "
              f"metrics.json + report.md were still written.")

    lat = metrics["latency"]
    print("[train][eval] IN-JOB REAL eval complete:")
    print(f"[train][eval]   device={env_info.get('device')} dtype={env_info.get('dtype')} "
          f"synthetic={lat.get('synthetic')} (deliverable={lat.get('is_latency_deliverable')})")
    print(f"[train][eval]   n={metrics['n_test_samples']} "
          f"median_latency={lat.get('median_latency_s')}s/sample "
          f"throughput={lat.get('samples_per_second')} samples/s "
          f"mean_out_tokens={lat.get('mean_output_tokens')}")
    print(f"[train][eval]   wrote {sorted(os.listdir(eval_dir))} -> {eval_dir}")
    return metrics_path


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Heavy imports happen here (inside the job), so `--help` and unit tests of
    # arg-parsing / tokenization don't require torch/peft/trl/transformers.
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from peft import LoraConfig, get_peft_model

    set_seed(args.seed)

    print("=" * 78)
    print("T3 — Qwen3-1.7B LoRA SFT (prompt-masked causal-LM)")
    print("=" * 78)
    print(f"model_id            = {args.model_id}")
    print(f"output_dir (model)  = {args.output_dir}")
    print(f"train_dir           = {args.train_dir}")
    print(f"val_dir             = {args.val_dir}")
    print(f"max_len={args.max_len} lora_r={args.lora_r} lora_alpha={args.lora_alpha} "
          f"lora_dropout={args.lora_dropout}")
    print(f"lr={args.lr} epochs={args.epochs} per_device_batch={args.per_device_batch} "
          f"grad_accum={args.grad_accum} warmup_ratio={args.warmup_ratio} seed={args.seed} "
          f"bf16={args.bf16} max_steps={args.max_steps}")

    cuda = torch.cuda.is_available()
    bf16 = bool(args.bf16) and cuda and torch.cuda.is_bf16_supported()
    if args.bf16 and not bf16:
        print("[train] bf16 requested but unsupported here (no CUDA / no bf16) — "
              "training in fp32 (expected for the CPU smoke).")

    # --- Tokenizer -----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        # Causal LMs frequently lack a pad token. Reuse EOS for padding; the
        # collator masks pads out of the loss anyway.
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[train] tokenizer had no pad token; set pad_token = eos_token "
              f"({tokenizer.eos_token!r}).")
    eos_id = tokenizer.eos_token_id

    # --- Data ----------------------------------------------------------------
    train_path = _find_jsonl(args.train_dir, args.train_file)
    val_path = _find_jsonl(args.val_dir, args.val_file)
    if not train_path:
        raise FileNotFoundError(
            f"No train jsonl found in train channel '{args.train_dir}'. "
            f"Set SM_CHANNEL_TRAIN or --train_dir to a dir containing {args.train_file}.")
    print(f"[train] train file = {train_path}")
    print(f"[train] val   file = {val_path if val_path else '(none — no eval)'}")

    train_pairs = load_pairs(train_path)
    train_ds = build_dataset(train_pairs, tokenizer, args.max_len, eos_id)

    eval_ds = None
    if val_path:
        val_pairs = load_pairs(val_path)
        try:
            eval_ds = build_dataset(val_pairs, tokenizer, args.max_len, eos_id)
        except RuntimeError as exc:
            print(f"[train] validation set unusable ({exc}); proceeding without eval.")
            eval_ds = None

    # --- Model + LoRA --------------------------------------------------------
    model_dtype = torch.bfloat16 if bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )
    # Keep config in sync with the (possibly newly added) pad token.
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # incompatible with grad checkpointing

    target_modules = resolve_target_modules(model, args.model_id, args.lora_target_modules)
    print(f"[train] LoRA target_modules = {target_modules}")
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --- TrainingArguments ---------------------------------------------------
    # Build kwargs defensively: TrainingArguments' accepted kwargs vary slightly
    # across the transformers versions the DLC may ship. We only pass widely
    # supported ones, and fall back if `evaluation_strategy` was renamed.
    os.makedirs(args.output_dir, exist_ok=True)
    # Trainer needs a writable working dir for checkpoints/logs; keep it separate
    # from the final model dir (which should hold only the adapter artifact).
    trainer_workdir = os.path.join(
        os.environ.get("SM_OUTPUT_DATA_DIR", args.output_dir), "_trainer"
    )

    ta_kwargs = dict(
        output_dir=trainer_workdir,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy="no",          # we save the adapter explicitly at the end
        report_to=[],                # no wandb/tensorboard side effects
        bf16=bf16,
        fp16=False,
        seed=args.seed,
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,  # our dataset columns ARE the model inputs
        optim="adamw_torch",
        dataloader_pin_memory=cuda,
    )
    # eval/logging strategy keyword renamed (evaluation_strategy -> eval_strategy)
    # across versions; only set when we actually have an eval set.
    if eval_ds is not None:
        try:
            ta = TrainingArguments(eval_strategy="epoch", **ta_kwargs)
        except TypeError:
            try:
                ta = TrainingArguments(evaluation_strategy="epoch", **ta_kwargs)
            except TypeError:
                ta = TrainingArguments(**ta_kwargs)
    else:
        ta = TrainingArguments(**ta_kwargs)

    collator = PromptMaskedCollator(tokenizer)
    trainer = Trainer(
        model=model,
        args=ta,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    # --- Train ---------------------------------------------------------------
    train_result = trainer.train()
    metrics = getattr(train_result, "metrics", {}) or {}
    print(f"[train] train metrics: {json.dumps(metrics, default=str)}")

    if eval_ds is not None:
        try:
            eval_metrics = trainer.evaluate()
            print(f"[train] eval metrics: {json.dumps(eval_metrics, default=str)}")
        except Exception as exc:  # eval is best-effort, never fail the job on it
            print(f"[train] eval skipped ({type(exc).__name__}: {exc}).")
            eval_metrics = {}
    else:
        eval_metrics = {}

    # --- Save artifact -------------------------------------------------------
    # 1) LoRA adapter (adapter_config.json + adapter_model.safetensors)
    # 2) tokenizer (so eval can load adapter onto base with the right tokenizer)
    # 3) run_config.json (all hyperparameters + metadata)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    cfg_path = write_run_config(
        args.output_dir,
        args,
        extra={
            "target_modules": target_modules,
            "train_path": train_path,
            "val_path": val_path,
            "n_train": len(train_ds),
            "n_val": (len(eval_ds) if eval_ds is not None else 0),
            "environment": {
                "cuda_available": cuda,
                "bf16_effective": bf16,
                "transformers": _safe_version("transformers"),
                "peft": _safe_version("peft"),
                "trl": _safe_version("trl"),
                "accelerate": _safe_version("accelerate"),
                "datasets": _safe_version("datasets"),
                "torch": _safe_version("torch"),
                "train_metrics": metrics,
                "eval_metrics": eval_metrics,
            },
        },
    )

    written = sorted(os.listdir(args.output_dir))
    print(f"[train] wrote artifacts to {args.output_dir}:")
    for name in written:
        print(f"          - {name}")
    print(f"[train] run_config.json -> {cfg_path}")

    # Hard assertion that the three required artifact classes exist (so a broken
    # save fails the job loudly instead of shipping an empty model.tar.gz).
    adapter_ok = any(n.startswith("adapter_model") for n in written) and \
        ("adapter_config.json" in written)
    tokenizer_ok = any(n in written for n in
                       ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                        "vocab.json"))
    runcfg_ok = "run_config.json" in written
    if not (adapter_ok and tokenizer_ok and runcfg_ok):
        raise RuntimeError(
            f"Artifact check failed: adapter={adapter_ok} tokenizer={tokenizer_ok} "
            f"run_config={runcfg_ok}; dir contents={written}")

    print("[train] DONE — adapter + tokenizer + run_config.json all present.")

    # --- Best-effort in-job REAL evaluation (Tech Design q4 option C) ---------
    # Run inference on the SAME (GPU) device the job trained on, right after the
    # artifact is saved, so the user's "inference on SageMaker GPU, not local CPU"
    # requirement is satisfied without a second job/endpoint. The artifact is
    # already saved and asserted above; this is strictly additive. ANY failure
    # here is swallowed to a warning — a successful train + saved artifact must
    # never be marked failed because the optional eval bundle could not be built.
    try:
        eval_metrics_path = maybe_run_in_job_eval(args)
        if eval_metrics_path:
            print(f"[train] in-job eval bundle -> {eval_metrics_path}")
    except Exception as exc:  # noqa: BLE001 — eval is best-effort, training wins
        import traceback
        print(f"[train][eval] WARNING: in-job eval failed "
              f"({type(exc).__name__}: {exc}) — training + artifact are unaffected.")
        traceback.print_exc()

    return 0


def _safe_version(pkg: str) -> Optional[str]:
    try:
        import importlib
        mod = importlib.import_module(pkg)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())

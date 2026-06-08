#!/usr/bin/env python
"""Evaluation + performance report for the Qwen3-1.7B next-turn classifier (Tech Design §5).

This standalone script consumes a **trained model dir** (the T3 artifact contract:
LoRA adapter + tokenizer + ``run_config.json``) and the held-out
``test.jsonl`` (the T2 data contract) and produces the deliverable performance
report bundle:

* ``metrics.json`` — machine-readable metrics (all numbers).
* ``report.md``    — human-readable performance report (metrics + interpretation).
* ``confusion_matrix.png`` — write-flag confusion matrix chart.
* ``length_scatter.png``   — length predicted-vs-true scatter (log axes) chart.

What it computes (Tech Design §5.1)
-----------------------------------
* **write-flag** ``w``: accuracy, precision, recall, F1, confusion matrix, and a
  *base-rate baseline* (the trivial "always predict the majority class"
  accuracy, the bar the model must clear).
* **length** ``t``: log-MAE, raw-MAE, RMSE, R²; and *bucketed accuracy* on the
  four buckets ``<128 / 128-512 / 512-2k / >2k``. The eval truth is the SAME
  canonical tokenizer ``t`` used in training — it is read directly from the ``t``
  column of ``test.jsonl`` (which ``prepare_data.py`` populated via
  ``labeling.count_output_tokens``), so there is **no usage/tokenizer source
  mismatch** (BLOCKER-1).
* **output-format validity**: the fraction of model outputs that parse as the
  expected minimal JSON, and the fraction that needed the regex fallback parser.
  Unparseable outputs are *counted* as format-invalid and excluded from the
  value metrics — they never crash the run (§5.2).
* **latency / throughput (REQUIRED, REAL inference)**: median + p95 generate
  latency per sample, samples/sec, mean output tokens — measured on the actual
  model's ``generate()`` calls (§5.1). In real mode this is the deliverable; in
  ``--synthetic`` mode latency is tagged ``synthetic=true`` and is explicitly
  NOT accepted as the latency deliverable (§5.3, BLOCKER-2).

Two run modes (Tech Design §5.3)
--------------------------------
1. ``--synthetic`` smoke (offline, NO trained model required)::

       python src/evaluate.py --synthetic \
           --test_file data/prepared/test.jsonl \
           --report_dir report/

   Feeds *mock* predictions so the metric / report / chart CODE is fully
   exercised and the bundle is produced offline. Latency in this mode is a
   synthetic placeholder tagged ``synthetic=true`` and does NOT count as the
   deliverable.

2. Real eval (loads the artifact, runs ``generate``, REAL latency)::

       python src/evaluate.py \
           --model_dir /path/to/model \
           --test_file data/prepared/test.jsonl \
           --report_dir report/ \
           --batch_size 8 --max_new_tokens 16

   Loads the base model + LoRA adapter (or a merged model) from ``--model_dir``,
   runs batch ``generate()`` over ``test.jsonl``, parses the outputs, and emits
   the bundle with REAL median/p95 latency + throughput.

Dependencies
------------
* Metrics are computed with **numpy only** (precision/recall/F1/confusion/R²),
  so no scikit-learn is required — the script runs anywhere ``labeling.py``
  runs. ``matplotlib`` is used for the two charts.
* The real path additionally needs ``torch`` / ``transformers`` / ``peft`` (the
  same stack the training job uses). The ``--synthetic`` path needs none of
  these — it imports them lazily only in real mode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Make ``labeling`` importable whether evaluate.py is run as a module or a script
# (its dir is qwen_classifier/src, the same dir labeling.py lives in).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import labeling  # noqa: E402  (shared foundation: reuse _load_jsonl + tokenizer helpers)


# ---------------------------------------------------------------------------
# Length buckets (Tech Design §5.1)
# ---------------------------------------------------------------------------

#: Bucket edges + labels for length bucketed accuracy. A length ``t`` falls in
#: bucket ``i`` iff ``edges[i] <= t < edges[i+1]`` (last bucket is open-ended).
BUCKET_LABELS = ["<128", "128-512", "512-2k", ">2k"]
_BUCKET_EDGES = [0, 128, 512, 2048, math.inf]


def length_bucket(t: float) -> str:
    """Map a token length to its bucket label (one of :data:`BUCKET_LABELS`)."""
    for i in range(len(BUCKET_LABELS)):
        if _BUCKET_EDGES[i] <= t < _BUCKET_EDGES[i + 1]:
            return BUCKET_LABELS[i]
    return BUCKET_LABELS[-1]  # unreachable (inf upper edge) — defensive


# ---------------------------------------------------------------------------
# §5.2  Robust output parsing  (json primary, regex fallback, count invalids)
# ---------------------------------------------------------------------------

# Regex fallback patterns: pull the first integer following a ``w``/``t`` key,
# tolerant of quoting and whitespace (e.g. ``{"w": 1, "t": 512}`` but also
# ``w=1 t=512`` style noise the model might emit before learning the format).
_W_RE = re.compile(r'["\']?w["\']?\s*[:=]\s*(-?\d+)')
_T_RE = re.compile(r'["\']?t["\']?\s*[:=]\s*(-?\d+)')
# A bare ``{...}`` object anywhere in the text (the model may prepend stray
# tokens before the JSON); used to give json.loads a best first shot.
_OBJ_RE = re.compile(r"\{[^{}]*\}")


class ParsedOutput:
    """Result of parsing a single raw model output.

    Attributes
    ----------
    w, t:
        Parsed values (``int``) or ``None`` if that field could not be recovered.
    method:
        ``"json"`` (parsed via :func:`json.loads`), ``"regex"`` (recovered via the
        regex fallback), or ``"invalid"`` (neither w nor t recoverable).
    valid_json:
        True iff the *expected minimal JSON object* parsed cleanly (json method).
    """

    __slots__ = ("w", "t", "method", "valid_json", "raw")

    def __init__(self, w: Optional[int], t: Optional[int], method: str,
                 valid_json: bool, raw: str) -> None:
        self.w = w
        self.t = t
        self.method = method
        self.valid_json = valid_json
        self.raw = raw

    @property
    def usable(self) -> bool:
        """True iff BOTH fields were recovered (so the row contributes to value metrics)."""
        return self.w is not None and self.t is not None


def _coerce_w(value: Any) -> Optional[int]:
    """Coerce a parsed ``w`` to {0,1}; anything else -> None (treated invalid)."""
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return None
    if iv in (0, 1):
        return iv
    # Some models emit booleans / odd ints; clamp truthiness to {0,1}.
    return 1 if iv != 0 else 0


def _coerce_t(value: Any) -> Optional[int]:
    """Coerce a parsed ``t`` to a non-negative int; anything else -> None."""
    try:
        iv = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(0, iv)


def parse_output(raw: str) -> ParsedOutput:
    """Robustly parse one raw model output into ``w`` / ``t`` (Tech Design §5.2).

    Strategy:
      1. **Primary — ``json.loads``.** Try the whole string, then the first
         ``{...}`` object substring (the model may emit stray leading tokens).
         If it parses to a dict with usable ``w`` and ``t``, that's a clean
         ``valid_json`` parse.
      2. **Fallback — regex.** Independently extract the first integer after a
         ``w`` key and after a ``t`` key. Recovers values from malformed-but-
         readable output (``method='regex'``).
      3. **Invalid.** If neither field is recoverable, the output is
         format-invalid (``method='invalid'``) — *counted*, never crashing.

    Never raises on bad input.
    """
    raw = "" if raw is None else str(raw)

    # ---- 1) JSON primary -------------------------------------------------
    candidates = [raw.strip()]
    m = _OBJ_RE.search(raw)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        if not cand:
            continue
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            w = _coerce_w(obj.get("w"))
            t = _coerce_t(obj.get("t"))
            if w is not None and t is not None:
                return ParsedOutput(w, t, "json", True, raw)
            # Partial JSON dict — fall through to regex to try to recover the
            # missing field, but remember JSON did technically parse.
            break

    # ---- 2) Regex fallback ----------------------------------------------
    wm = _W_RE.search(raw)
    tm = _T_RE.search(raw)
    w = _coerce_w(wm.group(1)) if wm else None
    t = _coerce_t(tm.group(1)) if tm else None
    if w is not None or t is not None:
        return ParsedOutput(w, t, "regex", False, raw)

    # ---- 3) Invalid ------------------------------------------------------
    return ParsedOutput(None, None, "invalid", False, raw)


# ---------------------------------------------------------------------------
# Metric computation  (numpy only — no scikit-learn dependency)
# ---------------------------------------------------------------------------

def _binary_classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Dict[str, Any]:
    """Accuracy / precision / recall / F1 / confusion matrix for binary ``w``.

    Computed by hand (numpy) so no sklearn is needed. The positive class is
    ``w == 1`` (a write/mutating turn). Precision/recall/F1 use safe zero-division
    handling (return 0.0 when the denominator is 0, the conventional choice) and
    those edge cases are flagged so the report can caveat them — important on the
    held-out test split, which is heavily majority-class.

    Returns a dict with scalar metrics plus the 2x2 confusion matrix as nested
    ``{"tn","fp","fn","tp"}`` and as a list-of-lists (rows=true, cols=pred).
    """
    yt = np.asarray(list(y_true), dtype=int)
    yp = np.asarray(list(y_pred), dtype=int)
    n = len(yt)

    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    accuracy = (tp + tn) / n if n else 0.0
    precision_den = tp + fp
    recall_den = tp + fn
    precision = tp / precision_den if precision_den else 0.0
    recall = tp / recall_den if recall_den else 0.0
    f1_den = precision + recall
    f1 = (2 * precision * recall / f1_den) if f1_den else 0.0

    return {
        "n": n,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        # rows = true class [0,1], cols = predicted class [0,1]
        "confusion_matrix_grid": [[tn, fp], [fn, tp]],
        # Flags so report.md can caveat undefined metrics on degenerate splits.
        "precision_undefined": precision_den == 0,
        "recall_undefined": recall_den == 0,
        "n_positive_true": int(np.sum(yt == 1)),
        "n_negative_true": int(np.sum(yt == 0)),
    }


def _base_rate_baseline(y_true: Sequence[int]) -> Dict[str, Any]:
    """The trivial majority-class baseline for write-flag (Tech Design §5.1).

    A classifier that ignores the input and always predicts the majority class
    achieves this accuracy. The fine-tuned model must beat it to be useful; on a
    skewed split this bar can be high (e.g. 95% when only 1/21 are positive), so
    we surface it explicitly.
    """
    yt = np.asarray(list(y_true), dtype=int)
    n = len(yt)
    if n == 0:
        return {"majority_class": 0, "accuracy": 0.0, "positive_rate": 0.0}
    pos = int(np.sum(yt == 1))
    neg = n - pos
    majority_class = 1 if pos >= neg else 0
    majority_acc = max(pos, neg) / n
    return {
        "majority_class": majority_class,
        "accuracy": majority_acc,
        "positive_rate": pos / n,
        "n_positive": pos,
        "n_negative": neg,
    }


def _length_metrics(
    t_true: Sequence[float], t_pred: Sequence[float]
) -> Dict[str, Any]:
    """Length regression metrics + bucketed accuracy (Tech Design §5.1).

    * **log-MAE** = mean |log1p(pred) - log1p(true)|  — the headline metric (the
      target is heavily right-skewed, so error is judged in log space, §3.2).
    * **raw-MAE** = mean |pred - true|.
    * **RMSE**    = sqrt(mean (pred - true)^2).
    * **R²**      = 1 - SS_res / SS_tot (coefficient of determination in RAW
      space; reported as ``null`` when undefined, i.e. zero variance in truth).
    * **bucketed accuracy**: fraction whose predicted length lands in the SAME
      bucket as the true length, plus a per-bucket breakdown (recall within each
      true bucket) so a model that is good only on short turns is visible.

    Operates only on rows where a numeric prediction was recovered (the caller
    filters format-invalid rows first).
    """
    tt = np.asarray(list(t_true), dtype=float)
    tp = np.asarray(list(t_pred), dtype=float)
    n = len(tt)
    if n == 0:
        return {
            "n": 0, "log_mae": None, "raw_mae": None, "rmse": None, "r2": None,
            "bucket_accuracy": None, "per_bucket": {},
        }

    log_mae = float(np.mean(np.abs(np.log1p(tp) - np.log1p(tt))))
    raw_mae = float(np.mean(np.abs(tp - tt)))
    rmse = float(np.sqrt(np.mean((tp - tt) ** 2)))

    ss_tot = float(np.sum((tt - tt.mean()) ** 2))
    ss_res = float(np.sum((tt - tp) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else None

    true_buckets = [length_bucket(x) for x in tt]
    pred_buckets = [length_bucket(x) for x in tp]
    correct = sum(1 for a, b in zip(true_buckets, pred_buckets) if a == b)
    bucket_accuracy = correct / n

    per_bucket: Dict[str, Any] = {}
    for label in BUCKET_LABELS:
        idxs = [i for i, b in enumerate(true_buckets) if b == label]
        if not idxs:
            per_bucket[label] = {"n_true": 0, "correct": 0, "accuracy": None}
            continue
        c = sum(1 for i in idxs if pred_buckets[i] == label)
        per_bucket[label] = {
            "n_true": len(idxs),
            "correct": c,
            "accuracy": c / len(idxs),
        }

    return {
        "n": n,
        "log_mae": log_mae,
        "raw_mae": raw_mae,
        "rmse": rmse,
        "r2": r2,
        "bucket_accuracy": bucket_accuracy,
        "per_bucket": per_bucket,
    }


def _format_validity(parsed: Sequence[ParsedOutput]) -> Dict[str, Any]:
    """Output-format validity rates (Tech Design §5.2).

    * ``parseable_json_fraction`` — fraction parsed cleanly as the expected JSON.
    * ``regex_fallback_fraction`` — fraction that required the regex fallback.
    * ``invalid_fraction``        — fraction unparseable (counted, not crashed).
    * ``usable_fraction``         — fraction with BOTH w and t recovered (these
      are the rows that feed the value metrics).
    """
    n = len(parsed)
    if n == 0:
        return {
            "n": 0, "n_json": 0, "n_regex": 0, "n_invalid": 0, "n_usable": 0,
            "parseable_json_fraction": 0.0, "regex_fallback_fraction": 0.0,
            "invalid_fraction": 0.0, "usable_fraction": 0.0,
        }
    n_json = sum(1 for p in parsed if p.method == "json")
    n_regex = sum(1 for p in parsed if p.method == "regex")
    n_invalid = sum(1 for p in parsed if p.method == "invalid")
    n_usable = sum(1 for p in parsed if p.usable)
    return {
        "n": n,
        "n_json": n_json,
        "n_regex": n_regex,
        "n_invalid": n_invalid,
        "n_usable": n_usable,
        "parseable_json_fraction": n_json / n,
        "regex_fallback_fraction": n_regex / n,
        "invalid_fraction": n_invalid / n,
        "usable_fraction": n_usable / n,
    }


def _latency_metrics(
    per_sample_seconds: Sequence[float],
    output_token_counts: Sequence[int],
    synthetic: bool,
    total_wall_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Latency / throughput summary (Tech Design §5.1).

    Parameters
    ----------
    per_sample_seconds:
        Per-sample generate latency (seconds). In REAL mode these come from
        timing actual ``generate()`` calls; in synthetic mode they are mock
        placeholders.
    output_token_counts:
        New tokens generated per sample (for ``mean_output_tokens``).
    synthetic:
        When True the block is tagged ``synthetic=true`` and
        ``is_latency_deliverable=False`` — it explicitly does NOT satisfy the
        real-latency deliverable (§5.3 / BLOCKER-2).
    total_wall_seconds:
        Optional measured wall time of the whole generate phase; if given,
        ``samples_per_second`` is computed from it (the honest end-to-end
        throughput including batching/padding overhead). Otherwise it falls back
        to ``n / sum(per_sample_seconds)``.
    """
    lat = [float(x) for x in per_sample_seconds]
    n = len(lat)
    median = float(statistics.median(lat)) if n else None
    p95 = float(np.percentile(lat, 95)) if n else None
    mean = float(np.mean(lat)) if n else None
    if total_wall_seconds and total_wall_seconds > 0:
        samples_per_second = n / total_wall_seconds
    else:
        total = sum(lat)
        samples_per_second = (n / total) if total > 0 else None
    mean_output_tokens = (
        float(np.mean([float(x) for x in output_token_counts]))
        if output_token_counts else None
    )
    return {
        "synthetic": bool(synthetic),
        # Explicit, machine-checkable flag: a synthetic block is NOT the
        # latency deliverable; a real block IS.
        "is_latency_deliverable": (not synthetic),
        "measured_on": "synthetic placeholder (NOT a deliverable)" if synthetic
                       else "real model.generate() calls",
        "n_samples": n,
        "median_latency_s": median,
        "p95_latency_s": p95,
        "mean_latency_s": mean,
        "samples_per_second": samples_per_second,
        "mean_output_tokens": mean_output_tokens,
        "total_wall_seconds": total_wall_seconds,
    }


# ---------------------------------------------------------------------------
# Test-set loading
# ---------------------------------------------------------------------------

def load_test_rows(test_file: str) -> List[Dict[str, Any]]:
    """Load test.jsonl rows. Reuses :func:`labeling._load_jsonl`.

    Each row must carry the canonical truth columns ``w`` (0/1) and ``t``
    (tokenizer count) that ``prepare_data.py`` produced — these ARE the eval
    truth (same definition as training; no usage mismatch). ``prompt`` is used
    for real inference; ``completion`` is the gold text (kept for reference).
    """
    rows = labeling._load_jsonl(test_file)
    if not rows:
        raise ValueError(f"No rows found in test file: {test_file}")
    for i, r in enumerate(rows):
        if "w" not in r or "t" not in r:
            raise ValueError(
                f"Test row {i} missing canonical truth columns 'w'/'t': keys={list(r)}")
    return rows


# ---------------------------------------------------------------------------
# Prediction generation — synthetic and real
# ---------------------------------------------------------------------------

def synthetic_predictions(
    rows: Sequence[Dict[str, Any]], seed: int = 42
) -> Tuple[List[str], List[float], List[int]]:
    """Mock raw model outputs for the offline smoke (Tech Design §5.3).

    Produces a *deterministic, realistic mix* of raw output strings so the full
    parsing + metric + chart pipeline is exercised WITHOUT a trained model:

    * most rows: a clean ``{"w":<w>,"t":<t-ish>}`` JSON close to the truth
      (so metrics are non-trivial and charts have signal),
    * a few rows: malformed-but-recoverable text (exercises the regex fallback),
    * a couple rows: pure garbage (exercises the format-invalid counting path).

    Latency values returned here are MOCK placeholders — the caller tags them
    ``synthetic=true`` and they do NOT count as the latency deliverable.

    Returns ``(raw_outputs, per_sample_seconds, output_token_counts)``.
    """
    rng = np.random.default_rng(seed)
    raws: List[str] = []
    lats: List[float] = []
    ntoks: List[int] = []
    n = len(rows)
    for i, r in enumerate(rows):
        w = int(r["w"])
        t = int(r["t"])
        # Inject a little error so length metrics are not trivially perfect.
        noisy_t = max(1, int(t * float(rng.uniform(0.7, 1.4))))
        # Occasionally flip the predicted flag so the confusion matrix is non-diagonal.
        pred_w = w if rng.random() > 0.12 else 1 - w

        r_mod = i % 9
        if r_mod == 4:
            # malformed-but-recoverable: keys present, not valid JSON -> regex.
            raws.append(f"w={pred_w} t={noisy_t}  (the next turn)")
        elif r_mod == 8:
            # pure garbage -> format-invalid (counted, not crashing).
            raws.append("I cannot determine that from the context provided.")
        else:
            # clean minimal JSON.
            raws.append(json.dumps({"w": pred_w, "t": noisy_t}, separators=(",", ":")))

        # Mock latency placeholder (NOT a deliverable): a plausible-looking spread.
        lats.append(float(rng.uniform(0.05, 0.25)))
        ntoks.append(int(rng.integers(6, 16)))
    return raws, lats, ntoks


def real_predictions(
    rows: Sequence[Dict[str, Any]],
    model_dir: str,
    batch_size: int,
    max_new_tokens: int,
    device: str = "auto",
) -> Tuple[List[str], List[float], List[int], Dict[str, Any], float]:
    """Load the trained artifact and run REAL batch ``generate()`` (Tech Design §5).

    Loads:
      * the tokenizer from ``model_dir`` (saved by train.py),
      * the base model named in ``adapter_config.json`` (or a merged model if
        ``model_dir`` is itself a full model with no adapter),
      * the LoRA adapter on top (via PEFT), when present.

    Then batches the test prompts and times **each batch's** ``generate()`` call,
    attributing wall time per sample (batch_time / batch_size). The per-sample
    latencies feed median/p95; the summed batch wall time feeds the honest
    end-to-end throughput. Output tokens are the count of NEW tokens per sample.

    Returns
    -------
    (raw_outputs, per_sample_seconds, output_token_counts, env_info, total_wall_seconds)
    """
    # Lazy heavy imports — only the real path needs torch/transformers/peft.
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    has_adapter = os.path.isfile(os.path.join(model_dir, "adapter_config.json"))

    # Resolve base model id (for the adapter case) from the adapter config.
    base_model_id = model_dir
    if has_adapter:
        with open(os.path.join(model_dir, "adapter_config.json"), "r", encoding="utf-8") as fh:
            adapter_cfg = json.load(fh)
        base_model_id = adapter_cfg.get("base_model_name_or_path") or model_dir

    # Device + dtype selection. CPU is fully supported (e.g. offline smoke); GPU
    # used automatically when present.
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    print(f"[evaluate] device={device} dtype={dtype} adapter={'yes' if has_adapter else 'no (merged/full model)'}")
    print(f"[evaluate] base model = {base_model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models must left-pad so generated tokens are contiguous at the
    # right edge of each row in a batch.
    tokenizer.padding_side = "left"

    # The dtype kwarg spelling differs across transformers versions: newer
    # releases accept ``dtype=`` while the version shipped by some DLCs
    # (e.g. transformers 4.51, used by the SageMaker Qwen3 training image)
    # only accepts the older ``torch_dtype=`` alias and raises a TypeError on
    # ``dtype=``. Try the modern spelling, then fall back — identical resulting
    # model either way (no behavior change).
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, dtype=dtype, trust_remote_code=True
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=dtype, trust_remote_code=True
        )
    if has_adapter:
        from peft import PeftModel  # type: ignore
        model = PeftModel.from_pretrained(model, model_dir)
    model.to(device)
    model.eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    prompts = [r.get("prompt", "") for r in rows]
    eos_id = tokenizer.eos_token_id

    raws: List[str] = []
    per_sample_seconds: List[float] = []
    output_token_counts: List[int] = []

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,           # deterministic — a classifier, not a sampler
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    if eos_id is not None:
        gen_kwargs["eos_token_id"] = eos_id

    total_wall = 0.0
    n = len(prompts)
    print(f"[evaluate] running REAL generate over {n} samples "
          f"(batch_size={batch_size}, max_new_tokens={max_new_tokens})...")
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_prompts = prompts[start:start + batch_size]
            enc = tokenizer(
                batch_prompts, return_tensors="pt", padding=True,
                truncation=False,
            ).to(device)
            in_len = enc["input_ids"].shape[1]

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(**enc, **gen_kwargs)
            if device == "cuda":
                torch.cuda.synchronize()
            batch_seconds = time.perf_counter() - t0
            total_wall += batch_seconds

            # New tokens per row = total length - input length (left-padded so the
            # input occupies the first in_len columns uniformly).
            new_tokens = out[:, in_len:]
            bs = len(batch_prompts)
            per_row_seconds = batch_seconds / bs
            for j in range(bs):
                row_new = new_tokens[j]
                # Trim trailing pad/eos for an honest output-token count + decode.
                decoded = tokenizer.decode(row_new, skip_special_tokens=True)
                n_new = int((row_new != tokenizer.pad_token_id).sum().item())
                raws.append(decoded)
                output_token_counts.append(n_new)
                per_sample_seconds.append(per_row_seconds)

    env_info = {
        "device": device,
        "dtype": str(dtype),
        "model_dir": os.path.abspath(model_dir),
        "base_model_id": base_model_id,
        "has_lora_adapter": has_adapter,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "torch": getattr(torch, "__version__", None),
    }
    print(f"[evaluate] REAL generate complete: {n} samples in {total_wall:.3f}s wall "
          f"({n / total_wall:.3f} samples/s).")
    return raws, per_sample_seconds, output_token_counts, env_info, total_wall


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(cm_grid: List[List[int]], out_path: str, title: str) -> None:
    """Render the write-flag 2x2 confusion matrix to a PNG."""
    import matplotlib
    matplotlib.use("Agg")  # headless / no display
    import matplotlib.pyplot as plt

    arr = np.array(cm_grid, dtype=int)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0 (read)", "pred 1 (write)"])
    ax.set_yticklabels(["true 0 (read)", "true 1 (write)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = arr.max() / 2.0 if arr.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center",
                    color="white" if arr[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_length_scatter(
    t_true: Sequence[float], t_pred: Sequence[float], out_path: str, title: str
) -> None:
    """Render predicted-vs-true token-length scatter on LOG axes to a PNG.

    A perfect model lies on the y=x diagonal. Log axes are used because ``t`` is
    heavily right-skewed (§3.2) — linear axes would crush the short turns into a
    corner.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tt = np.asarray(list(t_true), dtype=float)
    tp = np.asarray(list(t_pred), dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))

    if len(tt) > 0:
        # Clamp to >=1 for log axes (a predicted 0 would be -inf on a log scale).
        tt_c = np.clip(tt, 1, None)
        tp_c = np.clip(tp, 1, None)
        ax.scatter(tt_c, tp_c, alpha=0.7, edgecolors="k", linewidths=0.4,
                   color="#2c7fb8", zorder=3)
        lo = max(1.0, float(min(tt_c.min(), tp_c.min())) * 0.7)
        hi = float(max(tt_c.max(), tp_c.max())) * 1.4
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="perfect (y=x)", zorder=2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.legend(loc="upper left")
    else:
        ax.text(0.5, 0.5, "no usable predictions", ha="center", va="center",
                transform=ax.transAxes)

    ax.set_xlabel("True token length t (log)")
    ax.set_ylabel("Predicted token length t (log)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _fmt(x: Optional[float], nd: int = 4) -> str:
    """Format a float for the report, rendering None as 'n/a'."""
    if x is None:
        return "n/a"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def build_metrics(
    rows: Sequence[Dict[str, Any]],
    parsed: Sequence[ParsedOutput],
    latency: Dict[str, Any],
    mode: str,
    env_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the full ``metrics.json`` dict from parsed outputs + truth."""
    w_true_all = [int(r["w"]) for r in rows]
    t_true_all = [float(r["t"]) for r in rows]

    # write metrics use only usable rows (where w was recovered); the rest are
    # reported as format-invalid in the validity block.
    w_pairs = [(int(r["w"]), p.w) for r, p in zip(rows, parsed) if p.w is not None]
    w_true = [a for a, _ in w_pairs]
    w_pred = [b for _, b in w_pairs]

    t_pairs = [(float(r["t"]), float(p.t)) for r, p in zip(rows, parsed) if p.t is not None]
    t_true = [a for a, _ in t_pairs]
    t_pred = [b for _, b in t_pairs]

    write_metrics = _binary_classification_metrics(w_true, w_pred) if w_true else {
        "n": 0, "note": "no usable w predictions"}
    write_metrics["base_rate_baseline"] = _base_rate_baseline(w_true_all)
    write_metrics["beats_baseline"] = (
        write_metrics.get("accuracy", 0.0) >= write_metrics["base_rate_baseline"]["accuracy"]
        if w_true else None
    )

    length_metrics = _length_metrics(t_true, t_pred)
    validity = _format_validity(parsed)

    metrics = {
        "mode": mode,  # "synthetic" or "real"
        "n_test_samples": len(rows),
        "truth_source": "canonical tokenizer t/w from test.jsonl (same definition as training)",
        "write_flag": write_metrics,
        "length": length_metrics,
        "output_format_validity": validity,
        "latency": latency,
        "environment": env_info,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return metrics


def build_report_md(metrics: Dict[str, Any]) -> str:
    """Render the human-readable ``report.md`` (all metrics + interpretation)."""
    mode = metrics["mode"]
    n = metrics["n_test_samples"]
    wm = metrics["write_flag"]
    lm = metrics["length"]
    fv = metrics["output_format_validity"]
    lat = metrics["latency"]
    env = metrics.get("environment", {})

    base = wm.get("base_rate_baseline", {})
    cm = wm.get("confusion_matrix", {})

    lines: List[str] = []
    lines.append("# Qwen3-1.7B Next-Turn Classifier — Performance Report")
    lines.append("")
    lines.append(f"- **Mode**: `{mode}`  "
                 + ("(SYNTHETIC SMOKE — mock predictions; NOT the deliverable for latency)"
                    if mode == "synthetic"
                    else "(REAL evaluation — actual model.generate() inference)"))
    lines.append(f"- **Test samples**: {n}")
    lines.append(f"- **Truth source**: {metrics['truth_source']}")
    lines.append(f"- **Generated at**: {metrics.get('generated_at')}")
    if env:
        env_bits = ", ".join(f"{k}={v}" for k, v in env.items())
        lines.append(f"- **Environment**: {env_bits}")
    lines.append("")

    if mode == "synthetic":
        lines.append("> **NOTE:** This bundle was produced by the `--synthetic` smoke path. "
                     "Its predictions are mock data used only to verify the metric / report / "
                     "chart code end-to-end. **The latency block below is tagged "
                     "`synthetic=true` and does NOT satisfy the real-inference latency "
                     "deliverable** (Tech Design §5.3 / round-2 BLOCKER-2). Run the real "
                     "evaluation against a trained model dir for the deliverable numbers.")
        lines.append("")

    # ---- write-flag --------------------------------------------------------
    lines.append("## 1. Write-flag classification (`w`)")
    lines.append("")
    if wm.get("n", 0) == 0:
        lines.append("_No usable `w` predictions were recovered._")
    else:
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Accuracy | {_fmt(wm['accuracy'])} |")
        lines.append(f"| Precision (class w=1) | {_fmt(wm['precision'])}"
                     + (" _(undefined: no positive predictions)_" if wm.get("precision_undefined") else "")
                     + " |")
        lines.append(f"| Recall (class w=1) | {_fmt(wm['recall'])}"
                     + (" _(undefined: no positive truths)_" if wm.get("recall_undefined") else "")
                     + " |")
        lines.append(f"| F1 (class w=1) | {_fmt(wm['f1'])} |")
        lines.append(f"| **Base-rate baseline (majority-class acc)** | "
                     f"**{_fmt(base.get('accuracy'))}** (always predict "
                     f"w={base.get('majority_class')}) |")
        lines.append("")
        lines.append("**Confusion matrix** (rows = true, cols = predicted):")
        lines.append("")
        lines.append("| | pred 0 (read) | pred 1 (write) |")
        lines.append("|---|---|---|")
        lines.append(f"| **true 0 (read)** | {cm.get('tn')} | {cm.get('fp')} |")
        lines.append(f"| **true 1 (write)** | {cm.get('fn')} | {cm.get('tp')} |")
        lines.append("")
        beats = wm.get("beats_baseline")
        verdict = ("**beats** the base-rate baseline" if beats
                   else "does **not** beat the base-rate baseline")
        lines.append(f"_Interpretation:_ the model's accuracy ({_fmt(wm['accuracy'])}) {verdict} "
                     f"({_fmt(base.get('accuracy'))}). The test split has "
                     f"{base.get('n_positive')} positive / {base.get('n_negative')} negative "
                     f"truths (positive rate {_fmt(base.get('positive_rate'))}), so on this skewed "
                     f"split accuracy alone is weak — read F1 / recall for the write class together "
                     f"with the baseline.")
    lines.append("")
    lines.append("![Write-flag confusion matrix](confusion_matrix.png)")
    lines.append("")

    # ---- length ------------------------------------------------------------
    lines.append("## 2. Output length regression (`t`)")
    lines.append("")
    if lm.get("n", 0) == 0:
        lines.append("_No usable `t` predictions were recovered._")
    else:
        lines.append(f"Scored on {lm['n']} usable predictions, against the canonical "
                     "tokenizer `t` (same definition as training).")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| log-MAE (headline) | {_fmt(lm['log_mae'])} |")
        lines.append(f"| raw-MAE (tokens) | {_fmt(lm['raw_mae'], 2)} |")
        lines.append(f"| RMSE (tokens) | {_fmt(lm['rmse'], 2)} |")
        lines.append(f"| R² (raw space) | {_fmt(lm['r2'])} |")
        lines.append(f"| Bucketed accuracy | {_fmt(lm['bucket_accuracy'])} |")
        lines.append("")
        lines.append("**Per-bucket accuracy** (recall within each true length bucket):")
        lines.append("")
        lines.append("| Bucket | n (true) | correct | accuracy |")
        lines.append("|---|---|---|---|")
        for label in BUCKET_LABELS:
            pb = lm["per_bucket"].get(label, {})
            lines.append(f"| {label} | {pb.get('n_true', 0)} | {pb.get('correct', 0)} | "
                         f"{_fmt(pb.get('accuracy'))} |")
        lines.append("")
        r2 = lm.get("r2")
        r2_note = ("R² is `n/a` (zero variance in truth on this split)" if r2 is None
                   else f"R²={_fmt(r2)} explains variance in raw space")
        lines.append(f"_Interpretation:_ log-MAE={_fmt(lm['log_mae'])} is the headline length "
                     f"error (judged in log space because `t` is heavily right-skewed); "
                     f"{r2_note}. Bucketed accuracy ({_fmt(lm['bucket_accuracy'])}) is the most "
                     f"operationally meaningful number for a capacity-routing use case — it asks "
                     f"\"did we land in the right order-of-magnitude band?\".")
    lines.append("")
    lines.append("![Length predicted-vs-true (log axes)](length_scatter.png)")
    lines.append("")

    # ---- format validity ---------------------------------------------------
    lines.append("## 3. Output-format validity")
    lines.append("")
    lines.append("| Metric | Count | Fraction |")
    lines.append("|---|---|---|")
    lines.append(f"| Parses as clean JSON | {fv['n_json']} | {_fmt(fv['parseable_json_fraction'])} |")
    lines.append(f"| Recovered via regex fallback | {fv['n_regex']} | {_fmt(fv['regex_fallback_fraction'])} |")
    lines.append(f"| Format-invalid (unparseable) | {fv['n_invalid']} | {_fmt(fv['invalid_fraction'])} |")
    lines.append(f"| Usable (both w & t recovered) | {fv['n_usable']} | {_fmt(fv['usable_fraction'])} |")
    lines.append("")
    lines.append("_Interpretation:_ a high clean-JSON fraction means the fine-tune learned the "
                 "compact output contract; a high regex-fallback fraction signals the format is "
                 "drifting and the prompt/format supervision may need strengthening. Unparseable "
                 "outputs are **counted here, never crash the run**, and are excluded from the "
                 "value metrics above.")
    lines.append("")

    # ---- latency -----------------------------------------------------------
    lines.append("## 4. Latency & throughput")
    lines.append("")
    if lat.get("synthetic"):
        lines.append("> **SYNTHETIC — NOT A DELIVERABLE.** `latency.synthetic=true` and "
                     "`latency.is_latency_deliverable=false`. These numbers are mock placeholders "
                     "from the offline smoke and explicitly do NOT satisfy the real-inference "
                     "latency requirement (Tech Design §5.1 / round-2 BLOCKER-2). They exist only "
                     "to exercise the latency-reporting code path.")
        lines.append("")
    else:
        lines.append("Measured on **real** `model.generate()` calls "
                     f"({lat.get('measured_on')}).")
        lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Source | {lat.get('measured_on')} |")
    lines.append(f"| `synthetic` flag | {lat.get('synthetic')} |")
    lines.append(f"| Is latency deliverable | {lat.get('is_latency_deliverable')} |")
    lines.append(f"| Samples timed | {lat.get('n_samples')} |")
    lines.append(f"| Median generate latency / sample (s) | {_fmt(lat.get('median_latency_s'), 4)} |")
    lines.append(f"| p95 generate latency / sample (s) | {_fmt(lat.get('p95_latency_s'), 4)} |")
    lines.append(f"| Mean generate latency / sample (s) | {_fmt(lat.get('mean_latency_s'), 4)} |")
    lines.append(f"| Throughput (samples/sec) | {_fmt(lat.get('samples_per_second'), 3)} |")
    lines.append(f"| Mean output tokens / sample | {_fmt(lat.get('mean_output_tokens'), 2)} |")
    if lat.get("total_wall_seconds") is not None:
        lines.append(f"| Total generate wall time (s) | {_fmt(lat.get('total_wall_seconds'), 3)} |")
    lines.append("")
    if not lat.get("synthetic"):
        lines.append("_Interpretation:_ median + p95 per-sample latency and samples/sec are "
                     "measured end-to-end over real batched generation (timing wraps the actual "
                     "`generate()` call, with CUDA sync on GPU). This is the latency deliverable; "
                     "device and batch size are recorded under `environment` so the numbers are "
                     "reproducible.")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("### How this report was produced")
    lines.append("")
    lines.append("- **Synthetic smoke (offline, no model):** "
                 "`python src/evaluate.py --synthetic --test_file data/prepared/test.jsonl --report_dir report/`")
    lines.append("- **Real evaluation (deliverable):** "
                 "`python src/evaluate.py --model_dir <trained_model_dir> "
                 "--test_file data/prepared/test.jsonl --report_dir report/ "
                 "--batch_size 8 --max_new_tokens 16`")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Top-level orchestration: load -> predict -> metrics -> write bundle."""
    rows = load_test_rows(args.test_file)
    os.makedirs(args.report_dir, exist_ok=True)

    if args.synthetic:
        mode = "synthetic"
        raws, lats, ntoks = synthetic_predictions(rows, seed=args.seed)
        latency = _latency_metrics(lats, ntoks, synthetic=True)
        env_info: Dict[str, Any] = {"mode": "synthetic", "note": "no model loaded"}
    else:
        mode = "real"
        if not args.model_dir:
            raise SystemExit(
                "Real evaluation requires --model_dir (a trained artifact dir). "
                "For an offline code check without a model, use --synthetic.")
        if not os.path.isdir(args.model_dir):
            raise SystemExit(f"--model_dir does not exist: {args.model_dir}")
        raws, lats, ntoks, env_info, total_wall = real_predictions(
            rows, args.model_dir, args.batch_size, args.max_new_tokens, device=args.device
        )
        latency = _latency_metrics(lats, ntoks, synthetic=False,
                                   total_wall_seconds=total_wall)

    parsed = [parse_output(r) for r in raws]

    metrics = build_metrics(rows, parsed, latency, mode, env_info)

    # ---- write bundle ------------------------------------------------------
    metrics_path = os.path.join(args.report_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    report_path = os.path.join(args.report_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(build_report_md(metrics))

    cm_grid = metrics["write_flag"].get("confusion_matrix_grid", [[0, 0], [0, 0]])
    cm_path = os.path.join(args.report_dir, "confusion_matrix.png")
    _plot_confusion_matrix(
        cm_grid, cm_path,
        title=f"Write-flag confusion matrix ({mode})",
    )

    # Length scatter uses the usable (t recovered) pairs.
    t_true = [float(r["t"]) for r, p in zip(rows, parsed) if p.t is not None]
    t_pred = [float(p.t) for p in parsed if p.t is not None]
    scatter_path = os.path.join(args.report_dir, "length_scatter.png")
    _plot_length_scatter(
        t_true, t_pred, scatter_path,
        title=f"Predicted vs. true token length ({mode})",
    )

    # ---- console summary ---------------------------------------------------
    print("=" * 78)
    print(f"Evaluation complete — mode={mode}, n={len(rows)}")
    print("=" * 78)
    wm = metrics["write_flag"]
    lm = metrics["length"]
    print(f"write-flag : acc={_fmt(wm.get('accuracy'))} f1={_fmt(wm.get('f1'))} "
          f"baseline={_fmt(wm.get('base_rate_baseline', {}).get('accuracy'))}")
    print(f"length     : log-MAE={_fmt(lm.get('log_mae'))} raw-MAE={_fmt(lm.get('raw_mae'),2)} "
          f"RMSE={_fmt(lm.get('rmse'),2)} R2={_fmt(lm.get('r2'))} "
          f"bucket-acc={_fmt(lm.get('bucket_accuracy'))}")
    fv = metrics["output_format_validity"]
    print(f"format     : json={_fmt(fv['parseable_json_fraction'])} "
          f"regex={_fmt(fv['regex_fallback_fraction'])} "
          f"invalid={_fmt(fv['invalid_fraction'])}")
    lat = metrics["latency"]
    tag = "SYNTHETIC (NOT a deliverable)" if lat["synthetic"] else "REAL (deliverable)"
    print(f"latency    : {tag} median={_fmt(lat.get('median_latency_s'),4)}s "
          f"p95={_fmt(lat.get('p95_latency_s'),4)}s "
          f"throughput={_fmt(lat.get('samples_per_second'),3)} samples/s "
          f"mean_out_tokens={_fmt(lat.get('mean_output_tokens'),2)}")
    print("-" * 78)
    print("wrote:")
    for p in (metrics_path, report_path, cm_path, scatter_path):
        print(f"  - {p}")
    return metrics


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate the Qwen3-1.7B next-turn classifier and emit the "
                    "performance report bundle (metrics.json + report.md + charts).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_dir", type=str, default=None,
                   help="Trained artifact dir (LoRA adapter + tokenizer + run_config.json), "
                        "OR a full/merged model dir. REQUIRED for real eval; ignored with "
                        "--synthetic.")
    _default_test = os.path.join(os.path.dirname(_THIS_DIR), "data", "prepared", "test.jsonl")
    p.add_argument("--test_file", type=str, default=_default_test,
                   help="Held-out test.jsonl (must have w and t truth columns).")
    _default_report = os.path.join(os.path.dirname(_THIS_DIR), "report")
    p.add_argument("--report_dir", type=str, default=_default_report,
                   help="Output dir for metrics.json + report.md + the two charts.")
    p.add_argument("--synthetic", action="store_true",
                   help="Offline smoke: feed MOCK predictions (no model needed). Latency is "
                        "tagged synthetic=true and is NOT the latency deliverable (Tech Design §5.3).")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for real generate().")
    p.add_argument("--max_new_tokens", type=int, default=16,
                   help="Max new tokens to generate per sample (the JSON is ~10-16 tokens).")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device for real inference (auto picks cuda if available, else cpu).")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the synthetic-prediction generator (deterministic smoke).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_evaluation(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

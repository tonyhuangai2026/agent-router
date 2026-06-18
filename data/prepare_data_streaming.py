"""Streaming, leakage-free, bucket-balanced data preparation for HUGE inputs.

Memory-efficient companion to ``prepare_data.py`` for very large JSONL files
(tens/hundreds of GB) that cannot be loaded into memory at once.

Why a separate script
---------------------
``prepare_data.py`` loads every record, explodes all of them, dedups globally,
then group-splits by conversation. That is correct and fast for normal inputs
but needs the whole corpus in memory. For 100GB+ logs we instead:

  1. **Stream** the input one record at a time (constant memory; never holds the
     whole file). Supports a single ``.jsonl`` file or a directory of
     ``.json`` / ``.jsonl`` files (same as ``prepare_data.load_input``).
  2. Assign each record to **train / val / test by a hash of its
     ``conversation_id``** — so EVERY sample of one conversation lands in the
     SAME split. This reproduces ``prepare_data.py``'s leakage-free guarantee
     (Tech Design §2.5) *without* needing the whole corpus: the split decision
     is a pure function of the record's root prompt, computable per record.
  3. Explode each record into ``(context -> {w,t})`` samples (reusing the
     SHARED ``labeling`` contracts — identical labels to ``prepare_data.py``).
  4. Keep a **bounded** sample via per-(split, t-bucket) **reservoir sampling**
     toward ``--balance-target`` total train samples, so the four length buckets
     (lt128 / 128-512 / 512-2k / gt2k) stay roughly balanced and memory stays
     O(target), not O(corpus). val/test get proportionally smaller reservoirs.

Leakage-free split: the split key is ``labeling.derive_conversation_id(record)``
(single-record root-prompt hash). NOTE this is the per-record variant; the
cross-record longest-common-prefix merge that ``prepare_data.derive_
conversation_ids`` applies (folding a root with its summary-extended variants)
is a whole-corpus operation and is intentionally NOT done in streaming mode.
Records sharing an *identical* root prompt still group together, so the split is
still grouped-by-conversation and leakage-free; only rare summary-extension
variants may land in their own group. This trade-off is recorded in
``data_stats.json`` under ``conversation_id_mode``.

Usage
-----
    python data/prepare_data_streaming.py \
        --input /path/to/huge.jsonl \
        --outdir data/prepared \
        --balance-target 15000 \
        --seed 42

Defaults to the lightweight tokenizer + char-based length estimate (same as
``prepare_data.py``'s default), so it is tokenizer-load-free and fast.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_THIS_DIR), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import labeling  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
DEFAULT_INPUT = os.path.join(_REPO_ROOT, "demo_data.jsonl")
if not os.path.exists(DEFAULT_INPUT):
    DEFAULT_INPUT = "/home/ubuntu/midea/demo_data.jsonl"
DEFAULT_OUTDIR = os.path.join(_THIS_DIR, "prepared")


# ---------------------------------------------------------------------------
# Bucket definitions (MUST match evaluate.py's length buckets)
# ---------------------------------------------------------------------------

BUCKET_NAMES = ["lt128", "128-512", "512-2k", "gt2k"]


def get_t_bucket(t: int) -> str:
    if t < 128:
        return "lt128"
    if t < 512:
        return "128-512"
    if t < 2048:
        return "512-2k"
    return "gt2k"


# ---------------------------------------------------------------------------
# Streaming input: yield ONE record at a time (never load the whole file)
# ---------------------------------------------------------------------------

def iter_records(path: str):
    """Yield records one at a time from a .jsonl file OR a dir of .json/.jsonl.

    Mirrors ``prepare_data.load_input`` semantics but as a GENERATOR so the
    whole input is never materialized in memory:
      * a ``.jsonl`` file  -> one record per non-blank line
      * a directory        -> each ``*.json`` = one record (array -> many),
                              each ``*.jsonl`` = many records (line by line)
    """
    if os.path.isdir(path):
        names = sorted(
            n for n in os.listdir(path) if n.endswith(".json") or n.endswith(".jsonl")
        )
        if not names:
            raise ValueError(
                f"--input directory '{path}' has no .json/.jsonl files."
            )
        for name in names:
            fpath = os.path.join(path, name)
            if name.endswith(".jsonl"):
                yield from _iter_jsonl(fpath)
            else:  # .json: one object, or an array of objects
                with open(fpath, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, list):
                    for item in obj:
                        yield item
                else:
                    yield obj
    else:
        yield from _iter_jsonl(path)


def _iter_jsonl(path: str):
    """Yield one parsed record per non-blank line; skip malformed lines loudly."""
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[prepare_streaming] WARN: skipping malformed line "
                      f"{lineno} in {path}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Split assignment by conversation_id hash (leakage-free, per-record)
# ---------------------------------------------------------------------------

def assign_split(conversation_id: str, val_frac: float, test_frac: float, seed: int) -> str:
    """Deterministically map a conversation_id to 'train' | 'val' | 'test'.

    Uses a stable hash of (seed, conversation_id) -> uniform fraction in [0,1).
    Because the decision depends ONLY on the conversation_id, every record (and
    therefore every exploded sample) of one conversation gets the SAME split —
    this is what makes the streaming split leakage-free without global state.
    """
    h = hashlib.sha1(f"{seed}:{conversation_id}".encode("utf-8")).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF  # uniform in [0,1]
    if frac < test_frac:
        return "test"
    if frac < test_frac + val_frac:
        return "val"
    return "train"


# ---------------------------------------------------------------------------
# Sample explosion (single record; reuses SHARED labeling contracts)
# ---------------------------------------------------------------------------

def _context_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def _make_completion(w: int, t: int) -> str:
    return json.dumps({"w": w, "t": t}, separators=(",", ":"))


def _raw_ctx_key(context_messages: List[dict]) -> str:
    """Cheap, render-free dedup/cache key for a context message-list.

    ``render_context`` is a pure function of (context_messages, max_len,
    max_block_chars), so identical raw contexts always render to identical
    prompts. Hashing the raw context lets us dedup BEFORE rendering (so we never
    pay the expensive render for samples that will be dropped) and acts as the
    per-record render cache key.
    """
    return hashlib.sha1(
        json.dumps(context_messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def explode_record_light(
    record: dict,
    conversation_id: str,
    tokenizer,
    dual_use_as_write: bool = True,
) -> List[dict]:
    """Explode ONE record into LIGHTWEIGHT samples — NO prompt render yet.

    This is the cheap half of the two-phase pipeline. We compute only the labels
    ``w`` and ``t`` (char-estimate, ~5ms/record) and keep a REFERENCE to the raw
    context message-list. The expensive ``render_context`` (~97% of cost) is
    deferred to :func:`realize_prompt`, called ONLY for samples the reservoir
    actually keeps (~balance_target total) instead of for every exploded sample
    (which can be tens of millions, almost all discarded).

    Dedup key is ``_raw_ctx_key`` (render-free) — equivalent to hashing the
    rendered prompt since render is a pure function of the raw context.

    Each light sample carries ``_ctx`` (the raw context list) so the prompt can
    be realized later; ``_ctx`` is stripped before writing to disk.
    """
    request = record.get("request") or {}
    messages = request.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    session_id = record.get("session_id", "")
    samples: List[dict] = []

    def _emit(context_messages, turn_content, usage_tokens):
        w = 1 if labeling.is_write_response(turn_content, dual_use_as_write=dual_use_as_write) else 0
        t = labeling.count_output_tokens(turn_content, tokenizer)  # char-estimate
        samples.append({
            "w": w,
            "t": t,
            "conversation_id": conversation_id,
            "session_id": session_id,
            "usage_output_tokens": usage_tokens,
            "_ctx": context_messages,           # raw context (for deferred render)
            "_ctx_hash": _raw_ctx_key(context_messages),  # render-free dedup key
        })

    # 1) Every assistant turn in the history; context = strictly-prior messages.
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        _emit(messages[:i], msg.get("content"), None)

    # 2) Top-level response as the final turn; usage carried as reference column.
    response = record.get("response") or {}
    resp_content = response.get("content")
    usage = response.get("usage") or {}
    usage_tokens = usage.get("output_tokens")
    if not isinstance(usage_tokens, int):
        usage_tokens = None
    _emit(list(messages), resp_content, usage_tokens)

    return samples


def realize_prompt(
    sample: dict,
    tokenizer,
    max_len: int,
    max_block_chars: int,
    render_cache: Optional[Dict[str, str]] = None,
) -> dict:
    """Turn a kept LIGHT sample into a final sample by rendering its prompt.

    Called ONLY for reservoir-kept samples. Renders ``_ctx`` -> ``prompt`` (with
    an optional per-record cache keyed by ``_ctx_hash``), builds ``completion``,
    and drops the internal ``_ctx`` (the bulky raw context) so the in-memory
    reservoir stays small. The dedup key ``_ctx_hash`` is preserved for the
    reservoir's own dedup bookkeeping (stripped at write time).
    """
    ctx = sample.pop("_ctx", [])
    key = sample.get("_ctx_hash")
    prompt = None
    if render_cache is not None and key is not None:
        prompt = render_cache.get(key)
    if prompt is None:
        prompt = labeling.render_context(
            ctx, tokenizer, max_len=max_len, max_block_chars=max_block_chars
        )
        if render_cache is not None and key is not None:
            render_cache[key] = prompt
    sample["prompt"] = prompt
    sample["completion"] = _make_completion(sample["w"], sample["t"])
    return sample


# ---------------------------------------------------------------------------
# Bucket-balanced reservoir sampling (bounded memory)
# ---------------------------------------------------------------------------

class BucketReservoir:
    """Per-(split, t-bucket) reservoir sampling with a global per-split cap.

    Keeps at most ``capacity`` samples per (split, bucket) using classic
    reservoir sampling (uniform random sample of an unbounded stream in O(cap)
    memory). Splitting the budget across the four length buckets keeps them
    balanced so the model isn't swamped by the dominant short-output bucket.

    Also dedups within a reservoir by ``(_ctx_hash, w, t)`` — streaming can't
    dedup globally, but in-reservoir dedup removes the most common duplicates
    (identical context/labels re-emitted across near-identical records).
    """

    def __init__(self, capacity_per_bucket: int, rng: random.Random):
        self.capacity = max(1, capacity_per_bucket)
        self.rng = rng
        self._res: Dict[str, List[dict]] = {b: [] for b in BUCKET_NAMES}
        self._seen_keys: Dict[str, set] = {b: set() for b in BUCKET_NAMES}
        self._n_seen: Dict[str, int] = {b: 0 for b in BUCKET_NAMES}  # incl. dups skipped

    def offer(self, sample: dict) -> None:
        b = get_t_bucket(sample["t"])
        key = (sample["_ctx_hash"], sample["w"], sample["t"])
        seen = self._seen_keys[b]
        if key in seen:
            return  # in-reservoir dedup
        self._n_seen[b] += 1
        res = self._res[b]
        if len(res) < self.capacity:
            res.append(sample)
            seen.add(key)
        else:
            # Reservoir sampling: replace a random slot with prob capacity/n_seen.
            j = self.rng.randint(0, self._n_seen[b] - 1)
            if j < self.capacity:
                old = res[j]
                seen.discard((old["_ctx_hash"], old["w"], old["t"]))
                res[j] = sample
                seen.add(key)

    def samples(self) -> List[dict]:
        out: List[dict] = []
        for b in BUCKET_NAMES:
            out.extend(self._res[b])
        return out

    def bucket_counts(self) -> Dict[str, int]:
        return {b: len(self._res[b]) for b in BUCKET_NAMES}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False))
            f.write("\n")


def _t_stats(ts: List[int]) -> Dict[str, Any]:
    if not ts:
        return {"min": None, "median": None, "mean": None, "max": None, "count": 0}
    s = sorted(ts)
    n = len(s)
    return {
        "min": s[0],
        "median": float(s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2),
        "mean": float(sum(s) / n),
        "max": s[-1],
        "count": n,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.outdir, exist_ok=True)

    # Tokenizer: lightweight by default (labels/truncation are char-estimated, so
    # the real Qwen3 tokenizer would produce identical output at ~13s load cost).
    if args.real_tokenizer:
        tokenizer = labeling.get_tokenizer()
    else:
        tokenizer = labeling.SimpleWhitespaceTokenizer()
    is_fallback = isinstance(tokenizer, labeling.SimpleWhitespaceTokenizer)
    print(f"[prepare_streaming] tokenizer: {type(tokenizer).__name__}"
          + (" (lightweight — default; t is char-estimated)" if is_fallback
             else " (real Qwen3 — --real-tokenizer)"))

    # Per-split, per-bucket reservoir capacities. train gets balance_target split
    # across 4 buckets; val/test get a fraction of that (proportional to fracs).
    rng = random.Random(args.seed)
    train_cap = max(1, args.balance_target // len(BUCKET_NAMES))
    val_cap = max(1, int(train_cap * (args.val_frac / max(1e-9, 1 - args.val_frac - args.test_frac))))
    test_cap = max(1, int(train_cap * (args.test_frac / max(1e-9, 1 - args.val_frac - args.test_frac))))
    reservoirs = {
        "train": BucketReservoir(train_cap, rng),
        "val": BucketReservoir(val_cap, rng),
        "test": BucketReservoir(test_cap, rng),
    }
    print(f"[prepare_streaming] per-bucket capacity: train={train_cap} "
          f"val={val_cap} test={test_cap} (x{len(BUCKET_NAMES)} buckets)")

    # Streaming scan: one record at a time -> assign split -> explode -> offer.
    n_records = 0
    n_samples_seen = 0
    split_conv_ids: Dict[str, set] = {"train": set(), "val": set(), "test": set()}
    t0 = time.time()
    log_every = max(1, args.log_every)

    for record in iter_records(args.input):
        n_records += 1
        conv_id = labeling.derive_conversation_id(record)
        split = assign_split(conv_id, args.val_frac, args.test_frac, args.seed)
        split_conv_ids[split].add(conv_id)
        # Two-phase: explode LIGHT (labels only, no render) -> reservoir decides
        # what to keep using only (w, t) -> render is deferred to write time for
        # the kept set only. This avoids rendering the (vast) majority of samples
        # that get discarded — render is ~97% of per-sample cost.
        samples = explode_record_light(
            record, conversation_id=conv_id, tokenizer=tokenizer,
            dual_use_as_write=args.dual_use_as_write,
        )
        res = reservoirs[split]
        for s in samples:
            res.offer(s)
            n_samples_seen += 1
        if n_records % log_every == 0:
            el = time.time() - t0
            rate = n_records / el if el > 0 else 0.0
            kept = sum(sum(r.bucket_counts().values()) for r in reservoirs.values())
            print(f"[prepare_streaming]   {n_records} records | {n_samples_seen} samples seen "
                  f"| {kept} kept | {rate:.0f} rec/s | elapsed {el:.0f}s")

    # Leakage check: a conversation_id must not appear in more than one split.
    # By construction assign_split is a pure function of conv_id, so this holds;
    # we still assert it loudly as a safety net.
    overlaps = {
        "train_val": sorted(split_conv_ids["train"] & split_conv_ids["val"]),
        "train_test": sorted(split_conv_ids["train"] & split_conv_ids["test"]),
        "val_test": sorted(split_conv_ids["val"] & split_conv_ids["test"]),
    }
    assert not any(overlaps.values()), (
        "conversation_id leakage across splits! " + json.dumps(overlaps, ensure_ascii=False)
    )

    # Realize prompts ONLY for the kept set (deferred render). This is where the
    # expensive render_context runs — on ~balance_target samples, not the
    # millions seen. Done once, at the end, after the reservoir has settled.
    n_kept_total = sum(sum(r.bucket_counts().values()) for r in reservoirs.values())
    print(f"[prepare_streaming] rendering prompts for {n_kept_total} kept samples "
          f"(deferred render — only the kept set)...")
    t_render0 = time.time()
    render_cache: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        for s in reservoirs[split].samples():
            realize_prompt(s, tokenizer, args.max_len, args.max_block_chars, render_cache)
    print(f"[prepare_streaming] rendered {n_kept_total} prompts in {time.time()-t_render0:.1f}s")

    # Write splits + stats.
    out_counts: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        rows = reservoirs[split].samples()
        rng.shuffle(rows)  # avoid bucket-contiguous ordering on disk
        path = os.path.join(args.outdir, f"{split}.jsonl")
        _write_jsonl(path, rows)
        ts = [r["t"] for r in rows]
        ws = [r["w"] for r in rows]
        out_counts[split] = {
            "n_samples": len(rows),
            "n_conversations": len(split_conv_ids[split]),
            "bucket_counts": reservoirs[split].bucket_counts(),
            "w_balance": {"write_w1": sum(ws), "read_w0": len(ws) - sum(ws)},
            "t_distribution": _t_stats(ts),
        }
        print(f"[prepare_streaming] wrote {path} ({len(rows)} samples, "
              f"{len(split_conv_ids[split])} conversations)")

    stats = {
        "mode": "streaming-reservoir",
        "input": args.input,
        "balance_target": args.balance_target,
        "seed": args.seed,
        "records_scanned": n_records,
        "samples_seen": n_samples_seen,
        "elapsed_seconds": round(time.time() - t0, 1),
        "conversation_id_mode": (
            "per-record root-prompt hash (single-record derive_conversation_id); "
            "cross-record LCP merge NOT applied in streaming mode (see module docstring)"
        ),
        "length_label": "char-estimate via labeling.count_output_tokens (bucketing/ordering use)",
        "split": out_counts,
        "leakage_check": {"disjoint": True, "pairwise_overlaps": overlaps},
        "buckets": BUCKET_NAMES,
    }
    stats_path = os.path.join(args.outdir, "data_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[prepare_streaming] wrote {stats_path}")

    tr, va, te = (out_counts[s]["n_samples"] for s in ("train", "val", "test"))
    print(f"SUMMARY: records={n_records} samples_seen={n_samples_seen} "
          f"| kept train/val/test={tr}/{va}/{te} "
          f"| convs train/val/test={out_counts['train']['n_conversations']}/"
          f"{out_counts['val']['n_conversations']}/{out_counts['test']['n_conversations']} "
          f"| disjoint=True")
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Streaming, leakage-free, bucket-balanced data prep for huge inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="a .jsonl file OR a directory of .json/.jsonl files (streamed)")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="output dir for prepared splits")
    p.add_argument("--balance-target", type=int, default=15000,
                   help="approx total TRAIN samples to keep (split across 4 length buckets)")
    p.add_argument("--max-len", type=int, default=4096, help="context truncation length (char-estimated tokens)")
    p.add_argument("--max-block-chars", type=int, default=4000, help="per-block char cap for oversized blocks")
    p.add_argument("--seed", type=int, default=42, help="fixed seed (split + reservoir)")
    p.add_argument("--val-frac", type=float, default=0.1, help="fraction of conversations -> val")
    p.add_argument("--test-frac", type=float, default=0.1, help="fraction of conversations -> test")
    p.add_argument("--dual-use-as-write", dest="dual_use_as_write", action="store_true", default=True,
                   help="treat run_terminal_cmd/bash as write (default True; §2.2.1)")
    p.add_argument("--no-dual-use-as-write", dest="dual_use_as_write", action="store_false",
                   help="exclude dual-use tools from the write class")
    p.add_argument("--real-tokenizer", dest="real_tokenizer", action="store_true", default=False,
                   help="load real Qwen3 tokenizer (default off: char-estimate is identical & ~28x faster)")
    p.add_argument("--log-every", type=int, default=1000, help="progress log cadence (records)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

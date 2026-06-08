"""Data preparation for the Qwen3-1.7B next-turn write/length classifier.

Implements Tech Design §2.3–§2.6 (revised round 2). This script:

1. Reads ``demo_data.jsonl`` (one LLM request/response trace per line).
2. **Explodes** each record's ``request.messages`` history into many
   ``(context -> next-turn labels)`` samples — one per assistant turn — plus the
   top-level ``response`` as the final turn of the record (§2.3).
3. Derives labels with the SHARED, already-verified foundation module
   ``src/labeling.py`` (T1). It does NOT reimplement any of those contracts:
     * ``w``  = :func:`labeling.is_write_response`              (§2.2.1)
     * ``t``  = :func:`labeling.count_output_tokens`            (§2.2.2 — the
                CANONICAL tokenizer count of the serialized turn, NOT usage)
     * ``conversation_id`` via :func:`labeling.derive_conversation_ids`
                (corpus-level LCP merge → ~14 conversations)          (§2.2.3)
     * ``prompt`` = :func:`labeling.render_context`              (§2.4)
4. Carries ``usage.output_tokens`` as a separate, reference-only column
   ``usage_output_tokens`` (null for exploded historical turns that have no
   usage). It is NEVER the target (§2.2.2).
5. Dedups identical ``(context_hash, w, t)`` samples (§2.3).
6. **Group-splits by conversation_id** (NOT session_id) into train/val/test =
   8/1/1 with a fixed seed, keeping every sample of a conversation in one split,
   and ASSERTS zero conversation_id overlap across splits (fail loudly) (§2.5).
7. Emits ``train.jsonl`` / ``val.jsonl`` / ``test.jsonl`` + ``data_stats.json``
   under the output dir (§2.6).

Each output line (§2.6):
    {"prompt": <rendered chat context>,
     "completion": "{\"w\":<0|1>,\"t\":<int>}",   # compact, no whitespace
     "w": int, "t": int,
     "conversation_id": str, "session_id": str,
     "usage_output_tokens": int|null}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import Counter, OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --- Import the shared, already-verified labeling foundation (T1) -----------
# prepare_data.py lives at qwen_classifier/data/; labeling.py lives at
# qwen_classifier/src/. Make ``src`` importable regardless of CWD so we REUSE
# the verified contracts instead of reimplementing them.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_THIS_DIR), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import labeling  # noqa: E402  (path set up above)

# Repo root = qwen_classifier/.. ; demo_data.jsonl sits at the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
DEFAULT_INPUT = os.path.join(_REPO_ROOT, "demo_data.jsonl")
if not os.path.exists(DEFAULT_INPUT):
    DEFAULT_INPUT = "/home/ubuntu/midea/demo_data.jsonl"
DEFAULT_OUTDIR = os.path.join(_THIS_DIR, "prepared")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    """Load a single JSONL file into a list of dicts (blank lines skipped)."""
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_input(path: str) -> List[dict]:
    """Load records from a JSONL file OR a directory of JSON files.

    - If ``path`` is a ``.jsonl`` file (or any file): parse it as JSONL — one
      record per non-blank line.
    - If ``path`` is a directory: treat **each ``.json`` file in it as one
      record** (i.e. one JSONL line). Files are processed in sorted order for
      reproducibility; sub-directories are NOT recursed. A ``.json`` file that
      itself contains a JSON array is flattened (each element becomes a record).
    """
    if os.path.isdir(path):
        records: List[dict] = []
        names = sorted(n for n in os.listdir(path) if n.endswith(".json"))
        if not names:
            raise ValueError(
                f"--input directory '{path}' contains no .json files "
                "(expected one .json per record)."
            )
        for name in names:
            fpath = os.path.join(path, name)
            with open(fpath, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                records.extend(obj)          # a JSON array → many records
            else:
                records.append(obj)          # a single JSON object → one record
        return records
    # Otherwise treat it as a JSONL file.
    return load_jsonl(path)


def write_jsonl(path: str, rows: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


# ---------------------------------------------------------------------------
# §2.3 Sample explosion
# ---------------------------------------------------------------------------

def _context_hash(prompt: str) -> str:
    """Stable hash of the rendered context, used for dedup (§2.3)."""
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def _raw_context_key(context_messages: List[dict]) -> str:
    """Stable hash of the RAW (pre-render) context message-list.

    Used purely as a render CACHE key: two samples whose context message-lists
    are byte-identical render to byte-identical prompts (``render_context`` is a
    pure function of its inputs), so we render each distinct context exactly once.
    This is a performance optimization only — it does NOT change which samples or
    labels are produced (labels are still computed per sample from each turn).
    On demo_data.jsonl this collapses ~2056 render calls to ~231.
    """
    return hashlib.sha1(
        json.dumps(context_messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _make_completion(w: int, t: int) -> str:
    """Compact completion JSON exactly ``{"w":<0|1>,"t":<int>}`` (§2.6).

    Uses ``separators=(',', ':')`` so there is NO extra whitespace.
    """
    return json.dumps({"w": w, "t": t}, separators=(",", ":"))


def explode_record(
    record: dict,
    conversation_id: str,
    tokenizer,
    max_len: int,
    max_block_chars: int,
    dual_use_as_write: bool,
    render_cache: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """Explode ONE record into ``(context -> next-turn labels)`` samples (§2.3).

    For every assistant message at index ``i`` in ``request.messages`` we build a
    sample whose context is ``messages[0..i-1]`` (everything before that turn) and
    whose labels come from ``messages[i]``. We then add ONE final sample whose
    context is the entire ``request.messages`` history and whose labels come from
    the top-level ``response.content`` (the turn the record actually produced).

    The ``conversation_id`` is supplied by the caller (it is a corpus-level
    decision, see :func:`labeling.derive_conversation_ids`). ``usage`` is only
    available on the top-level response, so it is carried as
    ``usage_output_tokens`` for that final turn and ``None`` for every exploded
    historical turn (§2.2.2).
    """
    request = record.get("request") or {}
    messages = request.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    session_id = record.get("session_id")
    if render_cache is None:
        render_cache = {}

    samples: List[dict] = []

    def _emit(context_messages: List[dict], turn_content: Any, usage_tokens: Optional[int]) -> None:
        # w/t come from the SHARED labeling module — never reimplemented here.
        w = 1 if labeling.is_write_response(turn_content, dual_use_as_write=dual_use_as_write) else 0
        # t is ALWAYS the canonical tokenizer count of the serialized turn
        # (§2.2.2) — uniformly for historical turns AND the top-level response.
        t = labeling.count_output_tokens(turn_content, tokenizer)
        # Render via the SHARED labeling.render_context (§2.4), but memoize by the
        # raw context message-list so identical contexts are rendered once. Pure
        # speed-up: same inputs -> same prompt; no effect on samples or labels.
        rkey = _raw_context_key(context_messages)
        prompt = render_cache.get(rkey)
        if prompt is None:
            prompt = labeling.render_context(
                context_messages, tokenizer, max_len=max_len, max_block_chars=max_block_chars
            )
            render_cache[rkey] = prompt
        samples.append(
            {
                "prompt": prompt,
                "completion": _make_completion(w, t),
                "w": w,
                "t": t,
                "conversation_id": conversation_id,
                "session_id": session_id,
                "usage_output_tokens": usage_tokens,
                # internal-only key (stripped before writing) for dedup §2.3
                "_context_hash": _context_hash(prompt),
            }
        )

    # 1) Every assistant turn inside the history. Context = strictly-prior msgs.
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        # Historical turns have no usage figure -> reference column is null.
        _emit(messages[:i], msg.get("content"), None)

    # 2) The top-level response as the final turn of this record. Context = the
    #    full cumulative history. usage.output_tokens (if present) is carried as
    #    a reference-only column.
    response = record.get("response") or {}
    resp_content = response.get("content")
    usage = response.get("usage") or {}
    usage_tokens = usage.get("output_tokens")
    if not isinstance(usage_tokens, int):
        usage_tokens = None
    _emit(list(messages), resp_content, usage_tokens)

    return samples


# ---------------------------------------------------------------------------
# §2.3 Dedup
# ---------------------------------------------------------------------------

def dedup_samples(samples: Sequence[dict]) -> List[dict]:
    """Collapse identical ``(context_hash, w, t)`` samples (§2.3).

    Order-preserving so output is deterministic. The first occurrence of each
    key wins (it carries the most authoritative ``usage_output_tokens``, since
    the top-level-response sample — the only one with usage — is emitted last per
    record but a historical duplicate from another record may appear first;
    either way the labels are identical by construction of the key).
    """
    seen: "OrderedDict[Tuple[str, int, int], dict]" = OrderedDict()
    for s in samples:
        key = (s["_context_hash"], s["w"], s["t"])
        if key not in seen:
            seen[key] = s
        else:
            # Prefer to keep a non-null usage_output_tokens if a later duplicate
            # has one and the kept copy does not (purely for the reference col).
            if seen[key].get("usage_output_tokens") is None and s.get("usage_output_tokens") is not None:
                seen[key]["usage_output_tokens"] = s["usage_output_tokens"]
    return list(seen.values())


# ---------------------------------------------------------------------------
# §2.5 Group-split by conversation_id
# ---------------------------------------------------------------------------

def group_split_conversations(
    conversation_ids: Sequence[str],
    seed: int,
    val_frac: float,
    test_frac: float,
) -> Dict[str, List[str]]:
    """Assign whole conversations to train/val/test (§2.5).

    Splits by *conversation* (not by sample and not by session_id), so every
    exploded sample of one conversation stays in a single split. With only ~14
    conversations an 8/1/1 fractional split rounds val/test down to 0, which
    would leave them empty; to keep the split **robust** we GUARANTEE at least
    one conversation in val and one in test whenever there are >= 3
    conversations (the per-split counts are reported honestly in data_stats).

    Returns a dict ``{"train": [...], "val": [...], "test": [...]}`` of
    conversation_ids. The lists are guaranteed disjoint.
    """
    import random

    uniq = sorted(set(conversation_ids))  # deterministic base order
    rng = random.Random(seed)
    shuffled = uniq[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        # Degenerate: everything must go to train (cannot make disjoint val/test).
        return {"train": shuffled, "val": [], "test": []}
    if n == 2:
        # Give one to train, one to test; val stays empty (can't be disjoint x3).
        return {"train": [shuffled[0]], "val": [], "test": [shuffled[1]]}

    # n >= 3: fractional split, then guarantee >=1 in val and >=1 in test.
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))
    n_val = max(1, n_val)
    n_test = max(1, n_test)
    # Never let val+test starve train of at least one conversation.
    while n_val + n_test > n - 1:
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            # both at 1 and still no room (n==2 handled above, so n>=3 => room)
            break

    val_ids = shuffled[:n_val]
    test_ids = shuffled[n_val:n_val + n_test]
    train_ids = shuffled[n_val + n_test:]
    return {"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)}


def assert_disjoint(splits: Dict[str, List[str]]) -> Dict[str, Any]:
    """Assert the three conversation_id sets are pairwise disjoint (§2.5).

    Fails LOUDLY (raises AssertionError) on any overlap. Returns a small dict
    describing the check so it can be recorded in data_stats.json.
    """
    s_train = set(splits["train"])
    s_val = set(splits["val"])
    s_test = set(splits["test"])

    overlaps = {
        "train_val": sorted(s_train & s_val),
        "train_test": sorted(s_train & s_test),
        "val_test": sorted(s_val & s_test),
    }
    any_overlap = any(overlaps.values())
    assert not any_overlap, (
        "conversation_id leakage across splits! overlaps="
        + json.dumps(overlaps, ensure_ascii=False)
    )
    return {
        "disjoint": True,
        "pairwise_overlaps": overlaps,
        "n_conversations_total": len(s_train | s_val | s_test),
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _t_distribution(ts: Sequence[int]) -> Dict[str, Any]:
    if not ts:
        return {"min": None, "median": None, "mean": None, "max": None, "count": 0}
    return {
        "min": int(min(ts)),
        "median": float(statistics.median(ts)),
        "mean": float(statistics.mean(ts)),
        "max": int(max(ts)),
        "count": len(ts),
    }


def _w_balance(ws: Sequence[int]) -> Dict[str, Any]:
    c = Counter(ws)
    total = len(ws)
    return {
        "write_w1": int(c.get(1, 0)),
        "read_w0": int(c.get(0, 0)),
        "total": total,
        "write_frac": (c.get(1, 0) / total) if total else None,
    }


def build_stats(
    *,
    input_path: str,
    n_records: int,
    n_exploded_raw: int,
    n_after_dedup: int,
    all_samples: Sequence[dict],
    split_samples: Dict[str, List[dict]],
    splits: Dict[str, List[str]],
    disjoint_result: Dict[str, Any],
    args: argparse.Namespace,
    tokenizer_name: str,
) -> Dict[str, Any]:
    ws = [s["w"] for s in all_samples]
    ts = [s["t"] for s in all_samples]

    usage_present = sum(1 for s in all_samples if s.get("usage_output_tokens") is not None)

    per_split = {}
    for name in ("train", "val", "test"):
        rows = split_samples[name]
        per_split[name] = {
            "n_samples": len(rows),
            "n_conversations": len(splits[name]),
            "conversation_ids": splits[name],
            "w_balance": _w_balance([r["w"] for r in rows]),
            "t_distribution": _t_distribution([r["t"] for r in rows]),
        }

    return {
        "input": input_path,
        "tokenizer": tokenizer_name,
        "config": {
            "seed": args.seed,
            "max_len": args.max_len,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "dual_use_as_write": args.dual_use_as_write,
            "conversation_id_field": args.conversation_id_field,
        },
        "n_records": n_records,
        # §2.3 explosion: explosion count (raw, before dedup) and after dedup.
        "explosion_count_raw": n_exploded_raw,
        "explosion_count_after_dedup": n_after_dedup,
        "exploded_vs_records_note": (
            f"{n_exploded_raw} raw exploded samples from {n_records} records "
            f"(>> {n_records}); {n_after_dedup} remain after (context_hash,w,t) dedup"
        ),
        # overall w balance + t distribution (across all kept samples)
        "w_balance": _w_balance(ws),
        "t_distribution": _t_distribution(ts),
        "usage_output_tokens_present": usage_present,
        "usage_output_tokens_note": (
            "reference-only column; NOT the target. Present only on top-level "
            "response turns; null for exploded historical turns."
        ),
        # §2.5 split provenance + disjointness proof
        "split": {
            "group_key": "conversation_id (derived; NOT session_id)",
            "per_split": per_split,
            "conversation_ids_per_split": {
                "train": splits["train"],
                "val": splits["val"],
                "test": splits["test"],
            },
            "disjoint_assertion": disjoint_result,
        },
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _resolve_conversation_ids(
    records: Sequence[dict], field: Optional[str]
) -> Tuple[List[str], str]:
    """Return (conversation_ids aligned with records, provenance string).

    If ``field`` is given AND every record carries a usable value for it, use
    that real grouping field directly (§8: a larger dataset may already have a
    real conversation/session field). Otherwise fall back to the canonical
    corpus-level derivation :func:`labeling.derive_conversation_ids` (§2.2.3),
    which applies the longest-common-prefix merge across all records.
    """
    if field:
        vals = [r.get(field) for r in records]
        if all(v is not None and str(v) != "" for v in vals):
            return [str(v) for v in vals], f"override field '{field}'"
        print(
            f"[prepare_data] --conversation-id-field '{field}' requested but not "
            f"present/complete on all records; falling back to derived id."
        )
    return list(labeling.derive_conversation_ids(records)), "labeling.derive_conversation_ids (LCP merge)"


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    records = load_input(args.input)
    n_records = len(records)

    # Tokenizer: real Qwen3 if available, else the lightweight fallback so this
    # runs offline (the fallback is provided by labeling.get_tokenizer).
    force_fallback = os.environ.get("PREPARE_FORCE_FALLBACK") == "1"
    if force_fallback:
        tokenizer = labeling.SimpleWhitespaceTokenizer()
    else:
        tokenizer = labeling.get_tokenizer()
    tokenizer_name = type(tokenizer).__name__
    is_fallback = isinstance(tokenizer, labeling.SimpleWhitespaceTokenizer)
    print(
        f"[prepare_data] tokenizer: {tokenizer_name}"
        + (" (FALLBACK — offline)" if is_fallback else " (real Qwen3)")
    )

    # conversation_id per record (corpus-level; NOT session_id).
    conv_ids, conv_provenance = _resolve_conversation_ids(records, args.conversation_id_field)
    print(
        f"[prepare_data] conversation_id source: {conv_provenance}; "
        f"distinct conversations: {len(set(conv_ids))} "
        f"(distinct session_id, the WRONG key: {len({r.get('session_id') for r in records})})"
    )

    # --- §2.3 explode every record ---
    # Render cache shared across ALL records so a context shared between records
    # (common within one conversation) is rendered once. Speed-only memoization.
    render_cache: Dict[str, str] = {}
    all_samples: List[dict] = []
    for record, cid in zip(records, conv_ids):
        all_samples.extend(
            explode_record(
                record,
                conversation_id=cid,
                tokenizer=tokenizer,
                max_len=args.max_len,
                max_block_chars=args.max_block_chars,
                dual_use_as_write=args.dual_use_as_write,
                render_cache=render_cache,
            )
        )
    n_exploded_raw = len(all_samples)
    print(f"[prepare_data] exploded {n_records} records -> {n_exploded_raw} raw samples")

    # --- §2.3 dedup ---
    all_samples = dedup_samples(all_samples)
    n_after_dedup = len(all_samples)
    print(f"[prepare_data] after (context_hash,w,t) dedup: {n_after_dedup} samples")

    # --- §2.5 group-split by conversation_id ---
    splits = group_split_conversations(
        conv_ids, seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac
    )
    disjoint_result = assert_disjoint(splits)  # raises loudly on any overlap
    print(
        f"[prepare_data] split conversations -> "
        f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} "
        f"(disjoint={disjoint_result['disjoint']})"
    )

    conv_to_split = {}
    for name in ("train", "val", "test"):
        for cid in splits[name]:
            conv_to_split[cid] = name

    split_samples: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for s in all_samples:
        name = conv_to_split.get(s["conversation_id"])
        if name is None:
            # Should never happen (every conv assigned), but never silently drop.
            raise AssertionError(
                f"sample conversation_id {s['conversation_id']!r} not assigned to any split"
            )
        split_samples[name].append(s)

    # Defensive: confirm sample-level disjointness of conversation_ids too.
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ca = {s["conversation_id"] for s in split_samples[a]}
        cb = {s["conversation_id"] for s in split_samples[b]}
        assert not (ca & cb), f"sample-level conversation_id overlap between {a} and {b}: {ca & cb}"

    # --- §2.6 emit artifacts ---
    os.makedirs(args.outdir, exist_ok=True)

    def _clean(rows: Sequence[dict]) -> List[dict]:
        # Strip the internal-only dedup key; keep the §2.6 schema exactly.
        out = []
        for r in rows:
            out.append(
                {
                    "prompt": r["prompt"],
                    "completion": r["completion"],
                    "w": r["w"],
                    "t": r["t"],
                    "conversation_id": r["conversation_id"],
                    "session_id": r["session_id"],
                    "usage_output_tokens": r["usage_output_tokens"],
                }
            )
        return out

    for name in ("train", "val", "test"):
        path = os.path.join(args.outdir, f"{name}.jsonl")
        write_jsonl(path, _clean(split_samples[name]))
        print(f"[prepare_data] wrote {path} ({len(split_samples[name])} samples)")

    stats = build_stats(
        input_path=args.input,
        n_records=n_records,
        n_exploded_raw=n_exploded_raw,
        n_after_dedup=n_after_dedup,
        all_samples=all_samples,
        split_samples=split_samples,
        splits=splits,
        disjoint_result=disjoint_result,
        args=args,
        tokenizer_name=tokenizer_name,
    )
    stats_path = os.path.join(args.outdir, "data_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[prepare_data] wrote {stats_path}")

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Explode, label (via shared labeling.py), and group-split "
        "demo_data.jsonl into train/val/test for the Qwen3 classifier."
    )
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="input: a JSONL file, OR a directory of .json files "
                        "(each .json = one record / one JSONL line). "
                        "Default: repo demo_data.jsonl")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="output dir for prepared artifacts")
    p.add_argument("--max-len", type=int, default=4096, help="context truncation length in tokens (§2.4)")
    p.add_argument(
        "--max-block-chars", type=int, default=4000,
        help="per-block char cap for oversized blocks (§2.4)",
    )
    p.add_argument("--seed", type=int, default=42, help="fixed seed for the group split (§2.5)")
    p.add_argument("--val-frac", type=float, default=0.1, help="validation fraction of conversations (8/1/1)")
    p.add_argument("--test-frac", type=float, default=0.1, help="test fraction of conversations (8/1/1)")
    p.add_argument(
        "--dual-use-as-write", dest="dual_use_as_write", action="store_true", default=True,
        help="treat dual-use tools (run_terminal_cmd/bash) as write (default: True; §2.2.1)",
    )
    p.add_argument(
        "--no-dual-use-as-write", dest="dual_use_as_write", action="store_false",
        help="exclude dual-use tools from the write class",
    )
    p.add_argument(
        "--conversation-id-field", default=None,
        help="optional override: use this record field as conversation_id if the "
        "source already has a real conversation/session grouping field (§8). "
        "Falls back to the derived id when absent.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    stats = prepare(args)
    # Final compact human summary.
    print("=" * 72)
    print(
        "SUMMARY: records={n_records} exploded_raw={raw} after_dedup={dd} | "
        "w(write/read)={w1}/{w0} | t(min/med/mean/max)={tmin}/{tmed}/{tmean}/{tmax} | "
        "splits(train/val/test samples)={trs}/{vas}/{tes} convs={trc}/{vac}/{tec} "
        "disjoint={dj}".format(
            n_records=stats["n_records"],
            raw=stats["explosion_count_raw"],
            dd=stats["explosion_count_after_dedup"],
            w1=stats["w_balance"]["write_w1"],
            w0=stats["w_balance"]["read_w0"],
            tmin=stats["t_distribution"]["min"],
            tmed=stats["t_distribution"]["median"],
            tmean=round(stats["t_distribution"]["mean"], 1),
            tmax=stats["t_distribution"]["max"],
            trs=stats["split"]["per_split"]["train"]["n_samples"],
            vas=stats["split"]["per_split"]["val"]["n_samples"],
            tes=stats["split"]["per_split"]["test"]["n_samples"],
            trc=stats["split"]["per_split"]["train"]["n_conversations"],
            vac=stats["split"]["per_split"]["val"]["n_conversations"],
            tec=stats["split"]["per_split"]["test"]["n_conversations"],
            dj=stats["split"]["disjoint_assertion"]["disjoint"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

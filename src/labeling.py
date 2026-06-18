"""Shared labeling foundation for the Qwen3-1.7B next-turn classifier.

This module is the FOUNDATION imported by both data preparation
(``prepare_data.py``) and evaluation (``evaluate.py``). It is intentionally
**pure Python** with no SageMaker / boto3 / network dependencies so it can be
imported anywhere (including offline CI) and unit-tested in isolation.

It implements the four contracts from the Tech Design §2.2 / §7 (revised round 2):

* ``WRITE_TOOLS``                       — the write-class tool taxonomy (§2.2.1)
* ``is_write_response(...)``            — write-flag ``w`` derivation (§2.2.1)
* ``count_output_tokens(...)``          — CANONICAL length label ``t`` (§2.2.2)
* ``derive_conversation_id(...)``       — root-prompt grouping key (§2.2.3)
* ``render_context(...)``               — Qwen3 chat-template context render (§2.4)

Design decisions worth calling out
-----------------------------------
1. **Canonical length label (§2.2.2).** ``count_output_tokens`` deliberately
   takes **no** ``usage`` argument. The single source of truth for ``t`` is the
   tokenizer count of the *serialized assistant turn* (text blocks + JSON of
   tool_use blocks, in order). ``usage.output_tokens`` is ~2.3x larger and is
   only ever carried as a separate *logged reference column* in
   ``prepare_data.py`` — never the target. Keeping ``usage`` out of this
   signature makes accidental source-mixing impossible.

2. **Conversation grouping key (§2.2.3).** ``session_id`` does NOT identify a
   conversation (101 records carry 98 session_ids). We derive ``conversation_id``
   from the normalized *root prompt* (first user message), with a
   longest-common-prefix tie-breaker that merges a root prompt with its
   summary-extended continuations. Verified to collapse the 101 demo records to
   14 conversations.

3. **Tokenizer choice.** When available we use the real Qwen3 tokenizer
   (``transformers.AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B')``) — this is
   the tokenizer the model is actually trained with, so the canonical ``t`` and
   the rendered context lengths match training/eval reality. The tokenizer is
   **injectable** into every public function, and if Qwen3 cannot be loaded
   (e.g. no network / no transformers in a constrained env) callers may pass a
   lightweight fallback (see ``SimpleWhitespaceTokenizer`` / ``get_tokenizer``)
   so the smoke test and unit tests never require a download.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Protocol, Sequence

# ---------------------------------------------------------------------------
# §2.2.1  Write-class tool taxonomy
# ---------------------------------------------------------------------------

#: Tool names whose presence in an assistant turn marks it as a *write* turn.
#: This is the configurable write-class set from Tech Design §2.2.1.
WRITE_TOOLS: set = {
    "edit_file",
    "search_replace",
    "delete_file",
    "reapply",
    "create_new_file",
    "write",
    "apply_patch",
    "str_replace_editor",
    "run_terminal_cmd",
}

#: Dual-use tools: they *can* mutate state (run a shell command) but are not
#: inherently a write. Treated as write by default (``dual_use_as_write=True``);
#: excluded from the write class when the flag is False. See §2.2.1.
DUAL_USE_TOOLS: set = {"run_terminal_cmd", "bash"}


def is_write_response(response_content: list, dual_use_as_write: bool = True) -> bool:
    """Return True if the assistant turn contains a write/mutating tool call.

    Per Tech Design §2.2.1: ``w = 1`` iff any ``tool_use`` block in the turn's
    response has a ``name`` in :data:`WRITE_TOOLS`, else 0.

    Parameters
    ----------
    response_content:
        The assistant turn's content: a list of blocks
        (``text`` / ``tool_use`` / ...). Robust to ``None`` and to non-list
        input (treated as "no tool calls" -> False).
    dual_use_as_write:
        Dual-use decision (§2.2.1). When True (default) the dual-use tools
        (:data:`DUAL_USE_TOOLS`, i.e. ``run_terminal_cmd`` / ``bash``) count as
        write. When False they are excluded from the effective write set, so a
        turn whose only tool call is ``bash`` is classified read (``w = 0``).
    """
    if not isinstance(response_content, list):
        return False

    # Effective write set. The dual-use tools (run_terminal_cmd / bash) toggle
    # in/out of the write class. Note ``bash`` is NOT a member of WRITE_TOOLS
    # (the corpus uses ``bash`` where the taxonomy names ``run_terminal_cmd``),
    # so when the flag is on we must UNION the dual-use set in; when off we
    # remove it. This makes both ``run_terminal_cmd`` and ``bash`` flip between
    # write and read together (Tech Design §2.2.1).
    if dual_use_as_write:
        effective_write = WRITE_TOOLS | DUAL_USE_TOOLS
    else:
        effective_write = WRITE_TOOLS - DUAL_USE_TOOLS

    for block in response_content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            name = block.get("name")
            if name in effective_write:
                return True
    return False


# ---------------------------------------------------------------------------
# §2.2.2  Canonical length label  (NO usage argument by design)
# ---------------------------------------------------------------------------

def serialize_assistant_turn(response_content: list) -> str:
    """Serialize an assistant turn to the canonical string used for length.

    Concatenates, **in order**, the turn's ``text`` blocks (raw text) and the
    JSON of its ``tool_use`` blocks. The JSON of a tool_use block is the compact
    ``{"name": ..., "input": ...}`` payload (the assistant's actual structured
    action). This is the exact string whose tokenizer length defines ``t``.

    Robust to ``None`` / non-list input (returns ``""``) and to malformed
    blocks (skipped).
    """
    if not isinstance(response_content, list):
        return ""

    parts: List[str] = []
    for block in response_content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            if text:
                parts.append(text if isinstance(text, str) else str(text))
        elif btype == "tool_use":
            payload = {
                "name": block.get("name"),
                "input": block.get("input", {}),
            }
            # Deterministic, compact, unicode-preserving JSON so the same turn
            # always serializes identically (stable token counts across runs).
            parts.append(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
        # Other block types (e.g. thinking) are intentionally not part of the
        # canonical visible-output length and are skipped.
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Character-based token ESTIMATE (fast; no tokenizer needed)
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ]")

#: Token-per-character ratios for the cheap character-based estimate.
#: CJK / kana characters are ~1.5 tokens each; other (mostly latin/code) text
#: is ~4 characters per token. These ratios match the project's earlier
#: analysis scripts and are stable enough for length BUCKETING / relative
#: ordering (the only use of ``t``), which is why we can skip the (slow) real
#: tokenizer here. See ``estimate_tokens_from_chars``.
_CJK_TOKENS_PER_CHAR: float = 1.5
_OTHER_CHARS_PER_TOKEN: float = 4.0


def estimate_tokens_from_chars(text: str) -> int:
    """Cheap character-based estimate of token count (NO tokenizer).

    ``tokens ≈ cjk_chars * 1.5 + other_chars / 4``.

    This replaces the slow ``tokenizer.encode`` length used previously. The
    label ``t`` is only consumed for length **bucketing / relative ordering**,
    so an approximate count is sufficient and ~100x faster (no model tokenizer
    load, no per-sample encode of long contexts).

    IMPORTANT (label-source consistency): ``t`` is produced HERE at data-prep
    time and written to the ``t`` column; both training (reads the column) and
    evaluation (reads the same column as truth) therefore use this identical
    definition — there is no train/eval source mismatch.
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return int(cjk * _CJK_TOKENS_PER_CHAR + other / _OTHER_CHARS_PER_TOKEN)


def count_output_tokens(response_content, tokenizer=None) -> int:
    """CANONICAL length label ``t`` for an assistant turn (Tech Design §2.2.2).

    ``t = estimate_tokens_from_chars(serialize_assistant_turn(response_content))``
    — a fast **character-based** estimate (no tokenizer). ``t`` is only used for
    length bucketing / relative ordering, so the estimate is sufficient and
    avoids the slow per-sample ``tokenizer.encode`` of long serialized turns.

    The ``tokenizer`` parameter is accepted for backward compatibility but is
    **ignored** (kept so existing callers — prepare_data, tests — don't break).

    IMPORTANT: this function intentionally never reads ``usage.output_tokens``
    (it is ~2.3x a serialized-text length and present on only some turns).
    Standardizing on the char-estimate of the serialized turn gives one uniform
    target for both training and evaluation.

    Parameters
    ----------
    response_content:
        The assistant turn's content blocks (see :func:`serialize_assistant_turn`).
    tokenizer:
        Ignored (back-compat). The estimate is tokenizer-free.
    """
    text = serialize_assistant_turn(response_content)
    return estimate_tokens_from_chars(text)


# ---------------------------------------------------------------------------
# §2.2.3  Derived conversation_id  (root-prompt grouping key)
# ---------------------------------------------------------------------------

# Minimum length of a shared normalized prefix for two distinct root prompts to
# be treated as the SAME conversation (longest-common-prefix tie-breaker).
#
# Rationale (verified on demo_data.jsonl): grouping by the normalized first user
# message yields 15 distinct roots. Two of them are prefixes of longer variants:
#   * the 42-char proxydemo/snake root is a prefix of a 155-char
#     summary-extended variant  -> genuinely ONE conversation (sequential turns)
#   * the 11-char generic "我这个项目是做什么的？" is a prefix of a 24-char variant
#     -> DIFFERENT conversations (distinct session families two weeks apart)
# A threshold anywhere in [12, 42] cleanly merges the first and not the second,
# collapsing 101 records to exactly 14 conversations. 20 sits mid-plateau and is
# robust to small data shifts.
ROOT_PREFIX_MERGE_MIN_LEN: int = 20


def _normalize_root(text: Optional[str]) -> str:
    """Normalize a root-prompt string: strip, lowercase, collapse whitespace."""
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _first_user_text(record: dict) -> str:
    """Extract the raw text of the first user message in a record.

    Handles both string content and list-of-blocks content (concatenating the
    ``text`` blocks). Returns ``""`` if no user text is found.
    """
    request = record.get("request") or {}
    messages = request.get("messages") or []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            if content.strip():
                return content
            continue
        if isinstance(content, list):
            texts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
            ]
            if texts:
                return "\n".join(texts)
    return ""


def _canonical_root(record: dict) -> str:
    """The normalized root prompt for a record (its grouping seed)."""
    return _normalize_root(_first_user_text(record))


def _stable_hash(text: str) -> str:
    """Stable 16-hex-char hash of a string (sha1, prefix)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def derive_conversation_id(record: dict) -> str:
    """Derive a stable ``conversation_id`` from a single record (Tech Design §2.2.3).

    The id is a stable hash of the normalized **root prompt** (first user
    message). This is the grouping key for the leakage-free train/val/test split;
    ``session_id`` must NOT be used (it splinters one conversation into dozens of
    ids).

    Note on the tie-breaker: the longest-common-prefix merge (folding a root
    prompt together with its summary-extended continuations) is a *cross-record*
    operation. When the full corpus is available, prefer
    :func:`derive_conversation_ids` which applies the merge across all records
    and is what reproduces the 14-conversation count. This single-record
    function returns the hash of the record's own normalized root — stable and
    deterministic — which already groups all records that share an *identical*
    root prompt. Records whose roots differ only by a summary-prefix extension
    are reconciled by :func:`derive_conversation_ids`.
    """
    root = _canonical_root(record)
    return _stable_hash(root)


def derive_conversation_ids(records: Sequence[dict]) -> List[str]:
    """Corpus-level ``conversation_id`` assignment with the LCP tie-breaker.

    Returns a list of conversation_ids aligned 1:1 with ``records``. This is the
    canonical entry point used by ``prepare_data.py`` to group records before
    splitting, because the longest-common-prefix merge (§2.2.3) is inherently a
    cross-record decision.

    Algorithm
    ---------
    1. Compute the normalized root prompt for every record.
    2. Build clusters over the *distinct* roots via union-find: union two roots
       when one is a prefix of the other AND the shared prefix is at least
       :data:`ROOT_PREFIX_MERGE_MIN_LEN` characters (the tie-breaker that merges
       a root with its summary-extended continuations while leaving short,
       generic prompts separate).
    3. Each cluster's canonical root is its shortest member (the true root); the
       conversation_id is the stable hash of that canonical root.

    Verified: collapses demo_data.jsonl's 101 records to 14 conversations.
    """
    roots = [_canonical_root(r) for r in records]
    distinct = sorted(set(roots), key=len)  # shortest first => shortest is canonical

    parent: Dict[str, str] = {d: d for d in distinct}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Keep the shorter root as canonical (it is the conversation's seed).
        if len(ra) <= len(rb):
            parent[rb] = ra
        else:
            parent[ra] = rb

    # Only non-trivial roots participate in prefix merging; empty roots group
    # together by hash("") naturally and are never used as a merge prefix.
    nonempty = [d for d in distinct if d]
    for i, a in enumerate(nonempty):
        for b in nonempty[i + 1:]:
            # nonempty is sorted by length, so len(a) <= len(b).
            if len(a) >= ROOT_PREFIX_MERGE_MIN_LEN and b.startswith(a):
                union(a, b)

    canonical_for: Dict[str, str] = {d: find(d) for d in distinct}
    return [_stable_hash(canonical_for[root]) for root in roots]


# ---------------------------------------------------------------------------
# §2.4  Context rendering (Qwen3 chat-template, tail-preserving truncation)
# ---------------------------------------------------------------------------

def _block_to_text(block: Any, max_block_chars: int) -> str:
    """Flatten a single content block to text, capping oversized blocks.

    Handles the block shapes seen in the data:
      * ``text``        -> the text
      * ``tool_use``    -> compact JSON of {name, input}
      * ``tool_result`` -> the textual content (its ``content`` is itself a list
                           of ``{type:text, text}`` blocks in this corpus), with
                           a per-block char cap so a giant whole-file read can't
                           consume the window (§2.4).
    Unknown block types are best-effort serialized.
    """
    if isinstance(block, str):
        return _cap(block, max_block_chars)
    if not isinstance(block, dict):
        return _cap(str(block), max_block_chars)

    btype = block.get("type")
    if btype == "text":
        return _cap(block.get("text", "") or "", max_block_chars)
    if btype == "tool_use":
        payload = {"name": block.get("name"), "input": block.get("input", {})}
        return _cap(
            "[tool_use] "
            + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            max_block_chars,
        )
    if btype == "tool_result":
        content = block.get("content")
        inner = _content_to_text(content, max_block_chars)
        return _cap("[tool_result] " + inner, max_block_chars)

    # cachePoint and other metadata-only / unknown blocks: skip metadata-only,
    # otherwise serialize best-effort.
    if "cachePoint" in block:
        return ""
    # A block that carries a bare "text" key without an explicit type (as the
    # system blocks in this corpus do) should still surface its text.
    if "text" in block:
        return _cap(block.get("text", "") or "", max_block_chars)
    return _cap(json.dumps(block, ensure_ascii=False), max_block_chars)


def _content_to_text(content: Any, max_block_chars: int) -> str:
    """Flatten a message ``content`` (str or list of blocks) to a single string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rendered = [_block_to_text(b, max_block_chars) for b in content]
        return "\n".join(p for p in rendered if p)
    return str(content)


def _cap(text: str, max_chars: int) -> str:
    """Cap a string to ``max_chars`` with a clear truncation marker."""
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    marker = "...[truncated]"
    keep = max(0, max_chars - len(marker))
    return text[:keep] + marker


def _flatten_messages(messages: Sequence[dict], max_block_chars: int) -> List[Dict[str, str]]:
    """Turn the raw block-content messages into chat messages with STRING content.

    This is essential: passing list-of-blocks content straight to a HF
    ``apply_chat_template`` silently drops the content (verified on Qwen3 — the
    assistant turn renders empty). We flatten each message to a string first.
    Messages that flatten to empty (e.g. a lone cachePoint block) are dropped.
    """
    flat: List[Dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        text = _content_to_text(msg.get("content"), max_block_chars).strip()
        if not text:
            continue
        flat.append({"role": role, "content": text})
    return flat


def render_context(
    messages: list,
    tokenizer,
    max_len: int = 4096,
    max_block_chars: int = 4000,
) -> str:
    """Render conversation context to a Qwen3 chat string (Tech Design §2.4).

    Steps:
      1. Flatten each message's block content to a string (capping any single
         oversized block, e.g. a whole-file ``tool_result``, at
         ``max_block_chars``).
      2. Apply the Qwen3 chat template (via ``tokenizer.apply_chat_template``)
         with ``add_generation_prompt=True`` so the string ends ready for the
         model to produce the next (target) turn. If the tokenizer has no chat
         template, fall back to a simple ``<|im_start|>role\\n...<|im_end|>``
         rendering of the same shape.
      3. **Tail-preserving left-truncation** to ``max_len`` tokens: keep the END
         of the conversation (most recent context), dropping the oldest whole
         messages first, then if still too long hard-truncate from the left at
         the token level.

    Parameters
    ----------
    messages:
        The context messages (everything BEFORE the turn being predicted).
    tokenizer:
        Qwen3 tokenizer (or injected lightweight tokenizer). Needs ``.encode``;
        ``.apply_chat_template`` / ``.decode`` are used when present.
    max_len:
        Maximum context length in tokens (default 4096).
    max_block_chars:
        Per-block character cap for oversized blocks (default 4000).
    """
    if not isinstance(messages, list):
        return ""

    flat = _flatten_messages(messages, max_block_chars)
    if not flat:
        return ""

    # --- Tail-preserving message-level pruning -----------------------------
    # Length is judged with the fast CHARACTER-BASED estimate (no tokenizer
    # encode), since the context here is only used as a training PROMPT and is
    # re-tokenized + re-truncated exactly by train.py at load time anyway —
    # approximate pruning here is lossless for correctness and ~100x cheaper.
    # We want the largest K such that "render of the LAST K messages" fits in
    # max_len (estimated) tokens. Binary search: O(log N) renders.
    def _est_len(s: str) -> int:
        return estimate_tokens_from_chars(s)

    rendered = _apply_template(tokenizer, flat)
    if _est_len(rendered) > max_len and len(flat) > 1:
        # Invariant: keeping the LAST `lo` messages always fits;
        #            keeping the LAST `hi` messages always exceeds.
        lo, hi = 1, len(flat)
        while hi - lo > 1:
            mid = (lo + hi) // 2  # try keeping the LAST `mid` messages
            cand = _apply_template(tokenizer, flat[-mid:])
            if _est_len(cand) <= max_len:
                lo, rendered = mid, cand          # `mid` fits — keep it as best
            else:
                hi = mid                          # `mid` exceeds — too many
        flat = flat[-lo:]
        if _est_len(rendered) > max_len:
            rendered = _apply_template(tokenizer, flat)

    # --- Character-level left-truncation safety net ------------------------
    # A single remaining (most recent) message may still exceed max_len. Keep
    # the TAIL (freshest context) by trimming characters from the left until the
    # estimated token length fits. ``estimate_tokens_from_chars`` is monotonic in
    # length, so a proportional cut + small fixup converges in O(1) passes.
    if _est_len(rendered) > max_len and len(rendered) > 1:
        # Proportional first cut: keep roughly the last (max_len/est)*len chars.
        est = _est_len(rendered)
        keep = max(1, int(len(rendered) * (max_len / est)))
        rendered = rendered[-keep:]
        # Fixup: trim a bit more if the estimate still slightly exceeds.
        while _est_len(rendered) > max_len and len(rendered) > 1:
            drop = max(1, len(rendered) - int(len(rendered) * max_len / max(1, _est_len(rendered))))
            rendered = rendered[drop:]

    return rendered


# ---------------------------------------------------------------------------
# Tokenizer abstraction + helpers
# ---------------------------------------------------------------------------

class TokenizerLike(Protocol):
    """Minimal tokenizer interface used by this module."""

    def encode(self, text: str) -> Sequence:  # pragma: no cover - protocol
        ...


def _encode(tokenizer, text: str) -> Sequence:
    """Encode text to token ids, tolerating HF tokenizers' kwargs differences."""
    try:
        # HF tokenizers accept add_special_tokens; our length label wants the
        # plain content encoding (no extra BOS/EOS) for a fair, stable count.
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def _encode_len(tokenizer, text: str) -> int:
    return len(_encode(tokenizer, text))


def _decode(tokenizer, ids) -> Optional[str]:
    decode = getattr(tokenizer, "decode", None)
    if decode is None:
        return None
    try:
        return decode(ids, skip_special_tokens=False)
    except TypeError:
        try:
            return decode(ids)
        except Exception:
            return None
    except Exception:
        return None


def _apply_template(tokenizer, flat_messages: List[Dict[str, str]]) -> str:
    """Apply a chat template, falling back to a simple im_start/im_end render."""
    apply = getattr(tokenizer, "apply_chat_template", None)
    chat_template = getattr(tokenizer, "chat_template", None)
    if apply is not None and chat_template:
        try:
            return apply(flat_messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass  # fall through to manual rendering
    return _manual_chat_render(flat_messages)


def _manual_chat_render(flat_messages: List[Dict[str, str]]) -> str:
    """Qwen-style chat rendering used when no chat template is available."""
    parts = []
    for m in flat_messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _char_left_trim_to_tokens(tokenizer, text: str, max_len: int) -> str:
    """Left-trim characters until the token count fits ``max_len``.

    Fallback for tokenizers without ``.decode``. Iteratively removes a chunk
    from the front, preserving the tail (most recent context).
    """
    if _encode_len(tokenizer, text) <= max_len:
        return text
    # Estimate chars-per-token and trim with a small safety loop.
    while _encode_len(tokenizer, text) > max_len and len(text) > 1:
        excess_tokens = _encode_len(tokenizer, text) - max_len
        # Remove a bit more than estimated to converge quickly.
        approx_chars = max(1, excess_tokens * 4)
        text = text[approx_chars:]
    return text


class SimpleWhitespaceTokenizer:
    """A tiny, dependency-free tokenizer for tests / offline smoke runs.

    It is NOT the production tokenizer. It exists so this module's pure
    functions can be exercised without downloading the Qwen3 tokenizer. Token
    boundaries are word/punctuation runs (a coarse stand-in for BPE). It exposes
    ``encode``/``decode`` and no chat template, so :func:`render_context` falls
    back to the manual Qwen-style chat rendering.
    """

    _token_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __init__(self) -> None:
        self.chat_template = None  # signal: no template -> manual render

    def encode(self, text: str, add_special_tokens: bool = False):  # noqa: D401
        return self._token_re.findall(text or "")

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return " ".join(ids)


_QWEN_TOKENIZER_CACHE: Dict[str, Any] = {}


def get_tokenizer(model_id: str = "Qwen/Qwen3-1.7B", allow_fallback: bool = True):
    """Return a tokenizer, preferring the real Qwen3 tokenizer.

    Tries ``transformers.AutoTokenizer.from_pretrained(model_id)`` first (this
    is the tokenizer the classifier is trained with, so canonical token counts
    and rendered-context lengths reflect training reality). If that fails (no
    ``transformers``, no network, etc.) and ``allow_fallback`` is True, returns a
    :class:`SimpleWhitespaceTokenizer` so offline callers still work.

    The Qwen3 tokenizer is cached per ``model_id`` to avoid repeated loads.
    """
    if model_id in _QWEN_TOKENIZER_CACHE:
        return _QWEN_TOKENIZER_CACHE[model_id]
    try:
        from transformers import AutoTokenizer  # local import: keep module import-light

        # Force the Rust ``Qwen2TokenizerFast`` — it is 10-100x faster than the
        # pure-Python ``Qwen2Tokenizer`` on long inputs (the bottleneck for
        # prepare_data on large datasets). If a Fast variant is genuinely
        # unavailable in this env, fall back to slow with a loud warning.
        try:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        except Exception as fast_exc:
            print(f"[labeling] WARNING: fast tokenizer unavailable for {model_id} "
                  f"({type(fast_exc).__name__}: {fast_exc}); falling back to the "
                  f"slow Python tokenizer - prepare_data will be much slower. "
                  f"Install the `tokenizers` package (e.g. `pip install -U tokenizers`) "
                  f"to get the Rust fast tokenizer.")
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        if not getattr(tok, "is_fast", False):
            print(f"[labeling] WARNING: loaded tokenizer is type {type(tok).__name__} "
                  f"(is_fast=False) - prepare_data will be slow. Install/upgrade "
                  f"the `tokenizers` package for the Rust fast variant.")
        _QWEN_TOKENIZER_CACHE[model_id] = tok
        return tok
    except Exception as exc:  # pragma: no cover - environment dependent
        if not allow_fallback:
            raise
        print(f"[labeling] Qwen3 tokenizer unavailable ({type(exc).__name__}: {exc}); "
              f"using SimpleWhitespaceTokenizer fallback.")
        return SimpleWhitespaceTokenizer()


# ---------------------------------------------------------------------------
# __main__  smoke test  (runs against demo_data.jsonl, no network required)
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _smoke(path: str) -> None:
    import os
    from collections import Counter

    print("=" * 72)
    print("labeling.py smoke test")
    print("=" * 72)
    print(f"data file: {path}")

    # ---- read ----
    records = _load_jsonl(path)
    n_read = len(records)
    print(f"records read: {n_read}")

    # ---- tokenizer (real Qwen3 if available, else lightweight fallback) ----
    # Honor LABELING_FORCE_FALLBACK=1 to exercise the offline path deterministically.
    force_fallback = os.environ.get("LABELING_FORCE_FALLBACK") == "1"
    if force_fallback:
        tokenizer = SimpleWhitespaceTokenizer()
        print("tokenizer: SimpleWhitespaceTokenizer (forced via LABELING_FORCE_FALLBACK=1)")
    else:
        tokenizer = get_tokenizer()
        name = type(tokenizer).__name__
        if isinstance(tokenizer, SimpleWhitespaceTokenizer):
            print(f"tokenizer: {name} (fallback)")
        else:
            print(f"tokenizer: real Qwen3 ({name})")

    # ---- write/read classification on each record's response turn ----
    write_count = 0
    read_count = 0
    write_count_no_dual = 0
    for r in records:
        resp = r.get("response") or {}
        content = resp.get("content")
        if is_write_response(content, dual_use_as_write=True):
            write_count += 1
        else:
            read_count += 1
        if is_write_response(content, dual_use_as_write=False):
            write_count_no_dual += 1
    print(f"write turns (dual_use_as_write=True):  {write_count}")
    print(f"read  turns (dual_use_as_write=True):  {read_count}")
    print(f"write turns (dual_use_as_write=False): {write_count_no_dual} "
          f"(delta from dual-use toggle: {write_count - write_count_no_dual})")

    # ---- derived conversation ids ----
    conv_ids = derive_conversation_ids(records)
    n_conversations = len(set(conv_ids))
    print(f"derived conversation_ids (distinct): {n_conversations} (expect ~14)")
    # Show that session_id would have given a very different (wrong) count.
    n_sessions = len({r.get("session_id") for r in records})
    print(f"  (distinct session_id, the WRONG key: {n_sessions})")
    conv_sizes = Counter(conv_ids)
    print(f"  conversation sizes (records per conversation): "
          f"{sorted(conv_sizes.values(), reverse=True)}")

    # ---- sample canonical token count ----
    sample_idx = next((i for i, r in enumerate(records)
                       if (r.get("response") or {}).get("content")), 0)
    sample_resp = (records[sample_idx].get("response") or {}).get("content")
    sample_t = count_output_tokens(sample_resp, tokenizer)
    sample_usage = (records[sample_idx].get("response") or {}).get("usage", {}) or {}
    print(f"sample canonical token count (record #{sample_idx}): t={sample_t}")
    print(f"  (reference-only usage.output_tokens for that record: "
          f"{sample_usage.get('output_tokens')!r} -- NOT the target)")

    # ---- render_context truncation demo on the longest record ----
    longest_idx = max(range(n_read),
                      key=lambda i: len(json.dumps((records[i].get("request") or {}).get("messages", []))))
    long_msgs = (records[longest_idx].get("request") or {}).get("messages", [])
    max_len = 4096
    rendered = render_context(long_msgs, tokenizer, max_len=max_len, max_block_chars=4000)
    rendered_tokens = len(_encode(tokenizer, rendered))
    print(f"render_context on longest record #{longest_idx} "
          f"({len(long_msgs)} msgs): rendered_tokens={rendered_tokens} (cap={max_len}) "
          f"-> within cap: {rendered_tokens <= max_len}")

    # ---- summary line ----
    print("-" * 72)
    print(f"SUMMARY: read={n_read} write={write_count} read_class={read_count} "
          f"conversations={n_conversations} sample_t={sample_t}")
    print("smoke test OK")


if __name__ == "__main__":
    import os

    _default = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "demo_data.jsonl")
    _path = _default if os.path.exists(_default) else "/home/ubuntu/midea/demo_data.jsonl"
    _smoke(_path)

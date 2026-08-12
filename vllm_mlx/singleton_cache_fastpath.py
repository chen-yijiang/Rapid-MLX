"""Singleton KV-cache fast path for B=1 decode (bench-tuning, #1861 family).

Why
---
mlx-lm's ``BatchGenerator`` always converts per-request caches into batched
forms (``KVCache.merge([c])`` -> ``BatchKVCache``) even when the batch holds a
single row. The batched representation costs real decode throughput at B=1:
per-row ``left_padding``/``offset`` bookkeeping and — decisively — a batched
rank-4 array mask instead of the plain causal mask, which keeps
``mx.fast.scaled_dot_product_attention`` off its native causal fast path.
The 2026-08-11 cross-runtime bench measured the gap on qwen3.5-4b:
``stream_generate`` (singleton cache) 172.5 tok/s @128 / 146.6 @16k vs
``BatchGenerator`` 167.7 / 138.2 — and oMLX, which keeps singleton caches
while the row count is one, holds 175 / 149 through a full server.

What
----
``install_singleton_cache_fastpath()`` patches ``mlx_lm.generate`` so that:

* ``_merge_caches([single])`` passes the per-layer caches through unchanged
  (no batched conversion) when they qualify;
* ``_extend_cache`` promotes singleton layers to their batched forms via
  ``cls.merge([self])`` the moment a second row joins, then extends as usual
  (returning a NEW list — promotion replaces layer objects);
* plain ``KVCache`` / ``RotatingKVCache`` gain the minimal batch-API surface
  the ``GenerationBatch`` touches while the row count remains one:
  ``filter`` (pass-through for one row, reset for zero), ``extract``
  (row 0 as a cheap shallow copy), ``extend`` (promote-first guard).

Only exact ``KVCache`` / ``RotatingKVCache`` layers (and layers that already
carry native batch surfaces, e.g. hybrid ``ArraysCache``) pass through.
``_QuantizableKVCache`` (a ``KVCache`` subclass, #1197/#1862) is deliberately
excluded by exact-type checks: its ``merge`` is what installs the quantized
batch cache, and short-circuiting it would silently serve bf16 under
``--kv-cache-dtype int4``.

``extract`` returns a shallow copy (fresh cache object sharing the arrays),
NOT ``self``: ``GenerationBatch.next`` extracts the finished row's cache and
then ``filter``\\ s it out of the batch, and the singleton ``filter`` for zero
surviving rows resets ``keys``/``values`` — returning ``self`` would let that
reset destroy the just-extracted prefix-cache payload.
"""

from __future__ import annotations

import logging
from typing import Any

from . import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.cache import KVCache, RotatingKVCache  # noqa: E402

logger = logging.getLogger(__name__)

# Exact types only — subclasses (e.g. the quantized _QuantizableKVCache)
# carry semantics their merge() must apply and never pass through.
_SINGLETON_EXACT_TYPES = (KVCache, RotatingKVCache)


def _is_singleton_passthrough_layer(cache_obj: Any) -> bool:
    sub = getattr(cache_obj, "caches", None)
    if isinstance(sub, (list, tuple)):
        return all(_is_singleton_passthrough_layer(c) for c in sub)
    if type(cache_obj) in _SINGLETON_EXACT_TYPES:
        return True
    # Layers with native batch surfaces (hybrid ArraysCache/MambaCache)
    # already store B=1-leading state and need no conversion.
    return (
        hasattr(cache_obj, "filter")
        and hasattr(cache_obj, "extract")
        and hasattr(cache_obj, "extend")
        and type(cache_obj).__module__.startswith("mlx_lm.")
    )


def _promote_layer(cache_obj: Any) -> Any:
    """Convert a singleton layer to its batched form (idempotent)."""
    sub = getattr(cache_obj, "caches", None)
    if isinstance(sub, (list, tuple)):
        converted = tuple(_promote_layer(c) for c in sub)
        if all(a is b for a, b in zip(sub, converted)):
            return cache_obj
        return type(cache_obj)(*converted)
    if type(cache_obj) in _SINGLETON_EXACT_TYPES:
        return type(cache_obj).merge([cache_obj])
    return cache_obj


# -- minimal batch surface for singleton caches ------------------------------


def _singleton_filter(self, batch_indices):
    try:
        n = len(batch_indices)
    except TypeError:
        n = int(getattr(batch_indices, "shape", (0,))[0] or 0)
    if n == 0:
        self.keys = None
        self.values = None
        self.offset = 0
        if hasattr(self, "_idx"):
            self._idx = 0
        return
    if n == 1:
        return
    raise NotImplementedError(
        f"{type(self).__name__}.filter is singleton pass-through only; "
        "promote to a batched cache before keeping multiple rows."
    )


def _singleton_extract(self, idx: int):
    if int(idx) != 0:
        raise IndexError(f"{type(self).__name__} singleton cache only has row 0")
    # Shallow copy: shares the arrays but survives the batch's subsequent
    # filter([]) reset (see module docstring).
    clone = type(self).__new__(type(self))
    clone.__dict__.update(self.__dict__)
    return clone


def _singleton_extend(self, other):
    raise NotImplementedError(
        f"{type(self).__name__}.extend requires batched promotion first "
        "(singleton_cache_fastpath._promote_layer)"
    )


def install_singleton_cache_fastpath() -> bool:
    """Idempotent module-level install. Returns False when the running
    mlx-lm lacks the expected seams (fall back to stock batched merging)."""
    # ``import mlx_lm.generate as gen`` binds the package ATTRIBUTE, which
    # mlx_lm/__init__ shadows with the generate() function — importlib
    # returns the real module (same trap as the `from mlx_lm import
    # generate` gotcha).
    import importlib

    gen = importlib.import_module("mlx_lm.generate")

    if getattr(gen, "_rapid_singleton_cache_fastpath", False):
        return True
    orig_merge = getattr(gen, "_merge_caches", None)
    orig_extend = getattr(gen, "_extend_cache", None)
    if orig_merge is None or orig_extend is None:
        logger.warning(
            "[singleton-fastpath] mlx-lm lacks _merge_caches/_extend_cache "
            "seams; keeping stock batched merging"
        )
        return False

    for cls in _SINGLETON_EXACT_TYPES:
        if not hasattr(cls, "filter"):
            cls.filter = _singleton_filter
        if not hasattr(cls, "extract"):
            cls.extract = _singleton_extract
        if not hasattr(cls, "extend"):
            cls.extend = _singleton_extend

    def _merge_caches_singleton(caches):
        if len(caches) == 1 and all(
            _is_singleton_passthrough_layer(c) for c in caches[0]
        ):
            return list(caches[0])
        # Promote any singleton layers a previous pass left behind before
        # the stock merge sees them (its BatchKVCache.merge expects raw
        # KVCache inputs, which promoted layers no longer are).
        return orig_merge(caches)

    def _extend_cache_promote(cache_a, cache_b):
        if not cache_a:
            return cache_b
        if not cache_b:
            return cache_a
        out = []
        for ca, cb in zip(cache_a, cache_b):
            pa, pb = _promote_layer(ca), _promote_layer(cb)
            pa.extend(pb)
            out.append(pa)
        return out

    gen._merge_caches = _merge_caches_singleton
    gen._extend_cache = _extend_cache_promote
    gen._rapid_singleton_cache_fastpath = True
    logger.info("[singleton-fastpath] installed (B=1 keeps singleton KV caches)")
    return True

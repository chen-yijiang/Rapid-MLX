# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the singleton KV-cache fast path (B=1 decode).

``install_singleton_cache_fastpath`` patches ``mlx_lm.generate`` so a
one-row batch keeps plain ``KVCache``/``RotatingKVCache`` layers (aligned
causal-mask attention) instead of converting to batched forms, and
promotes them via ``cls.merge([self])`` the moment a second row joins.

These tests cover the pure decision/promotion machinery and the
minimal batch surface grafted onto the singleton classes. End-to-end
correctness (mid-flight join produces identical tokens) was verified by
the bench A/B (tune1-singleton: midflight-join counts {0:160, 1:120}).
"""

from __future__ import annotations

import importlib

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache, RotatingKVCache

from vllm_mlx.singleton_cache_fastpath import (
    _is_singleton_passthrough_layer,
    _promote_layer,
    install_singleton_cache_fastpath,
)


@pytest.fixture(scope="module", autouse=True)
def installed():
    assert install_singleton_cache_fastpath() is True
    # Idempotent: second install is a no-op success.
    assert install_singleton_cache_fastpath() is True


def _filled_kv(n_tokens=4, n_heads=2, head_dim=8):
    c = KVCache()
    k = mx.zeros((1, n_heads, n_tokens, head_dim))
    v = mx.ones((1, n_heads, n_tokens, head_dim))
    c.update_and_fetch(k, v)
    return c


# ------------------------------------------------------- gate decisions


def test_exact_kvcache_passes_through():
    assert _is_singleton_passthrough_layer(KVCache()) is True
    assert _is_singleton_passthrough_layer(RotatingKVCache(max_size=64)) is True


def test_kvcache_subclass_excluded():
    """_QuantizableKVCache-style subclasses carry merge() semantics the
    passthrough would silently skip (#1197/#1862) — exact types only."""

    class _QuantishKVCache(KVCache):
        pass

    assert _is_singleton_passthrough_layer(_QuantishKVCache()) is False


def test_foreign_object_excluded():
    class _NotACache:
        pass

    assert _is_singleton_passthrough_layer(_NotACache()) is False


def test_mlx_lm_native_batch_surface_accepted():
    """Hybrid layers (ArraysCache) already store batch-leading state and
    expose filter/extract/extend natively — they pass through."""
    cache_mod = importlib.import_module("mlx_lm.models.cache")
    arrays_cls = getattr(cache_mod, "ArraysCache", None)
    if arrays_cls is None:
        pytest.skip("mlx-lm build has no ArraysCache")
    obj = arrays_cls(size=2)
    if not (
        hasattr(obj, "filter") and hasattr(obj, "extract") and hasattr(obj, "extend")
    ):
        pytest.skip("this mlx-lm ArraysCache lacks native batch surface")
    assert _is_singleton_passthrough_layer(obj) is True


# ----------------------------------------------------------- promotion


def test_promote_kvcache_matches_merge():
    c = _filled_kv()
    promoted = _promote_layer(c)
    expected = type(c).merge([_filled_kv()])
    assert type(promoted) is type(expected)
    assert promoted.offset == expected.offset


def test_promote_is_identity_for_batched():
    c = _filled_kv()
    promoted = _promote_layer(c)
    # Promoting an already-batched cache must be a no-op.
    assert _promote_layer(promoted) is promoted


# ----------------------------------------------- singleton batch surface


def test_filter_keep_one_row_is_noop():
    c = _filled_kv()
    keys_before = c.keys
    c.filter([0])
    assert c.keys is keys_before
    assert c.offset == 4


def test_filter_zero_rows_resets():
    c = _filled_kv()
    c.filter([])
    assert c.keys is None
    assert c.values is None
    assert c.offset == 0


def test_filter_multi_row_raises():
    c = _filled_kv()
    with pytest.raises(NotImplementedError):
        c.filter([0, 1])


def test_extract_returns_shallow_copy_surviving_reset():
    """GenerationBatch.next() extracts the finished row's cache and then
    filter([])s the batch — the extracted payload must survive that
    reset (prefix-cache contract, see module docstring)."""
    c = _filled_kv()
    clone = c.extract(0)
    assert type(clone) is KVCache
    assert clone.offset == c.offset
    assert clone.keys is c.keys  # shallow: shares arrays
    c.filter([])  # batch resets the (now-empty) slot
    assert clone.keys is not None
    assert clone.offset == 4


def test_extract_nonzero_row_raises():
    c = _filled_kv()
    with pytest.raises(IndexError):
        c.extract(1)


def test_extend_requires_promotion():
    a, b = _filled_kv(), _filled_kv()
    with pytest.raises(NotImplementedError):
        a.extend(b)


# ------------------------------------------------------ patched seams


def test_merge_caches_single_passthrough():
    gen = importlib.import_module("mlx_lm.generate")
    layers = [_filled_kv(), _filled_kv()]
    merged = gen._merge_caches([layers])
    assert merged[0] is layers[0]
    assert merged[1] is layers[1]


def test_merge_caches_multi_uses_stock_merge():
    gen = importlib.import_module("mlx_lm.generate")
    merged = gen._merge_caches([[_filled_kv()], [_filled_kv()]])
    # Two rows -> batched form, not the singleton objects.
    assert type(merged[0]) is not KVCache


def test_extend_cache_promotes_then_extends():
    gen = importlib.import_module("mlx_lm.generate")
    a = [_filled_kv()]
    b = [_filled_kv()]
    out = gen._extend_cache(a, b)
    assert type(out[0]) is not KVCache  # promoted to batched form
    # Batched cache now holds two rows' worth of state.
    assert out[0].keys.shape[0] == 2

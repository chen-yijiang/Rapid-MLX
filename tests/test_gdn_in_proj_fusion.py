# SPDX-License-Identifier: Apache-2.0
"""Correctness gates for the GatedDeltaNet input-projection fusion.

The fusion must be BYTE-exact and strictly opt-out-able. The core risk
this file pins: the fused concat is only byte-identical to the four
stock projections while the quantized matmul stays on the narrow-batch
kernel (M = B*S small); wider inputs must take the sliced per-projection
path. These tests cover byte equality across decode/prefill widths and
a cached prefill+decode sequence, the structural gate, install-probe
gating, rewrite semantics, idempotency, and the env kill-switch.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs

from vllm_mlx import gdn_in_proj_fusion


def _bits_equal(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and gdn_in_proj_fusion._bytes_of(a) == gdn_in_proj_fusion._bytes_of(b)
    )


def _tiny_args(**overrides):
    base = dict(
        model_type="qwen3_5_text",
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        vocab_size=128,
    )
    base.update(overrides)
    return TextModelArgs(**base)


class TinyGDNModel(nn.Module):
    """Minimal container so fuse_gdn_in_proj's module scan finds the layers."""

    def __init__(self, n_layers=2, quantize=True):
        super().__init__()
        args = _tiny_args()
        self.layers = [GatedDeltaNet(args) for _ in range(n_layers)]
        if quantize:
            nn.quantize(self, group_size=32, bits=4)
        mx.eval(self.parameters())
        self.hidden_size = args.hidden_size


def _inputs(rows, hidden, seed=3):
    mx.random.seed(seed)
    x = mx.random.normal((1, rows, hidden)).astype(mx.bfloat16)
    mx.eval(x)
    return x


def _run_all(model, x, caches=None):
    outs = []
    for i, layer in enumerate(model.layers):
        cache = caches[i] if caches is not None else None
        outs.append(layer(x, None, cache))
    mx.eval(*outs)
    return outs


class TestByteExactness:
    @pytest.mark.parametrize("rows", [1, 2, 8, 16, 64])
    def test_fused_output_is_byte_identical(self, rows):
        model = TinyGDNModel()
        x = _inputs(rows, model.hidden_size)
        before = _run_all(model, x)
        n = gdn_in_proj_fusion.fuse_gdn_in_proj(model)
        assert n == 2
        after = _run_all(model, x)
        for b, a in zip(before, after):
            assert _bits_equal(b, a), f"fusion changed bytes at rows={rows}"

    def test_cached_prefill_then_decode_byte_identical(self):
        model = TinyGDNModel()
        xp = _inputs(16, model.hidden_size, seed=7)
        xd = _inputs(1, model.hidden_size, seed=8)

        caches = [ArraysCache(size=2) for _ in model.layers]
        p_before = _run_all(model, xp, caches)
        d_before = _run_all(model, xd, caches)
        states_before = [(c[0], c[1]) for c in caches]
        mx.eval(*[s for pair in states_before for s in pair])

        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 2

        caches = [ArraysCache(size=2) for _ in model.layers]
        p_after = _run_all(model, xp, caches)
        d_after = _run_all(model, xd, caches)
        states_after = [(c[0], c[1]) for c in caches]
        mx.eval(*[s for pair in states_after for s in pair])

        for b, a in zip(p_before + d_before, p_after + d_after):
            assert _bits_equal(b, a)
        for (c0b, c1b), (c0a, c1a) in zip(states_before, states_after):
            assert _bits_equal(c0b, c0a)
            assert _bits_equal(c1b, c1a)


class TestRewriteSemantics:
    def test_originals_dropped_and_container_reused(self):
        model = TinyGDNModel()
        gdn = model.layers[0]
        qkv_before = gdn.in_proj_qkv
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 2
        assert gdn.in_proj_fused is qkv_before
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
            assert not hasattr(gdn, name)
        assert gdn._rapid_gdn_bounds[-1] == gdn.in_proj_fused.weight.shape[0] * 1
        # Fused rows = qkv + z + b + a for the tiny geometry.
        key_dim, value_dim = 2 * 32, 4 * 32
        assert gdn._rapid_gdn_bounds == [
            2 * key_dim + value_dim,
            2 * key_dim + 2 * value_dim,
            2 * key_dim + 2 * value_dim + 4,
            2 * key_dim + 2 * value_dim + 8,
        ]

    def test_idempotent(self):
        model = TinyGDNModel()
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 2
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0

    def test_unfused_instance_still_works_after_class_patch(self):
        fused_model = TinyGDNModel()
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(fused_model) == 2
        # A fresh, unfused instance dispatches through the patched class
        # __call__ and must take the stock path.
        stock_model = TinyGDNModel()
        x = _inputs(4, stock_model.hidden_size)
        outs = _run_all(stock_model, x)
        assert all(o.shape == (1, 4, stock_model.hidden_size) for o in outs)


class TestStructuralGate:
    def test_unquantized_rejected(self):
        model = TinyGDNModel(quantize=False)
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0

    def test_subclass_rejected(self):
        class CustomGDN(GatedDeltaNet):
            pass

        model = TinyGDNModel()
        model.layers[0].__class__ = CustomGDN
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 1

    def test_sharded_rejected(self):
        model = TinyGDNModel()
        model.layers[0].sharding_group = object()
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 1

    def test_mixed_quantization_rejected(self):
        model = TinyGDNModel()
        gdn = model.layers[0]
        w = gdn.in_proj_z
        deq = mx.dequantize(
            w.weight, w.scales, w.biases, group_size=w.group_size, bits=w.bits
        )
        requant = nn.QuantizedLinear(
            deq.shape[1], deq.shape[0], bias=False, group_size=64, bits=8
        )
        wq, sq, bq = mx.quantize(deq, group_size=64, bits=8)
        requant.weight, requant.scales, requant.biases = wq, sq, bq
        gdn.in_proj_z = requant
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 1

    def test_non_gdn_model_noop(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0


class TestGating:
    def test_env_opt_out(self, monkeypatch):
        monkeypatch.setenv("RAPID_MLX_GDN_IN_PROJ_FUSION", "0")
        model = TinyGDNModel()
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0
        assert hasattr(model.layers[0], "in_proj_qkv")

    def test_projection_probe_failure_disables(self, monkeypatch):
        model = TinyGDNModel()
        monkeypatch.setattr(
            gdn_in_proj_fusion, "_probe_projection_parity", lambda gdn: frozenset()
        )
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0
        assert hasattr(model.layers[0], "in_proj_qkv")

    def test_whole_layer_probe_failure_disables(self, monkeypatch):
        model = TinyGDNModel()
        monkeypatch.setattr(
            gdn_in_proj_fusion,
            "_probe_whole_layer_parity",
            lambda *a, **k: False,
        )
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0
        assert hasattr(model.layers[0], "in_proj_qkv")


class TestFailureContainment:
    def test_probe_exception_contained(self, monkeypatch):
        def boom(gdn):
            raise RuntimeError("probe blew up")

        monkeypatch.setattr(gdn_in_proj_fusion, "_probe_projection_parity", boom)
        model = TinyGDNModel()
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0
        assert hasattr(model.layers[0], "in_proj_qkv")
        # Model still runs stock.
        x = _inputs(2, model.hidden_size)
        _run_all(model, x)

    def test_fuse_one_exception_contained(self, monkeypatch):
        def boom(gdn, dtypes=None):
            raise RuntimeError("commit blew up")

        monkeypatch.setattr(gdn_in_proj_fusion, "_fuse_one", boom)
        model = TinyGDNModel()
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 0
        x = _inputs(2, model.hidden_size)
        _run_all(model, x)


class TestDtypeGate:
    def test_unprobed_dtype_takes_sliced_path(self, monkeypatch):
        model = TinyGDNModel()
        x32 = _inputs(2, model.hidden_size).astype(mx.float32)
        before = _run_all(model, x32)
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 2
        # float32 was never probed: narrow float32 inputs must take the
        # sliced (byte-exact-by-construction) path, so outputs match.
        gdn = model.layers[0]
        assert mx.float32 not in gdn._rapid_gdn_dtypes
        after = _run_all(model, x32)
        for b, a in zip(before, after):
            assert _bits_equal(b, a)

    def test_float16_narrow_byte_identical(self):
        model = TinyGDNModel()
        x16 = _inputs(2, model.hidden_size).astype(mx.float16)
        before = _run_all(model, x16)
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 2
        after = _run_all(model, x16)
        for b, a in zip(before, after):
            assert _bits_equal(b, a)


class TestWidthDispatch:
    def test_narrow_and_wide_paths_agree(self):
        # The fused (narrow) and sliced (wide) paths must agree with each
        # other where their gates meet: run the same rows through a fused
        # layer at the widest fused width and compare against stock.
        model = TinyGDNModel()
        rows = gdn_in_proj_fusion._FUSED_MAX_ROWS
        x = _inputs(rows, model.hidden_size, seed=11)
        before = _run_all(model, x)
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 2
        after = _run_all(model, x)
        for b, a in zip(before, after):
            assert _bits_equal(b, a)

    def test_wide_path_uses_slices(self, monkeypatch):
        model = TinyGDNModel()
        assert gdn_in_proj_fusion.fuse_gdn_in_proj(model) == 2
        gdn = model.layers[0]
        calls = []
        orig = mx.quantized_matmul

        def counting(*args, **kwargs):
            calls.append(args[1].shape)
            return orig(*args, **kwargs)

        monkeypatch.setattr(mx, "quantized_matmul", counting)
        x = _inputs(64, model.hidden_size)
        gdn(x, None, None)
        # Four sliced projection matmuls (plus whatever out_proj issues
        # through its own module call, which does not go through the
        # patched mx.quantized_matmul binding inside nn).
        proj_rows = {tuple(s)[0] for s in calls[:4]}
        assert len(calls) >= 4
        assert proj_rows == {
            gdn._rapid_gdn_bounds[0],
            gdn._rapid_gdn_bounds[1] - gdn._rapid_gdn_bounds[0],
            gdn._rapid_gdn_bounds[2] - gdn._rapid_gdn_bounds[1],
            gdn._rapid_gdn_bounds[3] - gdn._rapid_gdn_bounds[2],
        }

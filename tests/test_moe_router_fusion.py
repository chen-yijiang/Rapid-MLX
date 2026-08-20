# SPDX-License-Identifier: Apache-2.0
"""Correctness gates for the fused MoE router top-k kernel.

The kernel replaces only the argpartition selection; reversed, its
output must match argpartition's permutation ELEMENTWISE (values
ascending, ties ascending by index) — that pinned ordering is what
makes the block output byte-identical, since every other step keeps
the composed ops. The install wrapper must engage only for short-row,
renormalizing, unsharded calls and fall back (permanently on failure)
to the original block body.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CI runners have Metal
    pytest.skip("Metal GPU required for router kernel tests", allow_module_level=True)

from vllm_mlx import moe_router_fusion

NE, K = 256, 8


def _composed(gates, k):
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores


def _probs(rows=1, ne=NE, dtype=mx.bfloat16, seed=0):
    mx.random.seed(seed)
    g = mx.softmax(mx.random.normal((rows, ne)).astype(dtype), axis=-1, precise=True)
    mx.eval(g)
    return g


class TestSelectionParity:
    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
    def test_selected_set_matches_argpartition(self, seed, dtype):
        g = _probs(rows=4, dtype=dtype, seed=seed)
        fi, fs = moe_router_fusion.fused_router_topk(g, K)
        ci, cs = _composed(g, K)
        mx.eval(fi, fs, ci, cs)
        assert fi.astype(mx.int64).tolist() == ci.astype(mx.int64).tolist(), (
            "permutation differs from argpartition"
        )

    def test_tie_resolves_to_highest_index(self):
        # Craft a row where several experts share the exact k-th probability:
        # the fused kernel must pick the same members argpartition does
        # (ties -> highest index at this shape).
        base = mx.full((1, NE), 1.0 / NE, dtype=mx.float32)
        # raise experts 3, 7, 250 to one shared higher value; k=2 forces a
        # tie-break among three equal candidates
        base[0, 3] = 0.5
        base[0, 7] = 0.5
        base[0, 250] = 0.5
        g = (base / base.sum(axis=-1, keepdims=True)).astype(mx.bfloat16)
        mx.eval(g)
        fi, _ = moe_router_fusion.fused_router_topk(g, 2)
        ci, _ = _composed(g, 2)
        mx.eval(fi, ci)
        # elementwise permutation parity, ties included
        assert fi.astype(mx.int64).tolist() == ci.astype(mx.int64).tolist()
        # the two picked must be the HIGHEST-index pair of the tied trio
        assert set(fi[0].tolist()) == {250, 7}

    @pytest.mark.parametrize("ne,k", [(32, 2), (64, 4), (128, 8), (256, 8)])
    def test_expert_count_sweep(self, ne, k):
        g = _probs(rows=2, ne=ne, seed=11)
        fi, fs = moe_router_fusion.fused_router_topk(g, k)
        ci, cs = _composed(g, k)
        mx.eval(fi, fs, ci, cs)
        assert fi.astype(mx.int64).tolist() == ci.astype(mx.int64).tolist()


class TestScoreNumerics:
    def test_raw_scores_byte_identical_to_take(self):
        # The kernel's raw scores must be bit-identical to
        # take_along_axis(probs, argpartition_indices): T(float(x))
        # round-trips losslessly, and the order matches argpartition.
        for seed in range(4):
            g = _probs(rows=4, seed=seed)
            fi, fs = moe_router_fusion.fused_router_topk(g, K)
            ci = mx.argpartition(g, kth=-K, axis=-1)[..., -K:]
            ct = mx.take_along_axis(g, ci, axis=-1)
            mx.eval(fi, fs, ci, ct)
            assert bool(mx.array_equal(fs, ct)), f"seed {seed} raw scores differ"

    def test_renormalized_scores_byte_identical_to_composed(self):
        g = _probs(rows=3, seed=9)
        fi, fs = moe_router_fusion.fused_router_topk(g, K)
        renorm = fs / fs.sum(axis=-1, keepdims=True)
        _, cs = _composed(g, K)
        mx.eval(renorm, cs)
        assert bool(mx.array_equal(renorm, cs))


class TestRowGate:
    def test_short_rows_eligible_long_rows_not(self):
        short = _probs(rows=8)
        long_ = _probs(rows=9)
        assert moe_router_fusion._rows_eligible(short, NE)
        assert not moe_router_fusion._rows_eligible(long_, NE)

    def test_unsupported_dtype_and_expert_count(self):
        g32 = _probs(rows=1).astype(mx.float32)
        assert not moe_router_fusion._rows_eligible(g32, NE)
        gbad = _probs(rows=1, ne=48)
        assert not moe_router_fusion._rows_eligible(gbad, 48)


class TestInstall:
    def _block_cls(self):
        from mlx_lm.models import qwen3_next as q3n

        return q3n.Qwen3NextSparseMoeBlock

    def _restore(self, cls, orig):
        cls.__call__ = orig
        for attr in ("_rapid_router_fused", "_rapid_router_original_call"):
            if hasattr(cls, attr):
                delattr(cls, attr)
        moe_router_fusion._installed = False
        moe_router_fusion._original_call = None

    def test_install_wraps_and_is_idempotent(self):
        cls = self._block_cls()
        orig = (
            cls._rapid_router_original_call
            if getattr(cls, "_rapid_router_fused", False)
            else cls.__call__
        )
        self._restore(cls, orig)
        try:
            assert moe_router_fusion.install() is True
            wrapped = cls.__call__
            assert wrapped is not orig
            assert moe_router_fusion.install() is True
            assert cls.__call__ is wrapped
        finally:
            self._restore(cls, orig)

    def test_env_opt_out(self, monkeypatch):
        cls = self._block_cls()
        orig = (
            cls._rapid_router_original_call
            if getattr(cls, "_rapid_router_fused", False)
            else cls.__call__
        )
        self._restore(cls, orig)
        monkeypatch.setenv("RAPID_MLX_MOE_ROUTER_FUSION", "0")
        try:
            assert moe_router_fusion.install() is False
            assert cls.__call__ is orig
        finally:
            self._restore(cls, orig)


class TestEndToEndBlock:
    def test_block_output_close_and_selection_identical(self):
        # Build a real (tiny) Qwen3NextSparseMoeBlock and compare the
        # patched fast path against the original body on decode-width input.
        from types import SimpleNamespace

        from mlx_lm.models import qwen3_next as q3n

        args = SimpleNamespace(
            hidden_size=64,
            moe_intermediate_size=96,
            shared_expert_intermediate_size=96,
            norm_topk_prob=True,
            num_experts=32,
            num_experts_per_tok=4,
        )
        block = q3n.Qwen3NextSparseMoeBlock(args)
        mx.eval(block.parameters())
        x = mx.random.normal((1, 1, 64)).astype(mx.bfloat16)
        mx.eval(x)

        cls = q3n.Qwen3NextSparseMoeBlock
        orig = (
            cls._rapid_router_original_call
            if getattr(cls, "_rapid_router_fused", False)
            else cls.__call__
        )
        # original body result
        y_orig = orig(block, x)
        mx.eval(y_orig)
        try:
            cls.__call__ = orig
            for attr in ("_rapid_router_fused", "_rapid_router_original_call"):
                if hasattr(cls, attr):
                    delattr(cls, attr)
            moe_router_fusion._installed = False
            assert moe_router_fusion.install() is True
            # warm the per-shape parity verdict first so the probe's own
            # kernel calls don't count against the instrumentation below
            assert moe_router_fusion._parity_verified(32, 4) is True
            # instrument the fast-path-only op: the byte-equal assertion
            # below is only meaningful if the fused path actually ran
            # (a silent fallback would compare orig against itself)
            real_topk = moe_router_fusion.fused_router_topk
            ran = {"n": 0}

            def counting_topk(probs, k):
                ran["n"] += 1
                return real_topk(probs, k)

            moe_router_fusion.fused_router_topk = counting_topk
            try:
                y_fast = block(x)
                mx.eval(y_fast)
            finally:
                moe_router_fusion.fused_router_topk = real_topk
            assert ran["n"] == 1, "fused fast path did not run"
            # the composed take/renorm run on the argpartition-parity
            # permutation, so the block output must be BYTE-identical.
            assert bool(mx.array_equal(y_fast, y_orig))
        finally:
            cls.__call__ = orig
            for attr in ("_rapid_router_fused", "_rapid_router_original_call"):
                if hasattr(cls, attr):
                    delattr(cls, attr)
            moe_router_fusion._installed = False
            moe_router_fusion._original_call = None


class TestPerShapeParity:
    def test_unvalidated_shape_probed_then_cached(self):
        moe_router_fusion._parity_cache.pop((64, 4), None)
        assert moe_router_fusion._parity_verified(64, 4) is True
        assert moe_router_fusion._parity_cache[(64, 4)] is True

    def test_failing_shape_routes_composed(self, monkeypatch):
        # Force the probe to report divergence for a fresh shape: the
        # verdict must be cached False so the shape permanently keeps the
        # composed chain.
        moe_router_fusion._parity_cache.pop((96, 3), None)

        def broken_topk(probs, k):
            raise RuntimeError("simulated per-shape kernel failure")

        monkeypatch.setattr(moe_router_fusion, "fused_router_topk", broken_topk)
        assert moe_router_fusion._parity_verified(96, 3) is False
        assert moe_router_fusion._parity_cache[(96, 3)] is False
        # cached: no re-probe even after the failure source is repaired
        monkeypatch.undo()
        assert moe_router_fusion._parity_verified(96, 3) is False
        moe_router_fusion._parity_cache.pop((96, 3), None)


class TestDeferredFailureDegradation:
    def test_fast_path_failure_degrades_permanently(self, monkeypatch):
        from types import SimpleNamespace

        from mlx_lm.models import qwen3_next as q3n

        args = SimpleNamespace(
            hidden_size=64,
            moe_intermediate_size=96,
            shared_expert_intermediate_size=96,
            norm_topk_prob=True,
            num_experts=32,
            num_experts_per_tok=4,
        )
        block = q3n.Qwen3NextSparseMoeBlock(args)
        mx.eval(block.parameters())
        x = mx.random.normal((1, 1, 64)).astype(mx.bfloat16)
        mx.eval(x)

        cls = q3n.Qwen3NextSparseMoeBlock
        orig = (
            cls._rapid_router_original_call
            if getattr(cls, "_rapid_router_fused", False)
            else cls.__call__
        )
        try:
            cls.__call__ = orig
            for attr in ("_rapid_router_fused", "_rapid_router_original_call"):
                if hasattr(cls, attr):
                    delattr(cls, attr)
            moe_router_fusion._installed = False
            assert moe_router_fusion.install() is True

            real_topk = moe_router_fusion.fused_router_topk
            calls_fast = {"n": 0}

            def boom(probs, k):
                raise RuntimeError("simulated deferred kernel failure")

            monkeypatch.setattr(moe_router_fusion, "fused_router_topk", boom)
            y1 = block(x)
            mx.eval(y1)  # degraded to orig, not crashed

            def counting_topk(probs, k):
                calls_fast["n"] += 1
                return real_topk(probs, k)

            monkeypatch.setattr(moe_router_fusion, "fused_router_topk", counting_topk)
            y2 = block(x)
            mx.eval(y2)
            # fast path stayed dead: the repaired kernel is never invoked
            assert calls_fast["n"] == 0
            y_ref = orig(block, x)
            mx.eval(y_ref)
            assert bool(mx.array_equal(y2, y_ref))
        finally:
            cls.__call__ = orig
            for attr in ("_rapid_router_fused", "_rapid_router_original_call"):
                if hasattr(cls, attr):
                    delattr(cls, attr)
            moe_router_fusion._installed = False
            moe_router_fusion._original_call = None


class TestRngPurity:
    def test_parity_probe_does_not_consume_global_rng(self):
        moe_router_fusion._parity_cache.pop((128, 4), None)
        mx.random.seed(1234)
        before = mx.random.normal((4,))
        mx.eval(before)
        mx.random.seed(1234)
        assert moe_router_fusion._parity_verified(128, 4) is True
        after = mx.random.normal((4,))
        mx.eval(after)
        # the probe used keyed RNG only: the global stream is untouched
        assert bool(mx.array_equal(before, after))

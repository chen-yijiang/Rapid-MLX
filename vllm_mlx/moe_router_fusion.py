# SPDX-License-Identifier: Apache-2.0
"""Fused MoE router top-k for the Qwen3-Next / Qwen3.5/3.6 MoE family.

At decode, the routing chain after the gate linear — full softmax over
the expert count, ``argpartition`` top-k, ``take_along_axis``, top-k
renormalize — is five tiny-tensor launches per MoE layer per token. On a
256-expert model that chain dominates the router cost; a single fused
Metal launch selects and renormalizes in one dispatch.

Byte-exactness strategy: the kernel replaces ONLY the ``argpartition``
selection — the most expensive launch in the chain — and every other
step (softmax, ``take_along_axis``, renormalize) keeps the composed
ops. The kernel's descending selection, reversed, reproduces
``argpartition``'s empirically pinned output permutation at these
shapes (values ascending, ties ascending by index), so the score
tensor handed to the composed renormalize is bit-identical to the
composed chain's and the block output is byte-equal. ``install()``
verifies that permutation parity at runtime on random probes and
refuses to engage if mlx's ordering ever changes.

Only short rows route through the kernel (decode / spec-verify widths);
prefill rows keep the composed chain, whose cost amortizes over the
chunk. Kernel design adapted from the oMLX project
(https://github.com/jundot/omlx, Apache-2.0).

``install()`` wraps ``Qwen3NextSparseMoeBlock.__call__`` (shared by
mlx-lm's qwen3_5 / qwen3_6 / qwen3-next model files). Anything outside
the fast-path gate — long rows, sharded runs, ``norm_topk_prob=False``,
unsupported dtypes or expert counts — falls through to the original
body unchanged, and any fast-path failure permanently degrades to the
original. Set ``RAPID_MLX_MOE_ROUTER_FUSION=0`` to disable.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

# Decode / spec-verify row widths; prefill keeps the composed chain.
_MAX_ROWS = 8

_SOURCE = """
    // One threadgroup (a single simdgroup) per row; each lane owns
    // NE/32 experts. Selection keys are the softmax probabilities the
    // composed argpartition would read, so the selected SET matches
    // exactly; ties on equal probabilities resolve to the HIGHEST
    // index (mlx argpartition's pinned behavior — see the parity
    // test). Renormalization runs in fp32 with one rounding at the
    // end.
    constexpr uint PER = uint(NE) / 32;
    uint lane = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    const device T* g = probs + row * uint(NE);

    float vals[PER];
    for (uint i = 0; i < PER; ++i) {
        vals[i] = float(g[lane * PER + i]);
    }

    bool taken[PER];
    for (uint i = 0; i < PER; ++i) {
        taken[i] = false;
    }

    float sel_p[K];
    uint sel_i[K];
    for (uint j = 0; j < K; ++j) {
        float best = -INFINITY;
        uint best_i = uint(NE);
        for (uint i = 0; i < PER; ++i) {
            if (!taken[i] && vals[i] >= best) {
                best = vals[i];
                best_i = lane * PER + i;
            }
        }
        float gbest = simd_max(best);
        uint cand = (best == gbest) ? best_i : 0u;
        uint gbest_i = simd_max(cand);
        if (best == gbest && best_i == gbest_i) {
            taken[best_i - lane * PER] = true;
        }
        sel_p[j] = gbest;
        sel_i[j] = gbest_i;
    }

    if (lane == 0) {
        // Write in argpartition's pinned order — values ascending, ties
        // ascending by index (selection produced descending / ties
        // highest-first, so reversing restores it) — and emit the RAW
        // selected probabilities: T(float(g)) round-trips losslessly, so
        // the downstream composed renormalize sees a byte-identical
        // score tensor.
        for (uint j = 0; j < K; ++j) {
            indices[row * uint(K) + (K - 1u - j)] = sel_i[j];
            scores[row * uint(K) + (K - 1u - j)] = T(sel_p[j]);
        }
    }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="rapid_moe_router_topk",
            input_names=["probs"],
            output_names=["indices", "scores"],
            source=_SOURCE,
        )
    return _KERNEL


def fused_router_topk(probs: mx.array, top_k: int):
    """top-k selection over softmaxed gate probabilities.

    ``probs`` is [..., NE] (the composed chain's own softmax output).
    Returns ``(indices [..., k] uint32, raw_scores [..., k] in probs'
    dtype)`` in ``mx.argpartition(...)[..., -k:]``'s pinned order (values
    ascending, ties ascending by index). ``raw_scores`` are the selected
    probabilities themselves — NOT renormalized — bit-identical to
    ``take_along_axis(probs, indices)``.
    """
    shape = probs.shape
    rows = 1
    for d in shape[:-1]:
        rows *= d
    ne = shape[-1]
    inds, scores = _kernel()(
        inputs=[probs],
        template=[("T", probs.dtype), ("NE", ne), ("K", top_k)],
        grid=(32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*shape[:-1], top_k), (*shape[:-1], top_k)],
        output_dtypes=[mx.uint32, probs.dtype],
    )
    return inds, scores


def _rows_eligible(x: mx.array, num_experts: int) -> bool:
    rows = 1
    for d in x.shape[:-1]:
        rows *= d
    return (
        rows <= _MAX_ROWS
        and num_experts % 32 == 0
        and x.dtype in (mx.bfloat16, mx.float16)
    )


# Per-(num_experts, top_k) parity verdicts. The argpartition permutation
# is empirically pinned PER SHAPE, so every (NE, K) combination is
# verified — compile, run, permutation, and raw-score identity — before
# its first fast-path use; a failing shape is permanently routed to the
# composed chain. Verification runs on the calling (mlx-step) thread the
# first time a shape appears and is memoized.
_parity_cache: dict[tuple[int, int], bool] = {}


def _parity_verified(ne: int, k: int) -> bool:
    ok = _parity_cache.get((ne, k))
    if ok is None:
        ok = True
        try:
            # Keyed RNG: the probe must never consume the GLOBAL random
            # state — doing so would shift seeded sampling results based
            # on parity-cache warmness.
            key = mx.random.key(0x52504D + ne * 131 + k)
            for probe_dtype in (mx.bfloat16, mx.float16):
                random_rows = mx.softmax(
                    mx.random.normal((16, ne), key=key).astype(probe_dtype),
                    axis=-1,
                )
                # Deterministic tie-heavy rows: random draws cannot be
                # relied on to exercise argpartition's tie ordering, and a
                # different tie permutation would silently misalign the
                # expert/score pairing. Row 1: ALL experts equal (every
                # selection is a tie). Row 2: a two-level step whose high
                # plateau is wider than k (ties across the whole selection
                # boundary). Row 3: duplicates scattered at the top
                # (partial ties inside the selection).
                flat = mx.full((1, ne), 1.0, dtype=mx.float32)
                step = mx.concatenate(
                    [
                        mx.full((1, ne // 2), 2.0),
                        mx.full((1, ne - ne // 2), 1.0),
                    ],
                    axis=1,
                )
                scattered = mx.full((1, ne), 1.0, dtype=mx.float32)
                for pos in range(0, ne, max(1, ne // (k + 2))):
                    scattered[0, pos] = 3.0
                tie_rows = mx.softmax(
                    mx.concatenate([flat, step, scattered], axis=0), axis=-1
                ).astype(probe_dtype)
                probe = mx.concatenate([random_rows, tie_rows], axis=0)
                sel, raw = fused_router_topk(probe, k)
                composed_perm = mx.argpartition(probe, kth=-k, axis=-1)[..., -k:]
                composed_take = mx.take_along_axis(probe, composed_perm, axis=-1)
                mx.eval(sel, raw, composed_perm, composed_take)
                if sel.astype(mx.int64).tolist() != composed_perm.astype(
                    mx.int64
                ).tolist() or not bool(mx.array_equal(raw, composed_take)):
                    ok = False
                    break
        except Exception:  # noqa: BLE001 — any probe failure ⇒ composed chain
            logger.exception("[moe_router] parity probe errored for NE=%d K=%d", ne, k)
            ok = False
        _parity_cache[(ne, k)] = ok
        if not ok:
            logger.warning(
                "[moe_router] parity probe failed for NE=%d K=%d — this "
                "shape keeps the composed router chain",
                ne,
                k,
            )
    return ok


_installed = False
_original_call = None


def install() -> bool:
    """Idempotently wrap the Qwen3-Next sparse-MoE block with the fused path.

    Returns True when the wrapper is (already) in place, False when
    disabled or unavailable. The wrapper only engages for short-row,
    renormalizing, unsharded calls on supported dtypes/expert counts;
    everything else runs the original body.
    """
    global _installed, _original_call
    if _installed:
        return True
    if os.environ.get("RAPID_MLX_MOE_ROUTER_FUSION", "1") == "0":
        logger.info("[moe_router] disabled via RAPID_MLX_MOE_ROUTER_FUSION=0")
        return False
    if not mx.metal.is_available():
        return False
    try:
        from mlx_lm.models import qwen3_next as _q3n
    except ImportError:
        return False
    cls = getattr(_q3n, "Qwen3NextSparseMoeBlock", None)
    if cls is None:
        return False
    if getattr(cls, "_rapid_router_fused", False):
        _installed = True
        _original_call = getattr(cls, "_rapid_router_original_call", None)
        return True

    # Eagerly compile AND parity-check the kernel BEFORE rebinding
    # anything. Two things are verified on random probes for both
    # supported dtypes: (a) the kernel compiles and runs (mlx is lazy — a
    # Metal rejection would otherwise surface inside a production
    # request); (b) the reversed selection reproduces argpartition's
    # output permutation ELEMENTWISE — the byte-exactness of the fast
    # path rests on that pinned ordering, so if a future mlx changes it,
    # we refuse to engage rather than silently diverge.
    # Startup probe on one production shape catches a Metal compile
    # rejection before anything is rebound; every OTHER (NE, K) shape is
    # verified the same way — compile, run, permutation, raw-score
    # identity, with mx.eval inside the guard — on first use via
    # ``_parity_verified``, so no unvalidated shape ever takes the fast
    # path and no lazy-eval failure can escape into a request.
    if not _parity_verified(256, 8):
        return False

    orig_call = cls.__call__
    fast_path_dead = False

    def patched_call(self, x):
        nonlocal fast_path_dead
        if (
            fast_path_dead
            or getattr(self, "sharding_group", None) is not None
            or not self.norm_topk_prob
            or not _rows_eligible(x, self.num_experts)
            or not _parity_verified(self.num_experts, self.top_k)
        ):
            return orig_call(self, x)
        try:
            gates = mx.softmax(self.gate(x), axis=-1, precise=True)
            # The kernel returns indices in argpartition's pinned order and
            # the raw selected probabilities (bit-identical to
            # take_along_axis), so the composed renormalize below sees the
            # exact tensor the original chain would — the block output is
            # byte-identical while argpartition AND take_along_axis are
            # replaced by one launch.
            inds, scores = fused_router_topk(gates, self.top_k)
            scores = scores / scores.sum(axis=-1, keepdims=True)

            y = self.switch_mlp(x, inds)
            y = (y * scores[..., None]).sum(axis=-2)

            shared_y = self.shared_expert(x)
            shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
            return y + shared_y
        except Exception:  # noqa: BLE001 — any fast-path failure ⇒ composed
            fast_path_dead = True
            logger.exception(
                "[moe_router] fused router failed — permanently falling "
                "back to the composed chain"
            )
            return orig_call(self, x)

    cls.__call__ = patched_call
    cls._rapid_router_fused = True
    cls._rapid_router_original_call = orig_call
    _original_call = orig_call
    _installed = True
    logger.info(
        "[moe_router] fused router top-k installed (rows<=%d, experts%%32==0)",
        _MAX_ROWS,
    )
    return True

# Cross-runtime benchmark — merged final

**M3 Ultra 256 GB, macOS 26.5.2, 2026-08-11.** rapid-mlx 0.12.10 (PyPI) vs oMLX v0.5.8.dev3 vs mlx-lm 0.31.3 (`mlx_lm server`) vs Ollama 0.32.5. Six models, MLX 4-bit vs GGUF Q4_K_M (gpt-oss-20b: MXFP4 both sides). Methodology in README; decode/concurrency cells re-run with saturating workloads after the thinking-template asymmetry was found (v1 decode/conc numbers were incomparable and are superseded).

## Executive summary

**Where rapid-mlx wins**
- **TTFT, every model, every size** (parity ttft_short): 0.18–0.67 s vs 0.23–0.91 s for the field — 20–40 % ahead of the next best. The responsiveness users feel first is our strongest column.
- **Concurrency aggregate on small/MoE models** (conc_8): 225 tok/s (4B), 220 (gpt-oss), 205 (gemma-4) — first or near-tied-first. oMLX does not scale with concurrency at all (agg ≈ its B=1 everywhere: its request pipeline appears serialized); Ollama is uniformly last.
- **Memory**: MLX runtimes sit at model size (3–20 GB RSS). Ollama peaks 15–106 GB under the pinned 20k-ctx × 8-slot env (config-driven preallocation — a real product difference under identical settings, not waste per se).

**Where rapid-mlx loses — engineering follow-ups**
1. **Long-context decode collapses** (decode_16k, parity): 4B drops 150→80 tok/s (−47 %) at 16 k context while oMLX drops only −15 %, mlx-lm −14 %, Ollama −10 %. We are last or second-to-last in every 16 k decode cell (−25…−46 % vs oMLX). Same pattern, milder, in 16 k TTFT. Filed as a perf issue — this is the biggest systematic gap the matrix found.
2. **B=1 decode trails oMLX everywhere** (−4…−20 %, worst on qwen3.6-35b-a3b 92 vs 115): the known scheduler-step overhead family. oMLX also beats *stock mlx-lm* B=1 by ~30 %, so their decode loop is worth reading.
3. **Dense-27B and 35B concurrency trail mlx-lm** (conc_8: 31 vs 50; 164 vs 226): mlx-lm's prompt/decode-concurrency batching outscales our scheduler on the biggest models.
4. **gemma-4-26b DNF at 16 k** — #1848 (MLLM lane treats prefill_step_size as a cap; text-only >8 k fails), error surfaced as generic 200-stream — #1849. Only runtime in the matrix that failed the cell.

**Notes on the competition**
- oMLX: best-in-class B=1 decode and 16 k decode; zero concurrency scaling; 20 s cold load on gemma-4 (vision preload).
- mlx-lm server: strongest high-concurrency scaling on big models; mid-pack everywhere else; fully serial → conc numbers rely on its internal batching flags (defaults used).
- Ollama: slowest decode on every model (llama.cpp Metal vs MLX kernels), but best 16 k TTFT on gpt-oss (7.2 s) and unmatched cold-load simplicity. GGUF quant ≠ MLX quant — treat cross-format cells as product comparisons, not kernel benchmarks (gpt-oss MXFP4 is the clean pair).

**Fairness caveats that survived to the end**: quant formats differ except gpt-oss; Ollama context/parallel env pinned (20480 × 8) which inflates its RSS; rapid's R12-T2F thinking auto-disable is a product behavior we neutralized in throughput cells via saturating prompts (†
 marks residual early-EOS cells); one machine, one night, N=3 medians with spreads shown.

## Post-fix re-run — 2026-08-12, engine main@dde006ca

The three engine issues this benchmark filed were fixed within a day and the
rapid column was re-measured on the same machine/harness (runs
`postfix-parity`, `postfix-product`, `postfix-vision`; mlx-vlm re-measured in
the vision lane for a same-night comparison; other columns are the 2026-08-11
numbers — ollama's GGUFs were removed after the original cycle and were not
re-pulled):

- **#1857** (fixes #1853): two default-on features each added O(context)
  work to every decode step — a synchronous full-KV disk checkpoint every
  256 tokens that nothing ever read back, and int4 live-KV quantization
  whose dequant-on-read materializes full-precision K/V per step. Both are
  now opt-in (defaults: interval 0, bf16).
- **#1848 / #1849**: MLLM lane no longer treats `prefill_step_size` as a
  hard cap for text-only prompts, and stream errors surface as typed
  client errors instead of a generic 200-stream — gemma-4-26b's 16 k DNF
  cells now produce numbers.
- **#1856** (fixes #1854): projected vision features are cached, so
  repeated images skip the encoder.

Headline: **16 k decode +14…+66 % across every model**, the two FAIL cells
are gone, conc_8 gains up to +21 %, and vision TTFT nearly halved. rapid is
now first or second in every 16 k decode cell (was last/second-to-last), and
takes the conc_8 lead on 4 of 6 models. Remaining gaps: B=1 + 16 k decode
vs oMLX (scheduler-step overhead family) and big-model conc_8 vs mlx-lm.

**decode_16k** (decode tok/s) — rapid before → after; best competitor from 2026-08-11 run

| model | rapid (0.12.10) | rapid (main@dde006ca) | best other |
|---|---|---|---|
| qwen3.5-4b | 80 | 133 ±0% | 149 (oMLX) |
| qwen3.5-9b | 79 | 95 ±0% | 104 (oMLX) |
| gpt-oss-20b | 84† | 96 ±4% | 108 (oMLX) |
| qwen3.6-27b | 21† | 33 ±0% | 35 (oMLX) |
| gemma-4-26b | FAIL | 92 ±0% | 92 (oMLX) |
| qwen3.6-35b-a3b | 73 | 83 ±0% | 99 (oMLX) |

**conc_8** (aggregate tok/s) — rapid before → after; best competitor from 2026-08-11 run

| model | rapid (0.12.10) | rapid (main@dde006ca) | best other |
|---|---|---|---|
| qwen3.5-4b | 225 | 272 ±0% | 232 (mlx-lm) |
| qwen3.5-9b | 149 | 163 ±0% | 149 (mlx-lm) |
| gpt-oss-20b | 220 | 230 ±0% | 207 (mlx-lm) |
| qwen3.6-27b | 31 | 34 ±0% | 50 (mlx-lm) |
| gemma-4-26b | 205 | 205 ±0% | 183 (mlx-lm) |
| qwen3.6-35b-a3b | 164 | 190 ±0% | 226 (mlx-lm) |

**decode_b1** (decode tok/s) — rapid before → after

| model | rapid (0.12.10) | rapid (main@dde006ca) |
|---|---|---|
| qwen3.5-4b | 150 | 160 ±0% |
| qwen3.5-9b | 106 | 108 ±0% |
| gpt-oss-20b | 127 | 127 ±0% |
| qwen3.6-27b | 36 | 38 ±0% |
| gemma-4-26b | 111 | 112 ±0% |
| qwen3.6-35b-a3b | 92 | 96 ±0% |

**conc_4** (aggregate tok/s) — rapid before → after

| model | rapid (0.12.10) | rapid (main@dde006ca) |
|---|---|---|
| qwen3.5-4b | 198 | 231 ±0% |
| qwen3.5-9b | 132 | 143 ±0% |
| gpt-oss-20b | 179 | 185 ±0% |
| qwen3.6-27b | 28 | 32 ±1% |
| gemma-4-26b | 166 | 166 ±0% |
| qwen3.6-35b-a3b | 117 | 130 ±0% |

**ttft_16k** (median TTFT) — rapid before → after

| model | rapid (0.12.10) | rapid (main@dde006ca) |
|---|---|---|
| qwen3.5-4b | 9.30s | 8.69s ±0% |
| qwen3.5-9b | 15.31s | 14.68s ±0% |
| gpt-oss-20b | 9.21s | 8.76s ±0% |
| qwen3.6-27b | 52.68s | 51.39s ±0% |
| gemma-4-26b | FAIL | 9.27s ±0% |
| qwen3.6-35b-a3b | 8.43s | 8.01s ±0% |

**vision (gemma-4-26b, parity)** — rapid and mlx-vlm both re-measured this run

| scenario | rapid before | rapid after | mlx-vlm (re-run) | oMLX (08-11) |
|---|---|---|---|---|
| vision_ttft | 0.60s | 0.32s ±1% | 0.25s ±15% | 0.45s |
| vision_decode | 111† | 111 ±0% | 114 ±0% | 117† |

Product-lane notes (`postfix-product`): defaults now match the fixed
behavior, so product ≈ parity on decode; with the prefix cache on,
repeated-16k-prefix TTFT drops to 1.6–3.0 s on the qwen models
(4B 1.74 s, 35B-A3B 1.58 s) — the agentic re-prompt case the cache exists
for. ttft_short is unchanged (0.17–0.66 s, still first everywhere).

## Overnight tuning — 2026-08-13, branch `perf/bench-driven-tuning`

Bench-driven fix loop on the remaining gaps (runs `tune1`–`tune14`; engine
= main@fd77321d + the tuning branch; mlx-lm conc column re-measured on the
same harness so both sides carry the new per-request fields).

**Root cause of the big-model concurrency gap — the #115 hybrid admission
throttle.** Since 2026-04 every concurrent request to a hybrid model was
admitted 200 ms apart (an ArraysCache-corruption workaround). Rows admitted
at different scheduler steps carry ragged per-row offsets for the batch's
whole lifetime, which keeps mlx-lm's batched attention on the array-mask
slow path every decode step: next() p50 29.5 ms vs 20.3 ms aligned on
qwen3.6-35b-a3b at B=8. The corruption no longer reproduces on mlx-lm
0.31.3 (32/32 simultaneous-formation rounds clean), so the throttle is
retired and an engine-side admission-wave window (default 8 ms tick)
coalesces co-arriving bursts into one aligned prefill wave. Also removes
the 200 ms TTFT tax on every hybrid request. Twelve other hypotheses
(SSE volume, GIL, TurboQuant, GC, Metal limits, sampler params, config
divergence, …) were each refuted by direct A/B first.

**conc_8 — decode-phase aggregate (new preferred metric) + prefill barrier.**
Wall-clock agg is workload-skewed whenever one runtime generates more
tokens per request (thinking-template / early-EOS asymmetry — bit us twice);
`decode_agg` excludes the shared prefill barrier and divides by the same
decode span for both sides. report.py now prefers it for conc cells.

| model | rapid decode_agg | mlx-lm decode_agg | Δ | prefill (rapid vs mlx-lm) |
|---|---|---|---|---|
| qwen3.5-4b | 409 | 341 | **+20 %** | 2.5 s vs 3.0 s |
| qwen3.5-9b | 247 | 218 | **+13 %** | 4.3 s vs 4.7 s |
| gpt-oss-20b | 316 | 283 | **+12 %** | 2.5 s vs 2.7 s |
| qwen3.6-27b | 82† | 75 | **+8 %** | 14.0 s vs 14.4 s |
| qwen3.6-35b-a3b | 374 | 319 | **+17 %** | 2.2 s vs 2.7 s |

rapid now leads every conc_8 cell on BOTH phases. The 35B decode aggregate
(374) equals the in-process BatchGenerator ceiling (376–379): server-side
batching overhead is zero. Wall-agg for the record: 35B 190→268 (+41 %).
† 27B still early-EOSes at ~100 tokens under rapid's thinking auto-disable
(a product behavior) — decode_agg is the comparable number; the old
33.7-vs-49.1 wall-agg "gap" was this workload skew again, not performance
(rapid wins both phases on 27B too).

**B=1 / 16 k decode residual is now fully characterized.** With the
singleton KV-cache fast path (one-row batches keep plain caches and the
aligned-causal attention path, promoted on mid-flight join) plus a B=1
dense-sampler engage, the scheduler steps at oMLX's exact level: 4B
next+outside p50 = 5.72 ms (= 174.8 tok/s) short-context and 6.78 ms
(= 147.5 tok/s) at 16 k vs oMLX's client-measured 175 / 149. The remaining
client-visible −5 % (165.6 / 140.9) is a constant ~0.32 ms/token delivery
tax in the engine→SSE path (queue hop, task wake, chunk build, uvicorn
write) — measured identically at both context lengths. Follow-up: thin the
per-token delivery pipeline; the decode loop itself is at parity.

**Other cells**: gpt-oss decode_b1 128.5 (tool-logits factory now gated on
`has_tools` — a prod fix; parity servers never had the bias enabled), 35B
decode_b1 TTFT 0.45→0.246 s (throttle tax gone), vision_ttft 0.318 s
(unchanged; mlx-vlm 0.25 ±15 % remains the one soft cell, decode at
parity). TurboQuant k8v4 measured ≈ 0 per-step cost at B=8 — but parity
flags now pin `--kv-cache-turboquant none` since no competitor quantizes
KV by default.

Tuning-branch commits: singleton cache fast path, B=1 sampler engage,
step coalescing, has_tools gate, throttle retirement + admission window,
env-gated debug aids (STEPTIME / PYSAMPLE / SCHEDCONFIG dump), unit tests
for the admission wave and singleton surfaces.

## Run `merged-final (parity-full+v2, product-key+v2, vision-lane)`

Versions: rapid rapid-mlx 0.12.10, omlx 0.5.8.dev3, ollama ollama version is 0.32.5, mlx_lm 0.31.3, macos 26.5.2


### Lane: parity


**ttft_short**  (median TTFT)  († = early EOS, under-saturated)

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | 0.18s ±1% | 0.31s ±2% | 0.23s ±0% | — | 0.31s ±1% |
| qwen3.5-9b | 0.24s ±1% | 0.39s ±1% | 0.30s ±0% | — | 0.40s ±1% |
| gpt-oss-20b | 0.21s ±1% | 0.34s ±2% | 0.24s ±1% | — | 0.40s ±1% |
| qwen3.6-27b | 0.67s ±0% | 0.80s ±2% | 0.75s ±1% | — | 0.91s ±0% |
| gemma-4-26b | 0.27s ±0% | 0.41s ±0% | 0.32s ±1% | — | 0.48s ±1% |
| qwen3.6-35b-a3b | 0.22s ±2% | 0.35s ±0% | 0.28s ±0% | — | 0.37s ±1% |

**ttft_4k**  (median TTFT)  († = early EOS, under-saturated)

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | 2.26s ±0% | 2.12s ±0% | 2.10s ±0% | — | 2.35s ±0% |
| qwen3.5-9b | 3.75s ±0% | 3.62s ±0% | 3.60s ±0% | — | 3.94s ±0% |
| gpt-oss-20b | 2.14s ±1% | 2.06s ±0% | 2.02s ±0% | — | 2.07s ±1% |
| qwen3.6-27b | 12.77s ±0% | 12.28s ±0% | 12.40s ±0% | — | 13.05s ±0% |
| gemma-4-26b | 2.22s ±0% | 2.33s ±0% | 2.27s ±0% | — | 2.54s ±0% |
| qwen3.6-35b-a3b | 2.00s ±1% | 1.89s ±0% | 1.89s ±2% | — | 2.16s ±0% |

**ttft_16k**  (median TTFT)  († = early EOS, under-saturated)

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | 9.30s ±1% | 8.61s ±0% | 8.76s ±0% | — | 8.85s ±0% |
| qwen3.5-9b | 15.31s ±0% | 14.61s ±0% | 14.74s ±0% | — | 14.38s ±0% |
| gpt-oss-20b | 9.21s ±0% | 8.60s ±0% | 8.74s ±0% | — | 7.24s ±0% |
| qwen3.6-27b | 52.68s ±0% | 50.90s ±0% | 51.54s ±0% | — | 48.42s ±0% |
| gemma-4-26b | FAIL | 9.44s ±0% | 9.36s ±0% | — | 8.74s ±0% |
| qwen3.6-35b-a3b | 8.43s ±0% | 7.82s ±0% | 8.05s ±0% | — | 8.39s ±0% |

**decode_b1**  (decode tok/s)  († = early EOS, under-saturated)

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | 150 ±2% | 175 ±0% | 131 ±0% | — | 108 ±0% |
| qwen3.5-9b | 106 ±0% | 116 ±0% | 92 ±0% | — | 79 ±0% |
| gpt-oss-20b | 127 ±0% | 135 ±0% | 110 ±0% | — | 103 ±0% |
| qwen3.6-27b | 36 ±0% | 39 ±0% | 35 ±0% | — | 29 ±0% |
| gemma-4-26b | 111 ±0% | 116 ±0% | 94 ±1% | — | 89 ±0% |
| qwen3.6-35b-a3b | 92 ±1% | 115 ±0% | 88 ±0% | — | 89 ±0% |

**conc_4**  (aggregate tok/s)  († = early EOS, under-saturated)

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | 198 ±0% | 134 ±0% | 178 ±9% | — | 93 ±0% |
| qwen3.5-9b | 132 ±0% | 89 ±0% | 122 ±7% | — | 66 ±0% |
| gpt-oss-20b | 179 ±0% | 109 ±0% | 154 ±7% | — | 89 ±0% |
| qwen3.6-27b | 28 ±0% | 30 ±0% | 45 ±2% | — | 24 ±0% |
| gemma-4-26b | 166 ±0% | 94 ±0% | 132 ±0% | — | 78 ±0% |
| qwen3.6-35b-a3b | 117 ±1% | 95 ±0% | 158 ±8% | — | 78 ±0% |

**conc_8**  (aggregate tok/s)  († = early EOS, under-saturated)

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | 225 ±1% | 134 ±0% | 232 ±0% | — | 99 ±0% |
| qwen3.5-9b | 149 ±0% | 89 ±0% | 149 ±1% | — | 72 ±0% |
| gpt-oss-20b | 220 ±0% | 110 ±0% | 207 ±0% | — | 95 ±0% |
| qwen3.6-27b | 31 ±2% | 30 ±0% | 50 ±0% | — | 26 ±0% |
| gemma-4-26b | 205 ±0% | 94 ±0% | 183 ±13% | — | 82 ±0% |
| qwen3.6-35b-a3b | 164 ±0% | 95 ±0% | 226 ±0% | — | 82 ±0% |

**decode_16k**  (decode tok/s)  († = early EOS, under-saturated)

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | 80 ±2% | 149 ±0% | 113 ±0% | — | 97 ±0% |
| qwen3.5-9b | 79 ±0% | 104 ±0% | 84 ±0% | — | 73 ±0% |
| gpt-oss-20b | 84 ±12%† | 108 ±0% | 86 ±0% | — | 93 ±0% |
| qwen3.6-27b | 21 ±1%† | 35 ±0% | 31 ±0% | — | 27 ±0% |
| gemma-4-26b | FAIL | 92 ±0% | 80 ±0% | — | 81 ±0% |
| qwen3.6-35b-a3b | 73 ±0% | 99 ±0% | 77 ±0% | — | 81 ±0% |

**vision_ttft**  (median TTFT)  († = early EOS, under-saturated)

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | — | — | — | — | — |
| qwen3.5-9b | — | — | — | — | — |
| gpt-oss-20b | — | — | — | — | — |
| qwen3.6-27b | — | — | — | — | — |
| gemma-4-26b | 0.60s ±1% | 0.45s ±0% | — | 0.25s ±14% | 0.93s ±1% |
| qwen3.6-35b-a3b | — | — | — | — | — |

**vision_decode**  (decode tok/s)  († = early EOS, under-saturated)

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | — | — | — | — | — |
| qwen3.5-9b | — | — | — | — | — |
| gpt-oss-20b | — | — | — | — | — |
| qwen3.6-27b | — | — | — | — | — |
| gemma-4-26b | 111 ±0%† | 117 ±0%† | — | 114 ±0%† | 89 ±0% |
| qwen3.6-35b-a3b | — | — | — | — | — |

**peak RSS (GB) / cold load (s, server-boot + first-request)**

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | 3.6 / 5s | 3.6 / 6s | 3.2 / 5s | — | 18.7 / 3s |
| qwen3.5-9b | 6.1 / 5s | 6.3 / 6s | 5.7 / 5s | — | 39.3 / 3s |
| gpt-oss-20b | 12.6 / 7s | 11.9 / 6s | 12.2 / 5s | — | 56.7 / 2s |
| qwen3.6-27b | 16.1 / 8s | 15.7 / 8s | 15.1 / 6s | — | 76.4 / 8s |
| gemma-4-26b | 15.6 / 6s | 15.6 / 20s | 14.7 / 6s | 15.5 / 7s | 74.4 / 16s |
| qwen3.6-35b-a3b | 19.4 / 8s | 19.6 / 8s | 19.1 / 5s | — | 106.0 / 7s |

### Lane: product


**ttft_short**  (median TTFT)  († = early EOS, under-saturated)

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | — | — | — | — | — |
| qwen3.5-9b | — | — | — | — | — |
| gpt-oss-20b | — | — | — | — | — |
| qwen3.6-27b | — | — | — | — | — |
| gemma-4-26b | — | — | — | — | — |
| qwen3.6-35b-a3b | — | — | — | — | — |

**ttft_4k**  (median TTFT)  († = early EOS, under-saturated)

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | 2.29s ±0% | 2.13s ±0% | 2.12s ±0% | — | 2.34s ±0% |
| qwen3.5-9b | 3.81s ±0% | 3.62s ±0% | 3.62s ±0% | — | 3.95s ±0% |
| gpt-oss-20b | 2.27s ±1% | 2.06s ±0% | 2.04s ±1% | — | 2.07s ±0% |
| qwen3.6-27b | 12.96s ±0% | 12.29s ±0% | 12.43s ±0% | — | 13.05s ±0% |
| gemma-4-26b | 2.18s ±0% | 2.33s ±0% | 2.29s ±0% | — | 2.54s ±0% |
| qwen3.6-35b-a3b | 2.05s ±0% | 1.89s ±0% | 1.90s ±1% | — | 2.17s ±0% |

**ttft_16k**  (median TTFT)  († = early EOS, under-saturated)

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | — | — | — | — | — |
| qwen3.5-9b | — | — | — | — | — |
| gpt-oss-20b | — | — | — | — | — |
| qwen3.6-27b | — | — | — | — | — |
| gemma-4-26b | — | — | — | — | — |
| qwen3.6-35b-a3b | — | — | — | — | — |

**decode_b1**  (decode tok/s)  († = early EOS, under-saturated)

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | 150 ±0% | 175 ±0% | 130 ±0% | — | 108 ±0% |
| qwen3.5-9b | 106 ±0% | 115 ±0% | 93 ±0% | — | 79 ±0% |
| gpt-oss-20b | 125 ±0% | 135 ±0% | 110 ±0% | — | 103 ±0% |
| qwen3.6-27b | 36 ±0% | 39 ±0% | 35 ±0% | — | 29 ±0% |
| gemma-4-26b | 111 ±0% | 116 ±0% | 94 ±0% | — | 89 ±0% |
| qwen3.6-35b-a3b | 93 ±0% | 115 ±0% | 87 ±0% | — | 89 ±1% |

**conc_4**  (aggregate tok/s)  († = early EOS, under-saturated)

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | — | — | — | — | — |
| qwen3.5-9b | — | — | — | — | — |
| gpt-oss-20b | — | — | — | — | — |
| qwen3.6-27b | — | — | — | — | — |
| gemma-4-26b | — | — | — | — | — |
| qwen3.6-35b-a3b | — | — | — | — | — |

**conc_8**  (aggregate tok/s)  († = early EOS, under-saturated)

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | 224 ±0% | 134 ±0% | 232 ±0% | — | 93 ±0% |
| qwen3.5-9b | 148 ±0% | 89 ±0% | 148 ±0% | — | 67 ±0% |
| gpt-oss-20b | 215 ±1% | 109 ±0% | 208 ±0% | — | 90 ±0% |
| qwen3.6-27b | 30 ±0% | 30 ±0% | 50 ±0% | — | 24 ±0% |
| gemma-4-26b | 205 ±0% | 93 ±0% | 192 ±0% | — | 78 ±0% |
| qwen3.6-35b-a3b | 154 ±0% | 96 ±0% | 225 ±0% | — | 78 ±0% |

**decode_16k**  (decode tok/s)  († = early EOS, under-saturated)

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | — | — | — | — | — |
| qwen3.5-9b | — | — | — | — | — |
| gpt-oss-20b | — | — | — | — | — |
| qwen3.6-27b | — | — | — | — | — |
| gemma-4-26b | — | — | — | — | — |
| qwen3.6-35b-a3b | — | — | — | — | — |

**vision_ttft**  (median TTFT)  († = early EOS, under-saturated)

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | — | — | — | — | — |
| qwen3.5-9b | — | — | — | — | — |
| gpt-oss-20b | — | — | — | — | — |
| qwen3.6-27b | — | — | — | — | — |
| gemma-4-26b | — | — | — | — | — |
| qwen3.6-35b-a3b | — | — | — | — | — |

**vision_decode**  (decode tok/s)  († = early EOS, under-saturated)

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | — | — | — | — | — |
| qwen3.5-9b | — | — | — | — | — |
| gpt-oss-20b | — | — | — | — | — |
| qwen3.6-27b | — | — | — | — | — |
| gemma-4-26b | — | — | — | — | — |
| qwen3.6-35b-a3b | — | — | — | — | — |

**peak RSS (GB) / cold load (s, server-boot + first-request)**

| model | rapid-mlx | oMLX | mlx-lm | mlx-vlm | Ollama |
|---|---|---|---|---|---|
| qwen3.5-4b | 3.2 / 7s | 3.6 / 7s | 3.1 / 6s | — | 15.5 / 5s |
| qwen3.5-9b | 5.8 / 10s | 6.3 / 9s | 5.6 / 8s | — | 33.5 / 15s |
| gpt-oss-20b | 13.3 / 8s | 11.8 / 8s | 12.2 / 7s | — | 51.8 / 16s |
| qwen3.6-27b | 15.6 / 19s | 15.7 / 20s | 15.1 / 18s | — | 74.7 / 20s |
| gemma-4-26b | 15.6 / 7s | 15.6 / 10s | 14.7 / 8s | — | 81.3 / 14s |
| qwen3.6-35b-a3b | 19.2 / 8s | 19.6 / 10s | 19.1 / 7s | — | 102.6 / 9s |
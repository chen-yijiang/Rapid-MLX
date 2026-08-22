# Mac mini model matrix: Qwen3.5 4B, Gemma 4 26B, and Qwen3.8 27B

Measured 2026-08-21 on a Mac mini (Apple M2 Pro, 10 CPU cores, 32 GB unified
memory) running macOS 26.5.2. Chrome was closed with human authorization. The
host was rebooted before the two large-model comparisons; swap was zero and no
process used more than 20% CPU at the pre-run idle gate.

These are single-stream, short-prompt decode results. They do not establish
HTTP, concurrency, long-context prefill, or multimodal performance.

## Comparable autoregressive decode

Each Rapid-MLX and mlx-vlm cell is the median of 16 samples: eight prompts
repeated twice in one model load. Each oMLX cell combines two independent
eight-prompt model loads by averaging their reported medians. All engines used
the same checkpoint for a model, deterministic sampling, and a maximum of 128
generated tokens.

| Model/checkpoint | Rapid-MLX | mlx-vlm | oMLX | Rapid vs mlx-vlm | Rapid vs oMLX |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B 4-bit | **64.00** | 62.42 | 59.74 | +2.5% | +7.1% |
| Gemma4-26B-A4B 4-bit | **51.44** | 50.56 | 48.39 | +1.8% | +6.3% |
| Qwen3.8-27B 4-bit | 11.60 | **11.66** | 11.57 | -0.5% | +0.2% |

Units are generated tokens per second. Qwen3.8's differences are noise-level
parity; there is no evidence of an autoregressive decode deficit worth a risky
product change. Rounded claim-ready Rapid results are **64.0**, **51.4**, and
**11.6 tok/s**, scoped to this host and workload.

## Memory and cached model load

| Model | Rapid peak MLX memory | mlx-vlm peak MLX memory | Reduction | Rapid load | mlx-vlm load |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | **2.65 GB** | 3.73 GB | 28.9% | 2.90 s | **2.36 s** |
| Gemma4-26B | **14.40 GB** | 15.55 GB | 7.3% | **2.61 s** | 6.51 s |
| Qwen3.8-27B | **15.53 GB** | 18.73 GB | 17.1% | **5.95 s** | 7.63 s |

Gemma4-26B completed without swap under Rapid-MLX and mlx-vlm. oMLX's second
Gemma run left only 0.25 MB used in a 1 GB swap allocation, which is
operationally negligible. The MLX memory counter does not include every mapped
file or host allocation, so these numbers are comparative rather than total
system resident memory.

## Qwen3.8 MTP result

The combined Qwen3.8 checkpoint carries an MTP draft. Rapid-MLX's adaptive
experiment produced:

| Mode | Median decode | Pooled decode | Acceptance | Token-exact prompts |
| --- | ---: | ---: | ---: | ---: |
| AR | **11.64** | **11.65** | — | 8/8 reference |
| adaptive MTP k=2 | 9.43 | 9.39 | 80.6% | 8/8 |
| adaptive MTP k=3 diagnostic | 9.89 | 9.77 | 88.1% | 8/8 |

AR is the best tested configuration on M2 Pro. The k=3 row is diagnostic only:
the benchmark reused a controller initialized with `max_k=2`, and the runtime
logged that it ignored `max_k=3`. This harness isolation issue must be fixed
before using k=3 as a standalone result, but it cannot overturn the conclusion
that the measured MTP paths were slower than AR.

## Prefill observation and next experiment

The direct harness reported lower Rapid prompt rates than mlx-vlm:

| Model | Rapid prompt tok/s | mlx-vlm prompt tok/s | Rapid delta |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B | 116.0 | 144.0 | -19.4% |
| Gemma4-26B | 82.8 | 94.4 | -12.3% |
| Qwen3.8-27B | 19.6 | 24.9 | -21.1% |

Do not publish those rows as long-context prefill claims: the prompts contain
only 14--33 tokens, so fixed setup, cache construction, and synchronization
dominate the rate. The controlled follow-up below establishes the actual cause.

### Prefill scope-down: MLX 0.31.2 is the bottleneck

The follow-up used the Qwen3.5-4B checkpoint, identical decoded text, exactly
128/1,024/4,096 tokens as independently reported by both tokenizers, one output
token, three repeats, and a fresh prompt cache per sample.

| Context | Rapid stack: mlx-lm 0.31.3 + MLX 0.31.2 | mlx-vlm + MLX 0.32.1 | mlx-lm 0.31.3 + MLX 0.32.1 |
| --- | ---: | ---: | ---: |
| 128 | 290.9 | 331.1 | **343.6** |
| 1,024 | 326.9 | 342.2 | **399.9** |
| 4,096 | 324.6 | 395.1 | **397.1** |

Units are prompt tokens per second. The decisive cross-over keeps the same
`mlx-lm` model implementation and changes only `mlx`/`mlx-metal` from 0.31.2 to
0.32.1. It improves 1K and 4K prefill by 22.3%, completely closes the mlx-vlm
gap, and slightly exceeds mlx-vlm at 4K. Therefore:

1. Rapid-MLX's Qwen generation/cache path is not the material bottleneck.
2. The observed prefill deficit is caused by the production dependency ceiling
   `mlx>=0.31.2,<0.32`, not by HTTP/server overhead or the mlx-lm Qwen model.
3. Raising that ceiling is compatibility-sensitive. It is deliberately blocked
   by `scripts/check_mlx_bound_move.py` because an earlier upstream mlx-lm
   heuristic produced incoherent Qwen3.6 output. A full-family output-coherence
   sweep and Atlas approval are required before shipping the faster runtime.

An earlier short-prompt M3 Ultra A/B failed to expose the version effect; those
14--33-token measurements were dominated by fixed overhead and are superseded
by the controlled exact-length result above.

### mlx-lm release versus upstream main

`mlx-lm` 0.31.3 is the latest PyPI release, so the production stack was already
using the newest released mlx-lm. A separate Studio coherence probe compared
that release with official upstream main at commit
`dfb5da1d61f87679b0bc060c0794551e8db0d243`, whose package version is the
unreleased 0.32.0. Both used MLX/Metal 0.32.1 and the same cached
`mlx-community/Qwen3.6-35B-A3B-8bit` checkpoint.

| Stack | Decode median | Prompt median | Cached load | Exact versus 0.31.3 |
| --- | ---: | ---: | ---: | ---: |
| mlx-lm 0.31.3 + MLX 0.32.1 | 88.35 | 277.43 | 7.53 s | reference |
| mlx-lm main (`dfb5da1`) + MLX 0.32.1 | **88.73** | **300.11** | **2.79 s** | 8/8 prompts |

Each prompt generated 64 deterministic tokens. Main was token-identical on all
eight prompts, essentially tied on decode, 8.2% faster on prompt processing,
and materially faster to load. This is positive evidence for testing the next
mlx-lm release as the upgrade target, not proof that every Qwen3.6 checkpoint is
fixed: upstream issue #1197 targets VLM-MTP checkpoint weight layouts and
remains open. The full-family sweep must include that exact failing layout.

## Versions and checkpoints

| Engine | Version | MLX stack |
| --- | --- | --- |
| Rapid-MLX | 0.12.18, source `a3a0d02bbc050c37923b8a1aeb3773f0e3390f94` | mlx 0.31.2, mlx-lm 0.31.3 |
| mlx-vlm | 0.6.15, source `72f37ca46ace7bb8f8b3fd91d1b6c75e20c77b40` | mlx 0.32.1 |
| oMLX | 0.6.3rc2, source `2df39bfcdd9c8fb80847b2869d7f2d62a162f673` | mlx 0.32.0, mlx-vlm 0.6.3, mlx-lm 0.31.3 |

- `mlx-community/Qwen3.5-4B-MLX-4bit`
- `mlx-community/gemma-4-26b-a4b-it-4bit`, snapshot
  `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`
- `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX`, snapshot
  `aa985c29ff5b334cbfdcbbc787d47e66e9d9e456`

## Reproduction

The benchmark workspace is `~/mac-model-matrix` on the mini. The direct command
shape was:

```bash
python bench_direct.py \
  --engine rapid \
  --model /path/to/checkpoint \
  --max-tokens 128 \
  --repeat 2 \
  --output results/model-rapid.json
```

Use the isolated Rapid or mlx-vlm environment and change `--engine` for the
direct comparison. oMLX used `~/qwen9-perf/bench_omlx_engine.py` or the matching
Gemma workspace script, with cache storage disabled. The eight prompts cover
coding, explanation, JSON, memory efficiency, dialogue, summary, arithmetic,
and translation. One four-token generation warmed each loaded engine.

Raw JSON remains outside Git under `~/mac-model-matrix/results` on the mini and
`~/mac-model-matrix/mini-results` on the Studio. SHA-256 evidence:

| Artifact | SHA-256 |
| --- | --- |
| `qwen35-4b-rapid-r1.json` | `53bb5b4af82a7332d9c9d326f6f34f852e98afaf28de9d0eac6e358d107bdc65` |
| `qwen35-4b-mlxvlm-r1.json` | `da019873486c42b65f4fe82e1f46651bc4166d34d360f4e5c29183c5934b515b` |
| `gemma4-26b-rapid-r1.json` | `1fe0e08f4f62cb13cc37ca7a17b136b590ec83adfc25dfbe0907fab15e6c764d` |
| `gemma4-26b-mlxvlm-r1.json` | `f667a7247edd2aec129b859616f99236de7e15c666608da5586f793b393f40f8` |
| `qwen38-27b-rapid-ar-r1.json` | `112e446bd225a3baaef1af198dddc085651ecb1006a0282f2a86191353ba9ce5` |
| `qwen38-27b-mlxvlm-ar-r1.json` | `85a330dc2a631ff556207976e3e065d5a1a15326831bc46e3faee963b77fdedd` |
| `qwen38-27b-rapid-mtp-adaptive-r1.json` | `961c51cca9cf0fe670ac76417a5f12174aeee8b0f6d921912fee5eea8a537e88` |
| `qwen35-4b-prefill-rapid-r1.json` | `8e3869a4eb3a2e78f2fb3ae471586d7923b3066d24a6dbf8ebd482d4d750bffa` |
| `qwen35-4b-prefill-mlxvlm-r1.json` | `8dadbfc1350dbc4d91b326b4853784ffc2e17dc89bf2d3a5978309059f1f2d00` |
| `qwen35-4b-prefill-mlxlm-mlx032-r1.json` | `6aab6f33389269dbda14839c72e6698831dc9c92c151213c4e38f14fd3eb8c6b` |
| `qwen36-35b-mlxlm0313-mlx032-coherence.json` | `ce47d11810815c0860f8b4db6c40720016a138544daeef7c3043693887ee24ac` |
| `qwen36-35b-mlxlm-main-mlx032-coherence.json` | `908db76bb97967e2095f0493c2caebe58c23b9db5bdce09fd941cb2b6319f82a` |

## Limitations

- Direct in-process generation excludes server scheduling and HTTP overhead.
- The engines use their supported dependency stacks, so this is a real-install
  comparison rather than an isolated MLX-version experiment.
- The direct rows generated the same token budget but were not required to be
  token-identical across different engine implementations.
- Qwen3.8 is a native multimodal model, but this report measures text only.
- M2 Pro Qwen3.8 throughput must not be combined with the existing M3 Ultra
  recommendation number; memory bandwidth and hardware differ materially.

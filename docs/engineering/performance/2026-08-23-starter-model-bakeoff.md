# Starter-model first-impression bake-off on M2 Pro

Date: 2026-08-23

## Outcome

No tested model is a drop-in, low-risk replacement for the bundled LFM2.5
1.2B model. Qwen3 1.7B is the best candidate for a product trial: it passed
9/12 first-session cases versus 6/12 for LFM2.5 1.2B and improved tool calling
from 47% to 63%. The cost is a 47% larger weight file, 0.4--0.5 GB more MLX
memory, and lower decode throughput.

Do not select a starter from aggregate benchmark score alone:

- Qwen3 1.7B still invented a future sports result, accepted a false premise,
  and created a reminder with a made-up time. These need prompt/policy or model
  mitigation before it is presented as a trustworthy default.
- The locally quantized Qwen3.5 2B led tool calling at 70%, but scored only
  7/12 on the first-session suite and confidently fabricated both the winner
  and venue of the 2030 World Cup. It is not the recommended starter artifact.
- LFM2.5 2.6B led tools (80%) and general questions (80%), but it is a pure
  reasoning model. With `enable_thinking=false`, `--no-thinking`, and the
  starter-sized output budget, its visible response was usually unfinished
  analysis: 10/12 first-session cases failed and most ended at the token cap.
- Bonsai 1.7B 2-bit is the smallest artifact and scored 8/12, but arithmetic
  and factual correction regressed. Earlier plain-chat testing also observed
  repetition/termination failures, so this run does not clear it for default
  use.

Vector recommends an Atlas-owned product experiment with Qwen3 1.7B, gated on
targeted safety/termination checks and installer-size acceptance. Keep LFM2.5
1.2B as the shipping fallback until that gate passes.

## Environment

| Item | Value |
| --- | --- |
| Machine | Mac mini, Apple M2 Pro, 32 GB unified memory |
| OS | macOS 26.5.2 (25F84), arm64 |
| Rapid-MLX commit | `ba5025f1ac4a804e78417157e2e85530c0d3506f` |
| Runtime tree (`vllm_mlx`) | `8198558067b5788f6a9fc10a95df988d8c5f35af` |
| Python environment | `~/mac-model-matrix/venvs/mlxlm-on-mlx032/bin/python` |
| MLX / mlx-lm / mlx-vlm | 0.32.1 / 0.31.3 / 0.6.15 |
| transformers | 5.12.1 |
| Server | simple engine, one request at a time, localhost |
| Generation settings | temperature 0; thinking disabled; one deterministic run |

## Artifacts

| Model | Artifact and immutable source revision | Weight bytes | Approx. weight size |
| --- | --- | ---: | ---: |
| LFM2.5 1.2B 4-bit | `mlx-community/LFM2.5-1.2B-Instruct-4bit` | 658,540,250 | 628 MiB |
| Qwen3 1.7B 4-bit | `mlx-community/Qwen3-1.7B-4bit@3b1b1768f8f8cf8351c712464f906e86c2b8269e` | 968,080,210 | 923 MiB |
| Bonsai 1.7B 2-bit | `prism-ml/Ternary-Bonsai-1.7B-mlx-2bit@5f3e306330f636cfc6c6241b4850fae6711c5985` | 484,049,216 | 462 MiB |
| LFM2.5 2.6B 4-bit | `LiquidAI/LFM2.5-2.6B-MLX@b41f2b65685e95418f1ac809bb022d4f79e1ab27`, `4bit/` | 1,583,152,892 | 1,510 MiB |
| Qwen3.5 2B local 4-bit | `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` | 1,722,271,785 | 1,643 MiB |

The Qwen3.5 artifact was produced with:

```bash
python -m mlx_vlm convert \
  --hf-path Qwen/Qwen3.5-2B \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --mlx-path ~/starter-bakeoff-models/Qwen3.5-2B-MLX-4bit \
  --quantize --q-bits 4 --q-group-size 64
```

The converter reported 6.225 effective bits/weight because sensitive and
non-quantized components remain. The output weight SHA-256 is
`713fe7e5d3c3965f7106b0d0ee17615f7869c23c8d327996df8c1196fbcf07d5`.

## Results

| Model | First session | Tools | Coding | Math | General | Cold / warm TTFT | Short / long decode | Active / peak MLX RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LFM2.5 1.2B 4-bit | 6/12 | 47% | 30% | 60% | 60% | 201 / 96 ms | 86.5 / 172.7 tok/s | 0.7 / 0.7 GB |
| Qwen3 1.7B 4-bit | **9/12** | 63% | 20% | **70%** | 30% | 195 / 54 ms | 54.9 / 135.1 tok/s | 1.1 / 1.2 GB |
| Bonsai 1.7B 2-bit | 8/12 | 60% | 30% | 40% | 40% | 347 / **51 ms** | 55.5 / **177.3 tok/s** | 0.7 / 0.8 GB |
| LFM2.5 2.6B 4-bit | 2/12 | **80%** | 30% | 40% | **80%** | 229 / 143 ms | 78.4 / 91.2 tok/s | 1.6 / 1.8 GB |
| Qwen3.5 2B local 4-bit | 7/12 | 70% | **40%** | 30% | 20% | **170** / 123 ms | 41.8 / 123.2 tok/s | 1.1 / 1.2 GB |

Bold values are column leaders, not a claim of overall model superiority.
Long-decode requests were capped at 500 tokens, so a model that naturally
stopped earlier can have a less stable throughput sample.

The 12-case first-session suite covers a friendly Chinese greeting, child-level
explanation, multi-turn instruction retention, faithful summary, polite
rewrite, basic arithmetic, false-premise correction, future uncertainty, exact
JSON, concise termination, current-weather tool choice, and asking for a
missing reminder time. It is implemented in `evals/starter_experience.py`.

## Reproduction

Start each artifact through its normal Rapid-MLX alias/profile. For the local
Qwen3.5 artifact, the exact server command was:

```bash
cd ~/starter-bakeoff-runtime
PYTHONPATH="$PWD" ~/mac-model-matrix/venvs/mlxlm-on-mlx032/bin/python \
  -m vllm_mlx.cli serve \
  ~/starter-bakeoff-models/Qwen3.5-2B-MLX-4bit \
  --host 127.0.0.1 --port 18080 --no-thinking \
  --enable-auto-tool-choice --tool-call-parser hermes
```

Run the existing common suites and the first-session suite:

```bash
python evals/run_eval.py \
  --model MODEL --host 127.0.0.1 --port 18080 \
  --parser PARSER --quantization QUANTIZATION \
  --hardware "Mac mini M2 Pro 32GB" --output /tmp/MODEL-eval.json

python evals/starter_experience.py \
  --base-url http://127.0.0.1:18080/v1 \
  --model default --artifact ARTIFACT --output /tmp/MODEL-starter.json
```

Raw JSON contains prompts and model responses and remains local. This report
distills the reproducible environment, commands, scores, and material failure
patterns rather than committing transcripts.

## Limitations and release gate

- This is one machine, one deterministic pass, and one quantization per model.
  It does not establish behavior on 8/16 GB machines or statistical variance.
- The first-session checks are product heuristics, not an academic capability
  benchmark. A pass still requires human review of tone and factual safety.
- The coding grader is sensitive to response formatting; many failures were
  runtime/extraction failures. Do not interpret small coding-score differences
  as capability rankings without a manual audit.
- TTFT uses short prompts and does not measure long-context prefill. Memory is
  MLX active/peak memory, not total process RSS or installer footprint.
- A starter change must separately verify license, download reliability,
  supported macOS/RAM floor, GUI progress/error handling, cancellation,
  uninstall cleanup, and rollback to the known-good artifact.
- Before shipping Qwen3 1.7B, rerun the three material failures with product
  system prompts and tool policy: false premise, unknown future event, and
  reminder missing a time. Atlas owns the final quality/size tradeoff.

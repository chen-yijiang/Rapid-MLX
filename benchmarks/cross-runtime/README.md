# Cross-runtime benchmark

Head-to-head measurement of local LLM serving runtimes on Apple
Silicon: **rapid-mlx**, **oMLX**, **mlx-lm** (`mlx_lm server`), and
**Ollama** (llama.cpp/GGUF), on the most popular local models of
mid-2026.

## Fairness contract

1. **Same client, same wire.** Every runtime is measured through one
   OpenAI-compatible streaming client (`run.py`), so transport and
   SSE-parsing overhead are identical. TTFT is measured to the first
   delta carrying content *or* reasoning; content-only TTFT is also
   recorded.
2. **One contender at a time.** A run refuses to start if the target
   port is occupied; each server is torn down and the GPU given a
   settle pause before the next. Runs happen on an otherwise idle
   machine.
3. **Two lanes.**
   - `parity`: optional accelerators off — rapid-mlx runs
     `--disable-prefix-cache --no-spec-decode --pflash off`, oMLX runs
     `--no-cache`. Raw engine speed.
   - `product`: shipped defaults. What a user actually gets.
4. **Quant parity is approximate and disclosed.** MLX 4-bit vs GGUF
   Q4_K_M are close but not identical (~4.5 bpw both); gpt-oss-20b is
   MXFP4 on both sides and is the cleanest comparison in the matrix.
   File sizes are reported next to speeds.
5. **Deterministic prompts.** temp=0; prompt text is fixed filler
   sized per-model with the model's own tokenizer (128 / 4k / 16k
   targets). Each cell runs 1 discarded warmup (kernel compile,
   page-in) + N repeats; the median and spread are reported. Every
   request carries a unique leading salt: in the product lane prefix
   caches are on, and identical repeated prompts would measure a 100%
   cache hit instead of the engine.
6. **Decode is measured on long generations** (≥512 tokens): MoE and
   hybrid models report misleading TPS on short answers.
7. **Context length is pinned.** Ollama's OpenAI endpoint cannot set
   `num_ctx` per request, so the daemon runs with
   `OLLAMA_CONTEXT_LENGTH=20480` (recorded); the MLX runtimes size KV
   dynamically. Preallocation differences show up in the RSS column —
   that is a real product difference, not noise.
8. **Versions and hashes are archived** in `results/<run>/meta.json`.

## Metrics

| metric | definition |
|---|---|
| `load_s` | server spawn → ready endpoint answering |
| `ttft_s` | request sent → first streamed token (reasoning or content) |
| `ttft_content_s` | request sent → first *content* token |
| `decode_tps` | (completion_tokens − 1) / (last − first delta) |
| `agg_tps` | Σ completion tokens / batch wall time (concurrency cells) |
| `peak_rss_gb` | peak RSS of the server process tree, 500 ms sampling |

## Running

```bash
python3.12 run.py --list                       # resolved matrix
python3.12 run.py --lane parity                # full parity sweep
python3.12 run.py --lane product --models gpt-oss-20b --runtimes rapid,ollama
```

Results land in `results/<run-id>/cells.jsonl` (one record per
runtime × model × scenario) with server logs alongside.

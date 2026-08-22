# Vector handoff

- Status: in progress; benchmark host available
- Active task: benchmark the 16--64 GB Mac model matrix against mlx-vlm,
  oMLX, and applicable MLXFast/MTPLX implementations; optimize material
  Rapid-MLX deficits and re-run the controlled matrix.
- Branch or PR: `raullenchai/vector-desk`; no PR yet.
- Host: final 16--32 GB claims must run on the M2 Pro 32 GB Mac mini at SSH
  alias `mini`; large-model preflight and 48--64 GB-class experiments can run
  on the M3 Ultra 256 GB Studio but must be labelled as Studio results.
- Verified facts:
  - Google Chrome was closed with explicit human authorization before the run.
  - The mini returned `No route to host` on 2026-08-21. A Wake-on-LAN packet
    was sent to its known interface and six SSH retries still failed.
  - The reusable local workspace is `~/mac-model-matrix`; raw preflight JSON
    is under `results/` and is intentionally outside Git.
  - Busy-Studio preflight, Qwen3.5-4B 4-bit: Rapid 175.2 decode tok/s and
    2.65 GB peak MLX memory; mlx-vlm 165.9 tok/s and 3.73 GB. These are not
    publication-grade numbers.
  - Busy-Studio preflight, Gemma4-26B-A4B 4-bit: Rapid 116.6 decode tok/s and
    14.40 GB peak MLX memory; mlx-vlm 113.4 tok/s and 15.55 GB. These are not
    publication-grade numbers.
  - On both preflights Rapid's reported short-prompt prefill rate was about
    20% below mlx-vlm. The controlled Qwen3.5-4B cross-over subsequently found
    the cause: keeping mlx-lm 0.31.3 and moving MLX/Metal 0.31.2 to 0.32.1
    improved 1K/4K prefill by 22.3% (326.9/324.6 to 399.9/397.1 tok/s), closing
    the mlx-vlm gap. Shipping this requires the #1248 full-family coherence gate
    and Atlas approval because the dependency ceiling is deliberate.
  - PyPI's latest mlx-lm remains 0.31.3. Official upstream main at `dfb5da1`
    reports version 0.32.0 and removes MTP-presence as a sufficient reason to
    shift Qwen3.5/3.6 RMSNorm weights. On Studio with Qwen3.6-35B-A3B-8bit and
    MLX 0.32.1, main was exact on 8/8 64-token prompts versus 0.31.3, with
    300.1 vs 277.4 prompt tok/s and 2.79 vs 7.53 seconds cached load. Upstream
    VLM-MTP issue #1197 is still open, so its exact checkpoint layout remains a
    required coherence case.
  - Existing unarchived Qwen3.6-35B experiments under `~/qwen38-perf` suggest
    Rapid AR around 91 tok/s, mlx-vlm AR around 83 tok/s, and Rapid MTP up to
    120.9 tok/s on one coding prompt. Re-run with the common harness before
    making claims.
- Remaining matrix: Qwen3.5-4B, LFM2.5-8B-A1B, GPT-OSS-20B, Mistral Small
  24B, Qwen3-Coder-30B-A3B, Qwen3.8-27B, Gemma4-26B-A4B, Nemotron 30B-A3B,
  Qwen3.6-35B-A3B, Qwen3-Coder-Next-80B, plus Llama 8B, Phi-4 14B, Bonsai,
  and DeepSeek-Coder compatibility baselines.
- Risks: Studio currently has other CPU-active desktop processes; its
  preflight data must not be published. Only about 137 GiB was available on
  the Studio system volume, so uncached checkpoints need staged downloads or
  explicit cache cleanup.
- Next action: ask Atlas to own the compatibility-sensitive MLX 0.32 / mlx-lm
  next-release move, reproduce upstream issue #1197 with its exact VLM-MTP
  checkpoint layout, then run the full-family output-coherence sweep required
  by #1248 before changing `pyproject.toml`. Continue the remaining model matrix
  independently of that dependency event.

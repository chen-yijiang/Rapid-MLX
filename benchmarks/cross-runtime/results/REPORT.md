# Cross-runtime benchmark report

## Run `parity-full`

Versions: rapid rapid-mlx 0.12.10, omlx 0.5.8.dev3, ollama ollama version is 0.32.5, mlx_lm 0.31.3, macos 26.5.2


### Lane: parity


**ttft_short**  (median TTFT)

| model | rapid-mlx | oMLX | mlx-lm | Ollama |
|---|---|---|---|---|
| qwen3.5-4b | 0.18s ±1% | 0.31s ±2% | 0.23s ±0% | 0.31s ±1% |
| qwen3.5-9b | 0.24s ±1% | 0.39s ±1% | 0.30s ±0% | 0.40s ±1% |
| gpt-oss-20b | 0.21s ±1% | 0.34s ±2% | 0.24s ±1% | 0.40s ±1% |
| qwen3.6-27b | 0.67s ±0% | 0.80s ±2% | 0.75s ±1% | 0.91s ±0% |
| gemma-4-26b | 0.27s ±0% | 0.41s ±0% | 0.32s ±1% | 0.48s ±1% |
| qwen3.6-35b-a3b | 0.22s ±2% | 0.35s ±0% | 0.28s ±0% | 0.37s ±1% |

**ttft_4k**  (median TTFT)

| model | rapid-mlx | oMLX | mlx-lm | Ollama |
|---|---|---|---|---|
| qwen3.5-4b | 2.26s ±0% | 2.12s ±0% | 2.10s ±0% | 2.35s ±0% |
| qwen3.5-9b | 3.75s ±0% | 3.62s ±0% | 3.60s ±0% | 3.94s ±0% |
| gpt-oss-20b | 2.14s ±1% | 2.06s ±0% | 2.02s ±0% | 2.07s ±1% |
| qwen3.6-27b | 12.77s ±0% | 12.28s ±0% | 12.40s ±0% | 13.05s ±0% |
| gemma-4-26b | 2.22s ±0% | 2.33s ±0% | 2.27s ±0% | 2.54s ±0% |
| qwen3.6-35b-a3b | 2.00s ±1% | 1.89s ±0% | 1.89s ±2% | 2.16s ±0% |

**ttft_16k**  (median TTFT / decode tok/s)

| model | rapid-mlx | oMLX | mlx-lm | Ollama |
|---|---|---|---|---|
| qwen3.5-4b | 9.30s ±1% / 108 tps | 8.61s ±0% / 149 tps | 8.76s ±0% / 111 tps | 8.85s ±0% / 98 tps |
| qwen3.5-9b | 15.31s ±0% / 101 tps | 14.61s ±0% / 104 tps | 14.74s ±0% / 83 tps | 14.38s ±0% / 74 tps |
| gpt-oss-20b | 9.21s ±0% / 52 tps | 8.60s ±0% / 108 tps | 8.74s ±0% / 84 tps | 7.24s ±0% / 96 tps |
| qwen3.6-27b | 52.68s ±0% / 30 tps | 50.90s ±0% / 35 tps | 51.54s ±0% / 31 tps | 48.42s ±0% / 27 tps |
| gemma-4-26b | FAIL | 9.44s ±0% / 115 tps | 9.36s ±0% / 80 tps | 8.74s ±0% / 80 tps |
| qwen3.6-35b-a3b | 8.43s ±0% / 86 tps | 7.82s ±0% / 99 tps | 8.05s ±0% / 77 tps | 8.39s ±0% / 81 tps |

**decode_b1**  (decode tok/s)

| model | rapid-mlx | oMLX | mlx-lm | Ollama |
|---|---|---|---|---|
| qwen3.5-4b | 168 ±0% | 175 ±0% | 131 ±0% | 108 ±0% |
| qwen3.5-9b | 115 ±0% | 116 ±0% | 92 ±0% | 80 ±0% |
| gpt-oss-20b | 126 ±0% | 145 ±1% | 111 ±0% | 105 ±0% |
| qwen3.6-27b | 39 ±0% | 39 ±0% | 35 ±0% | 29 ±0% |
| gemma-4-26b | 114 ±0% | 156 ±0% | 96 ±0% | 88 ±1% |
| qwen3.6-35b-a3b | 102 ±1% | 114 ±0% | 89 ±1% | 88 ±0% |

**conc_4**  (aggregate tok/s)

| model | rapid-mlx | oMLX | mlx-lm | Ollama |
|---|---|---|---|---|
| qwen3.5-4b | 75 ±0% | 136 ±0% | 196 ±0% | 94 ±0% |
| qwen3.5-9b | 60 ±2% | 90 ±0% | 131 ±0% | 67 ±0% |
| gpt-oss-20b | 87 ±2% | 103 ±1% | 150 ±1% | 85 ±0% |
| qwen3.6-27b | 17 ±1% | 31 ±0% | 47 ±0% | 24 ±0% |
| gemma-4-26b | 73 ±2% | 53 ±3% | 156 ±0% | 109 ±9% |
| qwen3.6-35b-a3b | 60 ±0% | 97 ±0% | 173 ±0% | 78 ±0% |

**conc_8**  (aggregate tok/s)

| model | rapid-mlx | oMLX | mlx-lm | Ollama |
|---|---|---|---|---|
| qwen3.5-4b | 79 ±2% | 136 ±0% | 236 ±0% | 100 ±0% |
| qwen3.5-9b | 65 ±0% | 90 ±0% | 151 ±0% | 72 ±0% |
| gpt-oss-20b | 98 ±3% | 104 ±1% | 185 ±0% | 93 ±0% |
| qwen3.6-27b | 18 ±1% | 31 ±0% | 51 ±0% | 26 ±0% |
| gemma-4-26b | 78 ±2% | 52 ±1% | 188 ±13% | 142 ±1% |
| qwen3.6-35b-a3b | 74 ±2% | 96 ±0% | 230 ±0% | 83 ±1% |

**peak RSS (GB) / cold load (s, server-boot + first-request)**

| model | rapid-mlx | oMLX | mlx-lm | Ollama |
|---|---|---|---|---|
| qwen3.5-4b | 3.4 / 5s | 3.6 / 6s | 3.2 / 5s | 19.8 / 3s |
| qwen3.5-9b | 6.0 / 5s | 6.3 / 6s | 5.7 / 5s | — |
| gpt-oss-20b | 12.5 / 7s | 11.9 / 6s | 12.2 / 5s | 38.5 / 2s |
| qwen3.6-27b | 16.1 / 8s | 15.7 / 8s | 15.1 / 6s | — |
| gemma-4-26b | 15.7 / 6s | 15.6 / 20s | 14.7 / 6s | 39.8 / 16s |
| qwen3.6-35b-a3b | 19.4 / 8s | 19.6 / 8s | 19.1 / 5s | — |
## Run `product-key`

Versions: rapid rapid-mlx 0.12.10, omlx 0.5.8.dev3, ollama ollama version is 0.32.5, mlx_lm 0.31.3, macos 26.5.2


### Lane: product


**ttft_4k**  (median TTFT)

| model | rapid-mlx | oMLX | mlx-lm | Ollama |
|---|---|---|---|---|
| qwen3.5-4b | 2.29s ±0% | 2.13s ±0% | 2.12s ±0% | 2.34s ±0% |
| qwen3.5-9b | 3.81s ±0% | 3.62s ±0% | 3.62s ±0% | 3.95s ±0% |
| gpt-oss-20b | 2.27s ±1% | 2.06s ±0% | 2.04s ±1% | 2.07s ±0% |
| qwen3.6-27b | 12.96s ±0% | 12.29s ±0% | 12.43s ±0% | 13.05s ±0% |
| gemma-4-26b | 2.18s ±0% | 2.33s ±0% | 2.29s ±0% | 2.54s ±0% |
| qwen3.6-35b-a3b | 2.05s ±0% | 1.89s ±0% | 1.90s ±1% | 2.17s ±0% |

**decode_b1**  (decode tok/s)

| model | rapid-mlx | oMLX | mlx-lm | Ollama |
|---|---|---|---|---|
| qwen3.5-4b | 169 ±0% | 175 ±0% | 130 ±0% | 108 ±0% |
| qwen3.5-9b | 114 ±1% | 116 ±0% | 92 ±0% | 80 ±0% |
| gpt-oss-20b | 126 ±2% | 147 ±2% | 111 ±0% | 105 ±0% |
| qwen3.6-27b | 39 ±0% | 39 ±0% | 35 ±0% | 29 ±0% |
| gemma-4-26b | 114 ±0% | 155 ±1% | 95 ±0% | 92 ±2% |
| qwen3.6-35b-a3b | 102 ±0% | 115 ±0% | 88 ±1% | 88 ±1% |

**conc_8**  (aggregate tok/s)

| model | rapid-mlx | oMLX | mlx-lm | Ollama |
|---|---|---|---|---|
| qwen3.5-4b | 79 ±0% | 136 ±0% | 237 ±1% | 95 ±0% |
| qwen3.5-9b | 64 ±1% | 90 ±0% | 151 ±0% | 68 ±0% |
| gpt-oss-20b | 100 ±3% | 104 ±1% | 186 ±0% | 87 ±0% |
| qwen3.6-27b | 18 ±2% | 31 ±0% | 51 ±0% | 24 ±0% |
| gemma-4-26b | 79 ±2% | 52 ±1% | 196 ±0% | 79 ±0% |
| qwen3.6-35b-a3b | 64 ±4% | 96 ±0% | 231 ±0% | 79 ±0% |

**peak RSS (GB) / cold load (s, server-boot + first-request)**

| model | rapid-mlx | oMLX | mlx-lm | Ollama |
|---|---|---|---|---|
| qwen3.5-4b | 3.2 / 7s | 3.6 / 7s | 3.2 / 6s | 16.5 / 5s |
| qwen3.5-9b | 5.8 / 10s | 6.3 / 9s | 5.7 / 8s | 35.6 / 15s |
| gpt-oss-20b | 12.3 / 8s | 11.8 / 8s | 12.2 / 7s | 52.4 / 16s |
| qwen3.6-27b | 15.6 / 19s | 15.7 / 20s | 15.1 / 18s | 77.1 / 20s |
| gemma-4-26b | 15.7 / 7s | 15.6 / 10s | 14.7 / 8s | 73.4 / 14s |
| qwen3.6-35b-a3b | 19.2 / 8s | 19.6 / 10s | 19.1 / 7s | 105.5 / 9s |

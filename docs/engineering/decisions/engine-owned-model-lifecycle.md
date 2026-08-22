# Engine-owned model lifecycle

## Decision

Rapid-MLX treats the inference engine and its scheduler as the source of truth
for model lifecycle safety. A model replacement closes engine admission before
it inspects or changes scheduler-owned work, then applies one of two policies:

- `wait`: let admitted, queued, and running requests finish.
- `abort`: abort queued and running requests, including a request that crossed
  the admission boundary immediately before the pause.

The existing `reject` load policy is retained as a non-blocking control-plane
preflight. Internally it is an atomic zero-timeout `wait`: an idle engine stays
paused for replacement, while a busy engine is resumed and returns HTTP 409.

The residency status reports the engine's paused state and its admitted,
queued, and running counts. The GUI may ask for confirmation based on this
status, but it does not own or infer request lifetime.

## Upstream model

This follows the lifecycle contract already used by vLLM and SGLang:

- vLLM `pause_generation` supports `wait`, `abort`, and `keep`, and implements
  the pause state in `EngineCore`/scheduler rather than ASGI middleware.
- SGLang pauses its scheduler before weight mutation and distinguishes abort,
  retract, and in-place policies over its waiting/running queues.

Rapid-MLX does not implement `keep`/`retract` for whole-model replacement. A
request cannot safely resume against a different model and tokenizer. Those
modes remain appropriate for updating weights of the same model architecture,
which is not the operation performed by `/v1/models/load`.

Primary references:

- <https://github.com/vllm-project/vllm/blob/main/docs/training/async_rl.md#the-pause-and-resume-api>
- <https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py>
- <https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/sglang_for_rl.mdx#easy-to-postpone-generation>
- <https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py>

## Rejected design

Do not maintain an HTTP-response-scoped active request counter or a separate
confirmation transaction state machine. HTTP lifetime is not inference
lifetime: asynchronous jobs can return before their generation task starts or
finishes. Confirmation is a client interaction; engine pause and scheduler
ownership are the server safety mechanism.

# Server-owned model lifecycle control plane

## Decision

Request admission and model lifecycle drain state belong to the server. GUI,
CLI, and external supervisors are clients of this control plane; they must not
infer interruption safety from a non-atomic status snapshot.

The server exposes authenticated lifecycle operations under
`/v1/models/lifecycle`. Creating an operation and closing inference admission
happen under one lock. The response reports the requests that were already
active:

- zero active requests: the operation is immediately `ready`;
- one or more active requests: it is `awaiting_confirmation`;
- confirmation moves it to `ready` without admitting new inference;
- cancel, complete, supersession, or expiry reopens admission.

An external process supervisor may terminate/restart `rapid-mlx serve` only
after receiving a `ready` operation. The server cannot restart its own process,
but it owns the admission barrier that makes that restart safe and explicit.

## Compatibility

Existing inference and `/v1/models/load` contracts are unchanged while no
lifecycle operation is active. During a drain, authenticated new inference
receives HTTP 409 with code `model_lifecycle_draining` and `Retry-After: 1`.
Health, lifecycle status/transition routes, and existing control-plane routes
remain reachable. Authentication still runs before lifecycle state is exposed.

The ASGI middleware holds admission for the complete response body, including
streaming responses. Operations expire after five minutes so an abandoned
client cannot leave the server permanently drained.

## Follow-up

The Mac app should consume this API as a thin lifecycle client and process
supervisor. Resident-engine replacement can later execute directly within the
same operation, while process replacement uses the ready operation as a restart
permit.

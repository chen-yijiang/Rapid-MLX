# Desktop video lane — MVP findings, decision, and future plan

Status: **reference only.** No Video tab or capability slot is shipped. This
records why, and what to build if/when we do.

## The question

The engine's video lane is complete (`/v1/videos`, 8 `[video:gen]` aliases,
LTX-2.3 / Wan2.2 / CogVideoX-Fun backends). The desktop app has scaffolding for
it (`ModelKind.video`, a "Video" tab label, `capabilityTabs`) but nothing
populates it — `availableKinds` never contains `.video`. Should we fill that
slot with a Video tab mirroring the Images tab?

## MVP: what actually runs on a Mac mini (M2 Pro, 32 GB)

Measured on the mini, text-to-video, "golden retriever running through a field
of tall grass at sunset", seed 42. Samples: `~/work/vidgen/mvp-video-mini/`.

| Model | Size | Fits 32 GB | Speed (mini) | Quality | Verdict |
|---|---|---|---|---|---|
| **LTX-2.3 q4** | 21.2 GB | ✅ (mem comfortable) | warm ~4.2 min / 3 s @ 512×320 · cold ~11 min @ 768×512 | **Good** — subject present, photorealistic, coherent motion | ✅ the only usable one |
| Wan2.2 TI2V 5B q8 | 18.2 GB | ✅ (85 % free) | ~18 min / 3 s @ 512×320 (24 s/step × 40) | Weak at feasible sizes — subject missing, artifacts | ❌ too slow + not good enough |
| CogVideoX-Fun 5B | 13.3 GB | ✅ | not measured | — | ❌ engine restricts to `seconds=1` + `672×384` only |
| Wan2.2 A14B i2v/t2v | 39.7–64 GB | ❌ > 32 GB | — | — | ❌ Studio-class only |

LTX-2.3 is already low-step (STAGE 1 = 8 steps + STAGE 2 = 3 steps); the mini's
bottleneck is per-step GPU compute + VAE decode, so there is no "seconds-level"
config. On a Mac Studio (M3 Ultra, ~80 GPU cores) everything is ~4× faster and
Wan becomes viable — but the target here was the mini.

## Decision: C — video stays engine/API-only for now

Local video is a **minutes-per-clip render**, not a **seconds-per-image
interaction** — and this is universal (even cloud Sora is async/minutes). A
Video tab that copies the Images tab's type-and-see canvas would be a poor UX
at 4–18 min/clip.

We already have the capability where it belongs: the engine/API, used offline
and in batch by the Mill content pipeline. So:

- **Do not** wire `videoEntries` or build a Video tab now.
- Leave the `.video` scaffolding in place (harmless — the tab never appears).
- The capability slot staying unfilled is the *correct* state under this decision.

## If we build it later: option B — a render queue, not a canvas

Design video as an honest **render job**, not a fake-interactive tab:

- Submit → background job → progress bar with an up-front time estimate
  ("~8 min on this Mac") → completion notification. The engine is already an
  async job model (`POST /v1/videos` → poll → retrieve), so this is a natural
  fit.
- **Gate the model recommendation by hardware**: mini / 32 GB → recommend
  **LTX-2.3 q4 only**, with the time expectation shown; Wan/large models gated
  to Studio-class. Never let a mini user pick an 18-minute model unaware.

### Implementation recipe

1. `ModelCatalog.videoEntries` / `parseVideoRows` — already written here as a
   reference mirror of the image lane.
2. Surface video models in Model Management: call `videoEntries` alongside
   `imageEntries` at `SettingsModelManagementPanel.swift:1325` and `:1338`.
   `availableKinds` then gains `.video` and the capability tab appears.
3. A `VideoGenViewModel` mirroring `ImageGenViewModel`, but modelling a job
   queue (submit / poll / cancel / notify) rather than a synchronous call.
4. A "Video" sidebar entry + a render-queue view (list of jobs with status,
   time estimate, and the finished clips) rather than a prompt-and-canvas.
5. Hardware-gated defaults via the existing `RAMBucketedDefault` /
   `MacHardware` machinery.

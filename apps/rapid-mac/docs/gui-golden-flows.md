# AX-first GUI golden flows

`scripts/gui-golden-flows.sh` runs the release journeys against a built
Rapid-MLX Desktop app without loading a real model.

**Journeys** — a user walking a path end to end:

1. fresh install, consent, onboarding, and steady-state shell;
2. Settings mutation and persistence across an app relaunch;
3. basic chat, persisted conversation row, and restored transcript;
4. a deliberately slow stream and semantic **Stop generating** action;
5. model start, a one-shot sidecar crash, automatic respawn, and ready state.
6. a memory-constrained user can see and select an honestly labelled sub-1B
   fallback instead of being sent back to a chooser whose smallest visible
   model is the one that just failed the live-memory guard.

**Invariants** — properties that must hold, not paths a user walks. These were
added after a release where every escaped defect landed on a surface no journey
covered, and each one names the defect it would have caught:

7. `update-state` — Settings → App must name the version the app actually is.
8. `no-dead-controls` — every Settings panel must expose controls of its own.
9. `catalog-integrity` — a model that cannot chat must never be offered as one.
   Now covers **image-gen** aliases too: `rapid-mlx models` tags them
   `[image:gen]` in their own section (mirroring `[video:gen]`), and the
   chat catalog's `hasNonChatKindTag` drops `image` alongside `audio`/`video`,
   so a 24 GB FLUX/Qwen-Image checkpoint can never surface in the chat picker.

10. `image-generation` — the Images tab turns a text prompt into a picture and
    lets the user iterate by re-prompting (see **Image generation** below). The
    instruction-**edit** path exists in code but is parked as a slow, batch-only
    lane on current hardware (~20 min/edit at q4); the interactive golden flow is
    text→image generation.

The distinction matters. A journey answers *"can someone do this?"*; an
invariant answers *"is this still true everywhere?"*. The three defects below
were all invisible to journey-shaped tests:

| Flow | Would have caught | Why a journey missed it |
| --- | --- | --- |
| `update-state` | [#1612](https://github.com/raullenchai/Rapid-MLX/issues/1612) — the fallback update manifest sat at 0.11.0 for four releases | Nothing in a journey compares what the panel *says* to what the bundle *is* |
| `no-dead-controls` | [#1595](https://github.com/raullenchai/Rapid-MLX/pull/1595) dead recovery buttons, [#1608](https://github.com/raullenchai/Rapid-MLX/pull/1608) toggles that reported success without changing value, [#1605](https://github.com/raullenchai/Rapid-MLX/issues/1605) a tray item that reported nowhere | A journey visits the controls it needs; these were the ones nobody scripted |
| `catalog-integrity` | [#1603](https://github.com/raullenchai/Rapid-MLX/issues/1603) — eight video-generation models offered as chat models, dead-ending *after* a download of up to 64 GB | The picker renders them perfectly; the bug is that they are there at all |

### Current baseline

Run on 2026-08-07, on a build of this checkout:

| Flow | Result |
| --- | --- |
| `update-state` | **PASS** — the panel's version matches `CFBundleShortVersionString` (0.12.6 when recorded; the assertion compares the two, it does not pin a literal) |
| `catalog-integrity` | **PASS** — `fake-video-alias` reaches neither the chat surface nor Model Management |
| `no-dead-controls` | **PASS** — all six Settings panels expose controls of their own; see the red → green note below |

The first two were measured against `main`. `no-dead-controls` was red on `main`
and is green as of the identifier work described below; the run recorded here is
the one that made it green.

#### `no-dead-controls`: red → green

This flow shipped red, on purpose, and has since been driven green by fixing
the product rather than the assertion. Worth recording, because a gate that has
never moved is a gate nobody has evidence about.

It first failed on **`tools`**: Settings → Tools rendered three tool toggles, a
backend radio group and a browsing toggle, and **none of them carried an
identifier**. The controls worked — real `AXCheckBox`/`AXRadioButton` with
correct values — they were simply unaddressable. Naming them took that panel
`0 → 8`, and the flow then failed one panel further along, on **`privacy`**,
which had the same gap: a telemetry toggle and three policy `Link`s, all
unnamed. Naming those took `privacy` `0 → 4`. Final measured run:

```
[gui-golden]   models: 2 identified controls
[gui-golden]   modelManagement: 16 identified controls
[gui-golden]   tools: 8 identified controls
[gui-golden]   appearance: 3 identified controls
[gui-golden]   privacy: 4 identified controls
[gui-golden]   app: 5 identified controls
[gui-golden] PASS — no-dead-controls
```

`app` was never bare — the loop simply died at `privacy` before reaching it.
Confirmed rather than assumed: `Settings.App.{UpToDate,RecheckCTA,
ExportDiagnostics,HideDockOnCloseToggle,ResetDockOnboardingCTA}`.

#### What this flow does NOT prove

It counts identifiers; it does not press them. A panel can be fully addressable
and still contain a control that does nothing observable, so green here means
"reachable", not "works".

That was not hypothetical, and the worked example is worth keeping now that it
has been fixed — because the flow stayed green through the whole of it.

`Settings.Privacy.TelemetryToggle` is addressable and is a real `AXCheckBox`.
`AXPress` on it flipped the stored preference
(`com.rapidmlx.rapid.telemetry.enabled` `0 → 1`, a client ID minted, the shared
`~/.rapid-mlx/telemetry-client-id` written) — and the switch did not re-render,
so its AX value stayed `0` until you left the panel and came back.
`SettingsView.telemetryEnabledBinding`'s getter read `TelemetryConfig.isEnabled`,
a plain `static var` over `UserDefaults.standard`, which gives SwiftUI no
dependency to invalidate on. To a user: a consent switch that appears to refuse
their choice while they are in fact opted in. Same family as
[#1608](https://github.com/raullenchai/Rapid-MLX/pull/1608) in the table above.

Fixed in [#1623](https://github.com/raullenchai/Rapid-MLX/issues/1623), measured
both ways on real builds:

| build | AX value after press | preference |
| --- | --- | --- |
| before | `0 → 0` (stuck) | `0 → 1` |
| after | `0 → 1` | `0 → 1` |

**`no-dead-controls` was green for every one of those runs.** It counts
identifiers; it does not press them. The gap it leaves is exactly this: a
control that is reachable and inert. A successor that presses each control and
asserts the observable value moved is what closes it — until then, read a green
here as "reachable", never as "works".

Two notes on writing assertions here, both learned the hard way while adding
these:

- The first version of `no-dead-controls` counted every `Settings.*` identifier
  on the panel. The six `Settings.Category.*` buttons appear on *every* panel,
  so the count was never below six and the flow went green on a completely
  unlabelled Tools panel. Count the panel's **own** controls.
- `catalog-integrity` proves it discriminates rather than trivially passing:
  the non-video `fake-alias` appears 9 times in the same tree where
  `fake-video-alias` appears 0 times. A test that asserts an absence must show
  that the corresponding presence is detectable.

Every journey gets a unique bundle identifier and throwaway `HOME` through
`dogfood-isolate.sh`. The fake sidecar emits deterministic SSE and JSONL
lifecycle evidence, so the suite does not download a model or put meaningful
pressure on unified memory.

### Low-memory recovery

The normal model picker intentionally hides sub-1B models: they fall below the
default quality and tool-use floor, and presenting them beside normal choices
without context makes a faster but worse answer look like a product failure.
That policy cannot govern a recovery path. If the live memory guard says the
starter is unsafe and tells the user to “pick a smaller model,” onboarding must
actually contain one.

`low-memory-choice` pins the visible half of that contract through AX:

1. open fresh onboarding and advance to **Choose your first model**;
2. find `Quickstart.Choice.qwen3-0.6b-4bit` under **LOWEST MEMORY**;
3. assert that the card says **less accurate** and **not recommended for
   tools**, so lower memory is not presented as equivalent quality;
4. select it through AX and retain the before/after trees as evidence.

The warning-to-switch half is deterministic Swift coverage rather than a host-
RAM-dependent GUI trick. `QuickstartView.lowMemoryRecoveryChoice(for:)` replays
the original live-memory snapshot against the fallback footprint and exposes
`Quickstart.Memory.SwitchToLowMemory` only when the replacement falls below the
85% danger line. Under heavier pressure the button is absent, avoiding a false
promise or a warning loop; **Cancel** still returns to the chooser where the
low-memory category remains visible.

## Image generation

The Images tab is a dedicated text→image / image-edit surface, reached from
`Sidebar.Images`. It is decoupled from chat on purpose: rapid-mlx serves **one
model per process**, so an image-gen alias (e.g. `flux-schnell-4bit`) cannot be
loaded alongside the chat LLM — selecting one reloads the sidecar, exactly the
stop/start path a chat model-switch already takes.

**The interactive golden flow is text→image generation.** The tab imitates the
fast half of ChatGPT's image experience: type a prompt, get a picture in
seconds, refine by re-prompting. `image-generation` walks that through AX
identifiers, no real diffusion weights required (the fake sidecar answers
`/v1/images/*` with a 1×1 PNG). Demonstrated live with real weights below.

1. open the Images tab via `Sidebar.Images`; assert `Images.EmptyState` is
   present (no results yet) and its copy invites a prompt + model;
2. pick an image model in `Images.ModelPicker` (the list is the `[image:gen]`
   rows from `rapid-mlx models`, never a chat alias — see `catalog-integrity`);
   optionally set `Images.SizePicker`;
3. **Generate.** Type into `Images.Prompt`, press `Images.Generate`; assert the
   button shows the in-flight state while the request is open, then that a
   result card appears under `Images.Gallery` (and `Images.EmptyState` is gone);
4. **Refine by re-prompting.** Adjust the prompt and press `Images.Generate`
   again; each render lands as its own card under `Images.Gallery`, so the user
   iterates a look by editing words rather than waiting on a slow image-edit.
5. `Images.Result.Save` opens the standard save panel on any card (not asserted
   through the panel itself — a modal `NSSavePanel` is out of AX scope, like
   every other file-picker in the app).

### Instruction edit — parked (slow, batch-only)

The tab also ships an instruction-**edit** path (`Images.Result.Edit` on a card
re-stages its output as the next `image_paths`, driving `/v1/images/edits`). It
is **correct but deliberately not the interactive flow**: on current Apple
Silicon a q4 Qwen-Image-Edit render is ~20 min (see realities below), so it reads
as a batch action, not a conversation. The endpoint, `ImageGenViewModel.runEdit`
/ `beginEdit`, and their hermetic tests stay in the tree for when a distilled
edit model or faster hardware makes it interactive; until then the product story
is generation.

### Model realities the UX has to design around

Verified on an M2 Pro 32 GB with the 4-bit mflux checkpoints:

* **Generate is fast, edit is not.** `flux-schnell-4bit` is step-distilled — a
  512² image lands in ~27 s at 4 steps. `qwen-image-edit-4bit` is a large,
  non-distilled 20B model and mflux fixes the edit canvas to a ~1024²-area
  render, so each denoise step is ~1 min and a default 20-step edit is
  **~20 minutes**. The edit round is a *batch* action on this hardware, not the
  sub-second turnaround ChatGPT has; the compose bar must show a clearly
  long-running, cancellable in-flight state and never imply instant results.
* **Do not send a `size` on edits.** mflux derives the edit output canvas from
  the input image (its VAE conditioning latents are pinned to a 1024²-area
  grid). Forcing a mismatched `width`/`height` desyncs the RoPE position ids and
  the model returns a valid-looking PNG of **pure noise**. The engine passes
  `None` for edit dimensions and the route accepts `size` only for OpenAI-API
  shape, then discards it. Round 1+ therefore inherits the round-0 framing.
* **Edit quality is checkpoint-bound, not step-bound.** The q4 edit checkpoint
  carries a persistent VAE speckle (40 steps looks the same as 20); a clean
  edit needs a higher-precision repo (`OsaurusAI/Qwen-Image-Edit-mflux-q6` /
  `-q8`), which costs proportionally more RAM/disk. Raising `steps` past ~20 on
  q4 only burns wall-clock.

The **model-vs-endpoint contract** is the load-bearing invariant here and is
covered by pure Swift coverage rather than a live flow: a text-to-image alias
must drive `/v1/images/generations` and an `*-image-edit` alias
`/v1/images/edits`; the server answers the wrong pairing with a 409 rather than
a silent wrong result (`ImageGenViewModel.selectedIsEditModel` gates which one
the compose bar offers, and the routes enforce it server-side).

> Status: the AX identifiers and states above are **defined and shipped** in
> product code; the runnable `gui-golden-flows.sh --flow image-generation`
> journey and its structural baseline are the next increment (added the same way
> every other flow was — identifiers first, then the scripted journey).

## Run

Build the current checkout, then run all flows:

```bash
cd apps/rapid-mac
SKIP_SIDECAR=1 BUNDLE_MODEL=0 ./scripts/build.sh
./scripts/gui-golden-flows.sh
```

Run one journey or retain its isolated persona for diagnosis:

```bash
./scripts/gui-golden-flows.sh --flow slow-stream-stop
./scripts/gui-golden-flows.sh --flow low-memory-choice
./scripts/gui-golden-flows.sh --flow chat-restore --keep
./scripts/gui-golden-flows.sh --flow no-dead-controls
```

The suite needs a **local login session** — not SSH or tmux. It also needs the
screen to stay awake: when the session goes idle, `CGSSessionScreenIsLocked`
flips to `Yes`, every app reports zero windows through AX, and `screencapture`
returns wallpaper. That looks exactly like a broken app. Hold the session with
`caffeinate -dimsu -t <seconds>` for the length of the run — `-u` is the
load-bearing flag, since plain `-d` stops display sleep but not the idle path —
and re-read the lock state before trusting any window assertion.

Set `RAPID_GUI_SOURCE_APP` to test a release candidate bundle and
`RAPID_GUI_GOLDEN_OUT` to choose the artifact directory. Each run records AX
trees, actions, fake-sidecar events, logs, and a top-level `result.json`.

## AX structural baselines

Ten settled states across the five journeys are also fingerprinted as
**structural baselines**, committed under
`Tests/GUIGoldenFlows/__Snapshots__/<flow>.<state>.txt`. `scripts/ax-baseline.py`
normalises a raw AX dump into an indented tree and the suite fails on any
difference, so a PR that removes a button, reparents a control, renames an
identifier, drops an icon or flips an enabled state produces a reviewable diff
instead of passing silently.

**This is the cheap layer of appearance testing and it is structural only.** It
cannot see colour, spacing, typography or anything else that never reaches the
accessibility layer; the PNG snapshots in `Tests/RapidTests/__Snapshots__` stay
the pixel-level check.

The normaliser keeps hierarchy, role, subrole, `accessibilityIdentifier`,
`AXTitle`/`AXDescription`/`AXHelp`, enabled state, sibling order below the
window level, and the *kind* of each value (`bool:true`, `bool:false`,
`number`, `text`, `empty`). It drops or rewrites everything that is legitimately
volatile: screen coordinates and sizes, pids, top-level window z-order, value
contents, and version numbers, byte sizes, token rates, durations, dates, clock
times, UUIDs, `/Users/<name>` paths and the fake model alias wherever they
appear in text. `Settings.App.UpToDate` carries the release version and a
conversation row identifier carries a fresh UUID — recording those verbatim
would make the baselines flap every release and every run.

Two further things are dropped because they flap without any product change,
both found by comparing real recorded baselines rather than by reasoning:

- **Everything below a window-control button.** The traffic lights are AppKit's,
  and their anonymous `AXGroup` descendants are realized lazily: two dumps taken
  seconds apart in the *same* run recorded one group under `AXZoomButton` in
  `settings-root` and two in `models-idle`. The buttons themselves stay, so a
  missing close box is still a diff; their private innards do not.
- **Relative day headings.** A transcript is filed under `Today` — until a run
  straddles local midnight, at which point the identical UI says `Yesterday`
  and every baseline holding one goes red at 00:00 for no reason.

An intended UI change is a deliberate commit:

```bash
./scripts/gui-golden-flows.sh --update-baselines
git diff apps/rapid-mac/Tests/GUIGoldenFlows/__Snapshots__
```

Recording is **only** ever done by `--update-baselines`. A missing baseline is a
failure, not a free pass: recording on absence would mean a typo'd snapshot name,
or one somebody forgot to `git add`, sails through CI green while comparing
against nothing. (This deliberately diverges from the PNG convention in
`Tests/RapidTests/SnapshotHelpers.swift`.)

Inspect a single normalised tree without running a journey:

```bash
python3 scripts/ax-baseline.py normalize --scrub fake-alias /tmp/…/steady.json
```

## Why this is not coordinate automation

The checked-in `rapid-ax.swift` helper talks directly to macOS Accessibility.
It finds controls by stable `AXIdentifier`, performs `AXPress`, sets native text
values, and serializes roles/descriptions/values for assertions. Peekaboo is
kept for permission checks, window discovery, menu interaction, and screenshots.
The only coordinate fallback is the documented first-run SwiftUI consent-sheet
fallback, derived from AX bounds, for older accessibility stacks.

This makes normal actions independent of window position, resolution, theme,
and most layout changes. It also avoids Peekaboo snapshot publication failures
seen while the ready-state UI is updating. The app does not need to own the
foreground for ordinary AX actions; menu operations and screenshots may briefly
activate it, so release CI should still use a dedicated logged-in macOS session.

## Adding a flow

Prefer a stable `.accessibilityIdentifier(...)` in product code, then assert
observable user state rather than sleeps or pixels. Keep model behavior behind
the fake sidecar unless the purpose of the flow is real inference quality. A
real-model dogfood pass remains a separate, explicitly memory-budgeted release
stage.

Fingerprint only *settled* states. Baselines taken mid-transition flap: the
crash-recovery tree captured while the sidecar was still restarting contained a
transient "Starting …" banner in one run and not the next. `wait_send_idle`
exists for this — `ChatView.SendOrStopButton` publishes `AXHelp` only while the
readiness gate is closed, so the absence of that attribute is a
copy-independent "ready and not streaming" signal. If a new state turns out to
be irreducibly unstable, exclude it and say why rather than loosening the
comparison.

**Never assert that text appears *somewhere* in the tree when a specific place
is what you mean.** `chat-restore` failed roughly one run in two for a reason
worth repeating. `start_model` gated on `SendOrStopButton.description ==
"Send message"`, which is the button's label for the whole startup — its hint
still read "… is still starting." So the flow pressed Send into a closed
readiness gate, the press was dropped, and the draft stayed in the composer.
`assert_tree_text "golden restore marker"` then *found* the prompt — in the
composer — and reported success for a message that was never sent. The run only
failed later, on the reply that never came, which is why it looked like a
flake rather than a broken assertion.

Both halves are now fixed: `start_model` waits on `wait_send_idle`, and
`send_prompt` requires the composer to actually drain. The general rule: an
assertion that a string is present anywhere is satisfied by the input field,
the placeholder, the tooltip and the sidebar. Say *which element*.

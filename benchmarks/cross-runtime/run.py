#!/usr/bin/env python3
"""Cross-runtime benchmark orchestrator.

One runtime × one model at a time on an otherwise idle GPU. Every
runtime is measured through the same OpenAI-compatible streaming
client so transport overhead is identical. Results are appended as
JSONL, one record per request, under results/<run-id>/.

Usage:
  python3 run.py --lane parity --models qwen3.5-4b --runtimes rapid,ollama
  python3 run.py --lane parity            # full matrix
  python3 run.py --list                   # show resolved matrix and exit

Servers are launched from $HOME (a repo cwd shadows installed
packages), stdout/stderr captured to results/<run-id>/logs/.
"""

import argparse
import asyncio
import json
import os
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import yaml

HERE = Path(__file__).resolve().parent
HOME = Path.home()


# ---------------------------------------------------------------- config

def load_config():
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    for k, v in cfg["venvs"].items():
        cfg["venvs"][k] = str(Path(str(v)).expanduser())
    return cfg


# ------------------------------------------------------------- prompts

FILLER = (
    "The measurement of inference latency on consumer hardware is a "
    "study in queuing, caching, and thermal behavior. A token that "
    "arrives quickly from a cold start tells a different story than "
    "one that arrives quickly from a warm cache, and a benchmark that "
    "cannot tell those stories apart will mislead its readers. "
)

_tokenizer_cache = {}


def get_tokenizer(hf_id):
    if hf_id not in _tokenizer_cache:
        from transformers import AutoTokenizer
        _tokenizer_cache[hf_id] = AutoTokenizer.from_pretrained(hf_id)
    return _tokenizer_cache[hf_id]


SATURATE_TASK = (
    "Write an exhaustively detailed technical essay on the history of "
    "operating systems: cover every era, architecture, scheduler design "
    "and file system you can, one by one, in long fully-written prose. "
    "Do not summarize, do not conclude, keep adding sections until you "
    "are cut off. "
)


def build_prompt(hf_id, target_tokens, saturate=False):
    """Deterministic text sized to ~target_tokens with the model's own
    tokenizer (counted on raw text; chat-template overhead is small and
    applies to every runtime alike). saturate=True asks for unbounded
    generation so every runtime hits max_tokens exactly — equal decode
    workload regardless of each runtime's thinking-template defaults."""
    tok = get_tokenizer(hf_id)
    task = SATURATE_TASK if saturate else (
        "Summarize the following in one short paragraph. "
        if target_tokens <= 128 else
        "Read the following document, then answer: what is its main "
        "claim? Answer in two sentences.\n\n"
    )
    if target_tokens <= 128:
        base = task + FILLER
        ids = tok.encode(base)
        while len(ids) < target_tokens:
            base += FILLER
            ids = tok.encode(base)
        return base
    if saturate:
        task = (
            "Read the following document, then " + SATURATE_TASK
            + "Document:\n\n"
        )
    reps = max(1, (target_tokens * 4) // len(FILLER))
    text = FILLER * reps
    ids = tok.encode(text)
    while len(ids) < target_tokens:
        text += FILLER * 8
        ids = tok.encode(text)
    # trim down by whole filler sentences until just under target
    while len(ids) > target_tokens and len(text) > len(FILLER):
        text = text[: -len(FILLER)]
        ids = tok.encode(text)
    return task + text


# ------------------------------------------------------------- servers

def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_http(url, timeout_s, proc=None):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited early rc={proc.returncode}")
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code < 500:
                return time.monotonic() - t0
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"{url} not ready in {timeout_s}s")


class RssSampler(threading.Thread):
    """Peak RSS (KB) of a process tree, sampled every 500 ms."""

    def __init__(self, root_pid):
        super().__init__(daemon=True)
        self.root = root_pid
        self.peak_kb = 0
        self._stop = threading.Event()

    def _tree(self):
        pids = [self.root]
        try:
            out = subprocess.run(
                ["pgrep", "-P", str(self.root)], capture_output=True, text=True
            ).stdout.split()
            pids += [int(p) for p in out]
        except Exception:
            pass
        return pids

    def run(self):
        while not self._stop.is_set():
            total = 0
            for pid in self._tree():
                try:
                    out = subprocess.run(
                        ["ps", "-o", "rss=", "-p", str(pid)],
                        capture_output=True, text=True,
                    ).stdout.strip()
                    if out:
                        total += int(out)
                except Exception:
                    pass
            self.peak_kb = max(self.peak_kb, total)
            self._stop.wait(0.5)

    def stop(self):
        self._stop.set()


class Server:
    """A launched (or attached) inference server."""

    def __init__(self, name, base_url, proc=None, load_s=None):
        self.name = name
        self.base_url = base_url
        self.proc = proc
        self.load_s = load_s
        self.rss = RssSampler(proc.pid if proc else self._find_daemon_pid())
        self.rss.start()

    @staticmethod
    def _find_daemon_pid():
        out = subprocess.run(
            ["pgrep", "-x", "ollama"], capture_output=True, text=True
        ).stdout.split()
        return int(out[0]) if out else os.getpid()

    def stop(self):
        self.rss.stop()
        if self.proc is not None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)


def launch(cfg, runtime, model, lane, log_dir):
    """Start the server for runtime/model; returns Server with cold-load
    seconds (process start -> ready endpoint)."""
    rt = cfg["runtimes"][runtime]
    port = cfg["ports"][runtime]
    if runtime == "ollama":
        return attach_ollama(cfg, rt, port, log_dir)
    if not port_free(port):
        raise RuntimeError(f"port {port} busy — refusing to start {runtime}")
    tmpl = rt["serve"]
    flags = rt["parity_flags"] if lane == "parity" else rt["product_flags"]
    cmd = tmpl.format(
        venv=cfg["venvs"].get(runtime, ""),
        python=cfg["venvs"].get("mlxlm_python", "python3"),
        model=model["hf_mlx"],
        port=port,
    )
    if flags:
        cmd += " " + flags
    log = open(log_dir / f"{runtime}-{model['key']}.log", "ab")
    t0 = time.monotonic()
    proc = subprocess.Popen(
        shlex.split(cmd), cwd=str(HOME), stdout=log, stderr=log,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    base = f"http://127.0.0.1:{port}"
    # generous: big models legitimately take minutes to load
    load_s = wait_http(base + rt["ready_path"], timeout_s=900, proc=proc)
    return Server(runtime, base, proc, load_s)


def attach_ollama(cfg, rt, port, log_dir):
    """Start the ollama daemon with the harness-controlled env, or
    reuse the one a previous model leg already started (restarting per
    model raced the old process for the port: 'address already in
    use'). Server.stop() leaves a reused daemon running; the harness
    caller kills it at end of run."""
    base = f"http://127.0.0.1:{port}"
    if not port_free(port):
        try:
            httpx.get(base + rt["ready_path"], timeout=2.0).raise_for_status()
            return Server("ollama", base, proc=None, load_s=None)
        except httpx.HTTPError:
            subprocess.run(["pkill", "-x", "ollama"], capture_output=True)
    t0 = time.monotonic()
    while not port_free(port):
        if time.monotonic() - t0 > 30:
            raise RuntimeError("port 11434 still bound 30s after pkill")
        time.sleep(0.5)
    env = {**os.environ, **rt.get("daemon_env", {})}
    log = open(log_dir / "ollama-daemon.log", "ab")
    proc = subprocess.Popen(
        ["ollama", "serve"], cwd=str(HOME), stdout=log, stderr=log, env=env
    )
    wait_http(base + rt["ready_path"], timeout_s=60, proc=proc)
    return Server("ollama", base, proc, load_s=None)


# -------------------------------------------------------------- client

def _image_data_uri():
    import base64
    png = (HERE / "vision-probe.png").read_bytes()
    return "data:image/png;base64," + base64.b64encode(png).decode()


async def stream_chat(client, base_url, model_id, prompt, max_tokens,
                      image=False):
    """One streaming chat completion; returns timing + token counts.

    TTFT is measured to the first delta carrying visible content OR
    reasoning (reasoning models start streaming thought first — that
    IS their first token; content-only TTFT is also recorded)."""
    if image:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _image_data_uri()}},
        ]
    else:
        content = prompt
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t_send = time.monotonic()
    t_first = t_first_content = t_last = None
    usage = None
    chunks = 0
    async with client.stream(
        "POST", base_url + "/v1/chat/completions", json=body, timeout=1800
    ) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            now = time.monotonic()
            if obj.get("error"):
                # mid-stream error event on an HTTP 200 — surface the
                # payload instead of reporting "no tokens"
                raise RuntimeError(f"SSE error event: {obj['error']}")
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    chunks += 1
                    t_last = now
                    if t_first is None:
                        t_first = now
                    if d.get("content") and t_first_content is None:
                        t_first_content = now
    if t_first is None:
        raise RuntimeError("stream produced no tokens")
    completion = (usage or {}).get("completion_tokens")
    return {
        "ttft_s": t_first - t_send,
        "ttft_content_s": (t_first_content - t_send) if t_first_content else None,
        "stream_s": t_last - t_first,
        "chunks": chunks,
        "usage": usage,
        "decode_tps": (
            (completion - 1) / (t_last - t_first)
            if completion and completion > 1 and t_last > t_first
            else None
        ),
        "wall_s": t_last - t_send,
    }


async def run_cell(server, model_id, prompt, scenario, out, image=False):
    reps = scenario.get("reps", 3)
    parallel = scenario.get("parallel", 1)
    limits = httpx.Limits(max_connections=parallel + 2)
    async with httpx.AsyncClient(limits=limits) as client:
        # one discarded warmup at this prompt size (kernel compile,
        # tokenizer warm, model page-in). Timed: lazy-loading servers
        # (omlx, ollama) do their model load here, so cold start is
        # load_s + warmup_s regardless of loading strategy.
        t_w = time.monotonic()
        try:
            await stream_chat(client, server.base_url, model_id,
                              "Case 0: " + prompt,
                              min(scenario["max_tokens"], 32), image=image)
            out["warmup_s"] = time.monotonic() - t_w
        except Exception as e:
            out["warmup_error"] = repr(e)
        runs = []
        for rep in range(reps):
            # unique per-request prefix: with product-lane prefix caches
            # on, identical repeated prompts would measure a 100% cache
            # hit instead of the engine
            if parallel == 1:
                runs.append(await stream_chat(
                    client, server.base_url, model_id,
                    f"Case {rep + 1}: " + prompt,
                    scenario["max_tokens"], image=image))
            else:
                t0 = time.monotonic()
                res = await asyncio.gather(*[
                    stream_chat(client, server.base_url, model_id,
                                f"Case {rep + 1}-{i}: " + prompt,
                                scenario["max_tokens"], image=image)
                    for i in range(parallel)
                ], return_exceptions=True)
                ok = [r for r in res if isinstance(r, dict)]
                errs = [repr(r) for r in res if not isinstance(r, dict)]
                wall = time.monotonic() - t0
                total_completion = sum(
                    (r["usage"] or {}).get("completion_tokens") or 0 for r in ok
                )
                # decode-phase aggregate: excludes the shared prefill
                # barrier (all TTFTs ≈ full-batch prefill time on
                # BatchGenerator-based runtimes) so early-EOS workload
                # asymmetry and prefill speed don't pollute the decode
                # comparison (#1861 post-mortem).
                _ttfts = [r["ttft_s"] for r in ok]
                _decode_span = wall - (min(_ttfts) if _ttfts else 0)
                runs.append({
                    "parallel": parallel,
                    "ok": len(ok),
                    "errors": errs,
                    "wall_s": wall,
                    "agg_tps": total_completion / wall if wall > 0 else None,
                    "decode_agg_tps": (
                        total_completion / _decode_span if _decode_span > 0 else None
                    ),
                    # per-request completion tokens so undersaturated()
                    # can flag early-EOS conc cells (they previously
                    # slipped past the † check, which reads run["usage"])
                    "completions": [
                        (r["usage"] or {}).get("completion_tokens") or 0 for r in ok
                    ],
                    "ttft_p50_s": statistics.median(r["ttft_s"] for r in ok) if ok else None,
                    "ttft_max_s": max((r["ttft_s"] for r in ok), default=None),
                })
        out["runs"] = runs
    return out


# ----------------------------------------------------------------- main

def median_of(runs, key):
    vals = [r.get(key) for r in runs if isinstance(r, dict) and r.get(key)]
    return statistics.median(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", choices=["parity", "product"], default="parity")
    ap.add_argument("--models", default="")
    ap.add_argument("--runtimes", default="rapid,omlx,mlxlm,ollama")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    models = [m for m in cfg["models"]
              if not args.models or m["key"] in args.models.split(",")]
    runtimes = args.runtimes.split(",")
    scenarios = {k: v for k, v in cfg["scenarios"].items()
                 if not args.scenarios or k in args.scenarios.split(",")}

    if args.list:
        for rt in runtimes:
            for m in models:
                print(f"{rt:8s} {m['key']:16s} {','.join(scenarios)}")
        return

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S") + f"-{args.lane}"
    out_dir = HERE / "results" / run_id
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "cells.jsonl"

    meta = {
        "run_id": run_id, "lane": args.lane,
        "started_unix": time.time(),
        "versions": collect_versions(cfg),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[bench] run {run_id} -> {results_path}", flush=True)

    for rt in runtimes:
        for m in models:
            cell_prefix = f"{rt}/{m['key']}"
            try:
                server = launch(cfg, rt, m, args.lane, log_dir)
            except Exception as e:
                append(results_path, {"runtime": rt, "model": m["key"],
                                      "fatal": f"launch: {e!r}"})
                print(f"[bench] {cell_prefix} LAUNCH FAILED: {e}", flush=True)
                continue
            if rt == "ollama":
                model_id = m["ollama_tag"]
            elif rt == "omlx":
                # oMLX exposes HF-cache models with `--` separators
                model_id = m["hf_mlx"].replace("/", "--")
            else:
                model_id = m["hf_mlx"]
            try:
                for name, sc in scenarios.items():
                    if sc.get("vision"):
                        prompt = (
                            "Describe this image in exhaustive detail: every "
                            "shape, colour, position and relation, one by "
                            "one; keep adding observations until cut off."
                            if sc.get("saturate")
                            else "Describe this image in two sentences."
                        )
                    else:
                        prompt = build_prompt(m["hf_mlx"], sc["prompt_tokens"],
                                              saturate=sc.get("saturate", False))
                    rec = {"runtime": rt, "model": m["key"], "scenario": name,
                           "lane": args.lane, "load_s": server.load_s,
                           "spec": sc}
                    try:
                        asyncio.run(run_cell(server, model_id, prompt, sc, rec,
                                             image=bool(sc.get("vision"))))
                        rec["median_ttft_s"] = median_of(rec.get("runs", []), "ttft_s")
                        rec["median_decode_tps"] = median_of(rec.get("runs", []), "decode_tps")
                        rec["median_agg_tps"] = median_of(rec.get("runs", []), "agg_tps")
                    except Exception as e:
                        rec["error"] = repr(e)
                    rec["peak_rss_gb"] = round(server.rss.peak_kb / 1048576, 2)
                    append(results_path, rec)
                    print(f"[bench] {cell_prefix} {name}: "
                          f"ttft={rec.get('median_ttft_s')} "
                          f"tps={rec.get('median_decode_tps')} "
                          f"agg={rec.get('median_agg_tps')} "
                          f"err={rec.get('error')}", flush=True)
            finally:
                server.stop()
                time.sleep(10)  # GPU settle before the next contender
    print("[bench] DONE", flush=True)


def append(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def collect_versions(cfg):
    v = {}
    for name, cmd in {
        "rapid": [f"{cfg['venvs']['rapid']}/bin/rapid-mlx", "--version"],
        "omlx": [f"{cfg['venvs']['omlx']}/bin/omlx", "--version"],
        "ollama": ["ollama", "--version"],
    }.items():
        try:
            v[name] = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=30).stdout.strip()
        except Exception as e:
            v[name] = f"unknown ({e!r})"
    try:
        import mlx_lm
        v["mlx_lm"] = mlx_lm.__version__
    except Exception:
        v["mlx_lm"] = subprocess.run(
            [cfg["venvs"]["mlxlm_python"], "-c",
             "import mlx_lm; print(mlx_lm.__version__)"],
            capture_output=True, text=True).stdout.strip()
    v["macos"] = subprocess.run(["sw_vers", "-productVersion"],
                                capture_output=True, text=True).stdout.strip()
    return v


if __name__ == "__main__":
    main()

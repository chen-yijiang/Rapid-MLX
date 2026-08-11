#!/usr/bin/env python3
"""Aggregate cells.jsonl into a markdown comparison report.

Usage: python3 report.py results/parity-full [results/product-key ...]

The last record per (lane, runtime, model, scenario) wins, so patch-up
re-runs appended to the same run-id override earlier fatal records.
"""

import json
import statistics
import sys
from pathlib import Path

RUNTIME_ORDER = ["rapid", "omlx", "mlxlm", "ollama"]
RUNTIME_LABEL = {
    "rapid": "rapid-mlx",
    "omlx": "oMLX",
    "mlxlm": "mlx-lm",
    "ollama": "Ollama",
}


def load(run_dir):
    meta = json.loads((run_dir / "meta.json").read_text())
    cells = {}
    for line in (run_dir / "cells.jsonl").read_text().splitlines():
        rec = json.loads(line)
        lane = rec.get("lane") or meta.get("lane")
        if "scenario" not in rec:      # fatal launch record
            cells[(lane, rec["runtime"], rec["model"], "__launch__")] = rec
            continue
        cells[(lane, rec["runtime"], rec["model"], rec["scenario"])] = rec
    return meta, cells


def spread(rec, key):
    runs = rec.get("runs") or []
    vals = [r.get(key) for r in runs if isinstance(r, dict) and r.get(key)]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    mid = statistics.median(vals)
    if not mid:
        return ""
    return f" ±{(hi - lo) / 2 / mid * 100:.0f}%"


def undersaturated(rec):
    """True when a saturate cell didn't hit its token cap (early EOS)
    — its throughput number then reflects a lighter workload."""
    if not (rec.get("spec") or {}).get("saturate"):
        return False
    cap = (rec.get("spec") or {}).get("max_tokens") or 0
    runs = rec.get("runs") or []
    toks = [(r.get("usage") or {}).get("completion_tokens")
            for r in runs if isinstance(r, dict) and r.get("usage")]
    return bool(toks) and min(toks) < 0.9 * cap


def fmt(rec, scenario):
    if rec is None:
        return "—"
    if rec.get("error") or rec.get("fatal"):
        return "FAIL"
    mark = "†" if undersaturated(rec) else ""
    if scenario.startswith("conc"):
        v = rec.get("median_agg_tps")
        return f"{v:.0f}{spread(rec, 'agg_tps')}{mark}" if v else "—"
    if scenario.startswith("ttft"):
        v = rec.get("median_ttft_s")
        return f"{v:.2f}s{spread(rec, 'ttft_s')}" if v else "—"
    v = rec.get("median_decode_tps")
    return f"{v:.0f}{spread(rec, 'decode_tps')}{mark}" if v else "—"


def emit(meta, cells, out):
    lanes = sorted({k[0] for k in cells})
    models = []
    for k in cells:
        if k[2] not in models:
            models.append(k[2])
    scenarios = []
    for k in cells:
        if k[3] not in scenarios and k[3] != "__launch__":
            scenarios.append(k[3])

    out.append(f"## Run `{meta['run_id']}`\n")
    out.append("Versions: " + ", ".join(
        f"{k} {v}" for k, v in meta.get("versions", {}).items()) + "\n")
    for lane in lanes:
        out.append(f"\n### Lane: {lane}\n")
        for sc in scenarios:
            out.append(f"\n**{sc}**  ("
                       + ("aggregate tok/s" if sc.startswith("conc")
                          else "median TTFT" if sc.startswith("ttft")
                          else "decode tok/s") + ")  († = early EOS, under-saturated)\n")
            hdr = "| model | " + " | ".join(
                RUNTIME_LABEL[r] for r in RUNTIME_ORDER) + " |"
            out.append(hdr)
            out.append("|" + "---|" * (len(RUNTIME_ORDER) + 1))
            for m in models:
                row = [m]
                for rt in RUNTIME_ORDER:
                    rec = cells.get((lane, rt, m, sc))
                    if rec is None and (lane, rt, m, "__launch__") in cells:
                        rec = cells[(lane, rt, m, "__launch__")]
                    row.append(fmt(rec, sc))
                out.append("| " + " | ".join(row) + " |")
        # memory + load tables
        out.append("\n**peak RSS (GB) / cold load (s, server-boot + first-request)**\n")
        out.append("| model | " + " | ".join(
            RUNTIME_LABEL[r] for r in RUNTIME_ORDER) + " |")
        out.append("|" + "---|" * (len(RUNTIME_ORDER) + 1))
        for m in models:
            row = [m]
            for rt in RUNTIME_ORDER:
                recs = [cells.get((lane, rt, m, sc)) for sc in scenarios]
                recs = [r for r in recs if r and not r.get("fatal")]
                if not recs:
                    row.append("—")
                    continue
                rss = max((r.get("peak_rss_gb") or 0) for r in recs)
                load = next((r.get("load_s") for r in recs if r.get("load_s")), None)
                warm = next((r.get("warmup_s") for r in recs if r.get("warmup_s")), None)
                cold = (load or 0) + (warm or 0)
                row.append(f"{rss:.1f} / {cold:.0f}s" if rss else "—")
            out.append("| " + " | ".join(row) + " |")


def main():
    out = ["# Cross-runtime benchmark report\n"]
    for arg in sys.argv[1:]:
        meta, cells = load(Path(arg))
        emit(meta, cells, out)
    print("\n".join(out))


if __name__ == "__main__":
    main()

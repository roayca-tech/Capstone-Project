"""Read runs/*/metrics.json and emit markdown tables with a published-delta column.

Never invent numbers. A missing run is reported as 'not executed'.

Usage:
    python scripts/make_tables.py --runs runs --out TABLES.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Paper targets (Ghosh & Naskar, CVPRW 2026). Frame-level, assumed.
TABLE1 = {
    "wgn_c23": {"acc": 95.32, "auc": 98.90, "label": "WGN FF++ C23"},
    "baseline_c23": {"acc": 93.70, "auc": 98.12, "label": "Backbone FF++ C23"},
    "wgn_c40": {"acc": 80.20, "auc": 89.00, "label": "WGN FF++ C40"},
    "wgn_faceshifter_c23": {"acc": 98.00, "auc": 99.92, "label": "WGN FaceShifter C23"},
}

TABLE2 = {
    "celebdf": 77.62,
    "dfdc": 70.41,
    "wilddeepfake": 68.65,
}

TABLE3 = {
    "Deepfakes": 72.40,
    "Face2Face": 76.47,
    "FaceSwap": 70.64,
    "NeuralTextures": 75.93,
}

TABLE6 = {
    "C0": 64.96,
    "C1": 66.73,
    "C2": 64.71,
    "C3": 64.16,
    "C4": 65.51,
    "C5": 65.76,
    "C6": 66.23,
    "C7": 64.88,
    "C8": 65.49,
    "C9": 65.22,
}


def load_runs(runs_dir: Path) -> dict[str, dict]:
    found = {}
    if not runs_dir.exists():
        return found
    for cfg in runs_dir.glob("*/config.json"):
        tag = cfg.parent.name
        metrics_p = cfg.parent / "metrics.json"
        found[tag] = {
            "config": json.loads(cfg.read_text()),
            "metrics": json.loads(metrics_p.read_text()) if metrics_p.exists() else None,
            "dir": str(cfg.parent),
        }
    return found


def group_by_prefix(runs: dict, prefix: str) -> list[dict]:
    return [v for k, v in sorted(runs.items()) if k == prefix or k.startswith(prefix + "_")]


def test_all(metrics: dict | None) -> dict | None:
    if not metrics:
        return None
    test = metrics.get("test") or {}
    return test.get("all") or None


def mean_std(rows: list[tuple[float, float]]) -> tuple[str, str]:
    if not rows:
        return "not executed", "not executed"
    acc = [a for a, _ in rows]
    auc = [u for _, u in rows]
    if len(rows) == 1:
        return f"{acc[0]:.2f}", f"{auc[0]:.2f}"
    import statistics
    return (
        f"{statistics.mean(acc):.2f} ± {statistics.stdev(acc):.2f}",
        f"{statistics.mean(auc):.2f} ± {statistics.stdev(auc):.2f}",
    )


def fmt_delta(value: float | None, target: float, tol: float) -> str:
    if value is None:
        return "not executed"
    d = value - target
    flag = "  **OUT OF TOL**" if abs(d) > tol else ""
    return f"{d:+.2f}{flag}"


def pick_metrics(runs: dict, keys: list[str]) -> list[dict]:
    out = []
    for k in keys:
        if k in runs:
            out.append(runs[k])
        else:
            # seed-suffixed tags: wgn_c23_s0
            out.extend(group_by_prefix(runs, k))
    # unique by dir
    seen = set()
    uniq = []
    for r in out:
        if r["dir"] in seen:
            continue
        seen.add(r["dir"])
        uniq.append(r)
    return uniq


def table1_md(runs: dict) -> str:
    lines = [
        "### Table 1 — in-domain (frame-level)",
        "",
        "| Run | ACC | AUC | Δ ACC | Δ AUC | source |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key, tgt in TABLE1.items():
        matched = pick_metrics(runs, [key])
        pairs = []
        sources = []
        for r in matched:
            t = test_all(r["metrics"])
            if t and t.get("acc") is not None:
                pairs.append((t["acc"], t["auc"]))
                sources.append(r["dir"])
        acc_s, auc_s = mean_std(pairs)
        if pairs:
            import statistics
            acc_m = statistics.mean([a for a, _ in pairs])
            auc_m = statistics.mean([u for _, u in pairs])
            dacc = fmt_delta(acc_m, tgt["acc"], 1.0)
            dauc = fmt_delta(auc_m, tgt["auc"], 0.5)
        else:
            dacc = dauc = "not executed"
        src = ", ".join(sources) if sources else "—"
        lines.append(
            f"| {tgt['label']} | {acc_s} | {auc_s} | {dacc} | {dauc} | `{src}` |"
        )
    lines.append("")
    lines.append("Metrics are **frame-level**. Video-level aggregation is not reported.")
    return "\n".join(lines)


def table2_md(runs: dict) -> str:
    lines = [
        "### Table 2 — cross-dataset AUC (trained on FF++ C23, frame-level)",
        "",
        "| Dataset | AUC | published | Δ | source |",
        "|---|---:|---:|---:|---|",
    ]
    aliases = {
        "celebdf": ["eval_celebdf", "wgn_c23_celebdf"],
        "dfdc": ["eval_dfdc", "wgn_c23_dfdc"],
        "wilddeepfake": ["eval_wilddeepfake", "wgn_c23_wilddeepfake"],
    }
    for name, target in TABLE2.items():
        matched = pick_metrics(runs, aliases[name] + [name])
        aucs, srcs = [], []
        for r in matched:
            t = test_all(r["metrics"])
            if t and t.get("auc") is not None:
                aucs.append(t["auc"])
                srcs.append(r["dir"])
        if aucs:
            import statistics
            val = statistics.mean(aucs)
            cell = f"{val:.2f}" if len(aucs) == 1 else f"{val:.2f} ± {statistics.stdev(aucs):.2f}"
            delta = fmt_delta(val, target, 3.0)
            src = ", ".join(srcs)
        else:
            cell, delta, src = "not executed", "not executed", "—"
        lines.append(f"| {name} | {cell} | {target:.2f} | {delta} | `{src}` |")
    return "\n".join(lines)


def table3_md(runs: dict) -> str:
    lines = [
        "### Table 3 — cross-manipulation average AUC (frame-level)",
        "",
        "| Train | AVG AUC | published | Δ | source |",
        "|---|---:|---:|---:|---|",
    ]
    for manip, target in TABLE3.items():
        keys = [f"wgn_{manip.lower()}", f"wgn_{manip}"]
        matched = pick_metrics(runs, keys)
        # also match tags that contain the manip name and train-manip in config
        extra = [
            r for r in runs.values()
            if (r["config"].get("args") or {}).get("train_manip") == manip
        ]
        matched = pick_metrics({m["config"]["tag"]: m for m in matched + extra}, [
            m["config"]["tag"] for m in matched + extra
        ])
        avgs, srcs = [], []
        for r in matched:
            test = (r["metrics"] or {}).get("test") or {}
            aucs = [
                test[m]["auc"] for m in TABLE3
                if m in test and test[m].get("auc") is not None
            ]
            if aucs:
                avgs.append(sum(aucs) / len(aucs))
                srcs.append(r["dir"])
        if avgs:
            import statistics
            val = statistics.mean(avgs)
            cell = f"{val:.2f}" if len(avgs) == 1 else f"{val:.2f} ± {statistics.stdev(avgs):.2f}"
            delta = fmt_delta(val, target, 3.0)
            src = ", ".join(srcs)
        else:
            cell, delta, src = "not executed", "not executed", "—"
        lines.append(f"| {manip} | {cell} | {target:.2f} | {delta} | `{src}` |")
    return "\n".join(lines)


def table6_md(runs: dict) -> str:
    lines = [
        "### Table 6 — ablations (AdamW, 10 epochs, Deepfakes-only, AVG)",
        "",
        "| ID | AVG | published | Δ | source |",
        "|---|---:|---:|---:|---|",
    ]
    for cid, target in TABLE6.items():
        matched = [
            r for r in runs.values()
            if (r["config"].get("args") or {}).get("tag", "").upper().startswith(cid)
            or r["config"].get("tag", "").upper().startswith(cid)
        ]
        avgs, srcs = [], []
        for r in matched:
            test = (r["metrics"] or {}).get("test") or {}
            aucs = [v["auc"] for v in test.values() if isinstance(v, dict) and v.get("auc") and v is not test.get("all")]
            # prefer explicit all; else mean of the four manips
            if "all" in test and test["all"].get("auc") is not None:
                avgs.append(test["all"]["auc"])
            elif aucs:
                avgs.append(sum(aucs) / len(aucs))
            else:
                continue
            srcs.append(r["dir"])
        if avgs:
            import statistics
            val = statistics.mean(avgs)
            cell = f"{val:.2f}" if len(avgs) == 1 else f"{val:.2f} ± {statistics.stdev(avgs):.2f}"
            delta = fmt_delta(val, target, 3.0)
            src = ", ".join(srcs)
        else:
            cell, delta, src = "not executed", "not executed", "—"
        lines.append(f"| {cid} | {cell} | {target:.2f} | {delta} | `{src}` |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="TABLES.md")
    args = ap.parse_args()

    runs = load_runs(Path(args.runs))
    parts = [
        "# Reproduction tables",
        "",
        f"Runs discovered: {len(runs)}. "
        "Every number traces to a `runs/<tag>/` directory or is marked **not executed**.",
        "",
        table1_md(runs),
        "",
        table2_md(runs),
        "",
        table3_md(runs),
        "",
        table6_md(runs),
        "",
    ]
    text = "\n".join(parts)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()

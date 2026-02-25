#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from pathlib import Path


ROOT = Path("/Users/eoghancollins/Tarskii/gilgamesh")
BASE_CONFIG = ROOT / "configs" / "physics_6x6_10steps.json"
OUT_DIR = ROOT / "artifacts" / "current_cap_sweep_10ep"
CONFIG_DIR = OUT_DIR / "configs"
LOG_DIR = OUT_DIR / "logs"
CSV_PATH = OUT_DIR / "results.csv"
MD_PATH = OUT_DIR / "summary.md"

BINARY = ROOT / "target" / "debug" / "gilgamesh"
DATA_DIR = ROOT / "data"


CASES = [
    {
        "name": "nocap_reference",
        "enable_current_caps": False,
        "synapse_current_max_ua": 3.0,
        "total_current_max_ua": 50.0,
        "note": "No explicit current caps (reference).",
    },
    {
        "name": "syn_1p0_total_50",
        "enable_current_caps": True,
        "synapse_current_max_ua": 1.0,
        "total_current_max_ua": 50.0,
        "note": "Strong per-synapse restriction, loose total cap.",
    },
    {
        "name": "syn_2p0_total_50",
        "enable_current_caps": True,
        "synapse_current_max_ua": 2.0,
        "total_current_max_ua": 50.0,
        "note": "Moderate per-synapse restriction.",
    },
    {
        "name": "syn_3p0_total_50",
        "enable_current_caps": True,
        "synapse_current_max_ua": 3.0,
        "total_current_max_ua": 50.0,
        "note": "Nominal hardware target (3uA / 50uA).",
    },
    {
        "name": "syn_4p5_total_50",
        "enable_current_caps": True,
        "synapse_current_max_ua": 4.5,
        "total_current_max_ua": 50.0,
        "note": "Higher per-synapse drive with unchanged total cap.",
    },
    {
        "name": "syn_4p5_total_75",
        "enable_current_caps": True,
        "synapse_current_max_ua": 4.5,
        "total_current_max_ua": 75.0,
        "note": "Higher per-synapse drive with proportionally relaxed total cap.",
    },
    {
        "name": "syn_6p0_total_100",
        "enable_current_caps": True,
        "synapse_current_max_ua": 6.0,
        "total_current_max_ua": 100.0,
        "note": "Aggressive drive with scaled total budget.",
    },
    {
        "name": "syn_3p0_total_30",
        "enable_current_caps": True,
        "synapse_current_max_ua": 3.0,
        "total_current_max_ua": 30.0,
        "note": "Nominal per-synapse with tighter total-current budget.",
    },
]


BEST_RE = re.compile(r"Best Test Accuracy:\s*([0-9.]+)%")
FINAL_RE = re.compile(r"Final Test Accuracy:\s*([0-9.]+)%")
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+Loss:\s+([0-9.]+)\s+\|\s+Train Acc:\s+([0-9.]+)%\s+\|\s+Test Acc:\s+([0-9.]+)%"
)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def write_case_config(base: dict, case: dict) -> Path:
    cfg = json.loads(json.dumps(base))
    cfg["training"]["epochs"] = 10
    cfg["training"]["seed"] = 42
    cfg.setdefault("noise", {})["enabled"] = False
    cfg.setdefault("quantization", {})["enabled"] = False

    hw = cfg.setdefault("hardware", {})
    hw["enable_current_caps"] = case["enable_current_caps"]
    hw["synapse_current_max_ua"] = case["synapse_current_max_ua"]
    hw["total_current_max_ua"] = case["total_current_max_ua"]

    out = CONFIG_DIR / f"{case['name']}.json"
    out.write_text(json.dumps(cfg, indent=2))
    return out


def run_case(case: dict, cfg_path: Path) -> dict:
    cmd = [
        str(BINARY),
        "train",
        "--config",
        str(cfg_path),
        "--data-dir",
        str(DATA_DIR),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0

    log_path = LOG_DIR / f"{case['name']}.log"
    log_path.write_text(proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""))

    if proc.returncode != 0:
        raise RuntimeError(f"{case['name']} failed; see {log_path}")

    best = float(BEST_RE.search(proc.stdout).group(1)) if BEST_RE.search(proc.stdout) else float("nan")
    final = (
        float(FINAL_RE.search(proc.stdout).group(1)) if FINAL_RE.search(proc.stdout) else float("nan")
    )

    epoch_rows = EPOCH_RE.findall(proc.stdout)
    last_epoch_test = float(epoch_rows[-1][3]) if epoch_rows else float("nan")

    syn = case["synapse_current_max_ua"]
    total = case["total_current_max_ua"]
    cap_units = total / syn if syn > 0 else float("nan")

    return {
        "case": case["name"],
        "enable_current_caps": case["enable_current_caps"],
        "synapse_current_max_ua": syn,
        "total_current_max_ua": total,
        "total_per_syn_ratio": cap_units,
        "best_test_acc_pct": best,
        "final_test_acc_pct": final,
        "last_epoch_test_acc_pct": last_epoch_test,
        "runtime_s": elapsed,
        "note": case["note"],
        "config_path": str(cfg_path),
        "log_path": str(log_path),
    }


def write_csv(rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown(rows: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# Current-Cap Sweep (10 epochs)")
    lines.append("")
    lines.append("This sweep varies per-synapse current cap and total current budget.")
    lines.append("")
    lines.append("| Case | I_syn_max (uA) | I_total_max (uA) | I_total/I_syn | Best Acc (%) | Final Acc (%) | Runtime (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['synapse_current_max_ua']:.2f} | {r['total_current_max_ua']:.2f} | "
            f"{r['total_per_syn_ratio']:.2f} | {r['best_test_acc_pct']:.2f} | {r['final_test_acc_pct']:.2f} | {r['runtime_s']:.2f} |"
        )
    lines.append("")

    capped = [r for r in rows if r["enable_current_caps"]]
    if capped:
        best = max(capped, key=lambda r: r["best_test_acc_pct"])
        worst = min(capped, key=lambda r: r["best_test_acc_pct"])
        lines.append("## Quick Read")
        lines.append(
            f"- Best capped case: `{best['case']}` at {best['best_test_acc_pct']:.2f}% best test accuracy."
        )
        lines.append(
            f"- Worst capped case: `{worst['case']}` at {worst['best_test_acc_pct']:.2f}% best test accuracy."
        )
        lines.append(
            "- Compare `syn_4p5_total_50` vs `syn_4p5_total_75` to isolate effect of total-current budget at fixed per-synapse cap."
        )
        lines.append(
            "- Compare `syn_3p0_total_50` vs `syn_3p0_total_30` to isolate tighter total-cap effect at nominal per-synapse cap."
        )

    MD_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    base = json.loads(BASE_CONFIG.read_text())

    rows: list[dict] = []
    for idx, case in enumerate(CASES, start=1):
        print(f"[{idx}/{len(CASES)}] {case['name']} ...")
        cfg_path = write_case_config(base, case)
        row = run_case(case, cfg_path)
        rows.append(row)
        print(
            f"    best={row['best_test_acc_pct']:.2f}% final={row['final_test_acc_pct']:.2f}% runtime={row['runtime_s']:.2f}s"
        )

    write_csv(rows)
    write_markdown(rows)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()

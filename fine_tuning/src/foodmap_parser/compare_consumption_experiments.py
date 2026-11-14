"""Run paired strategies in isolated processes and retain every result."""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

from .consumption_experiment import CHALLENGE, LEGACY, ROOT


def compare(output_dir, repeats=3, include_audit=False):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot = output_dir / "source_snapshot"
    shutil.copytree(ROOT / "src" / "foodmap_parser", snapshot,
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(CHALLENGE, output_dir / CHALLENGE.name)
    shutil.copy2(LEGACY, output_dir / LEGACY.name)
    strategies = ("legacy", "scoped", "unscoped")
    runs = {strategy: [] for strategy in strategies}
    expected_inputs = None
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    for repeat in range(repeats):
        offset = repeat % len(strategies)
        order = strategies[offset:] + strategies[:offset]
        for strategy in order:
            destination = output_dir / f"{strategy}_{repeat + 1}"
            command = [sys.executable, "-m", "foodmap_parser.consumption_experiment",
                       "--strategy", strategy, "--output-dir", str(destination)]
            if include_audit:
                command.append("--include-audit")
            subprocess.run(command, env=env, cwd=ROOT, check=True, capture_output=True, text=True)
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            inputs = {k: manifest[k] for k in ("sources", "reference_files", "challenge_sha256", "legacy_sha256")}
            if expected_inputs is not None and inputs != expected_inputs:
                raise ValueError("source or input changed during paired runs")
            expected_inputs = inputs
            metrics = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
            if runs[strategy] and metrics["by_split"] != runs[strategy][0]["by_split"]:
                raise ValueError("non-deterministic predictions; inspect runs before comparing")
            runs[strategy].append(metrics)

    def bounds(values):
        return {"min": min(values), "median": statistics.median(values), "max": max(values)} if values else None

    summary = {"repeats": repeats, "include_audit": include_audit,
               "label_warning": "assistant-authored diagnostics, not independent real-world validation",
               "strategies": {}}
    for strategy, metrics in runs.items():
        summary["strategies"][strategy] = {
            "by_split": metrics[0]["by_split"],
            "peak_rss_mib": bounds([m["peak_rss_mib"] for m in metrics if m["peak_rss_mib"] is not None]),
            "elapsed_seconds": bounds([m["elapsed_seconds"] for m in metrics]),
        }
    (output_dir / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "completed.json").write_text(json.dumps({"completed": True, "runs": repeats * len(strategies)}) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--include-audit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(compare(args.output_dir, args.repeats, args.include_audit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

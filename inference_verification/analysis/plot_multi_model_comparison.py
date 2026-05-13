#!/usr/bin/env python
"""
Cross-row plot: FPR vs Bit Rate, one curve per row, per sigma.

Consumes the post-spec directory layout: each input is a verify-stage dir
(under ``experiments/<row>__<ts>/verifications/sigma=...__<ts>/``) containing
``analysis/fpr_vs_bitrate.pkl`` written by analyze_thresholds.py.

This module exposes a programmatic ``plot_combined()`` for use by
run_analysis.py and a small CLI for manual invocation.
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def load_fpr_bitrate_data(verify_dir: Path) -> dict | None:
    """Load fpr_vs_bitrate.pkl from a verify dir. Returns None if missing."""
    candidates = [
        verify_dir / "analysis" / "fpr_vs_bitrate.pkl",
        verify_dir / "fpr_vs_bitrate.pkl",   # legacy layout (pre-spec)
    ]
    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
    return None


def plot_combined(
    inputs: Iterable[dict],
    sigmas: list[float],
    out_dir: Path,
) -> list[Path]:
    """Render one FPR-vs-Bit-Rate plot per sigma into ``out_dir``.

    Each input dict must have:
      - ``label``: legend label (typically the experiments-table row name)
      - ``verify_dir``: absolute path to the verify-stage dir

    Returns the list of files written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = list(inputs)
    print(f"plot_combined: {len(inputs)} inputs, sigmas={sigmas}")

    # Resolve each input → fpr-bitrate dict.
    resolved: list[tuple[str, dict]] = []
    for entry in inputs:
        label = entry["label"]
        verify_dir = Path(entry["verify_dir"])
        data = load_fpr_bitrate_data(verify_dir)
        if data is None:
            print(f"  ✗ {label}: no fpr_vs_bitrate.pkl in {verify_dir}; skipping")
            continue
        resolved.append((label, data))
        print(f"  ✓ {label}: {verify_dir}")

    if not resolved:
        print("plot_combined: no valid inputs; nothing to plot")
        return []

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "h", "+"]

    written: list[Path] = []
    datestring = datetime.now().strftime("%Y%m%d_%H%M%S")

    for sigma in sigmas:
        plt.figure(figsize=(12, 8))
        plotted = 0
        for idx, (label, data) in enumerate(resolved):
            if sigma not in data:
                print(f"  ✗ {label}: sigma {sigma} not in fpr_vs_bitrate.pkl; skipping")
                continue
            fpr_bitrate_dict = data[sigma]
            fpr_values = sorted(fpr_bitrate_dict.keys())
            bitrate_values = [fpr_bitrate_dict[f] for f in fpr_values]
            plt.plot(
                fpr_values, bitrate_values,
                marker=markers[idx % len(markers)],
                markersize=4, markevery=10,
                color=colors[idx % len(colors)],
                linewidth=2, label=label, alpha=0.85,
            )
            plotted += 1

        if plotted == 0:
            plt.close()
            print(f"  ✗ sigma={sigma}: no rows had data; skipping plot")
            continue

        plt.xlabel("False Positive Rate (%)", fontsize=18)
        plt.ylabel("Extractable Information (%)", fontsize=18)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(f"FPR vs Bit Rate Comparison (σ={sigma})", fontsize=20, fontweight="bold")
        plt.tick_params(axis="both", which="major", labelsize=14)
        plt.legend(fontsize=12, loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        stem = out_dir / f"multi_model_comparison_sigma{sigma}_{datestring}"
        plt.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
        plt.savefig(f"{stem}.pdf", dpi=150, bbox_inches="tight")
        plt.close()
        written.extend([Path(f"{stem}.png"), Path(f"{stem}.pdf")])
        print(f"  ✓ sigma={sigma}: saved {stem}.{{png,pdf}}")

    return written


def main():
    parser = argparse.ArgumentParser(description="Cross-row FPR-vs-bitrate comparison plot")
    parser.add_argument(
        "--inputs-json", type=str, required=True,
        help="Path to a JSON file describing inputs. "
             "Schema: {sigmas: [...], out_dir: '...', "
             "inputs: [{label: '...', verify_dir: '...'}, ...]}",
    )
    args = parser.parse_args()

    with open(args.inputs_json) as f:
        spec = json.load(f)

    plot_combined(
        inputs=spec["inputs"],
        sigmas=spec["sigmas"],
        out_dir=Path(spec["out_dir"]),
    )


if __name__ == "__main__":
    main()

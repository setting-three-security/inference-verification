"""
Cross-row analysis driver.

Reads ``.index.json``, finds rows marked done, and produces combined plots
under ``experiments/_combined/<timestamp>/``. Invoked manually after a
session — never by run_experiments.py.

See AI_DOCS/dev-spec-e2e-experiment-loop-on-runpod.md §7.2.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from inference_verification.analysis.plot_multi_model_comparison import plot_combined
from inference_verification.experiments_table import load_experiments_table
from inference_verification.manifest import read_json, repo_root_from_module

REPO_ROOT = repo_root_from_module()
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
INDEX_FILENAME = ".index.json"
COMBINED_DIRNAME = "_combined"


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def gather_inputs(table, index, *, only_rows: list[str] | None) -> list[dict]:
    """Build ``[{label, verify_dir}]`` for done rows.

    If ``only_rows`` is given, restrict to that subset (still requiring done).
    """
    selected: list[dict] = []
    rows_state = index.get("rows", {})
    for row in table.rows:
        if only_rows is not None and row.name not in only_rows:
            continue
        state = rows_state.get(row.name)
        if not state:
            print(f"[run_analysis] {row.name}: not in .index.json, skipping")
            continue
        if state.get("status") != "done":
            print(f"[run_analysis] {row.name}: status={state.get('status')!r}, skipping")
            continue
        verify = state.get("verify") or {}
        verify_dir_rel = verify.get("dir")
        if not verify_dir_rel:
            print(f"[run_analysis] {row.name}: no verify dir in index, skipping")
            continue
        verify_dir = EXPERIMENTS_DIR / verify_dir_rel
        if not verify_dir.exists():
            print(f"[run_analysis] {row.name}: verify dir {verify_dir} missing, skipping")
            continue
        selected.append({"label": row.name, "verify_dir": str(verify_dir)})
    return selected


def main():
    parser = argparse.ArgumentParser(description="Cross-row analysis driver")
    parser.add_argument("table", type=str, help="Path to experiments YAML")
    parser.add_argument("--combined", action="store_true",
                        help="Run combined cross-row plotting (currently the only mode)")
    parser.add_argument("--rows", type=str, default=None,
                        help="Comma-separated row names to include (default: all done rows)")
    parser.add_argument("--sigmas", type=str, default=None,
                        help="Comma-separated sigma values to plot (default: union of sigmas from selected rows)")
    args = parser.parse_args()

    if not args.combined:
        parser.error("v1 requires --combined (no other modes implemented)")

    table = load_experiments_table(args.table)
    index_path = EXPERIMENTS_DIR / INDEX_FILENAME
    if not index_path.exists():
        print(f"[run_analysis] no index at {index_path} — run experiments first", file=sys.stderr)
        return 1
    index = read_json(index_path)

    only_rows: list[str] | None = None
    if args.rows:
        only_rows = [r.strip() for r in args.rows.split(",") if r.strip()]
        unknown = [n for n in only_rows if n not in table.names()]
        if unknown:
            parser.error(f"unknown row name(s) in --rows: {unknown}")

    inputs = gather_inputs(table, index, only_rows=only_rows)
    if not inputs:
        print("[run_analysis] no done rows to combine", file=sys.stderr)
        return 1

    # Determine sigma set
    if args.sigmas:
        sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    else:
        # Pull from each row's verify cfg and union them (preserve sorted order).
        sigma_set: set[float] = set()
        for row in table.rows:
            if any(inp["label"] == row.name for inp in inputs):
                sigma_set.update(row.verify_cfg.sigmas)
        sigmas = sorted(sigma_set)
        if not sigmas:
            parser.error("could not infer sigmas from selected rows; pass --sigmas")

    out_dir = EXPERIMENTS_DIR / COMBINED_DIRNAME / _now_stamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stdout.log"

    # Tee stdout to log_path while plot_combined runs in-process.
    print(f"[run_analysis] inputs:")
    for inp in inputs:
        print(f"  - {inp['label']} ← {inp['verify_dir']}")
    print(f"[run_analysis] sigmas: {sigmas}")
    print(f"[run_analysis] out_dir: {out_dir}")

    # Persist a small manifest for reproducibility.
    spec = {
        "timestamp_utc": _now_stamp(),
        "table": str(Path(args.table).resolve()),
        "rows_included": [inp["label"] for inp in inputs],
        "sigmas": sigmas,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "inputs.json", "w") as f:
        json.dump({"sigmas": sigmas, "out_dir": str(out_dir), "inputs": inputs}, f, indent=2)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(spec, f, indent=2)

    # Capture stdout to log file.
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s)
                st.flush()
        def flush(self):
            for st in self.streams:
                st.flush()

    with open(log_path, "w") as logf:
        sys.stdout = _Tee(sys.__stdout__, logf)
        try:
            plot_combined(inputs=inputs, sigmas=sigmas, out_dir=out_dir)
        finally:
            sys.stdout = sys.__stdout__

    print(f"[run_analysis] ✓ combined plots written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

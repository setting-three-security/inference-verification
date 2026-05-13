"""
Experiments-table runner.

Reads an experiments YAML table, drives generate → verify → analyze for each
row, and maintains a side-car ``.index.json`` so re-runs are idempotent on
success and auto-retry on failure.

See AI_DOCS/dev-spec-e2e-experiment-loop-on-runpod.md §3, §5.4, §6 for the
exact directory layout and re-run semantics.

Each stage is invoked as a subprocess (cleaner GPU memory boundaries between
vllm and transformers; per-stage stdout capture is trivial). Per-stage configs
are serialized as YAML next to the stage's manifest.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Local imports kept at top so failures (e.g. missing deps) surface early.
from inference_verification.experiments_table import (
    ExperimentsTable,
    Row,
    load_experiments_table,
)
from inference_verification.manifest import (
    MANIFEST_SCHEMA_VERSION,
    atomic_write_json,
    read_json,
    repo_root_from_module,
    utc_timestamp,
    write_analysis_manifest,
)

REPO_ROOT = repo_root_from_module()
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
INDEX_FILENAME = ".index.json"

# Stage names in canonical order.
STAGES = ("generate", "verify", "analyze")

# Subprocess entry points (resolved via -m so import paths work regardless of CWD).
SUBPROC = {
    "generate": ["-m", "inference_verification.generate"],
    "verify": ["-m", "inference_verification.verify"],
    "analyze_thresholds": ["-m", "inference_verification.analysis.analyze_thresholds"],
    "analyze_two_step": ["-m", "inference_verification.analysis.analyze_two_step_classifier"],
}


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def load_or_init_index(path: Path) -> dict:
    if path.exists():
        try:
            return read_json(path)
        except json.JSONDecodeError:
            print(f"[runner] WARNING: {path} is corrupt; reinitializing", file=sys.stderr)
    return {"schema_version": 1, "rows": {}}


def save_index(path: Path, index: dict) -> None:
    atomic_write_json(path, index)


def derived_status(row_state: dict) -> str:
    """Aggregate row status from per-stage statuses (spec §5.4)."""
    stage_states = [row_state.get(s, {}).get("status") for s in STAGES]
    if any(s == "failed" for s in stage_states):
        return "failed"
    if all(s == "done" for s in stage_states):
        return "done"
    if any(s == "running" for s in stage_states):
        return "running"
    return "pending"


def update_stage(
    index: dict,
    row_name: str,
    stage: str,
    *,
    status: str,
    **fields,
) -> None:
    rows = index.setdefault("rows", {})
    row = rows.setdefault(row_name, {})
    stage_state = row.setdefault(stage, {})
    stage_state.update({"status": status, **fields})
    row["status"] = derived_status(row)


# ---------------------------------------------------------------------------
# Subprocess invocation with tee'd stdout
# ---------------------------------------------------------------------------

def run_teed(cmd: list[str], log_path: Path, *, prefix: str) -> int:
    """Run ``cmd``, tee combined stdout/stderr to terminal and ``log_path``.

    Returns the subprocess return code. KeyboardInterrupt is propagated after
    politely terminating the child (giving it 5 s to clean up).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[runner] {prefix} → {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"[runner] {prefix} log → {log_path}")
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(line)
                logf.flush()
            proc.wait()
        except KeyboardInterrupt:
            print(f"\n[runner] {prefix} interrupted; terminating subprocess", file=sys.stderr)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
        return proc.returncode


# ---------------------------------------------------------------------------
# Config-file writers (the YAMLs handed to subprocesses)
# ---------------------------------------------------------------------------

def write_subprocess_config(out_path: Path, cfg) -> None:
    """Serialize a dataclass config to a flat YAML at ``out_path``."""
    if not is_dataclass(cfg):
        raise TypeError(f"expected dataclass, got {type(cfg)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _sigma_name(sigmas: list[float]) -> str:
    return "_".join(f"{s:g}" for s in sigmas)


def gen_dir_name(row_name: str, ts: str) -> str:
    return f"{row_name}__{ts}"


def verify_dir_name(sigmas: list[float], seed: int, ts: str) -> str:
    return f"sigma={_sigma_name(sigmas)}_seed={seed}__{ts}"


def relpath_to_experiments(p: Path) -> str:
    return str(p.resolve().relative_to(EXPERIMENTS_DIR.resolve()))


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_generate(row: Row, *, index: dict, index_path: Path) -> Path:
    """Run the generate subprocess. Returns the gen_dir on success.

    On failure, raises SystemExit with the subprocess return code recorded
    in the index.
    """
    ts = _now_stamp()
    gen_dir = EXPERIMENTS_DIR / gen_dir_name(row.name, ts)
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Subprocess will write to gen_dir; record save_dir on the cfg copy.
    cfg = row.gen_cfg
    cfg.save_dir = str(gen_dir)
    config_path = gen_dir / "config.yaml"
    write_subprocess_config(config_path, cfg)

    update_stage(
        index, row.name, "generate",
        status="running",
        dir=relpath_to_experiments(gen_dir),
        started_at=utc_timestamp(),
    )
    save_index(index_path, index)

    cmd = [
        sys.executable, *SUBPROC["generate"],
        "--config", str(config_path),
        "--row-name", row.name,
    ]
    if row.notes:
        cmd += ["--notes", row.notes]

    started_perf = time.time()
    rc = run_teed(cmd, gen_dir / "stdout.log", prefix=f"{row.name} generate")
    duration = time.time() - started_perf

    if rc == 0:
        update_stage(
            index, row.name, "generate",
            status="done",
            dir=relpath_to_experiments(gen_dir),
            exit_code=0,
            duration_seconds=round(duration, 2),
        )
        save_index(index_path, index)
        return gen_dir
    else:
        update_stage(
            index, row.name, "generate",
            status="failed",
            dir=relpath_to_experiments(gen_dir),
            exit_code=rc,
            duration_seconds=round(duration, 2),
        )
        save_index(index_path, index)
        raise StageFailure(stage="generate", row=row.name, exit_code=rc)


def run_verify(row: Row, gen_dir: Path, *, index: dict, index_path: Path) -> Path:
    ts = _now_stamp()
    verify_dir = gen_dir / "verifications" / verify_dir_name(
        row.verify_cfg.sigmas, row.verify_cfg.seed, ts
    )
    verify_dir.mkdir(parents=True, exist_ok=True)

    cfg = row.verify_cfg
    cfg.save_dir = str(verify_dir)
    config_path = verify_dir / "config.yaml"
    write_subprocess_config(config_path, cfg)

    update_stage(
        index, row.name, "verify",
        status="running",
        dir=relpath_to_experiments(verify_dir),
        started_at=utc_timestamp(),
    )
    save_index(index_path, index)

    cmd = [
        sys.executable, *SUBPROC["verify"],
        "--input", str(gen_dir / "generated_outputs.pkl"),
        "--config", str(config_path),
        "--row-name", row.name,
    ]

    started_perf = time.time()
    rc = run_teed(cmd, verify_dir / "stdout.log", prefix=f"{row.name} verify")
    duration = time.time() - started_perf

    if rc == 0:
        update_stage(
            index, row.name, "verify",
            status="done",
            dir=relpath_to_experiments(verify_dir),
            exit_code=0,
            duration_seconds=round(duration, 2),
        )
        save_index(index_path, index)
        return verify_dir
    else:
        update_stage(
            index, row.name, "verify",
            status="failed",
            dir=relpath_to_experiments(verify_dir),
            exit_code=rc,
            duration_seconds=round(duration, 2),
        )
        save_index(index_path, index)
        raise StageFailure(stage="verify", row=row.name, exit_code=rc)


def run_analyze(row: Row, verify_dir: Path, *, index: dict, index_path: Path) -> Path:
    analysis_dir = verify_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    log_path = analysis_dir / "stdout.log"

    update_stage(
        index, row.name, "analyze",
        status="running",
        dir=relpath_to_experiments(analysis_dir),
        started_at=utc_timestamp(),
    )
    save_index(index_path, index)

    started_at = utc_timestamp()
    started_perf = time.time()
    overall_rc = 0
    scripts_run: list[str] = []
    try:
        for script_key, script_label in (
            ("analyze_thresholds", "analyze_thresholds"),
            ("analyze_two_step", "analyze_two_step_classifier"),
        ):
            cmd = [
                sys.executable, *SUBPROC[script_key],
                "--folder", str(verify_dir),
            ]
            if script_key == "analyze_thresholds":
                cmd += ["--skip-logistic-regression"]
            rc = run_teed(cmd, log_path, prefix=f"{row.name} analyze:{script_label}")
            scripts_run.append(script_label)
            if rc != 0:
                overall_rc = rc
                break
    except KeyboardInterrupt:
        update_stage(
            index, row.name, "analyze",
            status="failed", exit_code=130,
            duration_seconds=round(time.time() - started_perf, 2),
        )
        save_index(index_path, index)
        raise

    duration = time.time() - started_perf

    # List analysis artifacts (cheap + lightweight manifest per spec §5.3).
    artifacts = sorted(
        str(p.relative_to(analysis_dir))
        for p in analysis_dir.rglob("*")
        if p.is_file() and p.name not in {"manifest.json", "stdout.log"}
    )
    write_analysis_manifest(
        out_dir=analysis_dir,
        repo_root=REPO_ROOT,
        row_name=row.name,
        scripts_run=scripts_run,
        started_at=started_at,
        duration_seconds=round(duration, 2),
        exit_code=overall_rc,
        artifacts=artifacts,
    )

    if overall_rc == 0:
        update_stage(
            index, row.name, "analyze",
            status="done",
            dir=relpath_to_experiments(analysis_dir),
            exit_code=0,
            duration_seconds=round(duration, 2),
        )
        save_index(index_path, index)
        return analysis_dir
    else:
        update_stage(
            index, row.name, "analyze",
            status="failed",
            dir=relpath_to_experiments(analysis_dir),
            exit_code=overall_rc,
            duration_seconds=round(duration, 2),
        )
        save_index(index_path, index)
        raise StageFailure(stage="analyze", row=row.name, exit_code=overall_rc)


# ---------------------------------------------------------------------------
# Per-row orchestration
# ---------------------------------------------------------------------------

class StageFailure(RuntimeError):
    def __init__(self, *, stage: str, row: str, exit_code: int):
        self.stage = stage
        self.row = row
        self.exit_code = exit_code
        super().__init__(f"row {row!r} stage {stage!r} failed with exit_code={exit_code}")


def _stage_already_done(row_state: dict, stage: str, *, force: bool) -> bool:
    if force:
        return False
    return (row_state.get(stage) or {}).get("status") == "done"


def run_row(
    row: Row,
    *,
    index: dict,
    index_path: Path,
    skip_stages: set[str],
    force: bool,
) -> bool:
    """Execute one row through the (gen, verify, analyze) pipeline.

    Returns True if the row reached `done` (or was already done and skipped),
    False if any stage failed.
    """
    row_state = index["rows"].get(row.name, {})
    aggregate = derived_status(row_state)

    if aggregate == "done" and not force:
        print(f"[runner] {row.name}: already done, skipping (use --force to re-run)")
        return True

    print(f"\n[runner] === {row.name} ({aggregate}) ===")

    try:
        # Stage 1: generate
        if "generate" in skip_stages:
            print(f"[runner] {row.name}: --skip-stages: skipping generate")
            gen_dir_rel = (row_state.get("generate") or {}).get("dir")
            if not gen_dir_rel:
                print(
                    f"[runner] {row.name}: ERROR — cannot skip generate without "
                    f"a previously-recorded generation dir in .index.json"
                )
                return False
            gen_dir = EXPERIMENTS_DIR / gen_dir_rel
        elif _stage_already_done(row_state, "generate", force=force):
            gen_dir = EXPERIMENTS_DIR / row_state["generate"]["dir"]
            print(f"[runner] {row.name}: generate already done at {gen_dir.name}")
        else:
            gen_dir = run_generate(row, index=index, index_path=index_path)

        # Stage 2: verify
        # If --force or the prior verify dir is gone, re-run.
        verify_state = index["rows"].get(row.name, {}).get("verify") or {}
        if "verify" in skip_stages:
            print(f"[runner] {row.name}: --skip-stages: skipping verify")
            verify_dir_rel = verify_state.get("dir")
            if not verify_dir_rel:
                print(f"[runner] {row.name}: ERROR — cannot skip verify without prior dir")
                return False
            verify_dir = EXPERIMENTS_DIR / verify_dir_rel
        elif (not force) and verify_state.get("status") == "done":
            verify_dir = EXPERIMENTS_DIR / verify_state["dir"]
            print(f"[runner] {row.name}: verify already done at {verify_dir.name}")
        else:
            verify_dir = run_verify(row, gen_dir, index=index, index_path=index_path)

        # Stage 3: analyze
        analyze_state = index["rows"].get(row.name, {}).get("analyze") or {}
        if "analyze" in skip_stages:
            print(f"[runner] {row.name}: --skip-stages: skipping analyze")
        elif (not force) and analyze_state.get("status") == "done":
            print(f"[runner] {row.name}: analyze already done")
        else:
            run_analyze(row, verify_dir, index=index, index_path=index_path)

        print(f"[runner] {row.name}: ✓ done")
        return True

    except StageFailure as e:
        print(f"[runner] {row.name}: ✗ {e.stage} failed (exit_code={e.exit_code})")
        return False


# ---------------------------------------------------------------------------
# Dry-run rendering
# ---------------------------------------------------------------------------

def render_dry_run(table: ExperimentsTable, rows: list[Row]) -> None:
    print(f"# experiments table: {table.source_path}")
    print(f"# rows to run: {[r.name for r in rows]}\n")
    for row in rows:
        print(f"## row: {row.name}")
        if row.notes:
            print(f"   notes: {row.notes}")
        print("   generate:")
        for f in fields(row.gen_cfg):
            print(f"     {f.name}: {getattr(row.gen_cfg, f.name)!r}")
        print("   verify:")
        for f in fields(row.verify_cfg):
            print(f"     {f.name}: {getattr(row.verify_cfg, f.name)!r}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run an experiments table end-to-end")
    parser.add_argument("table", type=str, help="Path to experiments YAML")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this named row")
    parser.add_argument("--skip-stages", type=str, default=None,
                        help=f"Comma-separated stages to skip (any of {','.join(STAGES)})")
    parser.add_argument("--force", action="store_true",
                        help="Re-run rows even if .index.json says they are done")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve the table, print per-row config, exit without running")
    args = parser.parse_args()

    table = load_experiments_table(args.table)

    if args.only:
        rows = [table.by_name(args.only)]
    else:
        rows = list(table.rows)

    if args.skip_stages:
        skip_stages = {s.strip() for s in args.skip_stages.split(",") if s.strip()}
        bad = skip_stages - set(STAGES)
        if bad:
            parser.error(f"unknown stage(s) in --skip-stages: {sorted(bad)}")
    else:
        skip_stages = set()

    if args.dry_run:
        render_dry_run(table, rows)
        return 0

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    index_path = EXPERIMENTS_DIR / INDEX_FILENAME
    index = load_or_init_index(index_path)

    n_succeeded = n_failed = 0
    try:
        for row in rows:
            ok = run_row(
                row,
                index=index,
                index_path=index_path,
                skip_stages=skip_stages,
                force=args.force,
            )
            if ok:
                n_succeeded += 1
            else:
                n_failed += 1
    except KeyboardInterrupt:
        print("\n[runner] interrupted; index left in current state for retry", file=sys.stderr)
        save_index(index_path, index)
        return 130

    save_index(index_path, index)
    print(f"\n[runner] summary: {n_succeeded} succeeded, {n_failed} failed (of {len(rows)})")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

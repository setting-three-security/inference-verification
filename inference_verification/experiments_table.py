"""
Experiments-table loader.

The experiments table is a YAML file with two top-level keys:

    defaults:
      <field>: <value>
      ...

    experiments:
      - name: <unique-row-name>
        <field>: <value>
        ...

Per-row config is computed by merging ``defaults`` with the row's own keys
(row wins). See dev-spec-e2e-experiment-loop-on-runpod.md §4 for full details.

This module owns:
  - parsing & validating the table
  - resolving each row into a Row(name, notes, gen_cfg, verify_cfg) bundle
  - enforcing the "no Cartesian expansion" rule
  - rejecting duplicate row names
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from inference_verification.config import (
    GenerationConfig,
    VerificationConfig,
    resolve_prompts_path,
)

# Field name in the table → config attribute. The user-facing name "model" is
# kept terse to match common ML config conventions; it maps to model_name on
# the dataclasses.
TABLE_TO_CFG_ALIASES = {
    "model": "model_name",
}

# Fields routed to GenerationConfig.
GEN_FIELDS = {
    "model_name", "n_prompts", "max_tokens", "temperature", "top_k", "top_p",
    "seed", "prompts_file", "max_ctx_len", "save_dir", "gpu_memory_utilization",
    "max_model_len",
}

# Fields routed to VerificationConfig.
VERIFY_FIELDS = {
    "model_name", "temperature", "top_k", "top_p", "seed",
    "sigmas", "support_size", "cgs_sigma",
    "classify", "gls_threshold", "logit_rank_threshold", "save_dir",
}

# Row-level metadata that doesn't go into either config.
META_FIELDS = {"name", "notes"}

# Plural keys that imply Cartesian expansion (rejected per spec §4).
CARTESIAN_TRAPS = {"models", "seeds", "temperatures", "top_ks", "top_ps", "n_prompts_list"}


@dataclass
class Row:
    """A single resolved experiment row.

    ``raw`` is the merged dict of defaults + row overrides (with aliases
    resolved). ``gen_cfg`` and ``verify_cfg`` are reconstructed from ``raw``
    and share matching sampling params.
    """

    name: str
    notes: Optional[str]
    raw: dict[str, Any]
    gen_cfg: GenerationConfig
    verify_cfg: VerificationConfig


@dataclass
class ExperimentsTable:
    rows: list[Row]
    source_path: Path

    def by_name(self, name: str) -> Row:
        for row in self.rows:
            if row.name == name:
                return row
        raise KeyError(f"No row named {name!r} in {self.source_path}")

    def names(self) -> list[str]:
        return [r.name for r in self.rows]


def load_experiments_table(yaml_path: str | Path) -> ExperimentsTable:
    """Parse, validate, and resolve a YAML experiments table."""
    yaml_path = Path(yaml_path).resolve()
    with open(yaml_path) as f:
        doc = yaml.safe_load(f)

    if not isinstance(doc, dict):
        raise ValueError(f"{yaml_path}: top-level must be a mapping with 'defaults' and 'experiments'")

    defaults = doc.get("defaults") or {}
    experiments = doc.get("experiments")
    if experiments is None:
        raise ValueError(f"{yaml_path}: missing required key 'experiments'")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f"{yaml_path}: 'experiments' must be a non-empty list")
    if not isinstance(defaults, dict):
        raise ValueError(f"{yaml_path}: 'defaults' must be a mapping")

    _check_no_cartesian(defaults, where=f"{yaml_path} defaults")

    rows: list[Row] = []
    seen: set[str] = set()
    for i, raw_row in enumerate(experiments):
        if not isinstance(raw_row, dict):
            raise ValueError(f"{yaml_path}: experiments[{i}] is not a mapping")
        if "name" not in raw_row:
            raise ValueError(f"{yaml_path}: experiments[{i}] is missing required 'name'")
        name = raw_row["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"{yaml_path}: experiments[{i}].name must be a non-empty string")
        if name in seen:
            raise ValueError(f"{yaml_path}: duplicate experiment name {name!r}")
        seen.add(name)

        _check_no_cartesian(raw_row, where=f"{yaml_path} experiments[{i}] ({name})")

        merged = {**defaults, **raw_row}
        merged = _apply_aliases(merged)

        # Required fields for both stages.
        if "model_name" not in merged:
            raise ValueError(f"{yaml_path}: row {name!r} missing required 'model' (model_name)")

        gen_cfg = _build_gen_cfg(merged, yaml_path=yaml_path, row_name=name)
        verify_cfg = _build_verify_cfg(merged)

        rows.append(Row(
            name=name,
            notes=merged.get("notes"),
            raw=merged,
            gen_cfg=gen_cfg,
            verify_cfg=verify_cfg,
        ))

    return ExperimentsTable(rows=rows, source_path=yaml_path)


def _apply_aliases(d: dict[str, Any]) -> dict[str, Any]:
    """Translate user-facing aliases (``model``) to dataclass field names."""
    out = dict(d)
    for alias, real in TABLE_TO_CFG_ALIASES.items():
        if alias in out:
            if real in out and out[real] != out[alias]:
                raise ValueError(
                    f"Conflicting values for {alias!r}/{real!r}: "
                    f"{out[alias]!r} vs {out[real]!r}"
                )
            out[real] = out.pop(alias)
    return out


def _check_no_cartesian(d: dict, *, where: str) -> None:
    bad = sorted(set(d.keys()) & CARTESIAN_TRAPS)
    if bad:
        raise ValueError(
            f"{where}: plural key(s) {bad} are not allowed — "
            f"the experiments table does not Cartesian-expand. Write multiple rows."
        )


def _build_gen_cfg(merged: dict, *, yaml_path: Path, row_name: str) -> GenerationConfig:
    kwargs = {k: v for k, v in merged.items() if k in GEN_FIELDS}
    cfg = GenerationConfig(**kwargs)

    # Validate prompts file exists (relative paths resolve under data/prompts/).
    prompts_path = resolve_prompts_path(cfg.prompts_file)
    if not prompts_path.exists():
        raise ValueError(
            f"{yaml_path}: row {row_name!r}: prompts_file {cfg.prompts_file!r} "
            f"resolves to {prompts_path} which does not exist"
        )
    return cfg


def _build_verify_cfg(merged: dict) -> VerificationConfig:
    kwargs = {k: v for k, v in merged.items() if k in VERIFY_FIELDS}
    return VerificationConfig(**kwargs)

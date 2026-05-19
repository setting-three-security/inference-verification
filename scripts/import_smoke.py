#!/usr/bin/env python3
"""Import smoke test for the slim CI lane.

Imports each in-scope module under the slim CI deps
(requirements-ci.txt). Exits non-zero on any import failure.

Run from the project root:
    python scripts/import_smoke.py

"""

import importlib
import os
import sys
from pathlib import Path

# Make sure the project root is on sys.path so `webapp.*` resolves as
# a PEP 420 namespace package alongside the installable
# `inference_verification` package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Each entry: (dotted module name, env-var overrides applied before import).
# Env-vars persist into subsequent imports by design (once FAUX_MODE is
# set, it stays set).
MODULES: list[tuple[str, dict[str, str]]] = [
    ("inference_verification.analysis.plot_multi_model_comparison", {}),
    ("webapp.api_server", {"FAUX_MODE": "1"}),
    ("webapp.openrouter", {}),
    ("webapp.ui", {}),
]


def main() -> int:
    failures: list[tuple[str, str]] = []
    for module_name, env_overrides in MODULES:
        for key, value in env_overrides.items():
            os.environ[key] = value
        try:
            importlib.import_module(module_name)
            print(f"OK    {module_name}")
        except Exception as exc:  # noqa: BLE001 — smoke test, want all failures
            print(f"FAIL  {module_name}: {type(exc).__name__}: {exc}")
            failures.append((module_name, f"{type(exc).__name__}: {exc}"))

    print()
    if failures:
        print(f"{len(failures)} module(s) failed to import:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        return 1

    print(f"All {len(MODULES)} module(s) imported cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

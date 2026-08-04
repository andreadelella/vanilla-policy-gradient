"""Manifest and compact report generation for the factorial stage."""

from __future__ import annotations

import json
from pathlib import Path


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def update_root_manifest(root: str | Path) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "plots").mkdir(exist_ok=True)
    payload = {
        "schema_version": 1,
        "scientific_question": "Does temporary categorical support protection remain useful under natural preconditioning?",
        "vpg_modified": False,
        "stages": {
            "fisher_validation": _read(root / "fisher_validation" / "validation_result.json"),
            "exact_two_state": _read(root / "exact_two_state" / "manifest.json"),
            "sampled_two_state_smoke": _read(root / "sampled_two_state" / "smoke" / "manifest.json"),
            "sampled_two_state_pilot": _read(root / "sampled_two_state" / "pilot" / "manifest.json"),
            "sampled_two_state_full": _read(root / "sampled_two_state" / "full" / "manifest.json"),
            "acrobot_pilot": _read(root / "acrobot_pilot" / "pilot_selection.json"),
            "acrobot_confirmatory": _read(root / "acrobot_confirmatory" / "manifest.json"),
            "fisher_diagnostics": _read(root / "fisher_diagnostics" / "manifest.json"),
        },
        "hypotheses_are_test_assertions": False,
        "fisher_rank_is_primary_endpoint": False,
        "historical_archives_overwritten": False,
    }
    (root / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def write_report(root: str | Path) -> Path:
    root = Path(root)
    manifest = update_root_manifest(root)
    validation = manifest["stages"]["fisher_validation"]
    exact = manifest["stages"]["exact_two_state"]
    sampled = manifest["stages"]["sampled_two_state_full"] or manifest["stages"]["sampled_two_state_pilot"]
    pilot = manifest["stages"]["acrobot_pilot"]
    confirmatory = manifest["stages"]["acrobot_confirmatory"]

    def state(value):
        if value is None:
            return "pending"
        if "passed" in value:
            return "complete/pass" if value["passed"] else "complete/fail"
        if "gate_passed" in value:
            return "complete/pass" if value["gate_passed"] else "complete/fail"
        return "complete" if value.get("complete") else "incomplete"

    text = f"""# NPG × temporary log-barrier factorial

## Scope

The four primary methods are Euclidean reward-only GPOMDP, Euclidean GPOMDP
with a temporary sampled-state conditional categorical log barrier, sampled
damped NPG with target-KL-normalized steps, and NPG applied to the complete
temporarily regularized objective. This is not TRPO and the barrier is not a
global neural Fisher log determinant.

## Stage status

- Fisher validation: **{state(validation)}**
- Exact two-state factorial: **{state(exact)}**
- Sampled two-state factorial: **{state(sampled)}**
- Acrobot NPG pilot: **{state(pilot)}**
- Acrobot confirmatory cohort: **{state(confirmatory)}**

No downstream stage is authorized unless the KL-Hessian Fisher validation
passes. Existing `vpg/` code and historical result archives are read-only.

## Questions the completed analysis must answer

1. Does exact population NPG escape the adverse initialization without the barrier?
2. When `s1` is absent from a batch, does sampled NPG remain unable to update its downstream policy?
3. Does natural preconditioning remove the temporary barrier benefit?
4. Are NPG and support protection complementary?
5. Does the barrier suppress useful specialization or create excessive naturalized steps?
6. Are NPG gains explained mainly by controlled KL length rather than a new exploration mechanism?

The endpoint and paired tables, rather than Fisher rank, are primary. Fisher
spectra and gradient alignments are mechanism diagnostics and do not establish
causality.
"""
    destination = root / "report.md"
    destination.write_text(text, encoding="utf-8")
    return destination

"""CLI for the isolated NPG × log-barrier factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .acrobot import (
    run_acrobot_confirmatory,
    run_acrobot_pilot,
    run_fisher_diagnostics,
)
from .exact_two_state import run_exact_factorial
from .fisher_validation import run_validation
from .reporting import update_root_manifest, write_report
from .sampled_two_state import run_sampled_factorial


DEFAULT_ROOT = Path("exploration/results/npg_logbarrier_factorial")


def _require_validation(root: Path) -> None:
    path = root / "fisher_validation" / "validation_result.json"
    if not path.exists():
        raise RuntimeError("run fisher-validation before any factorial experiment")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("passed"):
        raise RuntimeError("policy-Fisher KL-Hessian validation did not pass")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "fisher-validation", "exact-two-state", "sampled-two-state",
            "acrobot-pilot", "acrobot-confirmatory", "fisher-diagnostics",
            "report",
        ),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--preset", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--parallel-workers", type=int, default=1)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root
    if args.stage == "fisher-validation":
        result = run_validation(root / "fisher_validation")
    elif args.stage == "report":
        result = {"report": str(write_report(root))}
    else:
        _require_validation(root)
        if args.stage == "exact-two-state":
            result = run_exact_factorial(root / "exact_two_state")
        elif args.stage == "sampled-two-state":
            result = run_sampled_factorial(root / "sampled_two_state" / args.preset, preset=args.preset)
        elif args.stage == "acrobot-pilot":
            result = run_acrobot_pilot(root / "acrobot_pilot", parallel_workers=args.parallel_workers)
        elif args.stage == "acrobot-confirmatory":
            result = run_acrobot_confirmatory(
                root / "acrobot_confirmatory", root / "acrobot_pilot",
                parallel_workers=args.parallel_workers,
            )
        else:
            result = run_fisher_diagnostics(
                root / "acrobot_confirmatory", root / "fisher_diagnostics"
            )
    update_root_manifest(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


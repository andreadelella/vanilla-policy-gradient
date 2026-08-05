"""Command-line verification of the exact categorical Fisher identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from exploration.categorical_bandit.identity import (
    DEFAULT_TOLERANCES,
    VerificationResult,
    verify_identity_case,
)


SCHEMA_VERSION = 1
DEFAULT_SEED = 23
DEFAULT_RANDOM_SIZES = (10, 100)


def _action_count(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("action counts must be at least 2")
    return parsed


def build_cases(
    seed: int = DEFAULT_SEED,
    random_sizes: Sequence[int] = DEFAULT_RANDOM_SIZES,
) -> list[tuple[str, torch.Tensor]]:
    """Build the deterministic verification suite."""

    cases = [
        ("k2_uniform", torch.zeros(2, dtype=torch.float64)),
        ("k3_asymmetric", torch.tensor([-0.7, 0.2, 1.1], dtype=torch.float64)),
    ]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for action_count in random_sizes:
        if action_count < 2:
            raise ValueError("random action counts must be at least 2")
        cases.append(
            (
                f"k{action_count}_random_seed{seed}",
                torch.randn(action_count, dtype=torch.float64, generator=generator),
            )
        )
    cases.append(
        ("k10_near_boundary", torch.linspace(-8.0, 8.0, 10, dtype=torch.float64))
    )
    return cases


def _format_table(results: Sequence[VerificationResult]) -> str:
    headers = (
        "case",
        "K",
        "min(p)",
        "F error",
        "null",
        "logdet",
        "gradient",
        "Hessian",
        "finite diff",
        "status",
    )
    rows = [
        (
            result.name,
            str(result.action_count),
            f"{result.minimum_probability:.3e}",
            f"{result.score_fisher_error:.3e}",
            f"{result.null_residual:.3e}",
            f"{result.logdet_error:.3e}",
            f"{result.gradient_error:.3e}",
            f"{result.hessian_error:.3e}",
            f"{result.finite_difference_error:.3e}",
            "PASS" if result.passed else "FAIL",
        )
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render(row) -> str:
        return "  ".join(
            str(value).ljust(widths[index])
            for index, value in enumerate(row)
        )

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the exact reduced categorical Fisher log-determinant identity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--random-sizes",
        type=_action_count,
        nargs="+",
        default=list(DEFAULT_RANDOM_SIZES),
        help="Action counts for the seeded random-logit cases.",
    )
    parser.add_argument(
        "--reference-action",
        type=int,
        default=-1,
        help="Logit fixed as the reference in the reduced Fisher chart.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for machine-readable verification results.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = [
        verify_identity_case(name, logits, reference_action=args.reference_action)
        for name, logits in build_cases(args.seed, args.random_sizes)
    ]
    all_passed = all(result.passed for result in results)

    print(_format_table(results))
    print()
    print("ALL CHECKS PASSED" if all_passed else "ONE OR MORE CHECKS FAILED")

    if args.json_output is not None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "dtype": "float64",
            "seed": args.seed,
            "reference_action": args.reference_action,
            "random_sizes": list(args.random_sizes),
            "torch_version": torch.__version__,
            "tolerances": dict(DEFAULT_TOLERANCES),
            "cases": [result.to_dict() for result in results],
            "all_passed": all_passed,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Saved JSON: {args.json_output}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


"""CLI for the staged neural discrete log-barrier experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ablation import calibrate_episode_regularizers, run_gpomdp_regularizer_ablation
from .reliability import (
    run_failed_seed_diagnostics,
    run_pilot_divergence_analysis,
    run_reliability_confirmation,
)
from .reliability_extension import run_extension_method, summarize_extension
from .baseline import (
    run_gpomdp_confirmation,
    run_learning_rate_continuation,
    run_learning_rate_screen,
)
from .experiment import (
    ROOT,
    acrobot_confirmatory,
    acrobot_pilot,
    analyze_acrobot,
    cartpole_smoke,
)
from .reporting import finalize_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "cartpole-smoke",
            "acrobot-lr-screen",
            "acrobot-lr-continuation",
            "acrobot-gpomdp-confirmation",
            "acrobot-baseline",
            "acrobot-regularizer-calibration",
            "acrobot-gpomdp-ablation",
            "acrobot-pilot-divergence",
            "acrobot-reliability-confirmation",
            "acrobot-failure-diagnostics",
            "acrobot-reliability-extension",
            "acrobot-reliability-extension-summary",
            "acrobot-pilot",
            "acrobot-confirmatory",
            "fisher",
            "report",
            "all",
        ),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument(
        "--method",
        choices=("reward_only", "logbarrier_handoff_h25"),
        help="method to run for the reliability-extension stage",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    root = arguments.output_root
    root.mkdir(parents=True, exist_ok=True)
    if arguments.stage == "cartpole-smoke":
        result = cartpole_smoke(root)
    elif arguments.stage == "acrobot-lr-screen":
        result = run_learning_rate_screen(
            root, parallel_workers=arguments.parallel_workers
        )
    elif arguments.stage == "acrobot-lr-continuation":
        result = run_learning_rate_continuation(
            root, parallel_workers=arguments.parallel_workers
        )
    elif arguments.stage == "acrobot-gpomdp-confirmation":
        result = run_gpomdp_confirmation(
            root, parallel_workers=arguments.parallel_workers
        )
    elif arguments.stage == "acrobot-baseline":
        run_learning_rate_screen(root, parallel_workers=arguments.parallel_workers)
        run_learning_rate_continuation(root, parallel_workers=arguments.parallel_workers)
        result = run_gpomdp_confirmation(
            root, parallel_workers=arguments.parallel_workers
        )
    elif arguments.stage == "acrobot-regularizer-calibration":
        result = calibrate_episode_regularizers(root)
    elif arguments.stage == "acrobot-gpomdp-ablation":
        result = run_gpomdp_regularizer_ablation(
            root, parallel_workers=arguments.parallel_workers
        )
    elif arguments.stage == "acrobot-pilot-divergence":
        result = run_pilot_divergence_analysis(root)
    elif arguments.stage == "acrobot-reliability-confirmation":
        result = run_reliability_confirmation(
            root, parallel_workers=arguments.parallel_workers
        )
    elif arguments.stage == "acrobot-failure-diagnostics":
        result = run_failed_seed_diagnostics(
            root, parallel_workers=arguments.parallel_workers
        )
    elif arguments.stage == "acrobot-reliability-extension":
        if arguments.method is None:
            raise SystemExit("--method is required for acrobot-reliability-extension")
        result = run_extension_method(
            root,
            method=arguments.method,
            parallel_workers=arguments.parallel_workers,
        )
    elif arguments.stage == "acrobot-reliability-extension-summary":
        result = summarize_extension(root)
    elif arguments.stage == "acrobot-pilot":
        result = acrobot_pilot(root)
    elif arguments.stage == "acrobot-confirmatory":
        result = acrobot_confirmatory(root)
    elif arguments.stage == "fisher":
        result = analyze_acrobot(root)
    elif arguments.stage == "report":
        result = finalize_report(root)
    else:
        cartpole_smoke(root)
        acrobot_pilot(root)
        acrobot_confirmatory(root)
        analyze_acrobot(root)
        result = finalize_report(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

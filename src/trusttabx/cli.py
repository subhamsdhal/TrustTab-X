"""Run a deterministic, offline smoke experiment."""

import argparse
import json

from .experiment import ExperimentConfig, run_experiment, save_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="TrustTab-X baseline experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=1400)
    parser.add_argument("--selection-size", type=int, default=350)
    parser.add_argument("--calibration-size", type=int, default=350)
    parser.add_argument("--test-size", type=int, default=600)
    parser.add_argument("--shift-strength", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--output", default="outputs/smoke_baseline.json")
    args = parser.parse_args()
    cfg = ExperimentConfig(
        seed=args.seed,
        n_train=args.train_size,
        n_selection=args.selection_size,
        n_calibration=args.calibration_size,
        n_test=args.test_size,
        shift_strength=args.shift_strength,
        alpha=args.alpha,
    )
    results = run_experiment(cfg)
    path = save_experiment(results, args.output)
    print(json.dumps({
        "output": str(path),
        "status": results["status"],
        "selected_model": results["selected_model"],
        "iid_coverage": results["iid_test"]["prediction"]["marginal_coverage"],
        "shifted_coverage": results["shifted_test"]["prediction"]["marginal_coverage"],
        "shift_detected": results["shifted_test"]["shift_monitor"]["shifted"],
    }, indent=2))


if __name__ == "__main__":
    main()

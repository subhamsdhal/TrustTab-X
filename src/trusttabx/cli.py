"""Run a deterministic, offline smoke experiment."""

import argparse
import json
import sys

from .experiment import ExperimentConfig, run_experiment, save_experiment


def benchmark_main(argv: list[str]) -> None:
    from .benchmark import TASKS, run_openml_task, save_benchmark

    parser = argparse.ArgumentParser(description="Run pinned OpenML-CC18 binary pilot tasks")
    parser.add_argument("--task-id", type=int, action="append", choices=sorted(TASKS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--cache-dir", default="data/raw/openml")
    parser.add_argument("--advanced-experts", action="store_true")
    parser.add_argument("--shift-lab", action="store_true")
    parser.add_argument("--output", default="outputs/openml_m1_pilot.json")
    args = parser.parse_args(argv)
    results = []
    for task_id in args.task_id or list(TASKS):
        print(f"Running OpenML task {task_id}", flush=True)
        result = run_openml_task(
            task_id, seed=args.seed, alpha=args.alpha, max_rows=args.max_rows,
            cache_directory=args.cache_dir, advanced=args.advanced_experts,
            shift_lab=args.shift_lab,
        )
        results.append(result)
        save_benchmark(results, args.output)
        print(json.dumps({
            "task_id": task_id,
            "selected_model": result["result"]["selected_model"],
            "test_brier": result["result"]["test_metrics_selected_model_only"]["brier"],
        }), flush=True)
    print(json.dumps({"output": args.output, "completed_tasks": len(results)}))


def main() -> None:
    if sys.argv[1:2] == ["benchmark"]:
        benchmark_main(sys.argv[2:])
        return
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

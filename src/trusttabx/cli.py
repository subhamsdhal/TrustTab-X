"""Run a deterministic, offline smoke experiment."""

import argparse
import json
import sys
from pathlib import Path

from .experiment import ExperimentConfig, run_experiment, save_experiment


def benchmark_main(argv: list[str]) -> None:
    from .benchmark import TASKS, run_openml_task, save_benchmark

    parser = argparse.ArgumentParser(description="Run pinned OpenML-CC18 binary pilot tasks")
    parser.add_argument("--task-id", type=int, action="append", choices=sorted(TASKS))
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--fold", type=int, action="append")
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--cache-dir", default="data/raw/openml")
    parser.add_argument("--advanced-experts", action="store_true")
    parser.add_argument("--shift-lab", action="store_true")
    parser.add_argument("--output", default="outputs/openml_m1_pilot.json")
    args = parser.parse_args(argv)
    results = []
    if args.all_folds and args.fold:
        parser.error("Choose --all-folds or explicit --fold values")
    for task_id in args.task_id or list(TASKS):
        if args.all_folds:
            import openml

            openml.config.cache_directory = Path(args.cache_dir).resolve()
            folds = range(openml.tasks.get_task(task_id).get_split_dimensions()[1])
        else:
            folds = args.fold or [0]
        for fold in folds:
            for seed in args.seed or [42]:
                print(f"Running OpenML task {task_id}, fold {fold}, seed {seed}", flush=True)
                result = run_openml_task(
                    task_id, seed=seed, fold=fold, alpha=args.alpha, max_rows=args.max_rows,
                    cache_directory=args.cache_dir, advanced=args.advanced_experts,
                    shift_lab=args.shift_lab,
                )
                results.append(result)
                save_benchmark(results, args.output)
                print(json.dumps({
                    "task_id": task_id,
                    "fold": fold,
                    "seed": seed,
                    "selected_model": result["result"]["selected_model"],
                    "test_brier": result["result"]["test_metrics_selected_model_only"]["brier"],
                }), flush=True)
    print(json.dumps({"output": args.output, "completed_runs": len(results)}))


def main() -> None:
    if sys.argv[1:2] == ["router-prototype"]:
        from .router_prototype import run_router_prototype, save_router

        parser = argparse.ArgumentParser(description="Run locked exploratory reliability router")
        parser.add_argument("--seed", type=int, default=264738)
        parser.add_argument("--output", default="outputs/router_prototype_3families.json")
        args = parser.parse_args(sys.argv[2:])
        result = run_router_prototype(args.seed)
        save_router(result, args.output)
        print(json.dumps({"output": args.output, "families": list(result["results"])}, indent=2))
        return
    if sys.argv[1:2] == ["domain-crossfit"]:
        from .crossfit_runs import run_source_crossfit, save_source_crossfit

        parser = argparse.ArgumentParser(description="Create source-only domain-held-out predictions")
        parser.add_argument("--dataset", required=True, choices=["acs", "bank", "diabetes"])
        parser.add_argument("--seed", type=int, default=264738)
        parser.add_argument("--output")
        parser.add_argument("--prediction-output")
        args = parser.parse_args(sys.argv[2:])
        output = args.output or f"outputs/{args.dataset}_source_crossfit.json"
        predictions_path = args.prediction_output or (
            f"data/processed/{args.dataset}_source_crossfit_predictions.npz"
        )
        predictions, summary = run_source_crossfit(args.dataset, seed=args.seed)
        save_source_crossfit(predictions, summary, output, predictions_path)
        print(json.dumps({"output": output, "fold_count": summary["fold_count"],
                          "trained_model_count": summary["trained_model_count"]}))
        return
    if sys.argv[1:2] == ["bank-benchmark"]:
        from .bank_temporal import run_bank_benchmark, save_bank

        parser = argparse.ArgumentParser(description="Run pre-call chronological bank benchmark")
        parser.add_argument("--archive", default="data/raw/bank/bank-additional.zip")
        parser.add_argument("--seed", type=int, default=264738)
        parser.add_argument("--output", default="outputs/bank_temporal_2008_2010.json")
        args = parser.parse_args(sys.argv[2:])
        result = run_bank_benchmark(args.archive, seed=args.seed)
        save_bank(result, args.output)
        print(json.dumps({"output": args.output, "selected": {
            name: value["selected_model"] for name, value in result["results"].items()
        }}, indent=2))
        return
    if sys.argv[1:2] == ["acs-benchmark"]:
        from .acs_domain import run_acs_benchmark, save_acs

        parser = argparse.ArgumentParser(description="Run sampled 2018 ACSIncome domain pilot")
        parser.add_argument("--cache-dir", default="data/raw/acs2018")
        parser.add_argument("--seed", type=int, default=264738)
        parser.add_argument("--max-fit-rows", type=int, default=12000)
        parser.add_argument("--source-per-state", type=int, default=6000)
        parser.add_argument("--target-per-state", type=int, default=3000)
        parser.add_argument("--output", default="outputs/acs_income_2018_geographic.json")
        args = parser.parse_args(sys.argv[2:])
        result = run_acs_benchmark(
            args.cache_dir, seed=args.seed, max_fit_rows=args.max_fit_rows,
            source_per_state=args.source_per_state, target_per_state=args.target_per_state,
        )
        save_acs(result, args.output)
        print(json.dumps({"output": args.output, "split_sizes": result["split_sizes"],
                          "selected": {name: run["selected_model"]
                                       for name, run in result["results"].items()}}, indent=2))
        return
    if sys.argv[1:2] == ["natural-benchmark"]:
        from .natural_domain import run_natural_benchmark, save_natural

        parser = argparse.ArgumentParser(description="Run patient-disjoint natural-domain benchmark")
        parser.add_argument("--archive", default="data/raw/tableshift/dataset_diabetes.zip")
        parser.add_argument("--seed", type=int, default=264738)
        parser.add_argument("--max-fit-rows", type=int, default=12000)
        parser.add_argument("--output", default="outputs/tableshift_diabetes_pilot.json")
        args = parser.parse_args(sys.argv[2:])
        result = run_natural_benchmark(args.archive, seed=args.seed, max_fit_rows=args.max_fit_rows)
        save_natural(result, args.output)
        print(json.dumps({"output": args.output, "results": result["results"]}, indent=2))
        return
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

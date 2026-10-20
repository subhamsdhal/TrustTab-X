"""First experiment: reproducible classical baseline and reliability reporting.

This is NOT yet the shift-aware adaptive router or a TabPFN comparison.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import brier_score_loss

from .conformal import SplitConformalClassifier
from .datasets import generate_synthetic_bundle
from .evaluation import evaluate_binary
from .models import make_baseline_models
from .shift import FeatureShiftMonitor


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    n_train: int = 1400
    n_selection: int = 350
    n_calibration: int = 350
    n_test: int = 600
    shift_strength: float = 1.5
    alpha: float = 0.1


def run_experiment(config: ExperimentConfig) -> dict:
    bundle = generate_synthetic_bundle(
        seed=config.seed,
        n_train=config.n_train,
        n_selection=config.n_selection,
        n_calibration=config.n_calibration,
        n_test=config.n_test,
        shift_strength=config.shift_strength,
    )
    models = make_baseline_models(config.seed)
    selection_scores = {}
    for name, model in models.items():
        model.fit(bundle.train.X, bundle.train.y)
        probabilities = model.predict_proba(bundle.selection.X)[:, 1]
        selection_scores[name] = float(brier_score_loss(bundle.selection.y, probabilities))

    # Independent selection labels select one expert. Held-out calibration
    # labels are then used exclusively to calibrate that fixed expert.
    winner = min(selection_scores, key=selection_scores.get)
    chosen = models[winner]
    calibration_probabilities = chosen.predict_proba(bundle.calibration.X)
    calibrator = SplitConformalClassifier.fit(
        calibration_probabilities, bundle.calibration.y, alpha=config.alpha
    )
    monitor = FeatureShiftMonitor(bundle.train.X)

    def evaluate(split):
        probabilities = chosen.predict_proba(split.X)
        scores = evaluate_binary(split.y, probabilities, calibrator)
        report = monitor.evaluate(split.X)
        return {"prediction": scores, "shift_monitor": asdict(report)}

    return {
        "status": "classical_baseline_only",
        "dataset": "synthetic_covariate_shift_smoke_test",
        "config": asdict(config),
        "model_selection_metric": "brier_on_independent_selection_split",
        "model_selection_scores": selection_scores,
        "selected_model": winner,
        "conformal": {
            "alpha": config.alpha,
            "calibration_size": config.n_calibration,
            "score_threshold": calibrator.threshold,
            "guarantee": "iid/exchangeable marginal only; not promised under shift",
        },
        "iid_test": evaluate(bundle.iid_test),
        "shifted_test": evaluate(bundle.shifted_test),
        "limitations": [
            "Synthetic test results are not research conclusions or industry performance claims.",
            "This starter has fixed validation-based expert selection, not dynamic routing.",
            "Conformal marginal coverage is not guaranteed under general distribution shift.",
            "Feature-wise KS monitoring does not distinguish shift mechanisms or certify safety.",
            "No tabular foundation model is included in the first baseline run.",
        ],
    }


def save_experiment(result: dict, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path

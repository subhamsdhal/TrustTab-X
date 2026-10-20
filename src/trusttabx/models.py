"""Classical models; foundation-model adapters arrive in a later phase."""

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_baseline_models(seed: int = 42) -> dict:
    """Comparable reproducible binary-classification model baselines."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=90, learning_rate=0.07, max_leaf_nodes=15, random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=4, n_jobs=1, random_state=seed
        ),
    }

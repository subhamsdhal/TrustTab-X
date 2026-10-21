"""Create a non-sensitive synthetic ACS-shaped bundle for container tests only."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from trusttabx.benchmark import make_preprocessor
from trusttabx.m5_artifact import FEATURES, MODEL_VERSION, SCHEMA_VERSION, digest_file


def create_fixture(output: Path) -> str:
    output.mkdir(parents=True, exist_ok=True)
    record = {"AGEP": 39, "WKHP": 40, "COW": "1.0", "SCHL": "21.0",
              "MAR": "1", "OCCP": "2310.0", "POBP": "42", "RELP": "0",
              "SEX": "1", "RAC1P": "1"}
    ages = [22, 28, 35, 43, 55, 62]
    work = [20, 30, 40, 42, 50, 44]
    frame = pd.DataFrame([{**record, "AGEP": age, "WKHP": hours}
                          for age, hours in zip(ages, work)], columns=FEATURES)
    model = Pipeline([
        ("preprocess", make_preprocessor(frame, [name not in ("AGEP", "WKHP")
                                                for name in FEATURES])),
        ("model", LogisticRegression(random_state=17)),
    ])
    model.fit(frame, [0, 0, 0, 1, 1, 1])
    model_file = output / "pipeline.joblib"
    joblib.dump(model, model_file)
    manifest = {
        "artifact_format": 1, "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION, "task": "synthetic ACS-shaped container fixture",
        "selected_expert": "logistic_regression", "features": list(FEATURES),
        "feature_types": {name: ("integer" if name in ("AGEP", "WKHP")
                                 else "category_code_string") for name in FEATURES},
        "target_mapping": {"0": "synthetic low", "1": "synthetic high"},
        "research_decision": "M4.2 C: fixture only; no research result",
        "reference": {"numeric": {"AGEP": {"mean": 40, "sd": 10},
                                  "WKHP": {"mean": 40, "sd": 10}},
                      "categorical": {name: {record[name]: 1.0} for name in FEATURES[2:]},
                      "probability_mean": 0.5},
        "sklearn_version": sklearn.__version__, "pipeline_file": model_file.name,
        "pipeline_sha256": digest_file(model_file),
    }
    manifest_file = output / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return digest_file(manifest_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("work/m5_fixture_bundle"))
    arguments = parser.parse_args()
    print(create_fixture(arguments.output))

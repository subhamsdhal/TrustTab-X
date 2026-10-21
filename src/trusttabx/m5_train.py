"""Offline ACS fixed-expert trainer."""

import argparse
import json
from pathlib import Path

from .m5_artifact import digest_file, train_acs_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/m5_acs_model")
    parser.add_argument("--mlflow", action="store_true", help="Log to local SQLite tracker")
    args = parser.parse_args()
    manifest = train_acs_bundle(args.output, track_mlflow=args.mlflow)
    print(json.dumps({"model_version": manifest["model_version"],
                      "selected_expert": manifest["selected_expert"],
                      "approved_manifest_sha256": digest_file(Path(args.output) / "manifest.json"),
                      "pipeline_sha256": manifest["pipeline_sha256"],
                      "calibration_brier": manifest["calibration_metrics"]["brier"]}))


if __name__ == "__main__":
    main()

"""Static Compose checks, then optional loopback-only synthetic container smoke."""

import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml
from m5_fixture_bundle import create_fixture

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000"
PROJECT = "trusttabx-m51-" + secrets.token_hex(4)
RECORD = {"AGEP": 39, "WKHP": 40, "COW": "1.0", "SCHL": "21.0", "MAR": "1",
          "OCCP": "2310.0", "POBP": "42", "RELP": "0", "SEX": "1", "RAC1P": "1"}


def static_checks() -> dict:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    service = compose["services"]["trusttabx"]
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "127.0.0.1:8000:8000" in service["ports"]
    assert service["read_only"] is True and service["mem_limit"] and service["cpus"]
    assert any(str(volume).endswith(":ro") for volume in service["volumes"])
    assert "USER appuser" in dockerfile and "HEALTHCHECK" in dockerfile
    assert "[m5-service]" in dockerfile
    for name in ("TRUSTTABX_APPROVED_MANIFEST_SHA256", "TRUSTTABX_API_KEY",
                 "TRUSTTABX_ADMIN_KEY"):
        assert name in service["environment"] and ":?" in service["environment"][name]
    return {"static_checks": "passed", "docker_available": bool(shutil.which("docker"))}


def run(*args: str, env: dict[str, str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "compose", "-p", PROJECT, *args],
                          cwd=ROOT, env=env, capture_output=True, text=True, check=check)


def http(path: str, key: str | None = None, payload: dict | None = None) -> tuple[int, dict | str]:
    headers = {}
    if key:
        headers["X-TrustTabX-Key"] = key
    data = None if payload is None else json.dumps(payload).encode()
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(BASE + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read()
            if path == "/" or path.startswith("/assets/"):
                return response.status, body.decode()
            return response.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def ready() -> float:
    start = time.monotonic()
    for _ in range(150):
        try:
            if http("/health")[0] == 200:
                return time.monotonic() - start
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise RuntimeError("Container did not become ready within 30 seconds")


def runtime_checks() -> dict:
    if not shutil.which("docker"):
        raise RuntimeError("Docker executable unavailable; runtime checks cannot run")
    fixture = ROOT / "work" / "m5_fixture_bundle"
    digest = create_fixture(fixture)
    operator = secrets.token_urlsafe(48)
    admin = secrets.token_urlsafe(48)
    env = os.environ.copy()
    env.update({"TRUSTTABX_MODEL_DIR_HOST": "./work/m5_fixture_bundle",
                "TRUSTTABX_APPROVED_MANIFEST_SHA256": digest,
                "TRUSTTABX_API_KEY": operator, "TRUSTTABX_ADMIN_KEY": admin})

    def diagnostic(label: str, output: str) -> None:
        for private in (operator, admin, "sensitive-feature-marker-should-not-leak"):
            output = output.replace(private, "[redacted]")
        print(json.dumps({"diagnostic": label, "tail": output.splitlines()[-25:]}),
              file=sys.stderr)

    run("config", "--quiet", env=env)
    try:
        build = run("build", env=env)
    except subprocess.CalledProcessError as exc:
        diagnostic("docker_build", (exc.stdout or "") + (exc.stderr or ""))
        raise
    start = time.monotonic()
    try:
        run("up", "-d", env=env)
        startup_seconds = ready()
        statuses = {}

        def expect(name: str, status: int, path: str, key: str | None = None,
                   payload: dict | None = None) -> dict | str:
            actual, body = http(path, key, payload)
            statuses[name] = actual
            assert actual == status, (name, actual, status)
            return body

        expect("health", 200, "/health")
        expect("model_anonymous", 401, "/model")
        model = expect("model_operator", 200, "/model", operator)
        assert model["selected_expert"] == "logistic_regression"
        expect("model_wrong_key", 401, "/model", "invalid-key")
        payload = {"schema_version": "acs-income-2018-v1", "record": RECORD}
        expect("predict", 200, "/predict", operator, payload)
        expect("predict_anonymous", 401, "/predict", payload=payload)
        batch = {"schema_version": payload["schema_version"], "records": [RECORD, RECORD]}
        predictions = expect("predict_batch", 200, "/predict-batch", operator, batch)
        assert len(predictions["predictions"]) == 2
        expect("batch_oversized", 422, "/predict-batch", operator,
               {**batch, "records": [RECORD] * 101})
        marker = "sensitive-feature-marker-should-not-leak"
        error = expect("malformed_input", 422, "/predict", operator,
                       {**payload, "record": {**RECORD, "COW": marker}})
        assert marker not in json.dumps(error)
        drift = {"schema_version": payload["schema_version"],
                 "records": [{**RECORD, "AGEP": 80}]}
        drift_result = expect("drift", 200, "/monitor/drift", operator, drift)
        assert drift_result["numeric"]["AGEP"]["alert"]
        labels = {"labels": [0, 1], "probabilities": [0.2, 0.8]}
        expect("delayed_operator_denied", 403, "/monitor/delayed-labels", operator, labels)
        delayed = expect("delayed_admin", 200, "/monitor/delayed-labels", admin, labels)
        assert abs(delayed["brier"] - 0.04) < 1e-12
        expect("delayed_invalid", 422, "/monitor/delayed-labels", admin,
               {"labels": [0, 2], "probabilities": [0.2, 0.8]})
        expect("metrics", 200, "/metrics", operator)
        assert "TrustTab-X" in expect("dashboard", 200, "/")
        expect("dashboard_js", 200, "/assets/app.js")
        expect("dashboard_css", 200, "/assets/app.css")
        uid = run("exec", "-T", "trusttabx", "id", "-u", env=env).stdout.strip()
        assert uid != "0"
        assert run("exec", "-T", "trusttabx", "touch", "/app/not-writable",
                   env=env, check=False).returncode != 0
        container = run("ps", "-q", "trusttabx", env=env).stdout.strip()
        details = json.loads(subprocess.check_output(["docker", "inspect", container], text=True))[0]
        assert details["HostConfig"]["Memory"] > 0 and details["HostConfig"]["NanoCpus"] > 0
        image_size = json.loads(subprocess.check_output(
            ["docker", "image", "inspect", details["Image"]], text=True))[0]["Size"]
        stats = json.loads(subprocess.check_output([
            "docker", "stats", "--no-stream", "--format", "{{json .}}", container], text=True))
        rejected = run("run", "--rm", "--no-deps", "-e",
                       "TRUSTTABX_APPROVED_MANIFEST_SHA256=" + "0" * 64,
                       "trusttabx", "python", "-c",
                       "from trusttabx.m5_artifact import load_bundle; load_bundle('/models')",
                       env=env, check=False)
        assert rejected.returncode != 0
        pipeline = fixture / "pipeline.joblib"
        original = pipeline.read_bytes()
        try:
            pipeline.write_bytes(original + b"tampered")
            rejected_pipeline = run("run", "--rm", "--no-deps", "trusttabx", "python", "-c",
                                    "from trusttabx.m5_artifact import load_bundle; load_bundle('/models')",
                                    env=env, check=False)
            assert rejected_pipeline.returncode != 0
        finally:
            pipeline.write_bytes(original)
        assert hashlib.sha256(pipeline.read_bytes()).hexdigest() == hashlib.sha256(original).hexdigest()
        log_process = run("logs", "--no-color", "--tail", "200", "trusttabx", env=env)
        logs = log_process.stdout + log_process.stderr
        assert all(value not in logs for value in (operator, admin, marker))
        run("restart", "trusttabx", env=env)
        restart_seconds = ready()
        expect("predict_after_restart", 200, "/predict", operator, payload)
        return {"runtime_checks": "passed", "startup_seconds": startup_seconds,
                "restart_seconds": restart_seconds, "image_size_bytes": image_size,
                "nonroot_uid": uid, "container_memory_limit_bytes": details["HostConfig"]["Memory"],
                "container_memory_usage": stats["MemUsage"],
                "container_cpu_nano": details["HostConfig"]["NanoCpus"],
                "response_statuses": statuses,
                "build_completed": build.returncode == 0,
                "tampered_manifest_rejected": True, "tampered_pipeline_rejected": True,
                "logs_exclude_keys_and_marker": True,
                "smoke_elapsed_seconds": time.monotonic() - start}
    except Exception:
        logs = run("logs", "--no-color", "--tail", "100", "trusttabx", env=env,
                   check=False)
        diagnostic("container_logs", logs.stdout + logs.stderr)
        raise
    finally:
        run("down", env=env, check=False)
        assert not run("ps", "-q", "trusttabx", env=env).stdout.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true", help="Build and exercise local Docker")
    args = parser.parse_args()
    result = static_checks()
    if args.runtime:
        result.update(runtime_checks())
    print(json.dumps(result, indent=2))

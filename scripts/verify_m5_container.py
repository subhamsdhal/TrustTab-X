"""Static Compose checks, then optional loopback-only synthetic container smoke."""

import argparse
import json
import os
import secrets
import shutil
import subprocess
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
    run("config", "--quiet", env=env)
    run("build", env=env)
    start = time.monotonic()
    run("up", "-d", env=env)
    try:
        startup_seconds = ready()
        assert http("/model")[0] == 401
        assert http("/model", operator)[0] == 200
        payload = {"schema_version": "acs-income-2018-v1", "record": RECORD}
        assert http("/predict", operator, payload)[0] == 200
        batch = {"schema_version": payload["schema_version"], "records": [RECORD, RECORD]}
        assert http("/predict-batch", operator, batch)[0] == 200
        drift = {"schema_version": payload["schema_version"], "records": [RECORD]}
        assert http("/monitor/drift", operator, drift)[0] == 200
        assert http("/metrics", operator)[0] == 200
        assert "TrustTab-X" in http("/")[1]
        assert http("/assets/app.js")[0] == 200
        assert http("/assets/app.css")[0] == 200
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
        run("restart", "trusttabx", env=env)
        restart_seconds = ready()
        assert http("/predict", operator, payload)[0] == 200
        return {"runtime_checks": "passed", "startup_seconds": startup_seconds,
                "restart_seconds": restart_seconds, "image_size_bytes": image_size,
                "nonroot_uid": uid, "container_memory_limit_bytes": details["HostConfig"]["Memory"],
                "container_memory_usage": stats["MemUsage"],
                "container_cpu_nano": details["HostConfig"]["NanoCpus"],
                "smoke_elapsed_seconds": time.monotonic() - start}
    finally:
        run("down", env=env, check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true", help="Build and exercise local Docker")
    args = parser.parse_args()
    result = static_checks()
    if args.runtime:
        result.update(runtime_checks())
    print(json.dumps(result, indent=2))

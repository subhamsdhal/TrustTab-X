"""Measured sequential loopback HTTP smoke for the approved local ACS bundle."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

import psutil

RECORD = {"AGEP": 39, "WKHP": 40, "COW": "1.0", "SCHL": "21.0", "MAR": "1",
          "OCCP": "2310.0", "POBP": "42", "RELP": "0", "SEX": "1", "RAC1P": "1"}


def fetch(url: str, key: str | None = None, payload: dict | None = None):
    headers = {}
    if key:
        headers["X-TrustTabX-Key"] = key
    data = json.dumps(payload).encode() if payload else None
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read()
            return response.status, json.loads(body) if body.startswith(b"{") else body.decode()
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--port", type=int, default=8877)
    args = parser.parse_args()
    if not 1 <= args.requests <= 10000:
        raise ValueError("requests must be 1..10000")
    key = os.environ["TRUSTTABX_API_KEY"]
    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "trusttabx.m5_api:app",
                                "--host", "127.0.0.1", "--port", str(args.port),
                                "--no-access-log", "--timeout-keep-alive", "5"],
                               env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{args.port}"
    try:
        startup = time.perf_counter()
        for _ in range(150):
            try:
                if fetch(base + "/health")[0] == 200:
                    break
            except OSError:
                time.sleep(0.2)
        else:
            raise RuntimeError("service did not start within 30 seconds")
        startup_seconds = time.perf_counter() - startup
        assert fetch(base + "/model", key)[0] == 200
        assert fetch(base + "/model")[0] == 401
        body = {"schema_version": "acs-income-2018-v1", "record": RECORD}
        assert fetch(base + "/predict-batch", key, {
            "schema_version": body["schema_version"], "records": [RECORD, RECORD]})[0] == 200
        assert fetch(base + "/monitor/drift", key, {
            "schema_version": body["schema_version"], "records": [RECORD]})[0] == 200
        assert "TrustTab-X" in fetch(base + "/")[1]
        assert "const defaults" in fetch(base + "/assets/app.js")[1]
        latencies = []
        start = time.perf_counter()
        for _ in range(args.requests):
            tic = time.perf_counter()
            status, result = fetch(base + "/predict", key, body)
            latencies.append((time.perf_counter() - tic) * 1000)
            assert status == 200 and result["model_version"] == "m5-acs-fixed-1"
        elapsed = time.perf_counter() - start
        sorted_times = sorted(latencies)
        processes = [psutil.Process(process.pid)] + psutil.Process(process.pid).children(
            recursive=True)
        memory_mib = sum(item.memory_info().rss for item in processes) / 1048576
        status, metrics = fetch(base + "/metrics", key)
        assert status == 200
        print(json.dumps({"requests": args.requests, "concurrency": 1,
                          "startup_seconds": startup_seconds,
                          "http_p50_ms": statistics.median(latencies),
                          "http_p95_ms": sorted_times[int(0.95 * (len(sorted_times) - 1))],
                          "throughput_requests_per_second": args.requests / elapsed,
                          "process_tree_rss_mib": memory_mib, "service_errors": metrics["errors"],
                          "service_predictions": metrics["requests"],
                          "python_version": sys.version.split()[0]}, indent=2))
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()

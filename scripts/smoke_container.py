"""Verify the real container stays unavailable without mounted research evidence."""

import argparse
import http.client
import json
import subprocess
import time
import urllib.error
import urllib.request
import uuid


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True).strip()


def fetch(url, payload=None, *, timeout=3):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as response:
        return response.code, json.load(response)


STARTUP_ERRORS = (
    ConnectionRefusedError,
    ConnectionResetError,
    ConnectionAbortedError,
    TimeoutError,
    http.client.RemoteDisconnected,
)


def wait_for_startup(url, *, timeout=60, interval=0.5, max_attempts=121):
    """Wait for an HTTP response; application responses are never retried."""
    if timeout <= 0 or interval <= 0 or max_attempts < 1:
        raise ValueError("Startup timeout, interval and attempt limit must be positive")
    deadline = time.monotonic() + timeout
    last_error = None
    attempts = 0
    for _ in range(max_attempts):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        attempts += 1
        try:
            return fetch(url, timeout=min(3, remaining))
        except (*STARTUP_ERRORS, urllib.error.URLError) as exc:
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            if not isinstance(reason, STARTUP_ERRORS):
                raise
            last_error = exc
            remaining = deadline - time.monotonic()
            if remaining <= 0 or attempts == max_attempts:
                break
            time.sleep(min(interval, remaining))
    raise RuntimeError(
        f"Container API startup failed after {attempts} attempts "
        f"within a {timeout:g}s window at {url}; "
        f"last transport error: {type(last_error).__name__}: {last_error}"
    ) from last_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    name = f"credit-risk-smoke-{uuid.uuid4().hex[:12]}"
    docker(
        "run",
        "--detach",
        "--name",
        name,
        "--read-only",
        "--tmpfs",
        "/tmp",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--publish",
        "127.0.0.1::8000",
        args.image,
    )
    try:
        inspection = json.loads(docker("inspect", name))[0]
        assert inspection["HostConfig"]["ReadonlyRootfs"]
        docker("exec", name, "python", "-c", "import os; assert os.getuid() == 10001")
        port = inspection["NetworkSettings"]["Ports"]["8000/tcp"][0]["HostPort"]
        base = f"http://127.0.0.1:{port}"
        status, body = wait_for_startup(base + "/health")
        assert status == 503 and body["status"] == "unavailable"
        for route in ("score", "decision"):
            status, body = fetch(
                base + "/" + route, {"application_id": "container-check", "features": {}}
            )
            assert status == 503 and body["detail"] == "Verified scoring model is unavailable"
        print("PASS: real container starts non-root/read-only and fails closed without artifacts")
    except Exception:
        print(docker("logs", name))
        raise
    finally:
        subprocess.run(["docker", "rm", "--force", name], check=True)


if __name__ == "__main__":
    main()

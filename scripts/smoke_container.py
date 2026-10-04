"""Verify the real container stays unavailable without mounted research evidence."""

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
import uuid


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True).strip()


def fetch(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as response:
        return response.code, json.load(response)


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
        deadline = time.monotonic() + 60
        while True:
            try:
                status, body = fetch(base + "/health")
                break
            except (urllib.error.URLError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise RuntimeError("Container API did not start within 60 seconds") from None
                time.sleep(0.5)
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

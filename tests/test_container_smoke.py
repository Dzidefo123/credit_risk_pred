"""Startup transport retries stay bounded without concealing application failures."""

import http.client
import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "smoke_container", Path(__file__).resolve().parents[1] / "scripts/smoke_container.py"
)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)
URL = "http://127.0.0.1:12345/health"
READY = (503, {"status": "unavailable"})


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        now = 0.0

        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(smoke.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(smoke.time, "sleep", clock.sleep)
    return clock


def test_immediate_startup_success(monkeypatch, clock):
    calls = []

    def fetch(url, *, timeout):
        calls.append((url, timeout))
        return READY

    monkeypatch.setattr(smoke, "fetch", fetch)
    assert smoke.wait_for_startup(URL) == READY
    assert calls == [(URL, 3)]
    assert clock.now == 0


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionRefusedError("not listening"),
        ConnectionResetError("reset by peer"),
        ConnectionAbortedError("aborted"),
        TimeoutError("request timed out"),
        http.client.RemoteDisconnected("closed before response"),
        urllib.error.URLError(ConnectionResetError("wrapped reset")),
        urllib.error.URLError(ConnectionRefusedError("wrapped refusal")),
        urllib.error.URLError(TimeoutError("wrapped timeout")),
    ],
)
def test_transient_startup_failure_then_success(monkeypatch, clock, failure):
    calls = []

    def fetch(url, *, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise failure
        return READY

    monkeypatch.setattr(smoke, "fetch", fetch)
    assert smoke.wait_for_startup(URL) == READY
    assert calls == [3, 3]
    assert clock.now == 0.5


def test_attempt_limit_exhaustion_preserves_last_error(monkeypatch, clock):
    failure = ConnectionResetError("reset by peer")
    calls = []

    def fetch(url, *, timeout):
        calls.append(timeout)
        raise failure

    monkeypatch.setattr(smoke, "fetch", fetch)
    with pytest.raises(RuntimeError, match="after 3 attempts.*60s.*ConnectionResetError") as exc:
        smoke.wait_for_startup(URL, max_attempts=3)
    assert exc.value.__cause__ is failure
    assert len(calls) == 3
    assert clock.now == 1


def test_deadline_caps_sleep_and_prevents_another_attempt(monkeypatch, clock):
    calls = []

    def fetch(url, *, timeout):
        calls.append(timeout)
        raise ConnectionRefusedError("not listening")

    monkeypatch.setattr(smoke, "fetch", fetch)
    with pytest.raises(RuntimeError, match="after 2 attempts.*0.6s"):
        smoke.wait_for_startup(URL, timeout=0.6)
    assert calls == pytest.approx([0.6, 0.1])
    assert clock.now == pytest.approx(0.6)


def test_request_timeout_uses_remaining_startup_budget(monkeypatch, clock):
    calls = []

    def fetch(url, *, timeout):
        calls.append(timeout)
        clock.now += timeout
        raise TimeoutError("timed out")

    monkeypatch.setattr(smoke, "fetch", fetch)
    with pytest.raises(RuntimeError, match="after 1 attempts.*2s"):
        smoke.wait_for_startup(URL, timeout=2)
    assert calls == [2]
    assert clock.now == 2


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("invalid response"),
        PermissionError("permission denied"),
        urllib.error.URLError("name resolution failed"),
        urllib.error.URLError(PermissionError("permission denied")),
    ],
)
def test_non_transient_failure_is_not_retried(monkeypatch, clock, failure):
    calls = []

    def fetch(url, *, timeout):
        calls.append(timeout)
        raise failure

    monkeypatch.setattr(smoke, "fetch", fetch)
    with pytest.raises(type(failure)) as exc:
        smoke.wait_for_startup(URL)
    assert exc.value is failure
    assert len(calls) == 1
    assert clock.now == 0


def test_fetch_preserves_http_error_response_and_timeout(monkeypatch):
    calls = []

    def urlopen(request, *, timeout):
        calls.append(timeout)
        raise urllib.error.HTTPError(
            URL, 500, "application failed", {}, io.BytesIO(b'{"detail":"failed"}')
        )

    monkeypatch.setattr(smoke.urllib.request, "urlopen", urlopen)
    assert smoke.fetch(URL, timeout=0.2) == (500, {"detail": "failed"})
    assert calls == [0.2]


@pytest.mark.parametrize("response", [(500, {"detail": "failed"}), (503, {"status": "wrong"})])
def test_application_contract_failure_still_fails_smoke(monkeypatch, response, capsys):
    calls = []
    cleanup = []

    def docker(*args):
        calls.append(args)
        if args[0] == "inspect":
            return json.dumps(
                [
                    {
                        "HostConfig": {"ReadonlyRootfs": True},
                        "NetworkSettings": {"Ports": {"8000/tcp": [{"HostPort": "12345"}]}},
                    }
                ]
            )
        return "container diagnostic log" if args[0] == "logs" else ""

    monkeypatch.setattr(smoke, "docker", docker)
    monkeypatch.setattr(smoke, "fetch", lambda *args, **kwargs: response)
    monkeypatch.setattr(smoke.subprocess, "run", lambda args, **kwargs: cleanup.append(args))
    monkeypatch.setattr("sys.argv", ["smoke_container.py", "--image", "test-image"])
    with pytest.raises(AssertionError):
        smoke.main()
    assert any(call[0] == "logs" for call in calls)
    assert "container diagnostic log" in capsys.readouterr().out
    assert cleanup[0][:3] == ["docker", "rm", "--force"]


@pytest.mark.parametrize("options", [{"timeout": 0}, {"interval": 0}, {"max_attempts": 0}])
def test_invalid_retry_limits_are_rejected(options):
    with pytest.raises(ValueError, match="must be positive"):
        smoke.wait_for_startup(URL, **options)

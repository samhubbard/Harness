"""send(): retries and context-window guards, with a fake network (no server, no real waiting)."""

import httpx
import pytest

from harness import llm
from harness.llm import ContextOverflowError, build_request, send

OK = {"choices": [], "usage": {"prompt_tokens": 500}}


def response(status: int, json: dict | None = None, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json=json or {}, headers=headers, request=httpx.Request("POST", llm.ENDPOINT))


def fake_network(*outcomes):
    """Stand-in for httpx.post: each call returns (or raises) the next outcome."""
    queue, calls = list(outcomes), []

    def post(url, json, timeout):
        calls.append(json)
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    return post, calls


@pytest.mark.parametrize("first_failure", [
    httpx.ConnectError("connection refused"),   # server not up yet
    httpx.ReadTimeout("model still loading"),   # cold start
    response(429),                              # rate limited
    response(503),                              # server overloaded
])
def test_temporary_failures_are_retried(first_failure):
    post, calls = fake_network(first_failure, response(200, OK))
    waits = []
    assert send(build_request("climb 5000"), post=post, sleep=waits.append) == OK
    assert len(calls) == 2 and len(waits) == 1


@pytest.mark.parametrize("status", [400, 401, 404])
def test_permanent_failures_are_not_retried(status):
    post, calls = fake_network(response(status))
    with pytest.raises(httpx.HTTPStatusError):
        send(build_request("climb 5000"), post=post, sleep=lambda s: None)
    assert len(calls) == 1  # same request -> same answer, so don't ask again


def test_gives_up_after_max_attempts():
    post, calls = fake_network(*[httpx.ConnectError("down")] * llm.MAX_ATTEMPTS)
    waits = []
    with pytest.raises(httpx.ConnectError):
        send(build_request("climb 5000"), post=post, sleep=waits.append)
    assert len(calls) == llm.MAX_ATTEMPTS
    assert 1 <= waits[0] < 1.3 and 2 <= waits[1] < 2.3  # backoff doubles, with a little jitter


def test_retry_after_header_is_honored_but_capped():
    post, _ = fake_network(response(429, headers={"Retry-After": "7"}), response(429, headers={"Retry-After": "999"}),
                           response(200, OK))
    waits = []
    send(build_request("climb 5000"), post=post, sleep=waits.append)
    assert waits == [7, llm.MAX_WAIT_S]


def test_prompt_too_long_is_refused_before_sending():
    post, calls = fake_network()
    with pytest.raises(ContextOverflowError, match="too long"):
        send(build_request("Background notes. " * 2000), post=post)
    assert calls == []  # never sent


def test_server_side_truncation_is_detected():
    """Measured: a too-long prompt comes back counted as 2,050 tokens and gets a confident wrong answer."""
    body = build_request("Background notes. " * 300)  # passes the estimate, but pretend the server truncated
    post, _ = fake_network(response(200, {"choices": [], "usage": {"prompt_tokens": 100}}))
    with pytest.raises(ContextOverflowError, match="dropped part of the prompt"):
        send(body, post=post)

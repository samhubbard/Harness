"""Step 2: show the exact HTTP request and response on the wire.

Same call as 01_first_request.py, with thinking disabled, but httpx "event hooks"
print the raw request and response so nothing is hidden by a library.

Run:  uv run examples/02_inspect_http.py
"""

import json

import httpx

ENDPOINT = "http://localhost:11434/v1/chat/completions"


def show_request(request: httpx.Request) -> None:
    print("=" * 30, "REQUEST (client -> server)", "=" * 30)
    print(f"{request.method} {request.url.raw_path.decode()} HTTP/1.1")
    for name, value in request.headers.items():
        print(f"{name}: {value}")
    print()  # blank line separates headers from body
    print(json.dumps(json.loads(request.content), indent=2))


def show_response(response: httpx.Response) -> None:
    response.read()  # load the body so we can print it here
    print("=" * 30, "RESPONSE (server -> client)", "=" * 30)
    print(f"HTTP/1.1 {response.status_code} {response.reason_phrase}")
    for name, value in response.headers.items():
        print(f"{name}: {value}")
    print()
    print(json.dumps(response.json(), indent=2))


request_body = {
    "model": "qwen3.5:9b",
    "messages": [
        {"role": "system", "content": "You are a CH-47F Chinook flight command interpreter. Reply in one short sentence."},
        {"role": "user", "content": "Climb and maintain five thousand feet."},
    ],
    "temperature": 0,
    "reasoning_effort": "none",  # disable the hidden "thinking" pass
}

with httpx.Client(event_hooks={"request": [show_request], "response": [show_response]}, timeout=120) as client:
    client.post(ENDPOINT, json=request_body)

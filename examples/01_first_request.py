"""Step 1: one raw HTTP request to a model, no SDK.

Everything an AI "integration" does reduces to this: build a JSON request,
POST it to an inference endpoint, and pull text out of the JSON response.

Run:  uv run examples/01_first_request.py
"""

import time

import httpx

# Ollama serves an OpenAI-compatible API locally. Swapping this URL (plus an
# API key header) is most of what it takes to point at a hosted provider.
ENDPOINT = "http://localhost:11434/v1/chat/completions"
MODEL = "qwen3.5:9b"

request_body = {
    "model": MODEL,
    "messages": [
        # System message: standing instructions the user doesn't see.
        {"role": "system", "content": "You are a CH-47F Chinook flight command interpreter. Reply in one short sentence."},
        # User message: the actual input for this turn.
        {"role": "user", "content": "Climb and maintain five thousand feet."},
    ],
    # 0 = always pick the most likely next token: the least random output we can ask for.
    "temperature": 0,
}

start = time.perf_counter()
# Timeout matters: the first call loads the model into VRAM and can take a while.
response = httpx.post(ENDPOINT, json=request_body, timeout=120)
elapsed = time.perf_counter() - start

response.raise_for_status()  # turn HTTP 4xx/5xx into an exception instead of bad data
data = response.json()

print("Model reply:", data["choices"][0]["message"]["content"])
print("Finish reason:", data["choices"][0]["finish_reason"])  # "stop" = done; "length" = hit the token limit
print("Token usage:", data["usage"])  # what a hosted provider would bill you for
print(f"Latency: {elapsed:.2f}s")

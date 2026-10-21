"""Step 6: the knobs on every request: max_tokens, temperature, top_p, and the context window.

Run:  uv run examples/06_inference_params.py
"""

import httpx

from harness.llm import ENDPOINT, MODEL, build_request

BRIEF = [{"role": "user", "content": "In one sentence, give a creative callsign-style name for a training helicopter."}]


def post(body: dict) -> dict:
    return httpx.post(ENDPOINT, json=body, timeout=300).json()


def chat(messages: list[dict], **params) -> dict:
    return post({"model": MODEL, "messages": messages, "reasoning_effort": "none", **params})


print("=" * 20, "1. max_tokens: a hard cap on the REPLY length", "=" * 20)
body = build_request("Turn right heading zero niner zero, climb and maintain five thousand")
choice = post({**body, "max_tokens": 15})["choices"][0]
print(f"finish_reason={choice['finish_reason']!r}  content={choice['message']['content']!r}")
print("-> cut off mid-JSON. This is why parse_reply() checks finish_reason before parsing.\n")

print("=" * 20, "2. temperature: how adventurous the sampling is", "=" * 20)
for temperature in (0, 1.0):
    for _ in range(3):
        reply = chat(BRIEF, temperature=temperature)["choices"][0]["message"]["content"]
        print(f"temp={temperature}: {reply}")
print("-> temp 0 picks the most likely token every time: same input, same output.\n")

print("=" * 20, "3. top_p: only sample from the most likely tokens", "=" * 20)
for _ in range(3):
    reply = chat(BRIEF, temperature=1.0, top_p=0.05)["choices"][0]["message"]["content"]
    print(f"temp=1.0, top_p=0.05: {reply}")
print("-> top_p keeps only the top 5% of probability mass, so high temperature has little to choose from.\n")

print("=" * 20, "4. context window: everything sent + generated must fit", "=" * 20)
filler = "Log entry: routine systems check complete, all parameters nominal. " * 3000  # ~40k tokens
messages = [{"role": "user", "content": "The secret word is KESTREL.\n" + filler + "\nWhat is the secret word?"}]
envelope = chat(messages, temperature=0, max_tokens=20)
print(f"prompt_tokens={envelope['usage']['prompt_tokens']:,}  reply={envelope['choices'][0]['message']['content']!r}")
print("-> if prompt_tokens is far below what we sent, the server silently dropped part of the prompt.")

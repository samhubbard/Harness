# Learning Log

Running record of what I've learned building the Aviation AI Command Harness, and what still needs work.
Interview: PeopleTec AI/ML Lead, Fri 2026-10-23, 1:00 PM.

## Mon 10-19: Environment + first model request

**Setup**
- PC: RTX 4070 Ti SUPER (16 GB VRAM), Ryzen 7 7700X, 32 GB RAM. Enough for local inference of models up to ~14B (4-bit).
- Python 3.12 via `uv` (per-project `.venv`, lockfile). Ollama serves models locally on `localhost:11434`.
- Model: `qwen3.5:9b`, 6.6 GB on disk (9B params × ~4-bit quantization + overhead).
- No CUDA toolkit needed: the driver sets the max CUDA version; runtimes ship inside Ollama/PyTorch.

**Concepts**
- **Inference endpoint:** an HTTP URL that takes a JSON request and returns generated text. The OpenAI-compatible
  `/v1/chat/completions` shape is the de facto standard (OpenAI, Azure OpenAI, Ollama, vLLM, LM Studio).
- **Messages:** `system` = standing instructions; `user` = this turn's input; `assistant` = the model's reply.
- **Tokens:** the unit models read, write and bill by (roughly ¾ of an English word). `usage` reports prompt vs completion tokens.
- **Temperature 0:** least random output. Not a guarantee of identical output.
- **Cold start:** the first request loaded the model into VRAM: 54 s. Warm requests: 2–10 s.
- **Reasoning ("thinking") models:** qwen3.5 writes hidden reasoning before answering.
  Measured: thinking ON = 761 tokens / 10.3 s; OFF = 11 tokens / 2.3 s, same answer.
  Trade-off: accuracy on hard inputs vs latency/cost. On hosted APIs, reasoning tokens are billed.

**Still to nail down**
- [ ] Explain-back: why validate in Python if the model already has a JSON Schema?
- [ ] Decide: thinking on or off for the command interpreter?

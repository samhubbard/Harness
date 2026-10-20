# Learning Log

Interview: PeopleTec AI/ML Lead (Speech, Model Tuning & Guardrails), **Fri 2026-10-23, 1:00 PM**.
Goal: hold an intelligent technical conversation, be honest about limits, and show how fast I learn.
The harness is a teaching tool, not the goal. **Harness feature freeze: Wednesday noon.**

Reference map: [AI-103 study guide](https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/ai-103)
(Developing AI Apps and Agents on Azure; Microsoft Foundry-centric; replaced AI-102). Used as a map, not exam prep.

## Progress tracker

Status: ⬜ not started · 🟨 in progress · ✅ can explain it in my own words

| # | Priority | Topics | When | Status |
|---|---|---|---|---|
| 1 | Direct model integration | HTTP API, tokens, temperature, reasoning models | Mon | ✅ |
| 1 | Structured outputs + validation | JSON Schema, Pydantic, constrained decoding, guardrails, tests | Tue AM | 🟨 |
| 1 | Tool / function calling | Tool schemas, the model asks, the app executes | Wed AM | ⬜ |
| 2 | Azure fundamentals | Tenant/subscription/resource group, Entra ID, regions | Tue PM | ⬜ |
| 2 | Microsoft Foundry + Azure OpenAI | Projects, model deployments, endpoints, quotas | Tue PM | ⬜ |
| 2 | Azure security | Managed identity, keyless auth, Key Vault, private endpoints, RBAC | Wed AM | ⬜ |
| 2 | Azure Government | Sovereign cloud, separate endpoints, compliance (FedRAMP High, IL levels), feature lag | Tue PM | ⬜ |
| 2 | Azure hosting | App Service, Functions, Container Apps, Storage | Wed AM | ⬜ |
| 3 | Speech | ASR vs TTS, preprocessing, aviation vocab, WER exercise | Wed PM | ⬜ |
| 3 | Model tuning + eval | Train/val/test splits, fine-tuning vs prompting, LoRA, overfitting, regression testing | Wed PM | ⬜ |
| 4 | RAG | Chunking, embeddings, vector search, Azure AI Search, grounding | Thu AM | ⬜ |
| 5 | Web technologies | HTML/CSS/JS, React, REST, frontend vs backend | Thu AM | ⬜ |
| 6 | Infra | CUDA, VRAM, quantization, serving, local vs cloud, AWS (EC2/S3/Lambda/IAM), certs | Thu PM | ⬜ |
| 7 | Interview prep | Lucid stories, Tatsu, Army aviation, gaps, mock interview | Thu PM | ⬜ |
| — | Light review | Re-read this log, the answers I've written, then stop | Fri AM | ⬜ |

**Tue 10-20 decision:** Azure is on hold until I can explain the harness end to end. Schedule shifts accordingly.

## Explain-back answers
- ✅ **Why is `message.content` a string?** JSON is just text, and the model generates text. The harness parses
  it into schema classes. *Correction:* there are two parses. `response.json()` parses the server's envelope;
  `content` is model-written text that may NOT be valid JSON (prose, truncated), so parse #2 can fail and is wrapped in `try`/`except`.
- 🟨 **What did "Turn left heading 270" lose?** My insight: *a validator can only check what arrives; it can't
  see what's missing* (it catches commission errors, not omissions). *Correction:* both commands arrived; what was lost
  was **turn direction**. The schema has no field for it, so the model couldn't express it. Fix: the schema first, with readback as the backstop.

---

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
- **Decision (mine):** thinking OFF by default. Ambiguity calls for clarification, not more reasoning.
  Revisit only if a labeled test set shows accuracy gains worth the latency.
- **The API is stateless:** "memory" is the app resending the whole conversation each call, which costs more tokens each turn until the context window forces trimming.
- **Auth:** hosted APIs add `Authorization: Bearer <key>`. Keys live in `.env` (git-ignored), never in code.
- **finish_reason:** `stop` = complete; `length` = cut off. Always check before parsing.

## Tue 10-20: Structured output + validation

**What I built:** `harness/schema.py` (Pydantic contract), `harness/validator.py` (deterministic rules),
`harness/llm.py` (model call), `examples/03_structured_output.py`, `tests/test_validator.py` (17 tests, 0.1 s).

**Concepts**
- **JSON Schema:** a machine-readable description of valid JSON (types, required fields, allowed values).
  Pydantic generates it from Python classes: one source of truth for both the prompt and the parser.
- **Constrained decoding:** the server blocks any token that would break the schema. It guarantees *shape*, not *truth*.
- **Defense in depth:** schema (shape) → Pydantic parse (types) → validator (ranges, cross-field rules, policy) → (later) readback/confirmation (meaning).
- **All-or-nothing batches:** one bad command rejects the whole instruction.

**Findings from real runs**
1. **Over-refusal:** the first prompt made the model "clarify" even "climb and maintain five thousand."
   Fix: define what counts as clear and add 3 few-shot examples. Result: 6/6 inputs behaved correctly.
2. **Constrained decoding can corrupt values silently:** schema `maximum: 15000`, user said 25,000 →
   model output **2500** (3/3 runs). The grammar blocked the last "0". The result is valid, confident and wrong, and it passes range checks.
   → Keep limits OUT of the schema; let the model transcribe and the validator refuse.
3. **Prompt injection** ("SYSTEM OVERRIDE ... airspeed 999") produced valid JSON; the Python validator rejected it.
   The validator doesn't care how the model was persuaded.

**My one-liner:** *(write it here in my own words)*

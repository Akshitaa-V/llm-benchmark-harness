# llm-benchmark-harness

A small harness demonstrating LLM benchmarking methodology — fixed prompt set, multiple "models," consistent scoring, leaderboard output.

## Read this before anything else

**Most of the model responses here are hardcoded strings, not real LLM output.** Two of the three "models" (`mock-concise-v1` and `mock-detailed-v1`) are simulated on purpose, to demonstrate benchmarking methodology without depending on a paid API key or a large download. The third model, `ollama-llama3.2-1b`, is a real, locally running LLM call — see "Real Model Validation" below.

I built the mocks this way on purpose: testing a real hosted LLM needs a paid API key, and running a real model locally needs a multi-GB download — both outside what I originally wanted this project to depend on. What I wanted to prove I understood first was the *methodology* — how you structure a benchmark so results are fair and comparable across models. I later added the real local model to validate that the methodology actually held up against genuine LLM output, not just hardcoded strings.

## What's actually real here

- A fixed benchmark set: the same 3 prompts and reference keywords used for every model, so nothing is cherry-picked per model
- A scoring function that checks keyword coverage against a reference list
- Latency timing per response
- Aggregation into a per-model leaderboard, sorted by average score
- A real local LLM call via Ollama, scored with the exact same logic as the mocks (see below)

If you swapped the mock model functions for real API calls, the rest of the harness — scoring, timing, leaderboard — would work unchanged. That's deliberate: the methodology is the same whether the model behind it is real or mocked. The Ollama integration below is proof of that, not just a claim.

## Running it

```bash
python3 benchmark_harness.py
```

Sample output:

```
NOTE: All model responses below are hardcoded mock strings, not real LLM output.

[mock-detailed-v1] p1: coverage=1.00, latency=0.001ms
    response: The Model Context Protocol (MCP) is an open protocol...

=== Leaderboard (avg keyword coverage across all prompts) ===
  mock-detailed-v1: 1.00 average keyword coverage
  mock-concise-v1: 0.28 average keyword coverage
```

## Real Model Validation (Ollama)

To validate that the harness design works against an actual LLM (not just hardcoded mocks), a third entry was added to `MOCK_MODELS`: `ollama-llama3.2-1b`, which calls a locally running Ollama instance (`llama3.2:1b`, ~1.3GB, fully offline, no API key) via its REST API at `localhost:11434`.

This required zero changes to `run_benchmark()`, `score_keyword_coverage()`, or `print_leaderboard()` — the existing scoring and leaderboard logic worked unmodified against real generated text, confirming the harness was correctly decoupled from the mock implementation.

**Note on reproducibility:** across repeated runs, `ollama-llama3.2-1b`'s exact wording and keyword coverage score varied (0.17–0.25 average coverage observed across two runs), since Ollama samples with some randomness by default rather than answering deterministically like the hardcoded mocks. This is itself a useful benchmarking observation — real model evaluation needs multiple trials per prompt to account for this, unlike fixed mock baselines that always return the same string.

**Result:** `ollama-llama3.2-1b` scored 0.17–0.25 average keyword coverage across the 3-prompt set (across two runs), lower than `mock-detailed-v1` (1.00) and roughly in line with `mock-concise-v1` (0.28). The model partially answered the MCP question correctly in one run but confused "Claude Code Hook" and "RAG" with unrelated concepts in both runs (general networking/software terminology and a homelessness-support or waste-reduction program, respectively) — a realistic example of a small (1B parameter) general-purpose model lacking AI-tooling-specific knowledge. Latency also rose from ~0.001ms (instant mock lookup) to 6,500–20,000ms (real inference), as expected for a genuine model call versus a hardcoded string.

This is a small but real demonstration of head-to-head model evaluation: same fixed prompt set, same scoring method, applied consistently across mock baselines and a real local model — and a concrete illustration of why reproducibility (multiple trials, fixed sampling settings) matters once a real model is in the loop.

## What I'd add next

- Run more trials per prompt against Ollama to quantify response variance more rigorously, rather than relying on two runs
- Add a deterministic mode (`temperature=0`, fixed seed) to Ollama calls for more directly comparable repeated runs
- Add more metrics beyond keyword coverage — e.g. response length variance, basic factual-consistency checks
- Try a slightly larger local model (e.g. `llama3.2:3b`) to see whether coverage improves with model size on this same fixed prompt set
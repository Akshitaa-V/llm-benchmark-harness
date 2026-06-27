# llm-benchmark-harness

A small harness demonstrating LLM benchmarking methodology — fixed prompt set, multiple "models," consistent scoring, leaderboard output.

## Read this before anything else

**This does not call a real language model.** Every model response in this project is a hardcoded string written to simulate plausible outputs of different quality. There is no live LLM, no API call, anywhere in this code.

I built it this way on purpose: testing a real hosted LLM needs a paid API key, and running a real model locally needs a multi-GB download — both outside what I wanted this project to depend on. What I actually wanted to prove I understood was the *methodology* — how you structure a benchmark so results are fair and comparable — and that part is fully real here.

## What's actually real here

- A fixed benchmark set: the same 3 prompts and reference keywords used for every model, so nothing is cherry-picked per model
- A scoring function that checks keyword coverage against a reference list
- Latency timing per response
- Aggregation into a per-model leaderboard, sorted by average score

If you swapped the two mock model functions for real API calls, the rest of the harness — scoring, timing, leaderboard — would work unchanged. That's deliberate: the methodology is the same whether the model behind it is real or mocked.

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

## What I'd add next

- Swap in a real API call (with my own key) for one of the two "models," to compare real vs. mock output side by side
- Add more metrics beyond keyword coverage — e.g. response length variance, basic factual-consistency checks
- Run the same harness against a small local model instead of a hosted API, to avoid needing any key at all

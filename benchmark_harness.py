"""
benchmark_harness.py
=====================

A benchmarking harness that demonstrates LLM evaluation METHODOLOGY using
simulated (mock) model responses.

IMPORTANT, READ THIS FIRST:
This script does NOT call a real language model. There is no live LLM
anywhere in this code. Every "model response" below is a hardcoded string
written to simulate plausible outputs of different quality, so that the
scoring and comparison logic has something to run against.

What this DOES demonstrate, honestly:
  - How to structure a fixed benchmark set (prompts + reference answers)
  - How to score outputs against multiple metrics (keyword coverage,
    length, latency) in a way that's directly comparable across "models"
  - How to aggregate per-prompt scores into a per-model leaderboard
  - Good benchmarking practice: same test set, same metrics, same scoring
    method applied to every "model" being compared

What this does NOT demonstrate:
  - Actually calling or testing a real LLM
  - Real latency, cost, or quality measurements
  - Any claim about how a real model would actually perform

This was built specifically because testing a real hosted LLM requires a
paid API key, and running a real local LLM requires a multi-GB model
download -- both outside the scope of a small, dependency-free local
project. The mock structure here is the same structure you'd plug a real
model's API call into later; only the response-generation step would
change.
"""

import time
import requests
from dataclasses import dataclass, field

@dataclass
class BenchmarkPrompt:
    prompt_id: str
    prompt: str
    reference_keywords: list[str]  # words a good answer should mention


@dataclass
class ModelResult:
    model_name: str
    prompt_id: str
    response: str
    latency_seconds: float
    keyword_coverage: float = field(default=0.0)


# --- Fixed benchmark set: same prompts, same reference keywords, for every model ---

BENCHMARK_SET = [
    BenchmarkPrompt(
        prompt_id="p1",
        prompt="What is the Model Context Protocol (MCP)?",
        reference_keywords=["protocol", "tools", "server", "client"],
    ),
    BenchmarkPrompt(
        prompt_id="p2",
        prompt="What is a Claude Code Hook?",
        reference_keywords=["event", "automatic", "workflow"],
    ),
    BenchmarkPrompt(
        prompt_id="p3",
        prompt="What does RAG stand for and what problem does it solve?",
        reference_keywords=["retrieval", "generation", "context", "hallucination"],
    ),
]


# --- Simulated models. These are NOT real LLM calls -- see module docstring. ---


def mock_model_concise(prompt: str) -> str:
    """Simulates a model that gives short, somewhat thin answers."""
    canned = {
        "p1": "MCP is a protocol for AI tools.",
        "p2": "A hook runs code automatically.",
        "p3": "RAG retrieves text before generating an answer.",
    }
    prompt_id = _match_prompt_id(prompt)
    return canned.get(prompt_id, "I don't know.")


def mock_model_detailed(prompt: str) -> str:
    """Simulates a model that gives longer, more complete answers."""
    canned = {
        "p1": "The Model Context Protocol (MCP) is an open protocol that lets a client connect to a server exposing tools, so the client can call those tools without custom integration code.",
        "p2": "A Claude Code Hook runs custom logic automatically at a specific workflow event, such as before or after a tool call, without relying on the model to remember to do it.",
        "p3": "RAG stands for Retrieval-Augmented Generation. It retrieves relevant context before generation, which reduces hallucination compared to relying on the model's training alone.",
    }
    prompt_id = _match_prompt_id(prompt)
    return canned.get(prompt_id, "I don't know.")

def ollama_model(prompt: str, model: str = "llama3.2:1b") -> str:
    """
    Calls a real, locally running Ollama model and returns its response.
    Unlike the mock_model_* functions above, this makes an actual LLM call
    over Ollama's local REST API (no API key, no cloud, fully offline).
    Same input/output shape as the mock functions, so it drops into
    MOCK_MODELS and the existing scoring/leaderboard code below without
    any changes to run_benchmark() or print_leaderboard().
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def _match_prompt_id(prompt: str) -> str:
    for bp in BENCHMARK_SET:
        if bp.prompt == prompt:
            return bp.prompt_id
    return ""

MOCK_MODELS = {
    "mock-concise-v1": mock_model_concise,
    "mock-detailed-v1": mock_model_detailed,
    "ollama-llama3.2-1b": ollama_model,
}
# --- Scoring ---


def score_keyword_coverage(response: str, reference_keywords: list[str]) -> float:
    """Fraction of reference keywords that appear in the response (case-insensitive)."""
    response_lower = response.lower()
    hits = sum(1 for kw in reference_keywords if kw.lower() in response_lower)
    return hits / len(reference_keywords) if reference_keywords else 0.0


def run_benchmark() -> list[ModelResult]:
    results = []
    for model_name, model_fn in MOCK_MODELS.items():
        for bp in BENCHMARK_SET:
            start = time.perf_counter()
            response = model_fn(bp.prompt)
            latency = time.perf_counter() - start

            coverage = score_keyword_coverage(response, bp.reference_keywords)

            results.append(
                ModelResult(
                    model_name=model_name,
                    prompt_id=bp.prompt_id,
                    response=response,
                    latency_seconds=latency,
                    keyword_coverage=coverage,
                )
            )
    return results


def print_leaderboard(results: list[ModelResult]) -> None:
    by_model: dict[str, list[ModelResult]] = {}
    for r in results:
        by_model.setdefault(r.model_name, []).append(r)

    print("=== Per-prompt results ===\n")
    for r in results:
        print(f"[{r.model_name}] {r.prompt_id}: coverage={r.keyword_coverage:.2f}, "
              f"latency={r.latency_seconds*1000:.3f}ms")
        print(f"    response: {r.response}\n")

    print("=== Leaderboard (avg keyword coverage across all prompts) ===\n")
    averages = {
        model: sum(r.keyword_coverage for r in rs) / len(rs)
        for model, rs in by_model.items()
    }
    for model, avg in sorted(averages.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {avg:.2f} average keyword coverage")


if __name__ == "__main__":
    print(
        "NOTE: All model responses below are hardcoded mock strings, not real "
        "LLM output. This demonstrates benchmarking methodology only.\n"
    )
    results = run_benchmark()
    print_leaderboard(results)

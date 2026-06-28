"""
benchmark_harness.py
=====================

A benchmarking harness that demonstrates LLM evaluation methodology using
both simulated (mock) model responses and a real local LLM call (Ollama).

READ THIS FIRST:
Two of the three "models" below (mock-concise-v1, mock-detailed-v1) are
hardcoded strings written to simulate plausible outputs of different
quality -- there is no live LLM call for these. The third model,
ollama-llama3.2-1b, IS a real, locally running LLM call via Ollama's
REST API (no API key, no cloud, fully offline).

This version adds:
  - A grounding score: overlap between a response and the FULL reference
    answer text (distinct from keyword_coverage, which only checks a
    short hand-picked keyword list).
  - An estimated cost field: a word-count-based token proxy multiplied by
    a fixed, disclosed reference price per 1K tokens. This is clearly an
    ESTIMATE for demonstrating cost-tracking mechanics -- no paid API was
    called anywhere in this project, and no real invoice exists.
  - A pandas DataFrame view of all results, used for aggregation.
  - A numpy-based latency variance calculation per model, useful for
    spotting non-deterministic/unstable models (e.g. the real Ollama call,
    which has genuine variance, versus the mocks, which have none).
  - A scikit-learn mean_squared_error comparison between the
    keyword_coverage metric and the grounding metric, to check whether
    two different scoring methods agree on the same responses.
  - A pytest suite (test_benchmark_harness.py) and a GitHub Actions CI
    workflow that runs it on every push.
"""

import time
import requests
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass, field


@dataclass
class BenchmarkPrompt:
    prompt_id: str
    prompt: str
    reference_keywords: list[str]  # words a good answer should mention
    reference_answer: str  # full reference answer, used for grounding score


@dataclass
class ModelResult:
    model_name: str
    prompt_id: str
    response: str
    latency_seconds: float
    keyword_coverage: float = field(default=0.0)
    grounding: float = field(default=0.0)
    estimated_cost_usd: float = field(default=0.0)


# --- Fixed benchmark set: same prompts, same reference data, for every model ---

BENCHMARK_SET = [
    BenchmarkPrompt(
        prompt_id="p1",
        prompt="What is the Model Context Protocol (MCP)?",
        reference_keywords=["protocol", "tools", "server", "client"],
        reference_answer="The Model Context Protocol (MCP) is an open protocol that lets a client connect to a server exposing tools, so the client can call those tools without custom integration code.",
    ),
    BenchmarkPrompt(
        prompt_id="p2",
        prompt="What is a Claude Code Hook?",
        reference_keywords=["event", "automatic", "workflow"],
        reference_answer="A Claude Code Hook runs custom logic automatically at a specific workflow event, such as before or after a tool call, without relying on the model to remember to do it.",
    ),
    BenchmarkPrompt(
        prompt_id="p3",
        prompt="What does RAG stand for and what problem does it solve?",
        reference_keywords=["retrieval", "generation", "context", "hallucination"],
        reference_answer="RAG stands for Retrieval-Augmented Generation. It retrieves relevant context before generation, which reduces hallucination compared to relying on the model's training alone.",
    ),
]


# --- Simulated models (mock) and one real local LLM call (Ollama) ---


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

# Published per-1K-token pricing used only as a disclosed, fixed reference
# rate to demonstrate cost-estimation mechanics. This is NOT a real charged
# cost -- no paid API was called anywhere in this project. Rates are loosely
# based on publicly published small-model pricing as of 2026, for illustration.
COST_PER_1K_INPUT_TOKENS = 0.00015
COST_PER_1K_OUTPUT_TOKENS = 0.0006


def score_keyword_coverage(response: str, reference_keywords: list[str]) -> float:
    """Fraction of reference keywords that appear in the response (case-insensitive)."""
    response_lower = response.lower()
    hits = sum(1 for kw in reference_keywords if kw.lower() in response_lower)
    return hits / len(reference_keywords) if reference_keywords else 0.0


def score_grounding(response: str, reference_answer: str) -> float:
    """
    A simple grounding metric: fraction of the reference answer's
    significant words (length > 3, to skip filler words like 'the', 'is')
    that also appear in the response. Distinct from keyword_coverage,
    which only checks a short hand-picked keyword list -- this checks
    overlap against the full reference answer text.
    """
    ref_words = {w.lower().strip(".,()") for w in reference_answer.split() if len(w) > 3}
    resp_lower = response.lower()
    if not ref_words:
        return 0.0
    hits = sum(1 for w in ref_words if w in resp_lower)
    return round(hits / len(ref_words), 3)


def estimate_cost(prompt: str, response: str) -> float:
    """
    Estimates cost using a simple word-count proxy for tokens (roughly
    0.75 tokens per word) and a fixed, disclosed reference price per 1K
    tokens. This is an ESTIMATE for demonstrating cost-tracking mechanics,
    not a real billed cost -- no paid API was called.
    """
    input_tokens = len(prompt.split()) / 0.75
    output_tokens = len(response.split()) / 0.75
    cost = (input_tokens / 1000 * COST_PER_1K_INPUT_TOKENS) + \
           (output_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS)
    return round(cost, 6)


def run_benchmark() -> list[ModelResult]:
    results = []
    for model_name, model_fn in MOCK_MODELS.items():
        for bp in BENCHMARK_SET:
            start = time.perf_counter()
            response = model_fn(bp.prompt)
            latency = time.perf_counter() - start

            coverage = score_keyword_coverage(response, bp.reference_keywords)
            grounding = score_grounding(response, bp.reference_answer)
            cost = estimate_cost(bp.prompt, response)

            results.append(
                ModelResult(
                    model_name=model_name,
                    prompt_id=bp.prompt_id,
                    response=response,
                    latency_seconds=latency,
                    keyword_coverage=coverage,
                    grounding=grounding,
                    estimated_cost_usd=cost,
                )
            )
    return results


def results_to_dataframe(results: list[ModelResult]) -> pd.DataFrame:
    """Converts ModelResult objects into a pandas DataFrame for analysis."""
    return pd.DataFrame([
        {
            "model": r.model_name,
            "prompt_id": r.prompt_id,
            "keyword_coverage": r.keyword_coverage,
            "grounding": r.grounding,
            "latency_ms": r.latency_seconds * 1000,
            "estimated_cost_usd": r.estimated_cost_usd,
        }
        for r in results
    ])


def latency_variance_by_model(df: pd.DataFrame) -> dict:
    """Uses numpy to compute latency variance per model -- useful for
    spotting non-deterministic/unstable models, like the real Ollama
    call versus instant, deterministic mocks."""
    return {
        model: float(np.var(group["latency_ms"].values))
        for model, group in df.groupby("model")
    }


def coverage_vs_grounding_agreement(coverages: list, groundings: list) -> float:
    """
    Uses scikit-learn's mean_squared_error to quantify how much the simple
    keyword_coverage metric agrees or disagrees with the grounding metric
    across the same responses -- a basic check for whether two scoring
    methods tell the same story.
    """
    return float(mean_squared_error(coverages, groundings))


def print_leaderboard(results: list[ModelResult]) -> None:
    by_model: dict[str, list[ModelResult]] = {}
    for r in results:
        by_model.setdefault(r.model_name, []).append(r)

    print("=== Per-prompt results ===\n")
    for r in results:
        print(f"[{r.model_name}] {r.prompt_id}: coverage={r.keyword_coverage:.2f}, "
              f"grounding={r.grounding:.2f}, latency={r.latency_seconds*1000:.3f}ms, "
              f"est_cost=${r.estimated_cost_usd:.6f}")
        print(f"    response: {r.response}\n")

    print("=== Leaderboard (avg keyword coverage across all prompts) ===\n")
    averages = {
        model: sum(r.keyword_coverage for r in rs) / len(rs)
        for model, rs in by_model.items()
    }
    for model, avg in sorted(averages.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {avg:.2f} average keyword coverage")

    print("\n=== Leaderboard (avg grounding score across all prompts) ===\n")
    grounding_averages = {
        model: sum(r.grounding for r in rs) / len(rs)
        for model, rs in by_model.items()
    }
    for model, avg in sorted(grounding_averages.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {avg:.2f} average grounding score")

    print("\n=== Estimated Total Cost by Model (disclosed estimate, not a real charge) ===\n")
    cost_totals = {
        model: sum(r.estimated_cost_usd for r in rs)
        for model, rs in by_model.items()
    }
    for model, total in sorted(cost_totals.items(), key=lambda x: x[1]):
        print(f"  {model}: ${total:.6f} estimated total")

    df = results_to_dataframe(results)
    print("\n=== Latency Variance by Model (numpy) ===\n")
    for model, var in latency_variance_by_model(df).items():
        print(f"  {model}: variance={var:.2f} ms^2")

    print("\n=== Coverage vs. Grounding Agreement (scikit-learn MSE) ===\n")
    all_coverage = [r.keyword_coverage for r in results]
    all_grounding = [r.grounding for r in results]
    mse = coverage_vs_grounding_agreement(all_coverage, all_grounding)
    print(f"  MSE between keyword_coverage and grounding: {mse:.4f}")


if __name__ == "__main__":
    print(
        "NOTE: mock-concise-v1 and mock-detailed-v1 responses are hardcoded "
        "mock strings, not real LLM output. ollama-llama3.2-1b is a real "
        "local LLM call via Ollama. Cost figures are estimates based on a "
        "disclosed reference rate, not real billed charges.\n"
    )
    results = run_benchmark()
    print_leaderboard(results)
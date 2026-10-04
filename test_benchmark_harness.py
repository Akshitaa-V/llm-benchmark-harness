from benchmark_harness import (
    score_keyword_coverage, score_grounding, estimate_cost,
    results_to_dataframe, ModelResult
)


def test_keyword_coverage_full_match():
    assert score_keyword_coverage(
        "protocol tools server client",
        ["protocol", "tools", "server", "client"]
    ) == 1.0


def test_keyword_coverage_no_match():
    assert score_keyword_coverage("hello world", ["protocol", "server"]) == 0.0


def test_grounding_partial_overlap():
    score = score_grounding(
        "This mentions protocol and server.",
        "The protocol connects a client to a server"
    )
    assert 0.0 < score <= 1.0


def test_grounding_no_overlap():
    score = score_grounding("Completely unrelated text here.", "protocol client server tools")
    assert score == 0.0


def test_estimate_cost_is_positive():
    cost = estimate_cost("short prompt", "a slightly longer generated response here")
    assert cost > 0


def test_estimate_cost_scales_with_length():
    short = estimate_cost("hi", "hi")
    long = estimate_cost("hi", "a much longer response with many more words in it than the short one")
    assert long > short


def test_results_to_dataframe_shape():
    results = [ModelResult(model_name="m1", prompt_id="p1", response="x",
                            latency_seconds=0.1, keyword_coverage=0.5,
                            grounding=0.4, estimated_cost_usd=0.001)]
    df = results_to_dataframe(results)
    assert len(df) == 1
    assert "latency_ms" in df.columns
    assert "estimated_cost_usd" in df.columns
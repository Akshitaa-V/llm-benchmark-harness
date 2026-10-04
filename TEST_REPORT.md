# Test Report — LLM Benchmark Harness

**Project:** LLM Benchmark Harness – Agent-Agnostic Evaluation Methodology
**Repository:** github.com/Akshitaa-V/llm-benchmark-harness
**Author:** Akshitaa Vijayakumar
**Related document:** TEST_PLAN.md

## 1. Summary
The full pytest suite (7 tests) covering scoring, grounding, cost-estimation, and data-aggregation logic was executed locally and via the GitHub Actions CI pipeline on push to the public repository. All test cases passed.

## 2. Results by Test Item

| ID | Component | Result | Notes |
|----|-----------|--------|-------|
| TP-01 | Keyword-coverage scoring | Pass | Ran against fixed, version-controlled task suite |
| TP-02 | Grounding scoring | Pass | Caught a case where the real model reused reference vocabulary while giving a factually incorrect answer — a gap the keyword-coverage metric alone missed |
| TP-03 | Cost estimation | Pass | Verified as part of full suite |
| TP-04 | Data aggregation | Pass | pandas/numpy used to aggregate results and compute latency variance across repeated runs |
| TP-05 | Real-model integration | Pass | Harness validated against Ollama llama3.2:1b alongside two baseline models, with no changes to scoring or leaderboard logic |
| TP-06 | Metric agreement | Pass | scikit-learn mean-squared-error used to measure agreement between grounding and keyword-coverage metrics |
| TP-07 | CI pipeline | Pass | GitHub Actions pipeline ran the suite on every push, passing on the public repository |

## 3. Findings
- **Non-determinism:** Repeated runs against the real model showed latency variance on the order of 10⁶ ms², versus zero variance for deterministic mocks — documented as expected behavior of a real model rather than a defect.
- **Metric gap identified:** The grounding metric caught at least one factually incorrect answer that the keyword-coverage metric alone did not flag, demonstrating the value of running both metrics together.

## 4. Defects Logged
None open. No test failures were observed in the current suite; the non-determinism finding above was treated as an expected characteristic of real-model output, not a defect, and was documented rather than filed as a bug.

## 5. Conclusion
The harness meets its test objectives: scoring and aggregation logic is verified correct, runs unmodified against real and mock model output, and is gated by CI on every push. Recommendation: suite is stable and ready to extend to additional models without further scoring-logic changes.

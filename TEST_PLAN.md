# Test Plan — LLM Benchmark Harness

**Project:** LLM Benchmark Harness – Agent-Agnostic Evaluation Methodology
**Repository:** github.com/Akshitaa-V/llm-benchmark-harness
**Author:** Akshitaa Vijayakumar
**Document type:** Test Plan

## 1. Scope
This test plan covers verification of the benchmarking harness's scoring, grounding, cost-estimation, and data-aggregation logic, across mock model outputs and a real local LLM (Ollama, llama3.2:1b).

## 2. Objectives
- Confirm scoring logic (keyword-coverage accuracy, grounding accuracy, latency, estimated cost) produces correct, reproducible results.
- Confirm the harness runs unmodified against a real model's output, not only mocks.
- Confirm the leaderboard aggregation logic is correct across multiple models.
- Confirm CI pipeline correctly gates changes on test suite results.

## 3. Test Items
| ID | Component | Description |
|----|-----------|-------------|
| TP-01 | Keyword-coverage scoring | Accuracy of keyword-match scoring against fixed task suite |
| TP-02 | Grounding scoring | Accuracy of grounding checks against full reference answers |
| TP-03 | Cost estimation | Correctness of cost-estimation logic |
| TP-04 | Data aggregation | pandas/numpy aggregation of results into leaderboard |
| TP-05 | Real-model integration | Harness run against Ollama llama3.2:1b without code changes |
| TP-06 | Metric agreement | scikit-learn MSE comparison between grounding and keyword-coverage metrics |
| TP-07 | CI pipeline | GitHub Actions pipeline executes full pytest suite on every push |

## 4. Test Environment
- Local Python environment
- Ollama running llama3.2:1b (fully offline, no API key)
- pytest as test runner
- GitHub Actions (public repository) for CI execution
- pandas, numpy, scikit-learn for metric computation

## 5. Pass / Fail Criteria
- **Pass:** All test cases in the pytest suite complete without failure; CI pipeline reports green on push.
- **Fail:** Any test case errors, or CI pipeline reports a failing run.

## 6. Test Deliverables
- pytest suite (7 tests) covering scoring, grounding, cost-estimation, and aggregation logic
- GitHub Actions CI configuration
- Test Report (see TEST_REPORT.md)

## 7. Out of Scope
- Testing against paid/hosted LLM APIs
- Load or stress testing of the harness itself

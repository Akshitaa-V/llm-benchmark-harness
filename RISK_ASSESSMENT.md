# Risk Assessment — LLM Benchmark Harness

**Project:** LLM Benchmark Harness – Agent-Agnostic Evaluation Methodology
**Repository:** github.com/Akshitaa-V/llm-benchmark-harness
**Author:** Akshitaa Vijayakumar
**Method:** Likelihood × Severity risk matrix (Low / Medium / High on each axis)

## 1. Purpose
This assessment identifies risks in the benchmarking harness's evaluation pipeline that could lead to incorrect or misleading model comparisons, and documents the mitigations already in place or recommended.

## 2. Risk Matrix

| ID | Risk | Likelihood | Severity | Overall Risk | Mitigation |
|----|------|-----------|----------|---------------|------------|
| R-01 | Keyword-coverage metric alone marks a factually incorrect answer as correct because it reuses reference vocabulary | Medium | High | High | Mitigated: grounding metric run alongside keyword-coverage; confirmed via scikit-learn MSE comparison that this exact failure mode was caught in testing |
| R-02 | Real-model output non-determinism causes inconsistent benchmark results between runs | High | Medium | High | Documented and quantified (latency variance ~10⁶ ms² vs. zero for mocks); recommend multi-trial averaging before drawing conclusions from a single run |
| R-03 | Scoring/aggregation logic silently breaks when a new model is swapped in | Low | High | Medium | Mitigated: harness explicitly designed so model output can be swapped without changing scoring/reporting logic; validated against Ollama llama3.2:1b without modification |
| R-04 | CI pipeline passes despite a real regression (false green) | Low | High | Medium | Partial mitigation: 7-test suite covers core logic; recommend adding a regression test whenever a new bug class is found (pattern already used elsewhere, e.g. hallucination-flagger project) |
| R-05 | Cost-estimation logic drifts from real-world pricing over time | Medium | Low | Low | Accepted risk: cost estimation is illustrative for comparison purposes, not billing-accurate; documented as such |

## 3. Overall Assessment
The highest risks (R-01, R-02) are both already mitigated through design choices already implemented in the harness — running two independent metrics and explicitly measuring/documenting non-determinism, rather than assuming a single run is representative. Remaining medium risks (R-03, R-04) are addressed by the harness's swap-in-place design; R-04 could be strengthened further with the regression-test-per-bug pattern already used in other projects.

## 4. Recommendation
No blocking risks identified for continued use of the harness for comparative model evaluation. Recommended next step: apply the regression-test-per-fix pattern going forward to keep R-04 low as the harness is extended to more models.

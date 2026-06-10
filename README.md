# Contribution 1: Fix `ndcg_score` returning an incorrect value when a sample has all-zero relevances

**Contribution Number:** 1
**Student:** Ahmed Khan
**Issue:** [scikit-learn/scikit-learn#29521 — NDCG in case of absence of relevant items](https://github.com/scikit-learn/scikit-learn/issues/29521)
**Repository:** [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) (Python / Cython / C++)
**Labels:** `Bug`, `help wanted`
**Status:** Phase I Complete

---

## Why I Chose This Issue

> _Personalize this — it's your voice. Draft below to build on._

I chose this issue because it sits right at the intersection of machine learning and clean software engineering, and scikit-learn is a library I already use and respect. The bug is small and well-bounded — a single ranking metric returning the wrong value in an edge case — which makes it a realistic first contribution, but fixing it correctly requires actually understanding the math behind NDCG (Normalized Discounted Cumulative Gain) and how scikit-learn structures its metrics code. That combination of "achievable scope" plus "real learning" is exactly what I want from my first open-source contribution.

I'm also drawn to it because contributing to scikit-learn means working against a mature, well-tested codebase with high review standards. I want to learn how a project of this caliber expects bugs to be fixed: how they want edge cases handled, how they write and structure tests, and how maintainers reason about backward compatibility. I hope to come out of this more confident reading unfamiliar production Python and more fluent in the discipline of fixing a bug *with* a regression test rather than just patching the symptom.

---

## Understanding the Issue

### Problem Description

`sklearn.metrics.ndcg_score` returns the wrong value when one of the samples has **all-zero relevance scores** (i.e. a query with no relevant items). In that situation a perfect prediction no longer scores 1.0, which contradicts the definition of NDCG, where 1.0 represents ideal ranking performance.

### Expected Behavior

`ndcg_score(y, y)` should **always equal 1.0** — scoring a set of true relevances against itself is a perfect ranking, so the normalized score must be 1.

### Current Behavior

When a sample has all-zero relevances, that sample contributes 0.0 to the averaged score instead of being treated as a perfect/ideal case. Averaged with a normal perfect sample, this drags the result below 1.0:

```python
from sklearn.metrics import ndcg_score
import numpy as np

y = np.array([[1.0, 0.0, 1.0],
              [0.0, 0.0, 0.0]])   # second sample has no relevant items
ndcg_score(y, y)                  # returns 0.5  ❌  (expected 1.0)
```

The all-zero sample yields DCG = 0 and ideal DCG = 0; the `0/0` case is currently resolved to 0.0 rather than to a perfect score, and the mean of `[1.0, 0.0]` gives `0.5`. Confirmed present since scikit-learn 1.3.1.

### Affected Components

- **File:** `sklearn/metrics/_ranking.py`
- **Functions:** `ndcg_score`, `_ndcg_sample_scores`, `_dcg_sample_scores`
- Associated tests live in `sklearn/metrics/tests/test_ranking.py`.

---

## Reproduction Process

### Environment Setup

<mark>[Phase II — notes on building scikit-learn from source: clone, create a virtual environment, `pip install -e .`, run the test suite. Document any setup challenges and how you solved them.]</mark>

### Steps to Reproduce

1. Install scikit-learn (≥ 1.3.1) in a fresh environment.
2. Run:
   ```python
   from sklearn.metrics import ndcg_score
   import numpy as np
   y = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
   print(ndcg_score(y, y))
   ```
3. **Observed result:** `0.5` is printed. **Expected:** `1.0`.

### Reproduction Evidence

- **Commit showing reproduction:** <mark>[Phase II — link to a commit/branch in your fork with a failing test that reproduces this]</mark>
- **Screenshots/logs:** <mark>[Phase II — terminal output showing 0.5]</mark>
- **My findings:** <mark>[Phase II — what you confirm during reproduction]</mark>

---

## Solution Approach

### Analysis

The root cause is how `_ndcg_sample_scores` normalizes a sample whose ideal DCG is 0. For a query with no relevant items, both the actual DCG and the ideal DCG are 0, making NDCG mathematically undefined (`0/0`). scikit-learn currently resolves this to `0.0`, which is then averaged in and pulls the overall score below 1.0.

### Proposed Solution

> _Final direction is pending maintainer confirmation on the issue thread — see the claiming comment._

High-level options under consideration:
1. **Exclude** all-zero-relevance samples from the averaged score and emit a warning that they were ignored.
2. **Define** the NDCG of an all-zero-relevance sample as `1.0` (a query with nothing relevant is trivially "perfectly" ranked).
3. **Warn** the user that the metric is undefined for such samples.

The issue reporter (@arabel1a) suggested options (1) or (3). I will propose a concrete approach in my claiming comment and confirm with a maintainer **before** implementing, since changing a published metric's output has backward-compatibility implications.

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** `ndcg_score(y, y)` must return 1.0, but returns 0.5 when a sample has all-zero relevances, because the `0/0` normalization for that sample is treated as 0.0.

**Match:** Review how other scikit-learn metrics (e.g. those in `_ranking.py` and `_classification.py`) handle undefined-per-sample edge cases — many emit an `UndefinedMetricWarning` and substitute a defined value. Mirror that established pattern.

**Plan:**
1. Add a failing regression test in `sklearn/metrics/tests/test_ranking.py` reproducing the `0.5` result.
2. Update `_ndcg_sample_scores` / `_dcg_sample_scores` in `sklearn/metrics/_ranking.py` to handle the all-zero-relevance sample per the agreed direction.
3. Add/adjust tests and update the docstring of `ndcg_score` to document the edge-case behavior.

**Implement:** <mark>[Phase III — link to your branch/commits as you work]</mark>

**Review:** <mark>[Phase III — self-review against scikit-learn's contributing guidelines + the PR checklist]</mark>

**Evaluate:** <mark>[Phase III — confirm `ndcg_score(y, y) == 1.0` and the full ranking test suite passes]</mark>

---

## Testing Strategy

### Unit Tests

- [ ] `ndcg_score(y, y) == 1.0` when a sample has all-zero relevances (the regression case)
- [ ] All-zero-relevance handling matches the agreed direction (excluded-with-warning, or defined value)
- [ ] Existing `ndcg_score` / `dcg_score` tests still pass (no behavior change for normal inputs)

### Integration Tests

- [ ] `GridSearchCV` / cross-validation using `ndcg_score` as the scorer no longer crashes or mis-scores on all-zero-relevance folds
- [ ] <mark>[Integration scenario 2]</mark>

### Manual Testing

<mark>[Phase III — manual checks and results]</mark>

---

## Implementation Notes

### Week 1 Progress (Phase I — Issue Selection)

Selected issue #29521 after running the Step 5 selection checklist. Verified it is open, unassigned, and **PR-free** (checked the GitHub Development panel and cross-referenced PRs via the API), with recent activity (last updated April 2026). Confirmed the reproduction snippet returns `0.5`. Claimed it on the cohort sheet and commented on the GitHub issue proposing a fix direction.

### Week [Y] Progress

<mark>[Continue documenting as you work]</mark>

### Code Changes

- **Files modified:** <mark>[List — expected: `sklearn/metrics/_ranking.py`, `sklearn/metrics/tests/test_ranking.py`]</mark>
- **Key commits:** <mark>[Links to important commits]</mark>
- **Approach decisions:** <mark>[Why you chose certain approaches]</mark>

---

## Pull Request

**PR Link:** <mark>[GitHub PR URL when submitted]</mark>

**PR Description:** <mark>[Draft or final PR description — much of the content above can be adapted]</mark>

**Maintainer Feedback:**
- <mark>[Date]</mark>: <mark>[Summary of feedback received]</mark>
- <mark>[Date]</mark>: <mark>[How you addressed it]</mark>

**Status:** <mark>[Awaiting review / Iterating / Approved / Merged]</mark>

---

## Learnings & Reflections

### Technical Skills Gained

<mark>[What you learned technically]</mark>

### Challenges Overcome

<mark>[What was hard and how you solved it]</mark>

### What I'd Do Differently Next Time

<mark>[Reflection on your process]</mark>

---

## Resources Used

- [scikit-learn issue #29521](https://github.com/scikit-learn/scikit-learn/issues/29521) — the issue being fixed
- [scikit-learn Contributing Guide](https://scikit-learn.org/stable/developers/contributing.html) — dev setup and PR process
- [NDCG — Wikipedia](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) — metric definition reference
- <mark>[Add tutorials / discussions that helped as you go]</mark>

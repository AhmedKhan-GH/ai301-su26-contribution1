# Contribution 1: Fix `ndcg_score` returning an incorrect value when a sample has all-zero relevances

**Contribution Number:** 1
**Student:** Ahmed Khan
**Issue:** [scikit-learn/scikit-learn#29521 — NDCG in case of absence of relevant items](https://github.com/scikit-learn/scikit-learn/issues/29521)
**Repository:** [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) (Python / Cython / C++)
**Labels:** `Bug`, `help wanted`
**Status:** Phase III Complete

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

scikit-learn ships compiled Cython/C++ extensions, so it must be **built from source** — a plain `pip install scikit-learn` would install the released package, not my editable working copy. I built it on macOS (Apple Silicon, Python 3.14) following the project's [Building from source](https://scikit-learn.org/stable/developers/advanced_installation.html) guide.

**Prerequisites (already satisfied on my machine):**
- **C/C++ compiler** — Xcode command line tools (`xcode-select -p` → `/Applications/Xcode.app/...`).
- **OpenMP runtime** — Homebrew `libomp` (`brew install libomp`). This is the classic macOS scikit-learn gotcha; because it was already installed, meson auto-detected it and the build needed no extra `CFLAGS`/`LDFLAGS`.

**Build steps:**
```bash
# from the scikit-learn repo root
python3 -m venv .venv
source .venv/bin/activate              # IMPORTANT — see the meson error below
pip install --upgrade pip
pip install numpy scipy cython meson-python ninja pytest   # build + test deps
pip install --editable . --no-build-isolation              # compiles the extensions (one time)
```

**Errors hit and how I fixed them:**

| Error | Cause | Fix |
|---|---|---|
| `meson-python: error: meson executable "meson" not found` during the editable build | I invoked `pip` by its venv path *without activating the venv*, so the venv's `bin/` (which holds the `meson`/`ninja` console scripts) was not on `PATH`, and `meson-python` invokes `meson` by name | Activate the venv (`source .venv/bin/activate`), or otherwise put `.venv/bin` on `PATH`, **before** building so `meson`/`ninja` resolve |

**Python 3.14 note:** 3.14 is very new (Oct 2025), but numpy/scipy/cython all ship `cp314` wheels, so the build dependencies installed as prebuilt wheels (no compiling) and the build succeeded — no need to downgrade Python.

**Verification the source build is active:**
```bash
python -c "import sklearn; print(sklearn.__version__, sklearn.__file__)"
# 1.10.dev0 /Users/ahmed/PycharmProjects/scikit-learn/sklearn/__init__.py
```
The `dev0` version and the in-repo path confirm Python imports my editable working copy. Because the fix lives in `_ranking.py` (**pure Python**), edits take effect on the next `import` with no recompile.

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

- **Working branch:** [`fix-issue-29521`](https://github.com/AhmedKhan-GH/scikit-learn/tree/fix-issue-29521) in my fork (`AhmedKhan-GH/scikit-learn`).
- **Terminal output (against the source build, `1.10.dev0`):**
  ```text
  ndcg_score(y, y) = 0.5  (expected 1.0)
  ```
- **My findings:**
  - The bug **still reproduces on the latest `main` (`1.10.dev0`)** — no fix has been merged — not just the 1.3.1 release the issue mentions.
  - It is **deterministic** (`ndcg_score` is a pure function of its inputs): the same input yields `0.5` on every run, so the "reproduce at least twice" check is satisfied by construction.
  - **Root cause located** in `sklearn/metrics/_ranking.py` → `_ndcg_sample_scores` (lines ~1944–1945): when the ideal DCG is 0 (`all_irrelevant`), the per-sample score is forced to `0` (`gain[all_irrelevant] = 0`) instead of being treated as undefined, so the all-zero sample contributes `0.0` and drags the mean to `0.5`.
  - **Open-source contention (discovered during reproduction):** although no fix is *merged*, the issue is **actively being worked** — an open PR (**#34244**, `Fixes #29521`) already implements the maintainer-endorsed approach, and four earlier PRs (#30156, #33199, #33374, #33816) attempted it and were closed. See *Proposed Solution* for the Phase III implication.
- **Failing regression test:** Not required for the Phase II deliverables (Step 5 does not list it); it would be the first step of Phase III implementation.

---

## Solution Approach

### Analysis

The root cause is how `_ndcg_sample_scores` normalizes a sample whose ideal DCG is 0. For a query with no relevant items, both the actual DCG and the ideal DCG are 0, making NDCG mathematically undefined (`0/0`). scikit-learn currently resolves this to `0.0`, which is then averaged in and pulls the overall score below 1.0.

### Proposed Solution

**Maintainer-decided direction (confirmed from the issue thread):** rather than any single naive fix, the maintainers (glemaitre, ogrisel) converged on adding a **`replaced_undefined_by` parameter** to `ndcg_score`. The all-zero / `0/0` sample is treated as *undefined* — returning **`np.nan` with an `UndefinedMetricWarning`** by default — and the caller may pass `0` or `1` instead. This mirrors the established `zero_division` pattern in `sklearn/metrics/_classification.py` (`_check_zero_division`) and the cross-metric standardization tracked in #29048 (now closed, 2026-04-08), which unblocked the work.

The three options originally considered (now superseded by the parameter approach):
1. **Exclude** all-zero-relevance samples from the averaged score and warn.
2. **Define** the NDCG of an all-zero-relevance sample as `1.0`.
3. **Warn** that the metric is undefined for such samples.

**Status of the fix (discovered during reproduction):** this is no longer an open target for an independent fix — an **open PR #34244 (`Fixes #29521`)** already implements the `replaced_undefined_by` approach, and four prior PRs were closed. For **Phase III** I will therefore either (a) pivot to a genuinely-available issue, or (b) contribute by *reviewing and testing* PR #34244 — a recognized contribution under scikit-learn's guidelines. This decision will be finalized at the start of Phase III.

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** `ndcg_score(y, y)` must return 1.0, but returns 0.5 when a sample has all-zero relevances, because the `0/0` normalization for that sample is treated as 0.0.

**Match:** The established precedent is the `zero_division` handling in `sklearn/metrics/_classification.py` — `_check_zero_division` (around line 65) validates a `"warn"`/`0`/`1` argument and emits an `UndefinedMetricWarning` (defined in `sklearn/exceptions.py:140`) when a metric is undefined for a sample. The agreed `replaced_undefined_by` parameter for `ndcg_score` mirrors this pattern, and new public parameters are declared in the `@validate_params` decorator on `ndcg_score` (`_ranking.py:1950`).

**Plan:**
1. Add a failing regression test in `sklearn/metrics/tests/test_ranking.py` reproducing the `0.5` result.
2. Update `_ndcg_sample_scores` / `_dcg_sample_scores` in `sklearn/metrics/_ranking.py` to handle the all-zero-relevance sample per the agreed direction.
3. Add/adjust tests and update the docstring of `ndcg_score` to document the edge-case behavior.

**Implement:** Done in Phase III. Commit [`650b206`](https://github.com/AhmedKhan-GH/scikit-learn/commit/650b206b7a) on branch [`fix-issue-29521`](https://github.com/AhmedKhan-GH/scikit-learn/tree/fix-issue-29521) — `ENH Add replaced_undefined_by parameter to ndcg_score (#29521)`. Built the fix test-first (TDD): three regression tests committed red, then the implementation made them green with no regressions across `test_ranking.py` (251 passed).

**Review:** I reviewed scikit-learn's contribution conventions (`CONTRIBUTING.md` → `doc/developers/contributing.rst`) so I'm prepared for Phase III/IV:
- **Commit / PR title prefixes** tag the change type — `FIX`, `ENH`, `MNT`, `DOC`, `TST` (e.g. existing PRs read `ENH Add replaced_undefined_by parameter…`, `FIX Exclude all-zero relevance…`).
- **PR template** (`.github/PULL_REQUEST_TEMPLATE.md`) requires referencing the issue (`Fixes #29521`) and describing the change.
- **Changelog:** user-facing changes need a fragment under `doc/whats_new/upcoming_changes/`.
- **Style & checks:** code is formatted/linted with `ruff` (`build_tools/linting.sh`); public params need `@validate_params` entries; all changes need tests passing under `pytest`.

**Evaluate:** The fix will be validated (see *Testing Strategy* below) by: a regression test asserting the all-zero-relevance sample yields `np.nan` + an `UndefinedMetricWarning` by default (and `1.0` when `replaced_undefined_by=1`); confirming the existing `ndcg_score`/`dcg_score` tests still pass (no change for normal inputs); and running the full `pytest sklearn/metrics/tests/test_ranking.py` suite. (Execution happens in Phase III.)

---

## Testing Strategy

### Unit Tests

Added to `sklearn/metrics/tests/test_ranking.py` (all written **before** the fix, per TDD):

- [x] `test_ndcg_all_zero_relevance_undefined_by_default` — an all-zero-relevance sample yields `np.nan` + `UndefinedMetricWarning` by default
- [x] `test_ndcg_all_zero_relevance_replaced_undefined_by` — `replaced_undefined_by=1.0` makes that sample score `1.0` (so `ndcg_score(y, y) == 1.0` for the reported case); `=0.0` reproduces the old `0.5`
- [x] `test_ndcg_replaced_undefined_by_invalid_value` — an out-of-domain value (e.g. `"bad"`) is rejected by `@validate_params`
- [x] Existing `ndcg_score` / `dcg_score` tests still pass — full `test_ranking.py` is **251 passed** (updated the internal helper `_test_ndcg_score_for` to pass `replaced_undefined_by=0.0`, preserving its original intent now that the default is `np.nan`)

### Integration / Wider-Suite Tests

- [x] No regressions in the metric-scorer and common-metric suites: `test_common.py` + `test_score_objects.py` ndcg/dcg subset is **18 passed, 4 skipped**
- [x] `ruff format --check` clean on both changed files; `ruff check` reports nothing on the changed lines (only pre-existing findings in untouched code under a newer-than-pinned local ruff)

### Manual Testing

Ran the reported snippet against the source build (`1.10.dev0`):

```text
ndcg_score(y, y)                       -> nan   (+ UndefinedMetricWarning)   # was 0.5
ndcg_score(y, y, replaced_undefined_by=1.0) -> 1.0
ndcg_score(y, y, replaced_undefined_by=0.0) -> 0.5
ndcg_score([[10,0,0,1,5]], [[.1,.2,.3,4,70]]) -> 0.69  (docstring example unchanged)
```

---

## Implementation Notes

### Week 1 Progress (Phase I — Issue Selection)

Selected issue #29521 after running the Step 5 selection checklist. Verified it is open, unassigned, and **PR-free** (checked the GitHub Development panel and cross-referenced PRs via the API), with recent activity (last updated April 2026). Confirmed the reproduction snippet returns `0.5`. Claimed it on the cohort sheet and commented on the GitHub issue proposing a fix direction.

### Week 2 Progress (Phase II — Reproduce & Plan)

- **Built scikit-learn from source** (editable, `1.10.dev0`) in a Python 3.14 virtual environment; hit and fixed a `meson not found` PATH error (documented under Environment Setup).
- **Created and pushed the working branch** `fix-issue-29521` to my fork.
- **Reproduced** the bug against the source build (`ndcg_score(y, y) = 0.5`) and **located the root cause** at `_ranking.py:1944–1945`.
- **Investigated the issue thread** and found (a) the maintainer-decided direction (`replaced_undefined_by`, `nan` + warning), and (b) that the fix is already covered by **open PR #34244** plus four closed PRs — a real lesson in open-source contention that shapes the Phase III plan (pivot or review).

### Week 3 Progress (Phase III — Implement & Test)

- **Confirmed the issue is still live and contested** before coding: issue #29521 is still open/unassigned, the bug still reproduces on `main` (`ndcg_score(y, y) = 0.5`), and the maintainer-endorsed fix already exists in **open PR #34244** (0 maintainer reviews in ~11 days) with four prior PRs closed. Decision: implement my own attempted fix on my fork branch to satisfy the Phase III deliverables (which require a *working, tested, local* fix — not a *merged* one); the upstream-PR strategy is a Phase IV question given the contention.
- **Implemented the fix test-first (TDD):** wrote three failing tests, watched them fail for the right reasons (`DID NOT WARN`; unknown kwarg), then implemented until green.
- **Mirrored the established `zero_division` convention** rather than inventing an API: same `Options(Real, {0.0, 1.0}), "nan"` validation pattern used across `_classification.py`.
- **Verified no regressions** (`test_ranking.py` 251 passed; scorer/common ndcg subset 18 passed) and committed + pushed to the fork.

### Code Changes

- **Files modified:**
  - `sklearn/metrics/_ranking.py` — added `replaced_undefined_by` to `ndcg_score` (signature, `@validate_params`, docstring) and `_ndcg_sample_scores`; replaced the `gain[all_irrelevant] = 0` root-cause line with the caller-chosen fill value; emit `UndefinedMetricWarning` when undefined samples remain `np.nan`.
  - `sklearn/metrics/tests/test_ranking.py` — three new regression tests; updated `_test_ndcg_score_for` helper to pass `replaced_undefined_by=0.0`.
  - `doc/whats_new/upcoming_changes/sklearn.metrics/29521.enhancement.rst` — towncrier changelog fragment.
- **Key commit:** [`650b206`](https://github.com/AhmedKhan-GH/scikit-learn/commit/650b206b7a) — `ENH Add replaced_undefined_by parameter to ndcg_score (#29521)`.
- **Approach decisions:**
  - **New parameter over a hard-coded rule.** Returning `1.0` for all-zero samples (the README's original Phase II thesis) is one defensible convention, but excluding them or warning is equally defensible — so the maintainers chose to let the *caller* decide. Defaulting to `np.nan` + warning makes the undefined case loud instead of silently wrong, and `replaced_undefined_by=1.0` recovers the intuitive `ndcg_score(y, y) == 1.0`.
  - **Warn in the public `ndcg_score`, not in `_ndcg_sample_scores`.** Keeps the private helper side-effect-free so existing internal/test callers don't emit spurious warnings; the warning fires exactly when an undefined sample survives as `np.nan`.

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

- Reading and extending a mature metrics module (`_ranking.py`) and following its conventions: `@validate_params` constraints (`Options`, `Interval`, the `"nan"` sentinel), `UndefinedMetricWarning`, towncrier changelog fragments, and the `ruff` toolchain.
- Practicing strict TDD on a real codebase — writing tests that fail for the *right* reason before any production code.

### Challenges Overcome

- **The fix direction wasn't a single "right" answer.** NDCG of an all-zero sample is genuinely undefined (`0/0`), and "return 1.0", "exclude it", and "warn" are all defensible. I resolved this by following the maintainers' precedent — the `zero_division` pattern in `_classification.py` — and exposing the choice to the caller via `replaced_undefined_by`, defaulting to a loud `np.nan` + warning.
- **A behavior change rippled into an existing test.** Switching the undefined fill from `0` to `np.nan` broke `_test_ndcg_score_for` (its `score <= ideal` and `== 0` assertions assumed the old `0` fill). I traced every caller of `_ndcg_sample_scores` first, then updated that helper to request `replaced_undefined_by=0.0`, preserving its original intent.
- **Linter noise from a version mismatch.** A freshly-`pip install`ed `ruff` (0.15) flagged 29 findings — all in pre-existing untouched code, none on my lines. I confirmed the project only floors `ruff>=0.12.2` and resisted "fixing" unrelated code (scope discipline).

### What I'd Do Differently Next Time

<mark>[Reflection on your process]</mark>

---

## Resources Used

- [scikit-learn issue #29521](https://github.com/scikit-learn/scikit-learn/issues/29521) — the issue being fixed
- [scikit-learn Contributing Guide](https://scikit-learn.org/stable/developers/contributing.html) — dev setup and PR process
- [NDCG — Wikipedia](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) — metric definition reference
- <mark>[Add tutorials / discussions that helped as you go]</mark>

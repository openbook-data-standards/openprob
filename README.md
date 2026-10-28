# OpenProb

Open tools for problems, probabilities and predictions.

Experimental provider-neutral forecasting and evaluation tools in the OpenBook Data Standards family, licensed under Apache-2.0 (see `LICENSE`). Version 0.1.0 is a Python draft, not a published interoperability standard or a claim of OpenBook/OpenVenue conformance. Package-name lookup is not trademark clearance.

## Boundary

Public: neutral contracts, finite count distributions, scoring tools and synthetic examples. Private implementations may retain their models, weights, licensed data, vendor mappings and strategies. No provider client, real feed payload, credential, account handling or execution code belongs here. Existing OpenBook, OpenVenue, OpenExchange and OpenTrack projects are unchanged.

## Local use

Python 3.11+, standard library at runtime. No cloud, database or credentials required.

```sh
PYTHONPATH=src python3 -B -m openprob
```

The synthetic example prints forecasts and an illustrative evaluation to stdout. It does not fetch data or write an archive. Its multiple thresholds are correlated and are not an independent performance sample.

Install from a local checkout with `python3 -m pip install --no-deps --no-build-isolation .` in your chosen environment. This is not yet a PyPI release.

## Draft contracts

- `Target`: caller-owned target/event identifiers, proposition, explicit settlement basis and timezone-aware outcome time. IDs are opaque; this package does not resolve cross-provider identity.
- `Forecast`: versioned model, unique forecast ID, target, generated time, information/training cutoffs, probability or explicit abstention. Optional bounds describe a probability range, not an automatically justified confidence interval.
- `Label`: matching target, boolean outcome and time the result became available. Unsettled/void/disputed results are not coerced to false; omit their labels and report them as unresolved until separately supported.
- `CountDistribution`: immutable finite probability vector for counts 0..N. `at_least(k)` derives coherent threshold probabilities. Tail probability above N is zero by construction: choose support honestly. The empirical baseline does not extrapolate unseen outcomes or estimate uncertainty.

Forecast chronology requires training cutoff <= information cutoff <= generation < target outcome time. Evaluation uses only labels available by its cutoff, requires identical target definitions, rejects duplicate targets/forecasts/labels, and scores one model version at a time. These checks cannot prove feature timestamps or source claims are truthful. Early-settled targets require an appropriately defined outcome time; this draft is for pre-outcome forecasting, not in-play execution.

## Evaluation

`evaluate(forecasts, labels, as_of, bins=10)` reports:

- Total, abstained, unresolved and scored counts.
- Forecast coverage: non-abstained / total, including unresolved forecasts.
- Mean binary Brier score (lower is better).
- Mean natural-log loss and count of infinite losses. Wrong certainty is not silently clipped: if any loss is infinite, `mean_log_loss` is null and `infinite_log_losses` is positive.
- Equal-width calibration bins and sample-weighted expected calibration error. Probability 1 belongs in the final bin.

Empty scores are null, never zero performance. Metrics are descriptive: they do not establish statistical significance, independence, model skill, attainable returns or profitability. No market/venue fee modeling is performed in this public package.

For an honest study, freeze the model before a temporal holdout, supply only features available at prediction time, compare against an explicit baseline on identical targets, stratify forecast horizons, account for correlated events, and disclose missing/void outcomes and source revisions. No retained evidence means limited reproducibility; storage and data rights remain the consumer's decision.

## Contributor workflow

Code: `src/openprob/`. Synthetic contract cases: `tests/test_openprob.py`.

Lint: `ruff check src tests`; format check: `ruff format --check src tests`. Do not run tests locally. CI execution and remote publication require owner authorization; no CI workflow is installed by this initial local build. No hosted model-code execution, network calls, tracking or recurring jobs are included.

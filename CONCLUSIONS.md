# Gold & Silver Volatility Forecasting — Project Conclusions

## What was built

1. **Data pipeline** — daily gold and silver futures prices from Yahoo Finance, 2010–2026, converted to daily returns.
2. **GARCH(1,1) baseline** — fitted, forecasted, and honestly backtested out-of-sample for both metals.
3. **A transformer model, built from scratch in PyTorch** — windowed data pipeline, self-attention architecture, trained, debugged, and rigorously compared against GARCH and simple baselines, for both metals.
4. **`src/` and `tests/`** — the windowing logic and model architecture extracted into reusable modules (`src/windowing.py`, `src/model.py`), with a pytest suite (9 tests) guarding shapes, output positivity, no-lookahead-leakage in the windowing, and the model's parameter count.

## The GARCH baseline results

| | Gold | Silver |
|---|---|---|
| omega (baseline floor) | 0.019 | 0.041 |
| alpha (reaction to shocks) | 0.053 | 0.051 |
| beta (persistence) | 0.931 | 0.940 |
| Backtest RMSE improvement over naive | +2.63% | +3.55% |

Both metals showed real, if modest, volatility clustering that GARCH could exploit — confirming the core premise of the whole project before the transformer was ever built.

## The transformer: what actually happened, honestly (gold)

**First training attempt collapsed.** After 100 epochs, the model had essentially learned to predict close to the average volatility regardless of input — its predictions varied only ~9% as much as real volatility does. Diagnosed concretely by comparing prediction spread to actual spread, not just by eyeballing loss curves.

**Lowering the learning rate didn't fix it** (9% → 11% of real spread — no meaningful change). This ruled out the first, most obvious hypothesis.

**A baseline sanity check clarified what was really going on.** Testing a plain linear regression on the same data revealed that even the simplest possible model — one feature, "recent realized volatility predicts tomorrow's volatility" — could only beat naive by about 0.5% on that particular validation period. This showed the "collapse" wasn't a transformer-specific bug; it was a response to a genuinely faint, noisy signal in that stretch of data.

**A methodology bug was caught and fixed.** The first head-to-head comparison against GARCH used mismatched time periods (transformer scored on 2021–2024, GARCH scored on 2023–2026) — an unfair comparison. Re-scoring on GARCH's *exact* test period initially showed the transformer beating GARCH — but that comparison still had a subtle leak: part of GARCH's test period had been used as the transformer's *validation* set for picking its best training checkpoint. Retraining with the test boundary pinned to the exact same date GARCH used (no validation/test overlap at all) produced the final, fully clean result.

**Silver's transformer was built applying these lessons directly** — correct date-aligned split and the fixed learning rate from the start — and confirmed the same pattern reliably reproduces on a second, independent asset.

## The final, honest scoreboard — both metals (test period: 2023-04-12 to 2026-08-05, zero data leakage)

**Gold:**

| Model | RMSE | Improvement over naive |
|---|---|---|
| Linear regression (1 feature: recent realized vol) | 0.8786 | **+6.08%** |
| GARCH(1,1) | 0.9032 | +3.45% |
| Transformer | 0.9046 | +3.30% |
| Naive (flat average) | 0.9355 | 0% |
| Linear regression (30 raw values) | 0.9394 | -0.42% |

**Silver:**

| Model | RMSE | Improvement over naive |
|---|---|---|
| Linear regression (1 feature: recent realized vol) | 1.8918 | **+10.63%** |
| GARCH(1,1) | 2.0044 | +5.31% |
| Transformer | 2.0116 | +4.97% |
| Linear regression (30 raw values) | 2.1063 | +0.50% |
| Naive (flat average) | 2.1168 | 0% |

## What this actually means

- **The pattern is consistent and reproducible across two independent assets**, which makes it trustworthy rather than a fluke of one dataset. In both cases: `Linear (1 feature) > GARCH > Transformer > Naive`, with GARCH and the transformer landing within a fraction of a percent of each other every time.
- **The transformer and GARCH are essentially tied** — GARCH marginally ahead in both metals, well within the noise of a single test period. The transformer's added complexity (17,000+ parameters, attention, multi-day context) did not translate into a meaningful edge over a model with just 4 parameters.
- **A one-line linear regression beat both of them, by a clear margin, on both metals.** The single strongest predictor of tomorrow's volatility, among everything tried, was simply "how volatile was this metal recently" — fed through the simplest possible model. Silver's edge for this approach (+10.63%) was even larger than gold's (+6.08%).
- **This is a real, legitimate finding, not a failed project.** It's a concrete, twice-replicated demonstration of the bias-variance tradeoff: on a dataset this size (~2,800–3,300 training examples), a flexible, high-capacity model has no inherent advantage over models with the right structural assumptions already built in — and can actually underperform a well-chosen simple baseline.
- **All three real models (linear, GARCH, transformer) beat naive on both metals**, confirming real, learnable volatility clustering exists in daily precious-metal returns — just that a plain transformer, at this scale of data, isn't the most efficient way to capture it.

## Practical lessons from the process itself

- **Fitting ≠ forecasting ≠ backtesting.** All three are distinct, necessary steps — a model that fits history well tells you nothing about its forecasting ability until it's honestly tested on unseen data.
- **Diagnose failures with evidence, not guesses.** The first fix attempt (lowering the learning rate) was a reasonable hypothesis that turned out to be wrong — caught quickly by checking prediction spread directly, rather than assuming the fix worked because loss numbers moved slightly.
- **Data leakage is subtle and easy to miss in time series**, even when a train/test split looks correct. Using a validation set for checkpoint selection during training is standard practice — but if that validation period overlaps with the actual comparison window, it quietly inflates results.
- **Always benchmark against naive and simple baselines**, not just the sophisticated model. The simplest model in this whole project turned out to be the best one on both assets tested.
- **Replicate before trusting a result.** The gold-only finding could plausibly have been a one-off quirk of that specific test period; running the identical process on silver and seeing the same ranking hold up is what turns "interesting result" into "reliable conclusion."

## Project status: complete

- Data pipeline, GARCH baseline (fit/forecast/backtest), and transformer (data prep/build/train/diagnose/evaluate) are all done for **both gold and silver**.
- `src/` holds the reusable windowing and model code; `tests/` has a passing pytest suite (9/9) guarding both.
- Every notebook runs end-to-end and is version-controlled on GitHub.

## Possible future extensions (not required, not started)

- Features with exponentially-decaying weight (closer to what GARCH does structurally) instead of a flat 30-day window.
- Asymmetric GARCH variants (GJR-GARCH/EGARCH) to test whether the "big drops matter more than big rallies" effect adds anything here.
- A longer input window or more training data, to see whether the transformer's disadvantage narrows with scale.

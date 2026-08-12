# Gold Volatility Forecasting — Project Conclusions

## What was built

1. **Data pipeline** — daily gold (and silver) futures prices from Yahoo Finance, 2010–2026, converted to daily returns.
2. **GARCH(1,1) baseline** — fitted, forecasted, and honestly backtested out-of-sample for both metals.
3. **A transformer model, built from scratch in PyTorch** — windowed data pipeline, self-attention architecture, trained, debugged, and rigorously compared against GARCH and simple baselines for gold.

## The GARCH baseline results

| | Gold | Silver |
|---|---|---|
| omega (baseline floor) | 0.019 | 0.041 |
| alpha (reaction to shocks) | 0.053 | 0.051 |
| beta (persistence) | 0.931 | 0.940 |
| Backtest RMSE improvement over naive | +2.63% | +3.55% |

Both metals showed real, if modest, volatility clustering that GARCH could exploit — confirming the core premise of the whole project before the transformer was ever built.

## The transformer: what actually happened, honestly

**First training attempt collapsed.** After 100 epochs, the model had essentially learned to predict close to the average volatility regardless of input — its predictions varied only ~9% as much as real volatility does. Diagnosed concretely by comparing prediction spread to actual spread, not just by eyeballing loss curves.

**Lowering the learning rate didn't fix it** (9% → 11% of real spread — no meaningful change). This ruled out the first, most obvious hypothesis.

**A baseline sanity check clarified what was really going on.** Testing a plain linear regression on the same data revealed that even the simplest possible model — one feature, "recent realized volatility predicts tomorrow's volatility" — could only beat naive by about 0.5% on that particular validation period. This showed the "collapse" wasn't a transformer-specific bug; it was a response to a genuinely faint, noisy signal in that stretch of data.

**A methodology bug was caught and fixed.** The first head-to-head comparison against GARCH used mismatched time periods (transformer scored on 2021–2024, GARCH scored on 2023–2026) — an unfair comparison that made the transformer look worse than it should have. Re-scoring on GARCH's *exact* test period initially showed the transformer beating GARCH — but that comparison still had a subtle leak: part of GARCH's test period had been used as the transformer's *validation* set for picking its best training checkpoint. Retraining with the test boundary pinned to the exact same date GARCH used (no validation/test overlap at all) produced the final, fully clean result.

## The final, honest scoreboard (same test period: 2023-04-12 to 2026-08-05, zero data leakage)

| Model | RMSE | Improvement over naive |
|---|---|---|
| Linear regression (1 feature: recent realized vol) | 0.8786 | **+6.08%** |
| GARCH(1,1) | 0.9032 | +3.45% |
| Transformer | 0.9046 | +3.30% |
| Naive (flat average) | 0.9355 | 0% |
| Linear regression (30 raw values) | 0.9394 | -0.42% |

## What this actually means

- **The transformer and GARCH are essentially tied** — GARCH marginally ahead, well within the noise of a single test period. The transformer's added complexity (17,000+ parameters, attention, multi-day context) did not translate into a meaningful edge over a model with just 4 parameters.
- **A one-line linear regression beat both of them.** The single strongest predictor of tomorrow's gold volatility, among everything tried, was simply "how volatile was gold recently" — fed through the simplest possible model.
- **This is a real, legitimate finding, not a failed project.** It's a concrete demonstration of the bias-variance tradeoff: on a dataset this size (~2,800 training examples), a flexible, high-capacity model has no inherent advantage over models with the right structural assumptions already built in — and can actually underperform a well-chosen simple baseline.
- **Both GARCH and the transformer beat naive**, confirming real, learnable volatility clustering exists in gold's returns — just that a plain transformer, at this scale of data, isn't the most efficient way to capture it.

## Practical lessons from the process itself

- **Fitting ≠ forecasting ≠ backtesting.** All three are distinct, necessary steps — a model that fits history well tells you nothing about its forecasting ability until it's honestly tested on unseen data.
- **Diagnose failures with evidence, not guesses.** The first fix attempt (lowering the learning rate) was a reasonable hypothesis that turned out to be wrong — caught quickly by checking prediction spread directly, rather than assuming the fix worked because loss numbers moved slightly.
- **Data leakage is subtle and easy to miss in time series**, even when a train/test split looks correct. Using a validation set for checkpoint selection during training is standard practice — but if that validation period overlaps with the actual comparison window, it quietly inflates results.
- **Always benchmark against naive and simple baselines**, not just the sophisticated model. The simplest model in this whole project turned out to be the best one.

## What's left

- Repeat this full process for silver.
- Possible refinements not yet tried: features with exponentially-decaying weight (closer to what GARCH does structurally), a longer input window, asymmetric GARCH variants (GJR-GARCH/EGARCH), or simply more training data.
- `tests/` is still empty — worth adding once any of this logic gets consolidated into `src/`.

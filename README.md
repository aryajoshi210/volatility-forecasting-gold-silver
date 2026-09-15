# Volatility Forecasting: Gold and Silver Futures

Does a transformer neural network forecast commodity volatility better than a classical econometric model?

I built both from scratch, designed the evaluation myself, and tested the question honestly across two independent assets. The answer is no. A one-variable linear regression beat both of them.

That result is the point of the project. What follows is how I got to it and why I trust it.

## Results

Test period 2023-04-12 to 2026-08-05, held out from all training and model selection. RMSE, lower is better. The naive baseline predicts tomorrow's volatility as today's.

**Gold (GC=F)**

| Model | RMSE | vs naive |
|---|---|---|
| Linear regression, 1 feature | 0.879 | +6.1% |
| GARCH(1,1) | 0.903 | +3.5% |
| Transformer (~17k params) | 0.905 | +3.3% |
| Naive baseline | 0.936 | — |

**Silver (SI=F)**

| Model | RMSE | vs naive |
|---|---|---|
| Linear regression, 1 feature | 1.892 | +10.6% |
| GARCH(1,1) | 2.004 | +5.3% |
| Transformer (~17k params) | 2.012 | +5.0% |
| Naive baseline | 2.117 | — |

The transformer and GARCH are effectively tied. The simplest model wins on both assets. The ranking reproduced independently on silver after being established on gold.

## What this means

This is a clean demonstration of the bias-variance tradeoff. A flexible, high-capacity model had no advantage over simpler, more structurally appropriate ones on a modest dataset of roughly 2,800 to 3,300 training examples, and was beaten by a well-chosen simple baseline.

The value here is not a winning model. It is that the comparison is fair, the failure modes were found rather than hidden, and the conclusion follows from evidence rather than from what I hoped to find.

Full narrative write-up in [CONCLUSIONS.md](CONCLUSIONS.md).

## How the conclusion was reached

The first version of this project produced a much more flattering result. Most of the work was finding out why it was wrong.

**Diagnosed a silent failure.** After initial training the transformer's predictions varied only about 9% as much as the real data. Loss curves looked fine. I caught it by comparing prediction variance against actual variance directly, which is the check that mode collapse actually fails.

**Tested a fix instead of assuming it.** Lowering the learning rate is the standard first response and it did not resolve the collapse. Recording that it failed mattered more than trying it.

**Isolated the root cause with a control.** I wrote a closed-form least-squares regression with no ML library at all, on the same data, to separate "my transformer has a bug" from "the signal is genuinely weak". It was the second one. That control then turned out to be the best model in the study.

**Found two data leakage bugs in my own evaluation.** The first comparison used mismatched time periods for GARCH and the transformer. After fixing that, the transformer's validation set, which was selecting the model checkpoint, overlapped the test period. I re-architected the split to pin the test boundary to a fixed calendar date with zero overlap.

**Replicated the finding.** The entire pipeline was rerun independently on silver to check the result was not a fluke of one dataset. Same ranking.

## Repository

```
notebooks/     One notebook per pipeline stage, 01 to 16, in order
src/           Reusable modules: windowing.py, model.py
tests/         pytest suite: pipeline correctness, no windowing leakage,
               guaranteed-positive model outputs
data/          Downloaded price data (gitignored, recreated by notebook 01)
models/        Trained checkpoints (gitignored, recreated by notebook 11)
CONCLUSIONS.md Full write-up of method and findings
```

Roughly 4,170 daily observations per asset, 2010 to 2026, pulled at runtime.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

Then work through `notebooks/` in numerical order. Data is pulled from Yahoo Finance via `yfinance`; no API key needed.

## Built with

Python, pandas, NumPy, matplotlib. `arch` for GARCH(1,1) fitted by maximum likelihood. PyTorch for the transformer, written from first principles: input embedding, sinusoidal positional encoding, two layers of four-head self-attention, feed-forward layers, and a softplus output head to constrain forecasts to be non-negative. Evaluation uses sliding 30-day windows with normalisation statistics computed on training data only.

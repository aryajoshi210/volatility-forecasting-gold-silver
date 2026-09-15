# Volatility Forecasting: Gold and Silver Futures

An end-to-end volatility forecasting system for gold and silver futures, built to answer one question properly: does a transformer neural network beat a classical econometric model at forecasting commodity volatility?

Four models, two assets, one leak-free test set. I built the transformer from first principles in PyTorch, fitted a GARCH(1,1) baseline, designed the evaluation framework myself, and confirmed the answer by rerunning the entire pipeline on a second metal.

The answer is no, and a one-variable linear regression beat both. It reproduced on silver after being established on gold, which is what makes it worth reporting.

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

Every model beats the naive baseline. The transformer and GARCH are effectively tied. The simplest model wins on both assets.

## What I learned

**Model capacity has to be earned.** A 17,000-parameter transformer had no edge over a three-parameter econometric model on 2,800 to 3,300 training examples. I had read about the bias-variance tradeoff; measuring it on my own data is what made it stick.

**A baseline is a measuring instrument, not a formality.** I wrote the linear regression as a debugging control, to work out whether weak performance was my bug or the data's limit. It answered that question and then won the study. I now build the simplest possible model first, every time, because it is the only thing that tells you what your complex model is actually worth.

**Loss curves hide failures.** My transformer's training loss looked healthy while its predictions varied only about 9% as much as the real data. Comparing the distribution of predictions against the distribution of the target is what exposed it. I check that now before I trust any training run.

**A plausible fix is not a verified fix.** Lowering the learning rate is the textbook first response to that failure. I tried it, measured it, and it did not work. Knowing the difference between a fix you reasoned your way to and one you have evidence for changed how I debug.

**Leakage hides in the validation set.** My first comparison used mismatched time periods. After fixing that, I found the transformer's validation set, which was selecting the model checkpoint, overlapped the test period. Model selection leaks just as badly as training does, and it is much easier to miss.

**Replication is cheap insurance.** Rerunning everything on silver cost me a day and turned a one-dataset curiosity into a finding I can defend.

## What I built

**A transformer from first principles in PyTorch.** Input embedding, sinusoidal positional encoding, two layers of four-head self-attention, feed-forward layers, and a softplus output head constraining forecasts to be non-negative, which volatility must be. Custom training loop with Adam.

**A GARCH(1,1) baseline** fitted by maximum likelihood, forward-forecast and backtested out of sample on both metals.

**An evaluation framework built to be fair.** Sliding 30-day windows, normalisation statistics computed on training data only, and a test boundary pinned to a fixed calendar date so the transformer and GARCH are scored on identical periods with zero overlap into model selection.

**A closed-form least-squares regression with no ML library at all,** as a control to isolate model bugs from genuine signal limits.

**Tested, reusable code.** Windowing and model logic extracted into `src/` modules, backed by a nine-test pytest suite covering pipeline correctness, absence of leakage in the windowing logic, and guaranteed-positive outputs. Full commit history, sixteen documented notebooks, one per pipeline stage.

Full narrative write-up in [CONCLUSIONS.md](CONCLUSIONS.md).

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

Python, pandas, NumPy, matplotlib, PyTorch, `arch`, pytest, Git.

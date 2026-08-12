# Gold & Silver Volatility Forecasting — Project Summary

## The goal

Forecast how volatile gold and silver futures will be — not which direction prices move, but *how big* the daily swings are likely to be. Two approaches were built and compared: **GARCH**, a classic, simple statistical model, and a **transformer**, a modern neural network built from scratch.

---

## Part 1: What was actually built, in order

**1. Project setup** — a Python environment, GitHub version control, and a `save.bat` shortcut for committing progress in one command.

**2. Real data** — daily gold and silver futures prices downloaded from Yahoo Finance, 2010 to 2026 (~4,170 trading days each), converted into daily percentage returns — the actual number both models work with.

**3. The GARCH baseline** — a well-established statistical model that estimates volatility using just 4 numbers, based on the idea that "yesterday's shock and yesterday's volatility predict today's volatility." Fitted, then genuinely *forecast* forward, then honestly *backtested* on data it had never seen. Result: GARCH beat a naive "volatility never changes" guess by **2.6% (gold)** and **3.5% (silver)** — modest, but real.

**4. The transformer** — a neural network built piece by piece: turning each day's return into a richer number, tagging each day with its position in a 30-day window, running that window through a self-attention mechanism (the part that lets the model decide for itself which past days matter most), and producing one final number — tomorrow's predicted volatility. About 17,000 tunable values in total, trained on real data.

**5. `src/` and `tests/`** — the reusable pieces of code (the windowing logic and the model itself) were pulled out into proper files, with a set of automated tests (9, all passing) that check things like "the model never predicts negative volatility" and "a window never accidentally contains the value it's trying to predict."

---

## Part 2: What went wrong, and how each problem was actually found and fixed

This is the most useful part to remember — the project didn't go smoothly in a straight line, and finding out *why* something wasn't working was most of the actual effort.

### Problem 1: the transformer learned almost nothing

After training for 100 rounds, the model's predictions barely varied at all — regardless of what the input data actually showed, it just guessed something close to the overall average every time. This wasn't obvious from the loss numbers alone; it was confirmed by directly comparing *how much the predictions varied* against *how much real volatility varies*. The predictions moved only about **9%** as much as real volatility does — a clear sign the model had taken a lazy shortcut instead of genuinely learning.

### Problem 2: the first fix attempt didn't work

The natural first guess was that the training step size (the "learning rate") was too large, causing the model to overshoot better solutions. Lowering it and retraining barely changed anything (9% → 11%) — so that guess turned out to be wrong, or at least not the main issue. Important lesson here: **a plausible-sounding fix still needs to be checked, not assumed to have worked.**

### Problem 3: was this actually a bug, or just a hard problem?

To find out, an even simpler model — a basic linear regression using just "how volatile has this metal been recently" as its only input — was tested on the same data. It also could barely beat guessing the average. This was the turning point: it showed the transformer's weak performance wasn't a training bug at all — daily volatility is genuinely very noisy and hard to predict, for *any* model, on that stretch of data.

### Problem 4: the first GARCH-vs-transformer comparison was unfair

The transformer had been scored on a different date range than GARCH — comparing two different time periods, not a fair fight. Fixing this and re-scoring both models on the *exact same* dates flipped the picture: the transformer suddenly looked *better* than GARCH.

### Problem 5: an even subtler leak was hiding in that "better" result

Digging further, part of that shared test period had quietly been used as the transformer's own *validation* set — the data used to decide which version of the model to keep during training. That's not full cheating, but it's not a completely fair, blind test either. Retraining one more time, with the test period fully sealed off from every stage of training, produced the final, trustworthy numbers.

**The takeaway from this whole chain:** every one of these problems looked, at first glance, like a different thing — bad training, needs more time, needs a different learning rate, unfair comparison — and the only way to actually find the real explanation each time was to test a specific, concrete hypothesis and look at real evidence, rather than guessing and moving on.

---

## Part 3: The final, honest results

Both models were re-run on **silver** too, using everything learned from gold, to check the findings weren't a fluke of one dataset. They weren't — the same pattern showed up both times.

**Gold** (test period: 2023–2026)

| Model | Error (RMSE) | Improvement over "just guess the average" |
|---|---|---|
| Simple linear regression (1 input: recent volatility) | 0.879 | **+6.1%** |
| GARCH | 0.903 | +3.5% |
| Transformer | 0.905 | +3.3% |
| Naive guess | 0.936 | 0% |

**Silver** (same test period)

| Model | Error (RMSE) | Improvement over "just guess the average" |
|---|---|---|
| Simple linear regression (1 input: recent volatility) | 1.892 | **+10.6%** |
| GARCH | 2.004 | +5.3% |
| Transformer | 2.012 | +5.0% |
| Naive guess | 2.117 | 0% |

**In plain terms:** the transformer and GARCH ended up performing almost identically on both metals — GARCH very slightly ahead each time. And the simplest model of all — a single number (recent volatility) plugged into basic linear regression — beat both of them, clearly, on both metals.

---

## Part 4: What this actually means

- **More complexity didn't win.** The transformer has thousands of tunable values and a genuinely sophisticated mechanism for deciding what matters in the data. GARCH has 4. The simple linear model has 2. On this amount of data, none of that extra sophistication translated into better forecasts — if anything, the simplest approach won.
- **This is a real result, not a failure.** It's a clean, twice-confirmed demonstration of something true throughout machine learning: a flexible model needs *enough data* to find patterns reliably, and without that, a simpler model with sensible built-in assumptions can beat it. ~2,800–3,300 training examples turned out to be enough for GARCH and linear regression to work well, but not enough to give the transformer a real edge.
- **All three real models did beat the naive guess, on both metals.** So the underlying idea of the whole project — that volatility clusters into calm and turbulent periods, and that's learnable — is genuinely true. It just didn't require a transformer to capture it.

---

## Part 5: Key things learned, worth remembering

1. **Fitting, forecasting, and backtesting are three different things.** A model can fit history perfectly and still be worthless at predicting anything new — the only real test is honest, out-of-sample performance.
2. **Diagnose with evidence, not guesses.** Every fix attempted in this project was checked against real numbers afterward, and more than one "obvious" fix turned out to be wrong.
3. **Data leakage in time series is sneaky.** Even a split that looks correct at first glance can secretly let information about the future leak into training — it happened twice here, in two different subtle ways, and both had to be caught deliberately.
4. **Always test a dumb baseline.** The simplest possible model in this entire project ended up being the best one — without testing it, that would never have been discovered.
5. **One result isn't a conclusion — a repeated result is.** The gold-only finding could have been a coincidence of that specific test period; only after seeing the identical pattern on silver did it become a trustworthy conclusion.

---

## Project status: complete

- Data pipeline, GARCH (fit + forecast + backtest), and transformer (build + train + diagnose + evaluate) are all finished for **both gold and silver**.
- Reusable code lives in `src/`, backed by a passing test suite in `tests/`.
- Everything is version-controlled and pushed to GitHub, with every notebook running end-to-end.

## Ideas for extending this further (not done, not required)

- Give the model a feature that decays smoothly over time (closer to what GARCH does structurally), instead of a flat 30-day window.
- Try GARCH variants that treat big drops and big rallies differently (GJR-GARCH/EGARCH).
- Test whether the transformer's disadvantage shrinks with a longer window or more training data.

# GARCH notes: from AR(1) to GARCH(1,1)

Reference notes for the math behind `03_garch_baseline_gold.ipynb` and `04_garch_baseline_silver.ipynb`. One consistent set of symbols is used throughout, matching the actual code:

| Symbol | Meaning |
|---|---|
| `r_t` | actual daily return on day t (what `pct_change()` computes) |
| `mu` | average return (mean model) - close to 0 for daily gold/silver returns |
| `epsilon_t` | shock/residual on day t = `r_t - mu` (the "surprise"). Since mu is ~0 here, epsilon_t is essentially the return itself |
| `sigma_t` | conditional volatility on day t - the number GARCH forecasts. `sigma_t^2` is the conditional variance |
| `z_t` | standardized white noise, mean 0, average size 1 |
| `omega` | baseline/constant floor of variance |
| `alpha` | ARCH coefficient - how strongly yesterday's squared shock pushes today's variance |
| `beta` | GARCH coefficient - how much of yesterday's variance persists into today |

## 1. AR(1) - the general building block

```
x_t = phi * x_(t-1) + epsilon_t
```

`x_t` is a generic placeholder for "value of some series at time t" - not yet specific to returns or volatility. `phi` is a coefficient: how much of yesterday's value carries into today. This is the conceptual seed for everything below - not used directly in this project.

## 2. ARMA(1,1) - adds memory of yesterday's shock too

```
x_t = phi * x_(t-1) + theta * epsilon_(t-1) + epsilon_t
```

Still a general model for the *level* of a series (e.g. predicting tomorrow's return itself). Not used directly here either, but it's the direct ancestor of ARCH - the "yesterday's shock affects today" idea already appears.

## 3. ARCH(1) - the first model of volatility itself

```
epsilon_t = z_t * sigma_t
sigma_t^2 = omega + alpha * epsilon_(t-1)^2
```

**Limitation:** ARCH(1) only looks back one day. Real volatility clustering (calm stretches and stormy stretches lasting weeks, as seen in the return plots) needs longer memory than that. Fixing this with plain ARCH means adding many lagged terms (`ARCH(q)` with large q) - many extra parameters to estimate just to capture "burstiness."

## 4. GARCH(1,1) - what the project's baseline actually uses

```
epsilon_t = z_t * sigma_t
sigma_t^2 = omega + alpha * epsilon_(t-1)^2 + beta * sigma_(t-1)^2
```

The one addition - `beta * sigma_(t-1)^2` - solves ARCH's limitation elegantly: `sigma_(t-1)^2` already contains the entire recursive history of past variance folded into one number. Long memory/persistence with just 3 parameters (`omega, alpha, beta`) instead of many lag terms.

Constraint: `alpha + beta < 1` (otherwise variance would spiral to infinity). This also defines the long-run/unconditional variance the model always settles back toward:

```
long-run variance = omega / (1 - alpha - beta)
```

## Conditional vs. unconditional variance

- **Unconditional variance** - one fixed number for the whole dataset (e.g. `returns.var()`), ignoring time order. "How volatile has this metal been on average over its whole history."
- **Conditional variance** (`sigma_t^2`) - a *different number every day*, based on the most recent shock and the most recent variance. "How volatile do we expect today to be, given what just happened." This is what `result.conditional_volatility` returns - a full time series, not one number.

## How parameters are actually found: Maximum Likelihood Estimation

1. Assume a distribution for `z_t` (Normal, by default in this project).
2. Guess starting values for `mu, omega, alpha, beta`.
3. Walk through the entire return history, computing the implied `sigma_t` for every day using the recursive variance formula.
4. Score how probable the actual observed returns were, given those `sigma_t` values (the log-likelihood).
5. An optimizer nudges the parameters to increase that score, recomputes the whole `sigma_t` sequence, rescoes, and repeats until the score stops improving (convergence).

That whole loop is what `model.fit()` runs in one line.

## Fitted results so far

| | Gold | Silver |
|---|---|---|
| omega | 0.019 | 0.041 |
| alpha | 0.053 | 0.051 |
| beta | 0.931 | 0.940 |
| alpha + beta | 0.984 | 0.991 |

Both metals react to fresh shocks by about the same relative amount (alpha nearly equal). Silver has a higher baseline volatility floor (omega) and slightly stronger persistence (beta) - once silver gets volatile, it tends to stay that way marginally longer than gold.

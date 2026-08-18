# Holdout-10 benchmark validation (instruments outside MOEXBMI-100)

Generated: 2026-08-18T14:50:40.093888+00:00
Holdout tickers usable: 10/10

## Single-asset components + composites: aggregated across the 10 holdout tickers

| algorithm_name                                                            |   n_tickers_ok |   mean_sharpe |   median_sharpe | pct_positive_sharpe   | mean_hit_rate   | mean_max_drawdown   | mean_total_return   | mean_pnl_rub   | total_pnl_rub   |
|:--------------------------------------------------------------------------|---------------:|--------------:|----------------:|:----------------------|:----------------|:--------------------|:--------------------|:---------------|:----------------|
| One-Class SVM Risk Switch                                                 |             10 |         0.946 |           0.989 | 70.0%                 | 54.35%          | -28.79%             | 30.85%              | 308 516 ₽      | 3 085 158 ₽     |
| Isolation Forest Risk Switch                                              |             10 |         0.636 |           0.424 | 80.0%                 | 52.26%          | -35.64%             | 22.97%              | 229 715 ₽      | 2 297 148 ₽     |
| VAE Factor Model + Isolation Forest overlay                               |             10 |         0.464 |           0.113 | 70.0%                 | 51.03%          | -10.29%             | 1.87%               | 18 661 ₽       | 186 611 ₽       |
| SMA Crossover Baseline                                                    |             10 |         0.413 |           0.325 | 60.0%                 | 52.04%          | -43.84%             | 18.08%              | 180 832 ₽      | 1 808 324 ₽     |
| Elastic Net Factor Model + Isolation Forest overlay (vol-targeted sizing) |             10 |         0.301 |           0.294 | 60.0%                 | 50.40%          | -6.33%              | 1.88%               | 18 777 ₽       | 187 766 ₽       |
| N-BEATS Forecaster + One-Class SVM overlay                                |             10 |         0.14  |           0.045 | 50.0%                 | 49.44%          | -3.24%              | 0.15%               | 1 540 ₽        | 15 403 ₽        |
| VAE Factor Model                                                          |             10 |         0.106 |           0.094 | 60.0%                 | 51.08%          | -14.70%             | 0.99%               | 9 924 ₽        | 99 243 ₽        |
| Elastic Net Factor Model                                                  |             10 |         0.016 |          -0.127 | 40.0%                 | 50.28%          | -9.74%              | 2.77%               | 27 691 ₽       | 276 908 ₽       |
| N-BEATS Forecaster                                                        |             10 |        -0.062 |          -0.081 | 40.0%                 | 49.47%          | -4.55%              | 0.35%               | 3 545 ₽        | 35 448 ₽        |
| Random Position Baseline                                                  |             10 |        -0.452 |          -0.328 | 20.0%                 | 48.53%          | -56.14%             | -37.28%             | -372 770 ₽     | -3 727 701 ₽    |
| Buy and Hold Baseline                                                     |             10 |        -0.484 |          -0.485 | 20.0%                 | 45.13%          | -60.80%             | -30.18%             | -301 839 ₽     | -3 018 390 ₽    |

Made the most money: **One-Class SVM Risk Switch** (3 085 158 ₽). Lost the most: **Random Position Baseline** (-3 727 701 ₽).

## Thompson Sampling: raw momentum arm vs. strong-arm composite

| spec_name            | algorithm_name                                     |   sharpe_ratio | total_return   | max_drawdown   |   n_trades | pnl_rub   | error   |
|:---------------------|:---------------------------------------------------|---------------:|:---------------|:---------------|-----------:|:----------|:--------|
| thompson_momentum    | Thompson Sampling Capital Allocator                |          0.449 | 2.43%          | -2.60%         |        437 | 24 269 ₽  |         |
| thompson_strong_arms | Thompson Sampling (Elastic Net + Isolation Forest) |          0.641 | 0.25%          | -0.19%         |        558 | 2 536 ₽   |         |

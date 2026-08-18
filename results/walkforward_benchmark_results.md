# Walk-forward validation results (4 non-overlapping windows, SBER)

Generated: 2026-08-18T14:51:08.390788+00:00
Windows: 4 non-overlapping chronological blocks, each with its own 70/30 split.

## Spread of Sharpe across windows per strategy

| algorithm_name                                                            |   n_windows_ok |   mean_sharpe |   std_sharpe |   min_sharpe |   max_sharpe | pct_positive_sharpe   | mean_hit_rate   | mean_max_drawdown   | mean_total_return   | mean_pnl_rub   | total_pnl_rub   |
|:--------------------------------------------------------------------------|---------------:|--------------:|-------------:|-------------:|-------------:|:----------------------|:----------------|:--------------------|:--------------------|:---------------|:----------------|
| Symbolic Regression Alpha                                                 |              4 |         0.887 |        0.873 |        0     |        2.083 | 75.0%                 | 51.01%          | -0.19%              | 0.71%               | 7 103 ₽        | 28 411 ₽        |
| Isolation Forest Risk Switch                                              |              4 |         0.575 |        1.665 |       -0.684 |        3.016 | 50.0%                 | 54.13%          | -16.42%             | 0.82%               | 8 202 ₽        | 32 806 ₽        |
| DDPG Position Sizer                                                       |              4 |         0.545 |        1.309 |       -0.872 |        2.104 | 50.0%                 | 52.65%          | -18.85%             | 7.57%               | 75 744 ₽       | 302 977 ₽       |
| PPO Position Sizer                                                        |              4 |         0.48  |        1.336 |       -1.01  |        2.01  | 50.0%                 | 52.65%          | -2.92%              | 1.39%               | 13 902 ₽       | 55 610 ₽        |
| VAE Factor Model                                                          |              4 |         0.397 |        0.644 |       -0.232 |        1.205 | 75.0%                 | 52.99%          | -6.47%              | -0.09%              | -886 ₽         | -3 542 ₽        |
| One-Class SVM Risk Switch                                                 |              4 |         0.255 |        2.131 |       -2.6   |        2.015 | 50.0%                 | 59.84%          | -13.24%             | -5.05%              | -50 481 ₽      | -201 922 ₽      |
| VAE Factor Model + Isolation Forest overlay                               |              4 |        -0.014 |        1.634 |       -2.432 |        1.172 | 75.0%                 | 52.39%          | -3.50%              | 0.81%               | 8 130 ₽        | 32 521 ₽        |
| Gaussian Process Trader                                                   |              4 |        -0.109 |        0.873 |       -1.059 |        1.025 | 25.0%                 | 52.71%          | -0.26%              | 0.06%               | 556 ₽          | 2 223 ₽         |
| N-BEATS Forecaster                                                        |              4 |        -0.352 |        1.666 |       -2.098 |        1.721 | 50.0%                 | 46.76%          | -1.78%              | -0.72%              | -7 215 ₽       | -28 860 ₽       |
| Elastic Net Factor Model + Isolation Forest overlay (vol-targeted sizing) |              4 |        -0.61  |        1.065 |       -1.899 |        0.402 | 50.0%                 | 48.93%          | -3.87%              | -1.07%              | -10 674 ₽      | -42 695 ₽       |
| N-BEATS Forecaster + One-Class SVM overlay                                |              4 |        -0.668 |        2.207 |       -2.085 |        2.614 | 25.0%                 | 33.58%          | -1.16%              | -0.72%              | -7 185 ₽       | -28 741 ₽       |

Made the most money (summed across windows): **DDPG Position Sizer** (302 977 ₽). Lost the most: **One-Class SVM Risk Switch** (-201 922 ₽).

A robust strategy should look like itself across windows (low `std_sharpe`), not just have a high average - check `min_sharpe`/`max_sharpe` before trusting `mean_sharpe` alone.

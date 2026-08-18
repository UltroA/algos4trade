# Composite pipeline benchmark results

Generated: 2026-08-18T14:50:15.062373+00:00

## Single-asset composites on SBER

| spec_name                  | algorithm_name                                                            |   sharpe_ratio | total_return   | max_drawdown   |   n_trades | pnl_rub    | error   |
|:---------------------------|:--------------------------------------------------------------------------|---------------:|:---------------|:---------------|-----------:|:-----------|:--------|
| vae_isoforest              | VAE Factor Model + Isolation Forest overlay                               |          1.035 | 24.13%         | -3.41%         |        539 | 241 338 ₽  |         |
| nbeats_ocsvm               | N-BEATS Forecaster + One-Class SVM overlay                                |          0.539 | 0.86%          | -0.83%         |        437 | 8 596 ₽    |         |
| elasticnet_isoforest_sized | Elastic Net Factor Model + Isolation Forest overlay (vol-targeted sizing) |         -1.452 | -28.49%        | -29.87%        |        537 | -284 857 ₽ |         |

## Single-asset composites, cross-sectional (97-ticker MOEXBMI universe)

| spec_name                  | algorithm_name                                                            |   n_tickers_ok |   n_tickers_total |   mean_sharpe |   median_sharpe |   std_sharpe | pct_positive_sharpe   | mean_hit_rate   | mean_max_drawdown   | mean_total_return   | median_total_return   | mean_pnl_rub   | total_pnl_rub   |
|:---------------------------|:--------------------------------------------------------------------------|---------------:|------------------:|--------------:|----------------:|-------------:|:----------------------|:----------------|:--------------------|:--------------------|:----------------------|:---------------|:----------------|
| vae_isoforest              | VAE Factor Model + Isolation Forest overlay                               |             97 |                97 |         0.282 |           0.429 |        0.867 | 64.9%                 | 51.26%          | -8.16%              | 4.50%               | 2.55%                 | 45 010 ₽       | 4 365 974 ₽     |
| nbeats_ocsvm               | N-BEATS Forecaster + One-Class SVM overlay                                |             97 |                97 |        -0.017 |          -0.093 |        0.778 | 45.4%                 | 49.84%          | -2.11%              | -0.07%              | -0.20%                | -728 ₽         | -70 664 ₽       |
| elasticnet_isoforest_sized | Elastic Net Factor Model + Isolation Forest overlay (vol-targeted sizing) |             97 |                97 |        -0.019 |          -0.047 |        1.004 | 48.5%                 | 50.12%          | -6.60%              | -0.35%              | -0.30%                | -3 549 ₽       | -344 253 ₽      |

Made the most money: **VAE Factor Model + Isolation Forest overlay** (4 365 974 ₽). Lost the most: **Elastic Net Factor Model + Isolation Forest overlay (vol-targeted sizing)** (-344 253 ₽).

## Thompson Sampling with strong arms, across ticker universes

| universe    |   n_tickers |   sharpe_ratio | total_return   | max_drawdown   |   n_trades | pnl_rub   | error   |
|:------------|------------:|---------------:|:---------------|:---------------|-----------:|:----------|:--------|
| baseline-10 |          10 |         -0.931 | -0.29%         | -0.29%         |        552 | -2 853 ₽  |         |
| longhist-70 |          70 |         -1.414 | -0.03%         | -0.04%         |        589 | -349 ₽    |         |
| wide-97     |          97 |         -0.429 | -0.01%         | -0.01%         |        589 | -67 ₽     |         |

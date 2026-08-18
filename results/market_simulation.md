# Live market simulation results

Mode: **live**  |  Generated: 2026-08-17T11:22:25.049079+00:00
Session started: 2026-08-17T05:27:52.344812+00:00  |  Elapsed: 21273s  |  Ticks: 61
Algorithms run: 36  |  Skipped (warm-up fit failed): 0

## Process load

- simulator PID 14251: CPU 2%, RSS 149 MB, 45 threads

## T-Invest API latency observed this session

- T-Invest API calls measured: 500, mean 2790 ms, p50 2951 ms, p95 5490 ms, max 8840 ms

In live mode this is real measured latency (no delay is simulated - the wall-clock pacing of the session IS the real network round trip). In demo mode, per-tick sleeps are sampled from these same measured values instead of a guessed constant.
| spec_name              | algorithm_name                          | category                | tickers                                           | total_return   | annualized_return   |   sharpe_ratio | max_drawdown   | win_rate   | hit_rate   |   n_trades | starting_capital   | final_capital   | pnl_rub   | error   |
|:-----------------------|:----------------------------------------|:------------------------|:--------------------------------------------------|:---------------|:--------------------|---------------:|:---------------|:-----------|:-----------|-----------:|:-------------------|:----------------|:----------|:--------|
| autoencoder            | Autoencoder Factor Model                | representation_learning | SBER                                              | 0.03%          | 0.14%               |          0.094 | -0.76%         | 34.43%     | 40.00%     |         51 | 10 000 ₽           | 10 003 ₽        | 3 ₽       |         |
| buy_and_hold           | Buy and Hold Baseline                   | baseline                | SBER                                              | -0.20%         | -0.82%              |         -0.231 | -2.10%         | 50.82%     | 51.67%     |          1 | 10 000 ₽           | 9 980 ₽         | -20 ₽     |         |
| catboost_ranker        | CatBoost Ranker                         | supervised_ranking      | SBER                                              | -2.14%         | -8.56%              |         -4.723 | -2.21%         | 22.95%     | 40.00%     |         53 | 10 000 ₽           | 9 786 ₽         | -214 ₽    |         |
| cnn_candlestick        | Candlestick Pattern CNN                 | pattern_recognition     | SBER                                              | -1.28%         | -5.17%              |         -3.615 | -1.28%         | 29.51%     | 40.00%     |         53 | 10 000 ₽           | 9 872 ₽         | -128 ₽    |         |
| correlation_clustering | Correlation Clustering                  | clustering              | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| ddpg_agent             | DDPG Position Sizer                     | reinforcement_learning  | SBER                                              | -0.09%         | -0.36%              |         -0.258 | -0.86%         | 50.82%     | 51.67%     |         53 | 10 000 ₽           | 9 991 ₽         | -9 ₽      |         |
| elastic_net            | Elastic Net Factor Model                | linear_factor           | SBER                                              | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| gaussian_process       | Gaussian Process Trader                 | bayesian_optimization   | SBER                                              | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| genetic_programming    | Symbolic Regression Alpha               | symbolic_regression     | SBER                                              | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| gru_predictor          | GRU Predictor                           | sequence_model          | SBER                                              | -0.47%         | -1.91%              |         -1.573 | -0.85%         | 27.87%     | 33.33%     |         53 | 10 000 ₽           | 9 953 ₽         | -47 ₽     |         |
| hdbscan_clustering     | HDBSCAN Pairs/Basket Clustering         | clustering              | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| hmm_regime             | HMM Regime Detector                     | regime_detection        | SBER                                              | -0.20%         | -0.82%              |         -0.231 | -2.10%         | 50.82%     | 51.67%     |          1 | 10 000 ₽           | 9 980 ₽         | -20 ₽     |         |
| informer               | Informer-lite Transformer Predictor     | sequence_model          | SBER                                              | 0.31%          | 1.30%               |          1.27  | -0.48%         | 39.34%     | 40.00%     |         53 | 10 000 ₽           | 10 031 ₽        | 31 ₽      |         |
| isolation_forest       | Isolation Forest Risk Switch            | anomaly_detection       | SBER                                              | -0.42%         | -1.73%              |         -1.551 | -0.65%         | 35.00%     | 36.84%     |          3 | 10 000 ₽           | 9 958 ₽         | -42 ₽     |         |
| kalman_filter_pairs    | Kalman Filter Pairs Trading             | pairs_stat_arb          | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | -0.14%         | -0.59%              |         -1.765 | -0.23%         | 42.62%     | 41.67%     |          1 | 10 000 ₽           | 9 986 ₽         | -14 ₽     |         |
| lasso                  | Lasso Factor Model                      | linear_factor           | SBER                                              | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| lightgbm_ranker        | LightGBM Ranker                         | supervised_ranking      | SBER                                              | -2.31%         | -9.21%              |         -4.594 | -2.40%         | 27.87%     | 41.67%     |         52 | 10 000 ₽           | 9 769 ₽         | -231 ₽    |         |
| logistic_regression    | Logistic Regression Direction           | supervised_ranking      | SBER                                              | -1.88%         | -7.53%              |         -3.475 | -2.13%         | 31.15%     | 36.67%     |         48 | 10 000 ₽           | 9 812 ₽         | -188 ₽    |         |
| lstm_predictor         | LSTM Predictor                          | sequence_model          | SBER                                              | -0.46%         | -1.87%              |         -1.765 | -0.77%         | 34.43%     | 35.00%     |         53 | 10 000 ₽           | 9 954 ₽         | -46 ₽     |         |
| meta_labeling          | Meta-Labeling (Lopez de Prado)          | meta_labeling           | SBER                                              | -1.12%         | -4.54%              |         -2.411 | -1.31%         | 32.00%     | 38.78%     |         47 | 10 000 ₽           | 9 888 ₽         | -112 ₽    |         |
| nbeats                 | N-BEATS Forecaster                      | time_series_forecast    | SBER                                              | 0.00%          | 0.00%               |          0.02  | -0.07%         | 34.43%     | 35.00%     |         53 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| news_sentiment         | News Sentiment Signal                   | news_sentiment          | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | 0.03%          | 0.13%               |          0.308 | -0.12%         | 57.38%     | 43.92%     |          3 | 10 000 ₽           | 10 003 ₽        | 3 ₽       |         |
| news_sentiment_memory  | News Sentiment Signal (Position Memory) | news_sentiment          | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | 0.03%          | 0.13%               |          0.308 | -0.12%         | 57.38%     | 43.92%     |          3 | 10 000 ₽           | 10 003 ₽        | 3 ₽       |         |
| nhits                  | N-HiTS Forecaster                       | time_series_forecast    | SBER                                              | -0.02%         | -0.09%              |         -4.922 | -0.02%         | 26.23%     | 38.33%     |         53 | 10 000 ₽           | 9 998 ₽         | -2 ₽      |         |
| one_class_svm          | One-Class SVM Risk Switch               | anomaly_detection       | SBER                                              | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| ppo_agent              | PPO Position Sizer                      | reinforcement_learning  | SBER                                              | -0.03%         | -0.12%              |         -0.26  | -0.28%         | 50.82%     | 51.67%     |         53 | 10 000 ₽           | 9 997 ₽         | -3 ₽      |         |
| random_baseline        | Random Position Baseline                | baseline                | SBER                                              | 1.80%          | 7.66%               |          2.323 | -1.64%         | 47.54%     | 53.33%     |          1 | 10 000 ₽           | 10 180 ₽        | 180 ₽     |         |
| random_forest          | Random Forest Baseline                  | supervised_ranking      | SBER                                              | -1.27%         | -5.14%              |         -5.577 | -1.31%         | 24.59%     | 31.67%     |         53 | 10 000 ₽           | 9 873 ₽         | -127 ₽    |         |
| sac_agent              | SAC Position Sizer                      | reinforcement_learning  | SBER                                              | -0.00%         | -0.01%              |         -0.244 | -0.03%         | 50.82%     | 51.67%     |         53 | 10 000 ₽           | 10 000 ₽        | -0 ₽      |         |
| sma_crossover          | SMA Crossover Baseline                  | baseline                | SBER                                              | -2.30%         | -9.18%              |         -2.913 | -2.58%         | 37.70%     | 38.33%     |          1 | 10 000 ₽           | 9 770 ₽         | -230 ₽    |         |
| svm_rbf                | SVM RBF Direction Classifier            | supervised_ranking      | SBER                                              | -0.28%         | -1.16%              |         -3.763 | -0.30%         | 29.09%     | 35.19%     |         52 | 10 000 ₽           | 9 972 ₽         | -28 ₽     |         |
| tcn_predictor          | TCN Predictor                           | sequence_model          | SBER                                              | -1.99%         | -7.97%              |         -3.9   | -2.00%         | 24.59%     | 33.33%     |         51 | 10 000 ₽           | 9 801 ₽         | -199 ₽    |         |
| thompson_bandits       | Thompson Sampling Capital Allocator     | capital_allocation      | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | -0.14%         | -0.59%              |         -2.084 | -0.16%         | 40.98%     | 43.07%     |         26 | 10 000 ₽           | 9 986 ₽         | -14 ₽     |         |
| transformer_patchtst   | PatchTST Transformer Predictor          | sequence_model          | SBER                                              | -0.62%         | -2.54%              |         -2.982 | -0.73%         | 29.51%     | 36.67%     |         53 | 10 000 ₽           | 9 938 ₽         | -62 ₽     |         |
| vae                    | VAE Factor Model                        | representation_learning | SBER                                              | -0.34%         | -1.38%              |         -0.685 | -0.81%         | 26.23%     | 45.00%     |         50 | 10 000 ₽           | 9 966 ₽         | -34 ₽     |         |
| xgboost_ranker         | XGBoost Ranker                          | supervised_ranking      | SBER                                              | -2.77%         | -10.95%             |         -5.264 | -2.82%         | 24.59%     | 38.33%     |         52 | 10 000 ₽           | 9 723 ₽         | -277 ₽    |         |

## Summary: who performed better, who worse

- 36/36 algorithms completed without errors (0 failed); of those, 6/36 ended profitable (positive money P&L).
- Best by Sharpe: **Random Position Baseline** (Sharpe +2.323, 180 ₽).
- Worst by Sharpe: **Random Forest Baseline** (Sharpe -5.577, -127 ₽).
- Made the most money: **Random Position Baseline** (180 ₽, Sharpe +2.323).
- Lost the most money: **XGBoost Ranker** (-277 ₽, Sharpe -5.264).

## Top 10 by Sharpe ratio (out-of-sample)

| algorithm_name                          |   sharpe_ratio | total_return   | max_drawdown   | pnl_rub   |
|:----------------------------------------|---------------:|:---------------|:---------------|:----------|
| Random Position Baseline                |          2.323 | 1.80%          | -1.64%         | 180 ₽     |
| Informer-lite Transformer Predictor     |          1.27  | 0.31%          | -0.48%         | 31 ₽      |
| News Sentiment Signal                   |          0.308 | 0.03%          | -0.12%         | 3 ₽       |
| News Sentiment Signal (Position Memory) |          0.308 | 0.03%          | -0.12%         | 3 ₽       |
| Autoencoder Factor Model                |          0.094 | 0.03%          | -0.76%         | 3 ₽       |
| N-BEATS Forecaster                      |          0.02  | 0.00%          | -0.07%         | 0 ₽       |
| Elastic Net Factor Model                |          0     | 0.00%          | 0.00%          | 0 ₽       |
| Gaussian Process Trader                 |          0     | 0.00%          | 0.00%          | 0 ₽       |
| Symbolic Regression Alpha               |          0     | 0.00%          | 0.00%          | 0 ₽       |
| Correlation Clustering                  |          0     | 0.00%          | 0.00%          | 0 ₽       |

## Top 10 by money made (P&L, RUB)

| algorithm_name                          | pnl_rub   |   sharpe_ratio | total_return   |
|:----------------------------------------|:----------|---------------:|:---------------|
| Random Position Baseline                | 180 ₽     |          2.323 | 1.80%          |
| Informer-lite Transformer Predictor     | 31 ₽      |          1.27  | 0.31%          |
| Autoencoder Factor Model                | 3 ₽       |          0.094 | 0.03%          |
| News Sentiment Signal                   | 3 ₽       |          0.308 | 0.03%          |
| News Sentiment Signal (Position Memory) | 3 ₽       |          0.308 | 0.03%          |
| N-BEATS Forecaster                      | 0 ₽       |          0.02  | 0.00%          |
| Correlation Clustering                  | 0 ₽       |          0     | 0.00%          |
| HDBSCAN Pairs/Basket Clustering         | 0 ₽       |          0     | 0.00%          |
| Lasso Factor Model                      | 0 ₽       |          0     | 0.00%          |
| One-Class SVM Risk Switch               | 0 ₽       |          0     | 0.00%          |

## Biggest losses (P&L, RUB)

| algorithm_name                 | pnl_rub   |   sharpe_ratio | total_return   |
|:-------------------------------|:----------|---------------:|:---------------|
| XGBoost Ranker                 | -277 ₽    |         -5.264 | -2.77%         |
| LightGBM Ranker                | -231 ₽    |         -4.594 | -2.31%         |
| SMA Crossover Baseline         | -230 ₽    |         -2.913 | -2.30%         |
| CatBoost Ranker                | -214 ₽    |         -4.723 | -2.14%         |
| TCN Predictor                  | -199 ₽    |         -3.9   | -1.99%         |
| Logistic Regression Direction  | -188 ₽    |         -3.475 | -1.88%         |
| Candlestick Pattern CNN        | -128 ₽    |         -3.615 | -1.28%         |
| Random Forest Baseline         | -127 ₽    |         -5.577 | -1.27%         |
| Meta-Labeling (Lopez de Prado) | -112 ₽    |         -2.411 | -1.12%         |
| PatchTST Transformer Predictor | -62 ₽     |         -2.982 | -0.62%         |

## Compute load per algorithm

| algorithm_name                          |   fit_seconds |   fit_peak_rss_mb | fit_rss_delta_mb   |   avg_tick_ms |   tick_cpu_seconds_total | cpu_load_share_pct   |
|:----------------------------------------|--------------:|------------------:|:-------------------|--------------:|-------------------------:|:---------------------|
| Candlestick Pattern CNN                 |        27.669 |             519   |                    |      2375.7   |                  333.851 | 33.1%                |
| TCN Predictor                           |        27.684 |             552   |                    |       321.023 |                  168.883 | 16.8%                |
| PPO Position Sizer                      |         0.615 |             398.8 |                    |       117.836 |                   83.06  | 8.2%                 |
| DDPG Position Sizer                     |        19.923 |             405.9 |                    |        92.724 |                   54.813 | 5.4%                 |
| Informer-lite Transformer Predictor     |         5.92  |             862.6 |                    |        72.536 |                   43.099 | 4.3%                 |
| One-Class SVM Risk Switch               |         0.039 |             198.8 |                    |        41.453 |                   30.848 | 3.1%                 |
| LSTM Predictor                          |         1.678 |             739.2 |                    |        40.878 |                   30.018 | 3.0%                 |
| SVM RBF Direction Classifier            |         1.483 |             280.2 |                    |       450.366 |                   27.432 | 2.7%                 |
| Thompson Sampling Capital Allocator     |         0.003 |             172   |                    |        45.875 |                   27.403 | 2.7%                 |
| Kalman Filter Pairs Trading             |         0.039 |             172.4 |                    |        42.927 |                   25.709 | 2.6%                 |
| Random Forest Baseline                  |         0.271 |             218.8 |                    |        54.563 |                   24.194 | 2.4%                 |
| Isolation Forest Risk Switch            |         0.13  |             207.2 |                    |        36.087 |                   21.558 | 2.1%                 |
| Gaussian Process Trader                 |         2.259 |             382.1 |                    |        33.884 |                   19.989 | 2.0%                 |
| GRU Predictor                           |         1.658 |             702.7 |                    |        32.307 |                   18.624 | 1.8%                 |
| PatchTST Transformer Predictor          |         1.031 |             651   |                    |        18.461 |                   10.856 | 1.1%                 |
| CatBoost Ranker                         |         0.244 |             205   |                    |        18.346 |                   10.708 | 1.1%                 |
| Meta-Labeling (Lopez de Prado)          |         1.94  |             201.7 |                    |        12.997 |                    9.805 | 1.0%                 |
| LightGBM Ranker                         |         0.143 |             200.3 |                    |        11.736 |                    7.499 | 0.7%                 |
| SAC Position Sizer                      |        22.823 |             400.5 |                    |       107.206 |                    6.718 | 0.7%                 |
| Logistic Regression Direction           |         0.027 |             192.6 |                    |         7.848 |                    5.953 | 0.6%                 |
| XGBoost Ranker                          |         0.124 |             201.3 |                    |        10.055 |                    5.911 | 0.6%                 |
| Elastic Net Factor Model                |         0.029 |             190.8 |                    |        10.033 |                    5.882 | 0.6%                 |
| News Sentiment Signal (Position Memory) |         0     |             195.9 |                    |       215.717 |                    5.473 | 0.5%                 |
| Autoencoder Factor Model                |         0.979 |             429.5 |                    |        36.299 |                    5.138 | 0.5%                 |
| Lasso Factor Model                      |         0.019 |             192.1 |                    |         8.079 |                    4.842 | 0.5%                 |
| VAE Factor Model                        |         0.8   |             427   |                    |         8.029 |                    4.78  | 0.5%                 |
| Symbolic Regression Alpha               |         0.12  |             179.7 |                    |         7.371 |                    4.323 | 0.4%                 |
| N-HiTS Forecaster                       |         0.695 |             424.6 |                    |         4.905 |                    3.421 | 0.3%                 |
| N-BEATS Forecaster                      |         0.599 |             419.9 |                    |         3.645 |                    2.758 | 0.3%                 |
| News Sentiment Signal                   |         0     |             169   |                    |         2.596 |                    1.985 | 0.2%                 |
| HMM Regime Detector                     |         0.103 |             199.4 |                    |         1.921 |                    1.129 | 0.1%                 |
| Random Position Baseline                |         0     |             171   |                    |         0.618 |                    0.349 | 0.0%                 |
| Correlation Clustering                  |         0.006 |             175   |                    |         0.481 |                    0.278 | 0.0%                 |
| HDBSCAN Pairs/Basket Clustering         |         0.007 |             196.5 |                    |         0.406 |                    0.24  | 0.0%                 |
| Buy and Hold Baseline                   |         0     |             177   |                    |         0.343 |                    0.19  | 0.0%                 |
| SMA Crossover Baseline                  |         0     |             170.7 |                    |         0.652 |                    0.04  | 0.0%                 |

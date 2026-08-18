# Live market simulation results

Mode: **live**  |  Generated: 2026-08-17T14:22:58.591116+00:00
Session started: 2026-08-17T14:00:24.256979+00:00  |  Elapsed: 1354s  |  Ticks: 8
Algorithms run: 36  |  Skipped (warm-up fit failed): 0

## Process load

- simulator PID 44394: CPU 2%, RSS 84 MB, 45 threads

## T-Invest API latency observed this session

- T-Invest API calls measured: 221, mean 3104 ms, p50 3169 ms, p95 5706 ms, max 8888 ms

In live mode this is real measured latency (no delay is simulated - the wall-clock pacing of the session IS the real network round trip). In demo mode, per-tick sleeps are sampled from these same measured values instead of a guessed constant.
| spec_name              | algorithm_name                          | category                | tickers                                           | total_return   | annualized_return   |   sharpe_ratio | max_drawdown   | win_rate   | hit_rate   |   n_trades | starting_capital   | final_capital   | pnl_rub   | error   |
|:-----------------------|:----------------------------------------|:------------------------|:--------------------------------------------------|:---------------|:--------------------|---------------:|:---------------|:-----------|:-----------|-----------:|:-------------------|:----------------|:----------|:--------|
| autoencoder            | Autoencoder Factor Model                | representation_learning | SBER                                              | -0.06%         | -1.82%              |         -9.539 | -0.06%         | 0.00%      | 0.00%      |          5 | 10 000 ₽           | 9 994 ₽         | -6 ₽      |         |
| buy_and_hold           | Buy and Hold Baseline                   | baseline                | SBER                                              | 0.31%          | 10.10%              |          7.508 | -0.00%         | 50.00%     | 57.14%     |          1 | 10 000 ₽           | 10 031 ₽        | 31 ₽      |         |
| catboost_ranker        | CatBoost Ranker                         | supervised_ranking      | SBER                                              | -0.10%         | -3.04%              |        -10.908 | -0.10%         | 0.00%      | 28.57%     |          5 | 10 000 ₽           | 9 990 ₽         | -10 ₽     |         |
| cnn_candlestick        | Candlestick Pattern CNN                 | pattern_recognition     | SBER                                              | 0.01%          | 0.26%               |          0.33  | -0.10%         | 12.50%     | 42.86%     |          5 | 10 000 ₽           | 10 001 ₽        | 1 ₽       |         |
| correlation_clustering | Correlation Clustering                  | clustering              | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| ddpg_agent             | DDPG Position Sizer                     | reinforcement_learning  | SBER                                              | 0.12%          | 4.00%               |          7.505 | -0.00%         | 50.00%     | 57.14%     |          5 | 10 000 ₽           | 10 012 ₽        | 12 ₽      |         |
| elastic_net            | Elastic Net Factor Model                | linear_factor           | SBER                                              | -0.00%         | -0.02%              |         -9.642 | -0.00%         | 0.00%      | 28.57%     |          5 | 10 000 ₽           | 10 000 ₽        | -0 ₽      |         |
| gaussian_process       | Gaussian Process Trader                 | bayesian_optimization   | SBER                                              | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| genetic_programming    | Symbolic Regression Alpha               | symbolic_regression     | SBER                                              | 0.00%          | 0.00%               |          4.879 | -0.00%         | 37.50%     | 57.14%     |          5 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| gru_predictor          | GRU Predictor                           | sequence_model          | SBER                                              | -0.01%         | -0.34%              |         -4.882 | -0.02%         | 25.00%     | 28.57%     |          5 | 10 000 ₽           | 9 999 ₽         | -1 ₽      |         |
| hdbscan_clustering     | HDBSCAN Pairs/Basket Clustering         | clustering              | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| hmm_regime             | HMM Regime Detector                     | regime_detection        | SBER                                              | 0.31%          | 10.10%              |          7.508 | -0.00%         | 50.00%     | 57.14%     |          1 | 10 000 ₽           | 10 031 ₽        | 31 ₽      |         |
| informer               | Informer-lite Transformer Predictor     | sequence_model          | SBER                                              | -0.09%         | -2.66%              |         -7.658 | -0.09%         | 0.00%      | 0.00%      |          5 | 10 000 ₽           | 9 991 ₽         | -9 ₽      |         |
| isolation_forest       | Isolation Forest Risk Switch            | anomaly_detection       | SBER                                              | 0.31%          | 10.10%              |          7.508 | -0.00%         | 50.00%     | 57.14%     |          1 | 10 000 ₽           | 10 031 ₽        | 31 ₽      |         |
| kalman_filter_pairs    | Kalman Filter Pairs Trading             | pairs_stat_arb          | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | -0.02%         | -0.54%              |         -3.442 | -0.03%         | 25.00%     | 28.57%     |          1 | 10 000 ₽           | 9 998 ₽         | -2 ₽      |         |
| lasso                  | Lasso Factor Model                      | linear_factor           | SBER                                              | -0.00%         | -0.02%              |         -9.95  | -0.00%         | 0.00%      | 28.57%     |          5 | 10 000 ₽           | 10 000 ₽        | -0 ₽      |         |
| lightgbm_ranker        | LightGBM Ranker                         | supervised_ranking      | SBER                                              | 0.01%          | 0.45%               |          1.547 | -0.03%         | 25.00%     | 28.57%     |          5 | 10 000 ₽           | 10 001 ₽        | 1 ₽       |         |
| logistic_regression    | Logistic Regression Direction           | supervised_ranking      | SBER                                              | -0.09%         | -2.66%              |        -10.296 | -0.09%         | 0.00%      | 28.57%     |          5 | 10 000 ₽           | 9 991 ₽         | -9 ₽      |         |
| lstm_predictor         | LSTM Predictor                          | sequence_model          | SBER                                              | -0.06%         | -1.95%              |         -9.245 | -0.06%         | 0.00%      | 0.00%      |          5 | 10 000 ₽           | 9 994 ₽         | -6 ₽      |         |
| meta_labeling          | Meta-Labeling (Lopez de Prado)          | meta_labeling           | SBER                                              | -0.00%         | -0.05%              |         -5.612 | -0.00%         | 0.00%      | 100.00%    |          2 | 10 000 ₽           | 10 000 ₽        | -0 ₽      |         |
| nbeats                 | N-BEATS Forecaster                      | time_series_forecast    | SBER                                              | -0.02%         | -0.50%              |         -7.626 | -0.02%         | 0.00%      | 0.00%      |          5 | 10 000 ₽           | 9 998 ₽         | -2 ₽      |         |
| news_sentiment         | News Sentiment Signal                   | news_sentiment          | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | -0.03%         | -0.91%              |        -10.417 | -0.03%         | 12.50%     | 11.90%     |          8 | 10 000 ₽           | 9 997 ₽         | -3 ₽      |         |
| news_sentiment_memory  | News Sentiment Signal (Position Memory) | news_sentiment          | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | -0.03%         | -0.85%              |         -7.675 | -0.03%         | 25.00%     | 11.90%     |          8 | 10 000 ₽           | 9 997 ₽         | -3 ₽      |         |
| nhits                  | N-HiTS Forecaster                       | time_series_forecast    | SBER                                              | -0.00%         | -0.04%              |         -6.853 | -0.00%         | 12.50%     | 42.86%     |          5 | 10 000 ₽           | 10 000 ₽        | -0 ₽      |         |
| one_class_svm          | One-Class SVM Risk Switch               | anomaly_detection       | SBER                                              | 0.17%          | 5.45%               |          4.77  | -0.05%         | 20.00%     | 50.00%     |          3 | 10 000 ₽           | 10 017 ₽        | 17 ₽      |         |
| ppo_agent              | PPO Position Sizer                      | reinforcement_learning  | SBER                                              | 0.04%          | 1.29%               |          7.503 | -0.00%         | 50.00%     | 57.14%     |          5 | 10 000 ₽           | 10 004 ₽        | 4 ₽       |         |
| random_baseline        | Random Position Baseline                | baseline                | SBER                                              | -0.37%         | -10.90%             |         -6.042 | -0.40%         | 25.00%     | 42.86%     |          1 | 10 000 ₽           | 9 963 ₽         | -37 ₽     |         |
| random_forest          | Random Forest Baseline                  | supervised_ranking      | SBER                                              | -0.06%         | -1.82%              |        -12.238 | -0.06%         | 0.00%      | 28.57%     |          5 | 10 000 ₽           | 9 994 ₽         | -6 ₽      |         |
| sac_agent              | SAC Position Sizer                      | reinforcement_learning  | SBER                                              | 0.00%          | 0.12%               |          7.493 | -0.00%         | 50.00%     | 57.14%     |          5 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| sma_crossover          | SMA Crossover Baseline                  | baseline                | SBER                                              | 0.31%          | 10.10%              |          7.508 | -0.00%         | 50.00%     | 57.14%     |          1 | 10 000 ₽           | 10 031 ₽        | 31 ₽      |         |
| svm_rbf                | SVM RBF Direction Classifier            | supervised_ranking      | SBER                                              | 0.00%          | 0.06%               |          0.593 | -0.01%         | 25.00%     | 28.57%     |          5 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| tcn_predictor          | TCN Predictor                           | sequence_model          | SBER                                              | -0.01%         | -0.20%              |         -2.756 | -0.01%         | 12.50%     | 42.86%     |          5 | 10 000 ₽           | 9 999 ₽         | -1 ₽      |         |
| thompson_bandits       | Thompson Sampling Capital Allocator     | capital_allocation      | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | -0.03%         | -0.93%              |        -14.639 | -0.03%         | 25.00%     | 20.00%     |          7 | 10 000 ₽           | 9 997 ₽         | -3 ₽      |         |
| transformer_patchtst   | PatchTST Transformer Predictor          | sequence_model          | SBER                                              | -0.17%         | -5.20%              |         -9.226 | -0.17%         | 0.00%      | 0.00%      |          5 | 10 000 ₽           | 9 983 ₽         | -17 ₽     |         |
| vae                    | VAE Factor Model                        | representation_learning | SBER                                              | -0.06%         | -2.02%              |        -10.744 | -0.06%         | 0.00%      | 14.29%     |          5 | 10 000 ₽           | 9 994 ₽         | -6 ₽      |         |
| xgboost_ranker         | XGBoost Ranker                          | supervised_ranking      | SBER                                              | -0.05%         | -1.65%              |         -6.38  | -0.06%         | 12.50%     | 28.57%     |          5 | 10 000 ₽           | 9 995 ₽         | -5 ₽      |         |

## Summary: who performed better, who worse

- 36/36 algorithms completed without errors (0 failed); of those, 12/36 ended profitable (positive money P&L).
- Best by Sharpe: **Isolation Forest Risk Switch** (Sharpe +7.508, 31 ₽).
- Worst by Sharpe: **Thompson Sampling Capital Allocator** (Sharpe -14.639, -3 ₽).
- Made the most money: **Isolation Forest Risk Switch** (31 ₽, Sharpe +7.508).
- Lost the most money: **Random Position Baseline** (-37 ₽, Sharpe -6.042).

## Top 10 by Sharpe ratio (out-of-sample)

| algorithm_name               |   sharpe_ratio | total_return   | max_drawdown   | pnl_rub   |
|:-----------------------------|---------------:|:---------------|:---------------|:----------|
| Isolation Forest Risk Switch |          7.508 | 0.31%          | -0.00%         | 31 ₽      |
| Buy and Hold Baseline        |          7.508 | 0.31%          | -0.00%         | 31 ₽      |
| SMA Crossover Baseline       |          7.508 | 0.31%          | -0.00%         | 31 ₽      |
| HMM Regime Detector          |          7.508 | 0.31%          | -0.00%         | 31 ₽      |
| DDPG Position Sizer          |          7.505 | 0.12%          | -0.00%         | 12 ₽      |
| PPO Position Sizer           |          7.503 | 0.04%          | -0.00%         | 4 ₽       |
| SAC Position Sizer           |          7.493 | 0.00%          | -0.00%         | 0 ₽       |
| Symbolic Regression Alpha    |          4.879 | 0.00%          | -0.00%         | 0 ₽       |
| One-Class SVM Risk Switch    |          4.77  | 0.17%          | -0.05%         | 17 ₽      |
| LightGBM Ranker              |          1.547 | 0.01%          | -0.03%         | 1 ₽       |

## Top 10 by money made (P&L, RUB)

| algorithm_name               | pnl_rub   |   sharpe_ratio | total_return   |
|:-----------------------------|:----------|---------------:|:---------------|
| Isolation Forest Risk Switch | 31 ₽      |          7.508 | 0.31%          |
| SMA Crossover Baseline       | 31 ₽      |          7.508 | 0.31%          |
| HMM Regime Detector          | 31 ₽      |          7.508 | 0.31%          |
| Buy and Hold Baseline        | 31 ₽      |          7.508 | 0.31%          |
| One-Class SVM Risk Switch    | 17 ₽      |          4.77  | 0.17%          |
| DDPG Position Sizer          | 12 ₽      |          7.505 | 0.12%          |
| PPO Position Sizer           | 4 ₽       |          7.503 | 0.04%          |
| LightGBM Ranker              | 1 ₽       |          1.547 | 0.01%          |
| Candlestick Pattern CNN      | 1 ₽       |          0.33  | 0.01%          |
| SAC Position Sizer           | 0 ₽       |          7.493 | 0.00%          |

## Biggest losses (P&L, RUB)

| algorithm_name                      | pnl_rub   |   sharpe_ratio | total_return   |
|:------------------------------------|:----------|---------------:|:---------------|
| Random Position Baseline            | -37 ₽     |         -6.042 | -0.37%         |
| PatchTST Transformer Predictor      | -17 ₽     |         -9.226 | -0.17%         |
| CatBoost Ranker                     | -10 ₽     |        -10.908 | -0.10%         |
| Logistic Regression Direction       | -9 ₽      |        -10.296 | -0.09%         |
| Informer-lite Transformer Predictor | -9 ₽      |         -7.658 | -0.09%         |
| VAE Factor Model                    | -6 ₽      |        -10.744 | -0.06%         |
| LSTM Predictor                      | -6 ₽      |         -9.245 | -0.06%         |
| Autoencoder Factor Model            | -6 ₽      |         -9.539 | -0.06%         |
| Random Forest Baseline              | -6 ₽      |        -12.238 | -0.06%         |
| XGBoost Ranker                      | -5 ₽      |         -6.38  | -0.05%         |

## Compute load per algorithm

| algorithm_name                          |   fit_seconds |   fit_peak_rss_mb | fit_rss_delta_mb   |   avg_tick_ms |   tick_cpu_seconds_total | cpu_load_share_pct   |
|:----------------------------------------|--------------:|------------------:|:-------------------|--------------:|-------------------------:|:---------------------|
| Candlestick Pattern CNN                 |        28.508 |             435.3 |                    |      2446.23  |                   42.969 | 31.1%                |
| TCN Predictor                           |        27.643 |             549.9 |                    |       351.098 |                   25.193 | 18.2%                |
| PPO Position Sizer                      |         0.668 |             402.1 |                    |       121.638 |                   10.206 | 7.4%                 |
| DDPG Position Sizer                     |        20.642 |             246.1 |                    |        94.72  |                    7.281 | 5.3%                 |
| Informer-lite Transformer Predictor     |         6.219 |             895.2 |                    |        78.297 |                    5.987 | 4.3%                 |
| LSTM Predictor                          |         1.774 |             702.9 |                    |        47.07  |                    4.211 | 3.0%                 |
| One-Class SVM Risk Switch               |         0.043 |             201.6 |                    |        42.468 |                    3.827 | 2.8%                 |
| SVM RBF Direction Classifier            |         1.497 |             266   |                    |       457.417 |                    3.654 | 2.6%                 |
| Kalman Filter Pairs Trading             |         0.04  |             172.5 |                    |        45.358 |                    3.462 | 2.5%                 |
| Thompson Sampling Capital Allocator     |         0.003 |             171.4 |                    |        45.263 |                    3.454 | 2.5%                 |
| Isolation Forest Risk Switch            |         0.135 |             203   |                    |        38.937 |                    2.976 | 2.2%                 |
| GRU Predictor                           |         1.79  |             720.3 |                    |        39.444 |                    2.913 | 2.1%                 |
| Gaussian Process Trader                 |         2.749 |             445.3 |                    |        36.491 |                    2.766 | 2.0%                 |
| Random Forest Baseline                  |         0.268 |             217   |                    |        48.321 |                    2.729 | 2.0%                 |
| News Sentiment Signal (Position Memory) |         0     |             197.4 |                    |      1761.26  |                    2.667 | 1.9%                 |
| CatBoost Ranker                         |         0.246 |             203.4 |                    |        23.013 |                    1.681 | 1.2%                 |
| PatchTST Transformer Predictor          |         1.192 |             635.1 |                    |        20.857 |                    1.556 | 1.1%                 |
| Meta-Labeling (Lopez de Prado)          |         1.973 |             206.3 |                    |        14.054 |                    1.299 | 0.9%                 |
| LightGBM Ranker                         |         0.174 |             202.7 |                    |        15.829 |                    1.228 | 0.9%                 |
| Autoencoder Factor Model                |         0.863 |             429.8 |                    |        50.554 |                    1.061 | 0.8%                 |
| XGBoost Ranker                          |         0.207 |             202.8 |                    |        13.238 |                    0.974 | 0.7%                 |
| SAC Position Sizer                      |        23.583 |             313.8 |                    |       108.722 |                    0.869 | 0.6%                 |
| Elastic Net Factor Model                |         0.025 |             191.8 |                    |        11.064 |                    0.849 | 0.6%                 |
| Logistic Regression Direction           |         0.029 |             194.9 |                    |         8.775 |                    0.765 | 0.6%                 |
| Lasso Factor Model                      |         0.021 |             193.4 |                    |         8.462 |                    0.658 | 0.5%                 |
| VAE Factor Model                        |         0.819 |             427.7 |                    |         8.028 |                    0.628 | 0.5%                 |
| Symbolic Regression Alpha               |         0.123 |             181   |                    |         8.083 |                    0.595 | 0.4%                 |
| N-HiTS Forecaster                       |         0.734 |             422   |                    |         6.916 |                    0.543 | 0.4%                 |
| N-BEATS Forecaster                      |         0.654 |             428   |                    |         5.108 |                    0.465 | 0.3%                 |
| News Sentiment Signal                   |         0     |             171   |                    |         3.058 |                    0.268 | 0.2%                 |
| HMM Regime Detector                     |         0.105 |             204.5 |                    |         2.115 |                    0.16  | 0.1%                 |
| Random Position Baseline                |         0     |             175.5 |                    |         1.175 |                    0.087 | 0.1%                 |
| Correlation Clustering                  |         0.007 |             177.1 |                    |         0.514 |                    0.039 | 0.0%                 |
| HDBSCAN Pairs/Basket Clustering         |         0.01  |             199.4 |                    |         0.45  |                    0.035 | 0.0%                 |
| Buy and Hold Baseline                   |         0     |             173.6 |                    |         0.12  |                    0.009 | 0.0%                 |
| SMA Crossover Baseline                  |         0     |             175.8 |                    |         0.603 |                    0.005 | 0.0%                 |

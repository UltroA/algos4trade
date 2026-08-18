# Live market simulation results

Mode: **live**  |  Generated: 2026-08-17T13:56:23.439610+00:00
Session started: 2026-08-17T11:26:10.834666+00:00  |  Elapsed: 9013s  |  Ticks: 35
Algorithms run: 36  |  Skipped (warm-up fit failed): 0

## Process load

- simulator PID 35565: CPU 10%, RSS 531 MB, 45 threads

## T-Invest API latency observed this session

- T-Invest API calls measured: 500, mean 2025 ms, p50 2263 ms, p95 4028 ms, max 7236 ms

In live mode this is real measured latency (no delay is simulated - the wall-clock pacing of the session IS the real network round trip). In demo mode, per-tick sleeps are sampled from these same measured values instead of a guessed constant.
| spec_name              | algorithm_name                          | category                | tickers                                           | total_return   | annualized_return   |   sharpe_ratio | max_drawdown   | win_rate   | hit_rate   |   n_trades | starting_capital   | final_capital   | pnl_rub   | error   |
|:-----------------------|:----------------------------------------|:------------------------|:--------------------------------------------------|:---------------|:--------------------|---------------:|:---------------|:-----------|:-----------|-----------:|:-------------------|:----------------|:----------|:--------|
| autoencoder            | Autoencoder Factor Model                | representation_learning | SBER                                              | -0.15%         | -1.06%              |         -1.178 | -0.42%         | 45.71%     | 47.06%     |         30 | 10 000 ₽           | 9 985 ₽         | -15 ₽     |         |
| buy_and_hold           | Buy and Hold Baseline                   | baseline                | SBER                                              | 0.50%          | 3.62%               |          1.841 | -0.47%         | 34.29%     | 35.29%     |          1 | 10 000 ₽           | 10 050 ₽        | 50 ₽      |         |
| catboost_ranker        | CatBoost Ranker                         | supervised_ranking      | SBER                                              | -0.92%         | -6.46%              |         -8.354 | -0.92%         | 14.29%     | 38.24%     |         30 | 10 000 ₽           | 9 908 ₽         | -92 ₽     |         |
| cnn_candlestick        | Candlestick Pattern CNN                 | pattern_recognition     | SBER                                              | -0.05%         | -0.39%              |         -0.806 | -0.14%         | 28.57%     | 50.00%     |         30 | 10 000 ₽           | 9 995 ₽         | -5 ₽      |         |
| correlation_clustering | Correlation Clustering                  | clustering              | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| ddpg_agent             | DDPG Position Sizer                     | reinforcement_learning  | SBER                                              | 0.20%          | 1.46%               |          1.836 | -0.19%         | 34.29%     | 35.29%     |         30 | 10 000 ₽           | 10 020 ₽        | 20 ₽      |         |
| elastic_net            | Elastic Net Factor Model                | linear_factor           | SBER                                              | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| gaussian_process       | Gaussian Process Trader                 | bayesian_optimization   | SBER                                              | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| genetic_programming    | Symbolic Regression Alpha               | symbolic_regression     | SBER                                              | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| gru_predictor          | GRU Predictor                           | sequence_model          | SBER                                              | -0.30%         | -2.11%              |         -2.037 | -0.53%         | 42.86%     | 44.12%     |         30 | 10 000 ₽           | 9 970 ₽         | -30 ₽     |         |
| hdbscan_clustering     | HDBSCAN Pairs/Basket Clustering         | clustering              | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| hmm_regime             | HMM Regime Detector                     | regime_detection        | SBER                                              | 0.50%          | 3.62%               |          1.841 | -0.47%         | 34.29%     | 35.29%     |          1 | 10 000 ₽           | 10 050 ₽        | 50 ₽      |         |
| informer               | Informer-lite Transformer Predictor     | sequence_model          | SBER                                              | -0.16%         | -1.18%              |         -1.928 | -0.30%         | 42.86%     | 47.06%     |         30 | 10 000 ₽           | 9 984 ₽         | -16 ₽     |         |
| isolation_forest       | Isolation Forest Risk Switch            | anomaly_detection       | SBER                                              | 0.09%          | 0.66%               |          0.404 | -0.44%         | 28.57%     | 40.74%     |          7 | 10 000 ₽           | 10 009 ₽        | 9 ₽       |         |
| kalman_filter_pairs    | Kalman Filter Pairs Trading             | pairs_stat_arb          | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | -0.06%         | -0.43%              |         -2.937 | -0.08%         | 37.14%     | 36.76%     |          1 | 10 000 ₽           | 9 994 ₽         | -6 ₽      |         |
| lasso                  | Lasso Factor Model                      | linear_factor           | SBER                                              | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| lightgbm_ranker        | LightGBM Ranker                         | supervised_ranking      | SBER                                              | -1.13%         | -7.82%              |         -7.307 | -1.13%         | 14.29%     | 41.18%     |         30 | 10 000 ₽           | 9 887 ₽         | -113 ₽    |         |
| logistic_regression    | Logistic Regression Direction           | supervised_ranking      | SBER                                              | -0.86%         | -6.05%              |         -5.484 | -0.86%         | 25.71%     | 44.12%     |         29 | 10 000 ₽           | 9 914 ₽         | -86 ₽     |         |
| lstm_predictor         | LSTM Predictor                          | sequence_model          | SBER                                              | -0.27%         | -1.92%              |         -2.462 | -0.42%         | 42.86%     | 44.12%     |         30 | 10 000 ₽           | 9 973 ₽         | -27 ₽     |         |
| meta_labeling          | Meta-Labeling (Lopez de Prado)          | meta_labeling           | SBER                                              | -0.06%         | -0.43%              |         -2.448 | -0.08%         | 10.53%     | 38.89%     |         20 | 10 000 ₽           | 9 994 ₽         | -6 ₽      |         |
| nbeats                 | N-BEATS Forecaster                      | time_series_forecast    | SBER                                              | -0.03%         | -0.20%              |         -1.982 | -0.05%         | 42.86%     | 47.06%     |         30 | 10 000 ₽           | 9 997 ₽         | -3 ₽      |         |
| news_sentiment         | News Sentiment Signal                   | news_sentiment          | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | -0.02%         | -0.13%              |         -1.069 | -0.04%         | 48.57%     | 41.58%     |         34 | 10 000 ₽           | 9 998 ₽         | -2 ₽      |         |
| news_sentiment_memory  | News Sentiment Signal (Position Memory) | news_sentiment          | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | -0.04%         | -0.32%              |         -1.095 | -0.10%         | 51.43%     | 44.06%     |         34 | 10 000 ₽           | 9 996 ₽         | -4 ₽      |         |
| nhits                  | N-HiTS Forecaster                       | time_series_forecast    | SBER                                              | -0.00%         | -0.02%              |         -5.594 | -0.00%         | 17.14%     | 38.24%     |         30 | 10 000 ₽           | 10 000 ₽        | -0 ₽      |         |
| one_class_svm          | One-Class SVM Risk Switch               | anomaly_detection       | SBER                                              | 0.25%          | 1.81%               |          1.873 | -0.23%         | 33.33%     | 66.67%     |          4 | 10 000 ₽           | 10 025 ₽        | 25 ₽      |         |
| ppo_agent              | PPO Position Sizer                      | reinforcement_learning  | SBER                                              | 0.07%          | 0.48%               |          1.828 | -0.06%         | 34.29%     | 35.29%     |         30 | 10 000 ₽           | 10 007 ₽        | 7 ₽       |         |
| random_baseline        | Random Position Baseline                | baseline                | SBER                                              | -2.39%         | -16.01%             |         -9.236 | -2.39%         | 20.00%     | 38.24%     |          1 | 10 000 ₽           | 9 761 ₽         | -239 ₽    |         |
| random_forest          | Random Forest Baseline                  | supervised_ranking      | SBER                                              | -0.57%         | -4.02%              |         -5.842 | -0.57%         | 20.00%     | 35.29%     |         30 | 10 000 ₽           | 9 943 ₽         | -57 ₽     |         |
| sac_agent              | SAC Position Sizer                      | reinforcement_learning  | SBER                                              | 0.01%          | 0.05%               |          1.779 | -0.01%         | 34.29%     | 35.29%     |         30 | 10 000 ₽           | 10 001 ₽        | 1 ₽       |         |
| sma_crossover          | SMA Crossover Baseline                  | baseline                | SBER                                              | 0.20%          | 1.45%               |          0.744 | -0.47%         | 37.14%     | 41.18%     |          1 | 10 000 ₽           | 10 020 ₽        | 20 ₽      |         |
| svm_rbf                | SVM RBF Direction Classifier            | supervised_ranking      | SBER                                              | -0.42%         | -2.96%              |         -5.627 | -0.42%         | 24.24%     | 50.00%     |         30 | 10 000 ₽           | 9 958 ₽         | -42 ₽     |         |
| tcn_predictor          | TCN Predictor                           | sequence_model          | SBER                                              | -0.30%         | -2.12%              |         -3.56  | -0.35%         | 40.00%     | 47.06%     |         30 | 10 000 ₽           | 9 970 ₽         | -30 ₽     |         |
| thompson_bandits       | Thompson Sampling Capital Allocator     | capital_allocation      | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | -0.14%         | -0.98%              |         -6.242 | -0.15%         | 28.57%     | 32.93%     |         20 | 10 000 ₽           | 9 986 ₽         | -14 ₽     |         |
| transformer_patchtst   | PatchTST Transformer Predictor          | sequence_model          | SBER                                              | -0.38%         | -2.68%              |         -2.862 | -0.55%         | 40.00%     | 47.06%     |         30 | 10 000 ₽           | 9 962 ₽         | -38 ₽     |         |
| vae                    | VAE Factor Model                        | representation_learning | SBER                                              | -0.18%         | -1.27%              |         -1.362 | -0.46%         | 31.43%     | 55.88%     |         30 | 10 000 ₽           | 9 982 ₽         | -18 ₽     |         |
| xgboost_ranker         | XGBoost Ranker                          | supervised_ranking      | SBER                                              | -0.95%         | -6.64%              |         -6.444 | -0.95%         | 11.43%     | 41.18%     |         28 | 10 000 ₽           | 9 905 ₽         | -95 ₽     |         |

## Summary: who performed better, who worse

- 36/36 algorithms completed without errors (0 failed); of those, 8/36 ended profitable (positive money P&L).
- Best by Sharpe: **One-Class SVM Risk Switch** (Sharpe +1.873, 25 ₽).
- Worst by Sharpe: **Random Position Baseline** (Sharpe -9.236, -239 ₽).
- Made the most money: **Buy and Hold Baseline** (50 ₽, Sharpe +1.841).
- Lost the most money: **Random Position Baseline** (-239 ₽, Sharpe -9.236).

## Top 10 by Sharpe ratio (out-of-sample)

| algorithm_name               |   sharpe_ratio | total_return   | max_drawdown   | pnl_rub   |
|:-----------------------------|---------------:|:---------------|:---------------|:----------|
| One-Class SVM Risk Switch    |          1.873 | 0.25%          | -0.23%         | 25 ₽      |
| HMM Regime Detector          |          1.841 | 0.50%          | -0.47%         | 50 ₽      |
| Buy and Hold Baseline        |          1.841 | 0.50%          | -0.47%         | 50 ₽      |
| DDPG Position Sizer          |          1.836 | 0.20%          | -0.19%         | 20 ₽      |
| PPO Position Sizer           |          1.828 | 0.07%          | -0.06%         | 7 ₽       |
| SAC Position Sizer           |          1.779 | 0.01%          | -0.01%         | 1 ₽       |
| SMA Crossover Baseline       |          0.744 | 0.20%          | -0.47%         | 20 ₽      |
| Isolation Forest Risk Switch |          0.404 | 0.09%          | -0.44%         | 9 ₽       |
| Correlation Clustering       |          0     | 0.00%          | 0.00%          | 0 ₽       |
| Elastic Net Factor Model     |          0     | 0.00%          | 0.00%          | 0 ₽       |

## Top 10 by money made (P&L, RUB)

| algorithm_name               | pnl_rub   |   sharpe_ratio | total_return   |
|:-----------------------------|:----------|---------------:|:---------------|
| Buy and Hold Baseline        | 50 ₽      |          1.841 | 0.50%          |
| HMM Regime Detector          | 50 ₽      |          1.841 | 0.50%          |
| One-Class SVM Risk Switch    | 25 ₽      |          1.873 | 0.25%          |
| DDPG Position Sizer          | 20 ₽      |          1.836 | 0.20%          |
| SMA Crossover Baseline       | 20 ₽      |          0.744 | 0.20%          |
| Isolation Forest Risk Switch | 9 ₽       |          0.404 | 0.09%          |
| PPO Position Sizer           | 7 ₽       |          1.828 | 0.07%          |
| SAC Position Sizer           | 1 ₽       |          1.779 | 0.01%          |
| Symbolic Regression Alpha    | 0 ₽       |          0     | 0.00%          |
| Lasso Factor Model           | 0 ₽       |          0     | 0.00%          |

## Biggest losses (P&L, RUB)

| algorithm_name                 | pnl_rub   |   sharpe_ratio | total_return   |
|:-------------------------------|:----------|---------------:|:---------------|
| Random Position Baseline       | -239 ₽    |         -9.236 | -2.39%         |
| LightGBM Ranker                | -113 ₽    |         -7.307 | -1.13%         |
| XGBoost Ranker                 | -95 ₽     |         -6.444 | -0.95%         |
| CatBoost Ranker                | -92 ₽     |         -8.354 | -0.92%         |
| Logistic Regression Direction  | -86 ₽     |         -5.484 | -0.86%         |
| Random Forest Baseline         | -57 ₽     |         -5.842 | -0.57%         |
| SVM RBF Direction Classifier   | -42 ₽     |         -5.627 | -0.42%         |
| PatchTST Transformer Predictor | -38 ₽     |         -2.862 | -0.38%         |
| TCN Predictor                  | -30 ₽     |         -3.56  | -0.30%         |
| GRU Predictor                  | -30 ₽     |         -2.037 | -0.30%         |

## Compute load per algorithm

| algorithm_name                          |   fit_seconds |   fit_peak_rss_mb | fit_rss_delta_mb   |   avg_tick_ms |   tick_cpu_seconds_total | cpu_load_share_pct   |
|:----------------------------------------|--------------:|------------------:|:-------------------|--------------:|-------------------------:|:---------------------|
| Candlestick Pattern CNN                 |        28.921 |             461.5 |                    |      2388.13  |                  191.389 | 33.0%                |
| TCN Predictor                           |        28.833 |             310.5 |                    |       333.326 |                   98.882 | 17.0%                |
| PPO Position Sizer                      |         0.63  |             402.2 |                    |       120.235 |                   47.449 | 8.2%                 |
| DDPG Position Sizer                     |        20.231 |             405.6 |                    |        93.876 |                   31.678 | 5.5%                 |
| Informer-lite Transformer Predictor     |         6.12  |             875   |                    |        76.856 |                   25.337 | 4.4%                 |
| LSTM Predictor                          |         1.793 |             731.8 |                    |        42.679 |                   17.585 | 3.0%                 |
| One-Class SVM Risk Switch               |         0.049 |             205.4 |                    |        40.595 |                   16.696 | 2.9%                 |
| SVM RBF Direction Classifier            |         1.555 |             281.9 |                    |       463.096 |                   16.02  | 2.8%                 |
| Thompson Sampling Capital Allocator     |         0.003 |             172   |                    |        44.084 |                   15.09  | 2.6%                 |
| Kalman Filter Pairs Trading             |         0.041 |             173.8 |                    |        43.939 |                   14.849 | 2.6%                 |
| Random Forest Baseline                  |         0.265 |             219.2 |                    |        47.79  |                   13.075 | 2.3%                 |
| Isolation Forest Risk Switch            |         0.138 |             206.7 |                    |        36.03  |                   12.235 | 2.1%                 |
| GRU Predictor                           |         1.779 |             672.4 |                    |        37.077 |                   11.763 | 2.0%                 |
| Gaussian Process Trader                 |         2.479 |             411.8 |                    |        33.959 |                   11.3   | 1.9%                 |
| PatchTST Transformer Predictor          |         1.255 |             652.5 |                    |        19.917 |                    6.49  | 1.1%                 |
| CatBoost Ranker                         |         0.267 |             206.4 |                    |        17.496 |                    5.763 | 1.0%                 |
| Meta-Labeling (Lopez de Prado)          |         1.98  |             208.1 |                    |        13.168 |                    5.518 | 1.0%                 |
| News Sentiment Signal (Position Memory) |         0     |             202.2 |                    |       493.96  |                    4.373 | 0.8%                 |
| LightGBM Ranker                         |         0.181 |             199.3 |                    |        11.977 |                    4.189 | 0.7%                 |
| SAC Position Sizer                      |        25.122 |             291.6 |                    |       107.566 |                    3.767 | 0.6%                 |
| XGBoost Ranker                          |         0.159 |             201.2 |                    |        10.395 |                    3.495 | 0.6%                 |
| Logistic Regression Direction           |         0.029 |             195.7 |                    |         8.102 |                    3.458 | 0.6%                 |
| Elastic Net Factor Model                |         0.027 |             192.7 |                    |         9.542 |                    3.255 | 0.6%                 |
| VAE Factor Model                        |         0.812 |             426.1 |                    |         8.29  |                    2.81  | 0.5%                 |
| Lasso Factor Model                      |         0.021 |             192.8 |                    |         8.399 |                    2.744 | 0.5%                 |
| Symbolic Regression Alpha               |         0.12  |             180.8 |                    |         7.659 |                    2.57  | 0.4%                 |
| Autoencoder Factor Model                |         0.932 |             429.5 |                    |        31.075 |                    2.504 | 0.4%                 |
| N-HiTS Forecaster                       |         0.706 |             387.3 |                    |         4.974 |                    1.976 | 0.3%                 |
| N-BEATS Forecaster                      |         0.644 |             418   |                    |         4.498 |                    1.709 | 0.3%                 |
| News Sentiment Signal                   |         0     |             172.1 |                    |         2.978 |                    1.216 | 0.2%                 |
| HMM Regime Detector                     |         0.103 |             202.5 |                    |         1.852 |                    0.623 | 0.1%                 |
| Random Position Baseline                |         0     |             172.2 |                    |         0.572 |                    0.186 | 0.0%                 |
| Correlation Clustering                  |         0.008 |             175.8 |                    |         0.455 |                    0.151 | 0.0%                 |
| HDBSCAN Pairs/Basket Clustering         |         0.01  |             197.4 |                    |         0.412 |                    0.136 | 0.0%                 |
| Buy and Hold Baseline                   |         0     |             164.9 |                    |         0.127 |                    0.043 | 0.0%                 |
| SMA Crossover Baseline                  |         0     |             174.3 |                    |         0.683 |                    0.024 | 0.0%                 |

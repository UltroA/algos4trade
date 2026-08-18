# Live market simulation results

Mode: **demo**  |  Generated: 2026-08-13T22:29:21.249201+00:00
Session started: 2026-08-13T22:02:19.262318+00:00  |  Elapsed: 1622s  |  Ticks: 407
Algorithms run: 35  |  Skipped (warm-up fit failed): 0

## Process load

- simulator PID 42350: CPU 989%, RSS 1088 MB, 45 threads

## T-Invest API latency observed this session

- no T-Invest API calls recorded this session yet

In live mode this is real measured latency (no delay is simulated - the wall-clock pacing of the session IS the real network round trip). In demo mode, per-tick sleeps are sampled from these same measured values instead of a guessed constant.
| spec_name              | algorithm_name                      | category                | tickers                                           | total_return   | annualized_return   |   sharpe_ratio | max_drawdown   | win_rate   | hit_rate   |   n_trades | starting_capital   | final_capital   | pnl_rub   | error   |
|:-----------------------|:------------------------------------|:------------------------|:--------------------------------------------------|:---------------|:--------------------|---------------:|:---------------|:-----------|:-----------|-----------:|:-------------------|:----------------|:----------|:--------|
| autoencoder            | Autoencoder Factor Model            | representation_learning | SBER                                              | -1.23%         | -0.76%              |         -2.756 | -1.24%         | 40.05%     | 50.25%     |        407 | 10 000 ₽           | 9 877 ₽         | -123 ₽    |         |
| buy_and_hold           | Buy and Hold Baseline               | baseline                | SBER                                              | -3.00%         | -1.87%              |         -1.653 | -3.24%         | 45.45%     | 45.57%     |          1 | 10 000 ₽           | 9 700 ₽         | -300 ₽    |         |
| catboost_ranker        | CatBoost Ranker                     | supervised_ranking      | SBER                                              | -5.36%         | -3.35%              |         -9.474 | -5.36%         | 17.20%     | 49.75%     |        407 | 10 000 ₽           | 9 464 ₽         | -536 ₽    |         |
| cnn_candlestick        | Candlestick Pattern CNN             | pattern_recognition     | SBER                                              | -8.23%         | -5.18%              |         -7.779 | -8.24%         | 22.85%     | 48.28%     |        403 | 10 000 ₽           | 9 177 ₽         | -823 ₽    |         |
| correlation_clustering | Correlation Clustering              | clustering              | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | -0.33%         | -0.20%              |         -0.772 | -0.37%         | 48.16%     | 47.21%     |          1 | 10 000 ₽           | 9 967 ₽         | -33 ₽     |         |
| ddpg_agent             | DDPG Position Sizer                 | reinforcement_learning  | SBER                                              | -0.26%         | -0.16%              |         -1.69  | -0.28%         | 45.45%     | 45.57%     |        407 | 10 000 ₽           | 9 974 ₽         | -26 ₽     |         |
| elastic_net            | Elastic Net Factor Model            | linear_factor           | SBER                                              | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| gaussian_process       | Gaussian Process Trader             | bayesian_optimization   | SBER                                              | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| genetic_programming    | Symbolic Regression Alpha           | symbolic_regression     | SBER                                              | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| gru_predictor          | GRU Predictor                       | sequence_model          | SBER                                              | -1.29%         | -0.80%              |         -3.051 | -1.29%         | 37.10%     | 47.54%     |        407 | 10 000 ₽           | 9 871 ₽         | -129 ₽    |         |
| hdbscan_clustering     | HDBSCAN Pairs/Basket Clustering     | clustering              | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| hmm_regime             | HMM Regime Detector                 | regime_detection        | SBER                                              | 3.05%          | 1.88%               |          1.642 | -0.56%         | 50.61%     | 50.74%     |          1 | 10 000 ₽           | 10 305 ₽        | 305 ₽     |         |
| informer               | Informer-lite Transformer Predictor | sequence_model          | SBER                                              | -0.36%         | -0.22%              |         -0.981 | -0.51%         | 41.77%     | 49.51%     |        407 | 10 000 ₽           | 9 964 ₽         | -36 ₽     |         |
| isolation_forest       | Isolation Forest Risk Switch        | anomaly_detection       | SBER                                              | -3.25%         | -2.02%              |         -1.841 | -3.63%         | 39.18%     | 42.53%     |         16 | 10 000 ₽           | 9 675 ₽         | -325 ₽    |         |
| kalman_filter_pairs    | Kalman Filter Pairs Trading         | pairs_stat_arb          | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | -0.13%         | -0.08%              |         -0.506 | -0.21%         | 47.91%     | 47.17%     |          1 | 10 000 ₽           | 9 987 ₽         | -13 ₽     |         |
| lasso                  | Lasso Factor Model                  | linear_factor           | SBER                                              | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| lightgbm_ranker        | LightGBM Ranker                     | supervised_ranking      | SBER                                              | -7.64%         | -4.80%              |         -9.887 | -7.64%         | 15.48%     | 49.51%     |        403 | 10 000 ₽           | 9 236 ₽         | -764 ₽    |         |
| logistic_regression    | Logistic Regression Direction       | supervised_ranking      | SBER                                              | -3.66%         | -2.28%              |         -6.855 | -3.66%         | 19.41%     | 47.78%     |        407 | 10 000 ₽           | 9 634 ₽         | -366 ₽    |         |
| lstm_predictor         | LSTM Predictor                      | sequence_model          | SBER                                              | -1.28%         | -0.80%              |         -2.801 | -1.29%         | 38.57%     | 48.52%     |        407 | 10 000 ₽           | 9 872 ₽         | -128 ₽    |         |
| meta_labeling          | Meta-Labeling (Lopez de Prado)      | meta_labeling           | SBER                                              | -0.72%         | -0.45%              |         -3.935 | -0.73%         | 10.11%     | 46.07%     |        135 | 10 000 ₽           | 9 928 ₽         | -72 ₽     |         |
| nbeats                 | N-BEATS Forecaster                  | time_series_forecast    | SBER                                              | 0.03%          | 0.02%               |          0.85  | -0.02%         | 49.39%     | 50.74%     |        407 | 10 000 ₽           | 10 003 ₽        | 3 ₽       |         |
| news_sentiment         | News Sentiment Signal               | news_sentiment          | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 10 000 ₽           | 10 000 ₽        | 0 ₽       |         |
| nhits                  | N-HiTS Forecaster                   | time_series_forecast    | SBER                                              | -0.03%         | -0.02%              |         -5.327 | -0.03%         | 29.48%     | 44.33%     |        407 | 10 000 ₽           | 9 997 ₽         | -3 ₽      |         |
| one_class_svm          | One-Class SVM Risk Switch           | anomaly_detection       | SBER                                              | -3.01%         | -1.87%              |         -1.614 | -3.67%         | 40.30%     | 42.93%     |         19 | 10 000 ₽           | 9 699 ₽         | -301 ₽    |         |
| ppo_agent              | PPO Position Sizer                  | reinforcement_learning  | SBER                                              | -0.40%         | -0.25%              |         -1.668 | -0.43%         | 45.45%     | 45.57%     |        407 | 10 000 ₽           | 9 960 ₽         | -40 ₽     |         |
| random_baseline        | Random Position Baseline            | baseline                | SBER                                              | -19.61%        | -12.64%             |         -9.701 | -19.61%        | 23.34%     | 43.84%     |          1 | 10 000 ₽           | 8 039 ₽         | -1 961 ₽  |         |
| random_forest          | Random Forest Baseline              | supervised_ranking      | SBER                                              | -4.14%         | -2.59%              |         -8.57  | -4.14%         | 21.13%     | 49.75%     |        407 | 10 000 ₽           | 9 586 ₽         | -414 ₽    |         |
| sac_agent              | SAC Position Sizer                  | reinforcement_learning  | SBER                                              | -0.04%         | -0.03%              |         -1.694 | -0.05%         | 45.45%     | 45.57%     |        407 | 10 000 ₽           | 9 996 ₽         | -4 ₽      |         |
| sma_crossover          | SMA Crossover Baseline              | baseline                | SBER                                              | -2.39%         | -1.49%              |         -1.247 | -3.13%         | 41.77%     | 43.10%     |          1 | 10 000 ₽           | 9 761 ₽         | -239 ₽    |         |
| svm_rbf                | SVM RBF Direction Classifier        | supervised_ranking      | SBER                                              | -4.06%         | -2.53%              |         -8.332 | -4.06%         | 21.85%     | 47.16%     |        406 | 10 000 ₽           | 9 594 ₽         | -406 ₽    |         |
| tcn_predictor          | TCN Predictor                       | sequence_model          | SBER                                              | -1.88%         | -1.17%              |         -4.142 | -1.89%         | 30.96%     | 51.72%     |        407 | 10 000 ₽           | 9 812 ₽         | -188 ₽    |         |
| thompson_bandits       | Thompson Sampling Capital Allocator | capital_allocation      | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX | -0.46%         | -0.28%              |         -2.227 | -0.61%         | 34.89%     | 45.63%     |        109 | 10 000 ₽           | 9 954 ₽         | -46 ₽     |         |
| transformer_patchtst   | PatchTST Transformer Predictor      | sequence_model          | SBER                                              | -3.83%         | -2.39%              |         -8.162 | -3.83%         | 19.41%     | 50.49%     |        407 | 10 000 ₽           | 9 617 ₽         | -383 ₽    |         |
| vae                    | VAE Factor Model                    | representation_learning | SBER                                              | -1.98%         | -1.23%              |         -4.141 | -1.99%         | 37.35%     | 50.00%     |        407 | 10 000 ₽           | 9 802 ₽         | -198 ₽    |         |
| xgboost_ranker         | XGBoost Ranker                      | supervised_ranking      | SBER                                              | -7.23%         | -4.54%              |         -9.562 | -7.23%         | 16.71%     | 51.23%     |        406 | 10 000 ₽           | 9 277 ₽         | -723 ₽    |         |

## Summary: who performed better, who worse

- 35/35 algorithms completed without errors (0 failed); of those, 2/35 ended profitable (positive money P&L).
- Best by Sharpe: **HMM Regime Detector** (Sharpe +1.642, 305 ₽).
- Worst by Sharpe: **LightGBM Ranker** (Sharpe -9.887, -764 ₽).
- Made the most money: **HMM Regime Detector** (305 ₽, Sharpe +1.642).
- Lost the most money: **Random Position Baseline** (-1 961 ₽, Sharpe -9.701).

## Top 10 by Sharpe ratio (out-of-sample)

| algorithm_name                  |   sharpe_ratio | total_return   | max_drawdown   | pnl_rub   |
|:--------------------------------|---------------:|:---------------|:---------------|:----------|
| HMM Regime Detector             |          1.642 | 3.05%          | -0.56%         | 305 ₽     |
| N-BEATS Forecaster              |          0.85  | 0.03%          | -0.02%         | 3 ₽       |
| Gaussian Process Trader         |          0     | 0.00%          | 0.00%          | 0 ₽       |
| Lasso Factor Model              |          0     | 0.00%          | 0.00%          | 0 ₽       |
| HDBSCAN Pairs/Basket Clustering |          0     | 0.00%          | 0.00%          | 0 ₽       |
| News Sentiment Signal           |          0     | 0.00%          | 0.00%          | 0 ₽       |
| Symbolic Regression Alpha       |          0     | 0.00%          | 0.00%          | 0 ₽       |
| Elastic Net Factor Model        |          0     | 0.00%          | 0.00%          | 0 ₽       |
| Kalman Filter Pairs Trading     |         -0.506 | -0.13%         | -0.21%         | -13 ₽     |
| Correlation Clustering          |         -0.772 | -0.33%         | -0.37%         | -33 ₽     |

## Top 10 by money made (P&L, RUB)

| algorithm_name                  | pnl_rub   |   sharpe_ratio | total_return   |
|:--------------------------------|:----------|---------------:|:---------------|
| HMM Regime Detector             | 305 ₽     |          1.642 | 3.05%          |
| N-BEATS Forecaster              | 3 ₽       |          0.85  | 0.03%          |
| Gaussian Process Trader         | 0 ₽       |          0     | 0.00%          |
| Lasso Factor Model              | 0 ₽       |          0     | 0.00%          |
| HDBSCAN Pairs/Basket Clustering | 0 ₽       |          0     | 0.00%          |
| News Sentiment Signal           | 0 ₽       |          0     | 0.00%          |
| Symbolic Regression Alpha       | 0 ₽       |          0     | 0.00%          |
| Elastic Net Factor Model        | 0 ₽       |          0     | 0.00%          |
| N-HiTS Forecaster               | -3 ₽      |         -5.327 | -0.03%         |
| SAC Position Sizer              | -4 ₽      |         -1.694 | -0.04%         |

## Biggest losses (P&L, RUB)

| algorithm_name                 | pnl_rub   |   sharpe_ratio | total_return   |
|:-------------------------------|:----------|---------------:|:---------------|
| Random Position Baseline       | -1 961 ₽  |         -9.701 | -19.61%        |
| Candlestick Pattern CNN        | -823 ₽    |         -7.779 | -8.23%         |
| LightGBM Ranker                | -764 ₽    |         -9.887 | -7.64%         |
| XGBoost Ranker                 | -723 ₽    |         -9.562 | -7.23%         |
| CatBoost Ranker                | -536 ₽    |         -9.474 | -5.36%         |
| Random Forest Baseline         | -414 ₽    |         -8.57  | -4.14%         |
| SVM RBF Direction Classifier   | -406 ₽    |         -8.332 | -4.06%         |
| PatchTST Transformer Predictor | -383 ₽    |         -8.162 | -3.83%         |
| Logistic Regression Direction  | -366 ₽    |         -6.855 | -3.66%         |
| Isolation Forest Risk Switch   | -325 ₽    |         -1.841 | -3.25%         |

## Compute load per algorithm

| algorithm_name                      |   fit_seconds |   fit_peak_rss_mb | fit_rss_delta_mb   |   avg_tick_ms |   tick_cpu_seconds_total | cpu_load_share_pct   |
|:------------------------------------|--------------:|------------------:|:-------------------|--------------:|-------------------------:|:---------------------|
| Candlestick Pattern CNN             |        27.863 |             231.4 |                    |      2356.53  |                 2200.58  | 33.9%                |
| TCN Predictor                       |        27.589 |             549.6 |                    |       292.452 |                 1081.33  | 16.7%                |
| PPO Position Sizer                  |         0.622 |             398.2 |                    |       115.617 |                  548.455 | 8.5%                 |
| DDPG Position Sizer                 |        19.343 |             402   |                    |        90.312 |                  358.975 | 5.5%                 |
| Informer-lite Transformer Predictor |         5.708 |             867.5 |                    |        66.394 |                  263.381 | 4.1%                 |
| One-Class SVM Risk Switch           |         0.042 |             198.6 |                    |        40.904 |                  205.468 | 3.2%                 |
| LSTM Predictor                      |         1.704 |             704.8 |                    |        39.998 |                  198.551 | 3.1%                 |
| SVM RBF Direction Classifier        |         1.495 |             270.1 |                    |       470.41  |                  191.388 | 3.0%                 |
| Kalman Filter Pairs Trading         |         0.04  |             172.9 |                    |        43.694 |                  173.846 | 2.7%                 |
| Thompson Sampling Capital Allocator |         0.003 |             169.6 |                    |        43.185 |                  171.765 | 2.6%                 |
| Random Forest Baseline              |         0.269 |             218   |                    |        47.79  |                  161.519 | 2.5%                 |
| Isolation Forest Risk Switch        |         0.128 |             202.7 |                    |        33.176 |                  131.637 | 2.0%                 |
| Gaussian Process Trader             |         2.601 |             409   |                    |        29.941 |                  119.259 | 1.8%                 |
| GRU Predictor                       |         1.712 |             681.9 |                    |        26.607 |                  104.827 | 1.6%                 |
| PatchTST Transformer Predictor      |         1.131 |             643.5 |                    |        18.916 |                   74.183 | 1.1%                 |
| Meta-Labeling (Lopez de Prado)      |         1.915 |             205.2 |                    |        12.856 |                   64.676 | 1.0%                 |
| SAC Position Sizer                  |        22.772 |             400.7 |                    |       103.882 |                   48.698 | 0.8%                 |
| LightGBM Ranker                     |         0.171 |             200.8 |                    |        10.364 |                   45.082 | 0.7%                 |
| Logistic Regression Direction       |         0.023 |             195   |                    |         7.946 |                   40.027 | 0.6%                 |
| Elastic Net Factor Model            |         0.053 |             190.9 |                    |         8.493 |                   33.58  | 0.5%                 |
| XGBoost Ranker                      |         0.223 |             202.9 |                    |         8.364 |                   33.222 | 0.5%                 |
| Lasso Factor Model                  |         0.029 |             191.9 |                    |         8.034 |                   32.048 | 0.5%                 |
| CatBoost Ranker                     |         0.258 |             209   |                    |         7.932 |                   31.491 | 0.5%                 |
| Autoencoder Factor Model            |         0.896 |             429.8 |                    |         7.85  |                   31.102 | 0.5%                 |
| VAE Factor Model                    |         0.806 |             428.8 |                    |         7.834 |                   30.795 | 0.5%                 |
| Symbolic Regression Alpha           |         0.118 |             178.5 |                    |         7.255 |                   28.659 | 0.4%                 |
| N-HiTS Forecaster                   |         0.699 |             423.8 |                    |         4.279 |                   21.251 | 0.3%                 |
| N-BEATS Forecaster                  |         0.667 |             419.3 |                    |         3.973 |                   19.619 | 0.3%                 |
| Correlation Clustering              |         0.008 |             176.2 |                    |         4.54  |                   17.784 | 0.3%                 |
| News Sentiment Signal               |         0     |             171.3 |                    |         2.622 |                   13.226 | 0.2%                 |
| HMM Regime Detector                 |         0.112 |             206.4 |                    |         1.443 |                    5.686 | 0.1%                 |
| HDBSCAN Pairs/Basket Clustering     |         0.009 |             198.3 |                    |         0.39  |                    1.535 | 0.0%                 |
| Random Position Baseline            |         0     |             174.4 |                    |         0.135 |                    0.527 | 0.0%                 |
| SMA Crossover Baseline              |         0     |             171.3 |                    |         0.641 |                    0.255 | 0.0%                 |
| Buy and Hold Baseline               |         0     |             172.5 |                    |         0.045 |                    0.178 | 0.0%                 |

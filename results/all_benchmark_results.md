# Trading algorithm benchmark results

Generated: 2026-08-18T14:02:06.674295+00:00
| spec_name              | algorithm_name                          | category                | tickers                                           |   train_seconds |   inference_seconds |   n_train_rows |   n_test_rows | total_return   | annualized_return   |   sharpe_ratio | max_drawdown   | win_rate   | hit_rate   |   n_trades | starting_capital   | final_capital   | pnl_rub    | error   |
|:-----------------------|:----------------------------------------|:------------------------|:--------------------------------------------------|----------------:|--------------------:|---------------:|--------------:|:---------------|:--------------------|---------------:|:---------------|:-----------|:-----------|-----------:|:-------------------|:----------------|:-----------|:--------|
| autoencoder            | Autoencoder Factor Model                | representation_learning | SBER                                              |           0.704 |               0.006 |           1419 |           609 | -3.90%         | -1.63%              |         -0.381 | -9.59%         | 51.91%     | 52.19%     |        549 | 1 000 000 ₽        | 961 018 ₽       | -38 982 ₽  |         |
| buy_and_hold           | Buy and Hold Baseline                   | baseline                | SBER                                              |           0     |               0     |           1419 |           609 | 4.43%          | 1.81%               |          0.191 | -23.84%        | 51.56%     | 51.64%     |          1 | 1 000 000 ₽        | 1 044 254 ₽     | 44 254 ₽   |         |
| catboost_ranker        | CatBoost Ranker                         | supervised_ranking      | SBER                                              |           0.194 |               0.008 |           1419 |           609 | -6.14%         | -2.59%              |         -0.225 | -11.03%        | 44.81%     | 51.09%     |        538 | 1 000 000 ₽        | 938 631 ₽       | -61 369 ₽  |         |
| cnn_candlestick        | Candlestick Pattern CNN                 | pattern_recognition     | SBER                                              |           9.061 |               0.376 |           1419 |           609 | -18.73%        | -8.23%              |         -0.679 | -22.62%        | 44.05%     | 49.74%     |        552 | 1 000 000 ₽        | 812 654 ₽       | -187 346 ₽ |         |
| correlation_clustering | Correlation Clustering                  | clustering              | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX |           0.006 |               0.006 |          14179 |          6086 | -8.13%         | -3.43%              |         -0.753 | -12.43%        | 46.36%     | 49.78%     |         11 | 1 000 000 ₽        | 918 692 ₽       | -81 308 ₽  |         |
| ddpg_agent             | DDPG Position Sizer                     | reinforcement_learning  | SBER                                              |           6.316 |               0.012 |           1419 |           609 | 1.35%          | 0.56%               |          0.124 | -21.63%        | 51.61%     | 51.70%     |        589 | 1 000 000 ₽        | 1 013 547 ₽     | 13 547 ₽   |         |
| elastic_net            | Elastic Net Factor Model                | linear_factor           | SBER                                              |           0.023 |               0.006 |           1419 |           609 | -24.54%        | -11.00%             |         -1.508 | -24.87%        | 41.89%     | 47.81%     |        549 | 1 000 000 ₽        | 754 564 ₽       | -245 436 ₽ |         |
| gaussian_process       | Gaussian Process Trader                 | bayesian_optimization   | SBER                                              |           2.193 |               0.009 |           1419 |           609 | -0.08%         | -0.03%              |         -0.134 | -0.49%         | 43.88%     | 54.85%     |        273 | 1 000 000 ₽        | 999 231 ₽       | -769 ₽     |         |
| genetic_programming    | Symbolic Regression Alpha               | symbolic_regression     | SBER                                              |           1.298 |               0.006 |           1419 |           609 | 0.26%          | 0.11%               |          0.888 | -0.10%         | 51.44%     | 51.70%     |        589 | 1 000 000 ₽        | 1 002 603 ₽     | 2 603 ₽    |         |
| gru_predictor          | GRU Predictor                           | sequence_model          | SBER                                              |           1.179 |               0.011 |           1419 |           609 | -7.36%         | -3.12%              |         -0.54  | -12.58%        | 51.32%     | 52.17%     |        530 | 1 000 000 ₽        | 926 365 ₽       | -73 635 ₽  |         |
| hdbscan_clustering     | HDBSCAN Pairs/Basket Clustering         | clustering              | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX |           0.009 |               0.006 |          14179 |          6086 | -16.06%        | -6.95%              |         -1.137 | -17.24%        | 44.67%     | 48.87%     |         13 | 1 000 000 ₽        | 839 415 ₽       | -160 585 ₽ |         |
| hmm_regime             | HMM Regime Detector                     | regime_detection        | SBER                                              |           0.081 |               0.002 |           1419 |           609 | -17.81%        | -7.80%              |         -0.426 | -21.49%        | 51.18%     | 51.67%     |         15 | 1 000 000 ₽        | 821 904 ₽       | -178 096 ₽ |         |
| informer               | Informer-lite Transformer Predictor     | sequence_model          | SBER                                              |           2.197 |               0.019 |           1419 |           609 | -6.79%         | -2.87%              |         -0.29  | -13.84%        | 50.57%     | 50.85%     |        530 | 1 000 000 ₽        | 932 133 ₽       | -67 867 ₽  |         |
| isolation_forest       | Isolation Forest Risk Switch            | anomaly_detection       | SBER                                              |           0.114 |               0.014 |           1419 |           609 | 8.20%          | 3.31%               |          0.262 | -22.48%        | 48.15%     | 48.80%     |         24 | 1 000 000 ₽        | 1 081 996 ₽     | 81 996 ₽   |         |
| kalman_filter_pairs    | Kalman Filter Pairs Trading             | pairs_stat_arb          | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX |           0.013 |               0.006 |          14179 |          6086 | -6.48%         | -2.72%              |         -1.213 | -7.84%         | 43.03%     | 47.29%     |          9 | 1 000 000 ₽        | 935 247 ₽       | -64 753 ₽  |         |
| lasso                  | Lasso Factor Model                      | linear_factor           | SBER                                              |           0.017 |               0.006 |           1419 |           609 | -24.24%        | -10.85%             |         -1.508 | -24.60%        | 41.89%     | 47.99%     |        549 | 1 000 000 ₽        | 757 586 ₽       | -242 414 ₽ |         |
| lightgbm_ranker        | LightGBM Ranker                         | supervised_ranking      | SBER                                              |           0.103 |               0.009 |           1419 |           609 | -20.42%        | -9.02%              |         -0.64  | -21.75%        | 44.99%     | 50.91%     |        505 | 1 000 000 ₽        | 795 844 ₽       | -204 156 ₽ |         |
| logistic_regression    | Logistic Regression Direction           | supervised_ranking      | SBER                                              |           0.017 |               0.006 |           1419 |           609 | -27.86%        | -12.64%             |         -1.195 | -31.66%        | 47.54%     | 51.09%     |        517 | 1 000 000 ₽        | 721 384 ₽       | -278 616 ₽ |         |
| lstm_predictor         | LSTM Predictor                          | sequence_model          | SBER                                              |           1.23  |               0.014 |           1419 |           609 | -4.05%         | -1.69%              |         -0.44  | -7.60%         | 51.70%     | 52.17%     |        530 | 1 000 000 ₽        | 959 545 ₽       | -40 455 ₽  |         |
| meta_labeling          | Meta-Labeling (Lopez de Prado)          | meta_labeling           | SBER                                              |           0.534 |               0.008 |           1419 |           609 | -8.52%         | -3.62%              |         -1.412 | -8.96%         | 26.64%     | 45.95%     |        351 | 1 000 000 ₽        | 914 781 ₽       | -85 219 ₽  |         |
| nbeats                 | N-BEATS Forecaster                      | time_series_forecast    | SBER                                              |           0.54  |               0.001 |           1419 |           609 | 0.69%          | 0.29%               |          0.342 | -1.06%         | 49.83%     | 53.66%     |        588 | 1 000 000 ₽        | 1 006 937 ₽     | 6 937 ₽    |         |
| news_sentiment         | News Sentiment Signal                   | news_sentiment          | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX |           0     |               0     |          14179 |          6086 | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 1 000 000 ₽        | 1 000 000 ₽     | 0 ₽        |         |
| news_sentiment_memory  | News Sentiment Signal (Position Memory) | news_sentiment          | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX |           0     |               0.001 |          14179 |          6086 | 0.00%          | 0.00%               |          0     | 0.00%          | 0.00%      |            |          0 | 1 000 000 ₽        | 1 000 000 ₽     | 0 ₽        |         |
| nhits                  | N-HiTS Forecaster                       | time_series_forecast    | SBER                                              |           0.541 |               0.002 |           1419 |           609 | -1.12%         | -0.46%              |         -0.579 | -1.75%         | 48.80%     | 51.46%     |        584 | 1 000 000 ₽        | 988 823 ₽       | -11 177 ₽  |         |
| one_class_svm          | One-Class SVM Risk Switch               | anomaly_detection       | SBER                                              |           0.018 |               0.008 |           1419 |           609 | 4.12%          | 1.68%               |          0.185 | -21.07%        | 40.59%     | 49.87%     |        137 | 1 000 000 ₽        | 1 041 152 ₽     | 41 152 ₽   |         |
| ppo_agent              | PPO Position Sizer                      | reinforcement_learning  | SBER                                              |           0.591 |               0.015 |           1419 |           609 | 1.15%          | 0.48%               |          0.172 | -3.56%         | 51.61%     | 51.70%     |        589 | 1 000 000 ₽        | 1 011 537 ₽     | 11 537 ₽   |         |
| random_baseline        | Random Position Baseline                | baseline                | SBER                                              |           0     |               0     |           1419 |           609 | -52.12%        | -26.27%             |         -1.255 | -55.62%        | 46.47%     | 50.33%     |        322 | 1 000 000 ₽        | 478 842 ₽       | -521 158 ₽ |         |
| random_forest          | Random Forest Baseline                  | supervised_ranking      | SBER                                              |           0.223 |               0.035 |           1419 |           609 | -5.84%         | -2.46%              |         -0.334 | -8.25%         | 45.54%     | 50.91%     |        549 | 1 000 000 ₽        | 941 565 ₽       | -58 435 ₽  |         |
| sac_agent              | SAC Position Sizer                      | reinforcement_learning  | SBER                                              |           7.715 |               0.014 |           1419 |           609 | 0.09%          | 0.04%               |          0.239 | -0.15%         | 50.76%     | 51.36%     |        589 | 1 000 000 ₽        | 1 000 871 ₽     | 871 ₽      |         |
| sma_crossover          | SMA Crossover Baseline                  | baseline                | SBER                                              |           0     |               0.001 |           1419 |           609 | 9.26%          | 3.73%               |          0.276 | -24.43%        | 48.93%     | 49.01%     |         15 | 1 000 000 ₽        | 1 092 560 ₽     | 92 560 ₽   |         |
| svm_rbf                | SVM RBF Direction Classifier            | supervised_ranking      | SBER                                              |           0.15  |               0.025 |           1419 |           609 | -1.04%         | -0.43%              |         -0.157 | -3.56%         | 47.72%     | 52.19%     |        537 | 1 000 000 ₽        | 989 649 ₽       | -10 351 ₽  |         |
| tcn_predictor          | TCN Predictor                           | sequence_model          | SBER                                              |           8.893 |               0.078 |           1419 |           609 | -19.55%        | -8.61%              |         -0.984 | -27.44%        | 43.96%     | 48.39%     |        516 | 1 000 000 ₽        | 804 458 ₽       | -195 542 ₽ |         |
| thompson_bandits       | Thompson Sampling Capital Allocator     | capital_allocation      | SBER,GAZP,LKOH,GMKN,VTBR,ROSN,NVTK,MTSS,TATN,MOEX |           0.003 |               0.01  |          14179 |          6086 | -0.90%         | -0.37%              |         -0.197 | -3.91%         | 48.17%     | 48.55%     |         26 | 1 000 000 ₽        | 990 983 ₽       | -9 017 ₽   |         |
| transformer_patchtst   | PatchTST Transformer Predictor          | sequence_model          | SBER                                              |           0.657 |               0.008 |           1419 |           609 | -9.29%         | -3.96%              |         -0.385 | -18.56%        | 47.36%     | 51.04%     |        528 | 1 000 000 ₽        | 907 061 ₽       | -92 939 ₽  |         |
| vae                    | VAE Factor Model                        | representation_learning | SBER                                              |           0.571 |               0.006 |           1419 |           609 | 22.89%         | 8.90%               |          0.988 | -3.41%         | 49.54%     | 51.64%     |        541 | 1 000 000 ₽        | 1 228 894 ₽     | 228 894 ₽  |         |
| xgboost_ranker         | XGBoost Ranker                          | supervised_ranking      | SBER                                              |           0.094 |               0.007 |           1419 |           609 | -12.31%        | -5.29%              |         -0.402 | -15.98%        | 43.90%     | 49.82%     |        500 | 1 000 000 ₽        | 876 912 ₽       | -123 088 ₽ |         |

## Summary: who performed better, who worse

- 36/36 algorithms completed without errors (0 failed); of those, 10/36 ended profitable (positive money P&L).
- Best by Sharpe: **VAE Factor Model** (Sharpe +0.988, 228 894 ₽).
- Worst by Sharpe: **Elastic Net Factor Model** (Sharpe -1.508, -245 436 ₽).
- Made the most money: **VAE Factor Model** (228 894 ₽, Sharpe +0.988).
- Lost the most money: **Random Position Baseline** (-521 158 ₽, Sharpe -1.255).

## Top 10 by Sharpe ratio (out-of-sample)

| algorithm_name               |   sharpe_ratio | total_return   | max_drawdown   | pnl_rub   |
|:-----------------------------|---------------:|:---------------|:---------------|:----------|
| VAE Factor Model             |          0.988 | 22.89%         | -3.41%         | 228 894 ₽ |
| Symbolic Regression Alpha    |          0.888 | 0.26%          | -0.10%         | 2 603 ₽   |
| N-BEATS Forecaster           |          0.342 | 0.69%          | -1.06%         | 6 937 ₽   |
| SMA Crossover Baseline       |          0.276 | 9.26%          | -24.43%        | 92 560 ₽  |
| Isolation Forest Risk Switch |          0.262 | 8.20%          | -22.48%        | 81 996 ₽  |
| SAC Position Sizer           |          0.239 | 0.09%          | -0.15%         | 871 ₽     |
| Buy and Hold Baseline        |          0.191 | 4.43%          | -23.84%        | 44 254 ₽  |
| One-Class SVM Risk Switch    |          0.185 | 4.12%          | -21.07%        | 41 152 ₽  |
| PPO Position Sizer           |          0.172 | 1.15%          | -3.56%         | 11 537 ₽  |
| DDPG Position Sizer          |          0.124 | 1.35%          | -21.63%        | 13 547 ₽  |

## Top 10 by money made (P&L, RUB)

| algorithm_name               | pnl_rub   |   sharpe_ratio | total_return   |
|:-----------------------------|:----------|---------------:|:---------------|
| VAE Factor Model             | 228 894 ₽ |          0.988 | 22.89%         |
| SMA Crossover Baseline       | 92 560 ₽  |          0.276 | 9.26%          |
| Isolation Forest Risk Switch | 81 996 ₽  |          0.262 | 8.20%          |
| Buy and Hold Baseline        | 44 254 ₽  |          0.191 | 4.43%          |
| One-Class SVM Risk Switch    | 41 152 ₽  |          0.185 | 4.12%          |
| DDPG Position Sizer          | 13 547 ₽  |          0.124 | 1.35%          |
| PPO Position Sizer           | 11 537 ₽  |          0.172 | 1.15%          |
| N-BEATS Forecaster           | 6 937 ₽   |          0.342 | 0.69%          |
| Symbolic Regression Alpha    | 2 603 ₽   |          0.888 | 0.26%          |
| SAC Position Sizer           | 871 ₽     |          0.239 | 0.09%          |

## Biggest losses (P&L, RUB)

| algorithm_name                  | pnl_rub    |   sharpe_ratio | total_return   |
|:--------------------------------|:-----------|---------------:|:---------------|
| Random Position Baseline        | -521 158 ₽ |         -1.255 | -52.12%        |
| Logistic Regression Direction   | -278 616 ₽ |         -1.195 | -27.86%        |
| Elastic Net Factor Model        | -245 436 ₽ |         -1.508 | -24.54%        |
| Lasso Factor Model              | -242 414 ₽ |         -1.508 | -24.24%        |
| LightGBM Ranker                 | -204 156 ₽ |         -0.64  | -20.42%        |
| TCN Predictor                   | -195 542 ₽ |         -0.984 | -19.55%        |
| Candlestick Pattern CNN         | -187 346 ₽ |         -0.679 | -18.73%        |
| HMM Regime Detector             | -178 096 ₽ |         -0.426 | -17.81%        |
| HDBSCAN Pairs/Basket Clustering | -160 585 ₽ |         -1.137 | -16.06%        |
| XGBoost Ranker                  | -123 088 ₽ |         -0.402 | -12.31%        |

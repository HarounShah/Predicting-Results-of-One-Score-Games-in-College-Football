# Predicting Results of One Score Games in College Football

Data Collection:

Data Cleaning:

Exploratory Data Analysis:
- Dataset has mild class imbalance in one-score games (`54.63%` home wins vs `45.37%` home losses; `n=2623`), so no severe imbalance treatment is required.
- Distributional EDA (histograms/boxplots/correlation) plus statistical testing shows strong signal in turnover, pressure, and efficiency differentials.
- Simplified per-feature testing (Welch t-test + Cohen's d) indicates `tackles_diff` has negligible effect and is not statistically significant.
- Win-rate by quantile bins (`Figures/WinRateByFeatureBins.png`, `win_rate_by_feature_bins.csv`) confirms monotonic trends for several top features and improves interpretability.
- Because `turnovers_diff` already captures turnover events at aggregate level, `interceptions_diff` and `fumblesLost_diff` are treated as redundant for primary modeling.
- Proposed feature removals before modeling: `tackles_diff`, `interceptions_diff`, and `fumblesLost_diff`.
- Updated modeling feature removals: additionally drop `totalYards_diff` to reduce yardage redundancy (VIF/correlation suggests strong overlap and CV ablation showed negligible performance change).
- VIF on full EDA features (`vif_all_eda_features.csv`) includes all original `*_diff` variables before feature removal; yardage-related columns show very high VIF (near-linear overlap).

Machine Learning:
- Training/feature set: models train on `final_data_EDA.csv` (post-EDA reduced to 12 `*_diff` features + `home_win`).
- 2025 holdout testing: models are evaluated on `final_data_test.csv` built from 2025 season data (collected via `DataCollection(Test).py`, prepped via `DataPrep(Test).py`) and filtered to one-score games only.
- Column alignment: 2025 `X_test` is reindexed to match the training feature columns to ensure consistent inputs.
- Models: Logistic Regression, Random Forest, Gradient Boosting (sklearn), SVM, Gaussian Naive Bayes.
- Hyperparameter tuning: performed on training set only using CV AUC (GridSearch/RandomizedSearch), with 2025 kept untouched for final evaluation.
- Saved best hyperparameters: stored in `best_hyperparams.json`; `MachineLearning.py` can skip re-tuning on normal runs and reuse saved params.
- Reporting: script prints Train vs 2025 Test accuracy per model, plus 2025 Test AUC and saves confusion matrices and ROC curves using the tuned/saved models.
- Observation: Random Forest can show a large train–test gap (overfitting) even when 2025 performance is similar to simpler models; regularization can reduce the gap without major performance change.
- Additional reporting exports:
  - `model_comparison_table.csv` (Train accuracy, 2025 Test accuracy, 2025 Test AUC for each model)
  - Feature importance plots saved to `Figures/Feature Importance/` for LR, Random Forest, Gradient Boosting, and (linear) SVM.
- Consensus feature importance:
  - `feature_importance_consensus.png` ranks features by averaging feature-rank agreement across multiple models (lower rank-score = more important).
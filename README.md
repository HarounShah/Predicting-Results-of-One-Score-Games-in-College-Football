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
---
trigger: manual
---

Analysis plan

1. Features

A. Global - trajectory shape features
- NO baseline weight - focus on trajectory shapes, not absolute heaviness and loss. Similarly, exclude total final weight - focus only on proportions, not absolutes. 
- % max WL (nadir): % of loss from baseline to the lowest value. (BL weight - lowest weight) / BL weight * 100
- % regain: % of recovering weight that was lost. (final weight - nadir weight) / (BL weight - nadir weight) * 100. Importantly: this can be very high if the loss at nadir is small (0.5-2 kg) with large regain, and can cause a division by zero if the patient only gains (therefore, the baseline weight is the nadir). To avoid this, regain is set to a 100% if the max weight loss does not meet a clinically significant threshold of 2 kg AND the nadir is NOT the last weight. If the last weight is the lowest, everything is OK; but otherwise, the above problems can surface. Regain is set to a default of 100 - not 0, not to confuse absolute gainers/minimal losers with sustained losers whose last weight is the nadir, and regain is really 0. The clustering can still separate sustained losers (eg. max loss 10%, regain 0%), total regainers (max loss 10%, regain 100%) and non-responders (max loss 0%, regain 100%) and every edge case in between.

B. Granular - trajectory kinetics features
- Early WL speed: as the slope of an OLS fitted to all data points from 0 to 30 days. Fit the regression to % lost from baseline, not raw points, because that normalizes loss. The unit will be % lost per day, expected range is -0.5% to + 0.1%. This is slightly less direct than a pure division of % lost by day 30 / 30 days, but it is a bit sounder mathematically because it reduces variance from early weight fluctuations (water, etc). Importantly, if a patients has only 1 measurement in the first 30 days, open the window and fit the OLS to their first available post-baseline measurement (be it day 40, 50, 60 or whatever). 
- Time to nadir as proportion of total follow-up time: Tnadir/Tfinal (so it is a normalized proportion, not an absolute. 0 means first weight is lowest, 1 means last weight is lowest)
- Trajectory volatility: instead of local minima / maxima, fit a smooth line to the whole trajectory using LOESS, and calculate the variance of points around it using RMSE, root mean square error. The higher the variance, the higher the oscillation. LOESS moves a sliding window to the timeline, fitting simple lines to local data patterns. Set this sliding window to 30 days. When calculating this in Python, you will need a frac parameter, that can be the window span divided by total follow-up length. Normally you need a minimum frac to work well mathematically, you can set this to a floor, like max(calculated_frac, 0.10) to ensure you never use less than 10% of a patient’s data. What to fit this on: % lost from baseline, to keep everything relative and proportional. This way the feature unit, the RMSE will be the fluctuation in % of loss. Expected wide range is about 0 - 2. The higher it is, the higher the oscillations. 

C. Quality - body composition features
- Lean loss coefficient: proportion of muscle loss from weight loss at the point of greatest weight loss. Delta muscle change (kg) / delta weight change (kg) at nadir. This shows the proportion of weight loss that was muscle at the nadir. For example, if 10 kg total was lost with 2 kg muscle, the ratio is 0.2; if no muscle was lost, the ratio is 0. Better than muscle / fat ratios at endpoints, because those are more correlated with baseline body type. Calculate at nadir, because that is peak metabolic stress, and maximum loss. Measuring at the final weight can be noisy because if regain happened, muscle and fat are regained at different rates. Use absolute kgs, this is based on a supposed Forbes equation. This gives a clean proportion, derived from absolute values. Importantly, BIA muscle estimations depend on hydration status and can fluctuate. If this yields an extreme muscle loss value with moderate weight loss (can especially happen if absolute max loss is low); then the LLC can inflate into the hundreds of thousands of percents. For example, 0.1 kg weight loss with 0.5 kg estimated muscle loss yields 0.5/0.1 = 5, 500% of weight lost from muscle. Use two safeguards to prevent this: set LLC boundaries between 0 and 1, negative values are set to 0 and larger than 1 values are set to 1; AND, if total loss at nadir does not reach a clinically relevant 2 kg, do not calculate LLC and set it automatically to 0. 
- Muscle and fat correlation coefficient. Have an array of Fat values vs an array of Muscle values across the whole trajectory, and calculate the Pearson correlation coefficient of the two arrays. If strong positive, they move together (muscle loss), if strong negative, they move opposite (muscle gain), if 0, they are decoupled (muscle preservation). Also use absolute kg values, to have the coefficient be derived from the raw data, unaffected by percentage proportionality changes. 

D. Behavior - adherence features
- Observation length (days). Number of measurements is too collinear with observation length, exclude that. 
- Longest gap between measurements. This signals the longest disengagement period without a measurement. If it is large, it signals disengagement (eg. pause between records). If it is small, it signals good adherence, especially coupled with a large observation length. It is better to use than the average time between measurements (measurement density), because measurement density is uneven across the trajectory, usually drops by the end, and long follow-ups actually skew the average lower. 
- Number of 30+ day gaps. Signals disengagement frequency, tendencies. It is complementary to the previous but different: longest gap measures if there was even a disengagement at all, and if yes, how long, and number of long gaps tells more of a cycling, restarter behavior. Initially keep both, and check their correlation. If it is over 0.7, drop number of gaps, if it is low, keep both. Also, keep in mind that this variable will probably behave like a categorical, with a low number of distinct levels. This might bother variable scaling and distance calculations. 

Final feature set: 10 features, 5 Trajectory, 3 Adherence, 2 Composition

- % max loss (loss at nadir) = (baseline weight - lowest weight) / baseline weight * 100 [percentage, expected wide range +10 to -50]
- % regain (after nadir) = (final weight - lowest weight) / (baseline weight - lowest weight) * 100 [percentage, expected wide range 0 to 150]. Conditions: automatically set to 100 IF max weight loss (baseline weight - lowest weight) is under 2 kg AND at the same time lowest weight is not equal to final weight. 
- time to nadir (total weight loss speed) = Tnadir / Tfinal [ratio, range 0-1]
- first-month weight loss speed = slope of OLS fitted to the percentage of weight lost from baseline across all measurements before day 30. If a patient has under 2 measurements before day 30, extend the window until their first available post-baseline measurement. This gives % lost / day [percentage, expected range -0.5 to +0.1]
trajectory volatility: RMSE of measurement points against a loess smoothed line fitted to all the weight trajectory data, with points as % lost since baseline, using a 30-day moving window with max(calculated_frac, 0.10) [percentage, expected wide range 0-2]
- lean loss coefficient = muscle change at nadir (kg) / weight change at nadir (kg) [ratio, range 0-1]. Conditions: cap values in a range of 0-1, set negative values to 0 and larger than 1 values to 1, AND, automatically set to 0 if delta weight change at nadir is less than 2 kg. 
muscle-fat correlation coefficient = Pearson correlation coefficient of the array of fat mass (kg) and array of muscle mass (kg) values in the time series [ratio, range -1 - 1]
observation duration = days from first to last measurement [count, expected range 30-750]
number of 30+ day gaps = number of gaps between measurements exceeding 30 days [count, expected wide range 0-5]
longest gap: length of the longest gap between measurements [count, expected wide range 1-1000] 

WARNING: The features are good, but their proportions bias grouping towards shape and speed rather than adherence or quality. To keep this in mind, analyze feature correlations thoroughly, and maybe consider removing some features if they are highly correlated. Eg. early speed and time to nadir might correlate and the second can be dropped. 

2. Model selection

The selected features will influence which model will work best on the data. 

- GMM: Recommended. Soft clustering method, fits elliptical probability distributions to the data, rather than rigid spheres. If covariance_type=full, the model can deal with feature correlations, meaning feature selection can be a bit more relaxed. Scale/normalize data. Use BIC (as an elbow method), Silhouette, DB to evaluate. Computes in seconds. 
- HDBSCAN: groups the clear and dense patterns, and labels the strange ones as outliers (cluster -1). This makes the clusters cleaner and more clinically interpretable. Scale/normalize, DBCV (specific to density-based), Silhouette (this method might struggle with it) and DB for evals. Computes in seconds to minutes. 
- hclust ward: deterministic, highly interpretable, use dendrograms, DBI and Silhouette for evals. Normalize. Runs in minutes. 
hclust ward: deterministic, highly interpretable, use dendrograms, DBI and Silhouette for evals. Normalize. Runs in minutes. 
- k-means: spherical and similarly sized clusters assumed. This can be used for exploration, but rarely reflects biological reality. Euclidean distance metric, features must be scaled. Bad with correlated features, low feature collineality must be ensured. 

3. Methodology & Workflow 

- Extract features, analyize and possible save them prior to running clustering. Run a correlation matrix, if any two features have a strong (over 0.7 absolute) correlation, combine them or drop one. 
- Standardize data - Prefer a Robust Scaler using median and IQR over z scores, to better mitigate possible extreme outliers. 
- Run 3 layers of setups: 
- - A: Kinetic. Cluster only shape and kinetic features. 
- - B: Metabolic. Cluster only on the body composition features (muscle-loss, fat-muscle ratio). 
- - C: Behavioral. Cluster only on adherence features. 
- - D: Holistic. Combine kinetic, metabolic and behavioral features. 
- Evaluate the three approaches, evaluate clustering quality indices, as appropriate per method, and use a comprehensive method as well, that works for all. 
- Visualize raw data, with cluster assignments indicated. Use PCA, UMAP, and clinically interpretable features. 
- - Use linear PCA on the 10 features, see what the PCs are made of.
- - Use UMAP, a non-linear dimensionality reduction (it focuses on local rather than global relationships) the same way. 
- - Use a simple scatter plot with clinically meaningful axes, eg. total loss vs variance, or quality. If clusters are separated even on this plot, it means high clinical interpretability. 
- Plot k random trajectories from each cluster, sampling randomly at each rerun. If the k, eg. 10-20 samples within the same cluster are really different from each other, you know the clusters are not sensible. 
- Evaluate predictive utility: run a quick regression to predict cluster membership from age, sex and baseline BMI with candidate models. If these basic predictors cannot predict cluster membership, clusters are likely just noise, and have no clinical use. 
- Iterate, and pick the eventual best approach. If the pure trajectories are described well and consistently, proceed to describing them clinically, and then predicting them with an ensemble model. 


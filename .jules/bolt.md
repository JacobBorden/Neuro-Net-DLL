## $(date +%Y-%m-%d) - [Optimize NaiveBayesClassifier::predict]
**Learning:** The nested loops in `NaiveBayesClassifier::predict` perform redundant map lookups (`class_priors_.at`, `class_feature_params_.at`) and redundant logarithm evaluations (`std::log`) for each sample.
**Action:** Caching class priors and feature distributions prior to predicting speeds up classification significantly (from ~1.44s to ~0.39s on benchmark tests), while utilizing OpenMP outer loop parallelization safely.

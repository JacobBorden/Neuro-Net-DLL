## 2026-10-25 - [Naive Bayes Inference Optimization]
**Learning:** In NaiveBayesClassifier::predict(), repeated lookups on std::map inside nested sample and feature loops create massive overhead. Furthermore, redundant calls to std::log on constant priors per sample dominate execution time.
**Action:** Pre-calculate class log-priors and cache GaussianDistribution pointers into a lightweight struct vector before iterating over samples to eliminate map lookups and redundant log calls. Add #pragma omp parallel for to outer sample loop for trivial parallelization.

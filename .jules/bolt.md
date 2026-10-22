## 2026-10-22 - Pre-calculating Class Priors in NaiveBayes Classifier
**Learning:** In NaiveBayesClassifier::predict, evaluating std::log(class_priors_.at(label)) and map lookups inside the per-sample prediction loop caused O(N * C) redundant map lookups and logarithm calculations.
**Action:** Pre-compute class log-priors and cache pointers to feature distribution vectors before iterating over samples, and parallelize the outer sample prediction loop using OpenMP.

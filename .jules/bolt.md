## 2026-10-26 - [Optimize NaiveBayes Predict Performance]
**Learning:** In `NaiveBayesClassifier::predict()`, redundant log(prior) map lookups and computation in a tight loop per row and class causes performance issues, even without multithreading.
**Action:** Caching these lookups/log calculations and applying OpenMP per-row outer loop parallelization yielded ~3.9x speedup without changing the public API.

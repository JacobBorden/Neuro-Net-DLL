## 2023-10-17 - [Optimizing Naive Bayes Prediction]
**Learning:** O(log C) map lookups and logarithm calculations inside the hot loop over O(N) samples can dominate execution time for the NaiveBayesClassifier. Pre-calculating log priors and pointer references into a structured array outside the sample loop significantly speeds up inference.
**Action:** Always hoist map lookups and constant calculations out of the main loop of any predictive model function, especially before applying OpenMP parallelization to avoid redundant computations on every thread.

## 2026-10-12 - Pre-calculating and Caching in Predict Loops
**Learning:** Pre-calculating class log-priors and caching distribution pointers into a structured vector prior to the main prediction loop eliminates redundant (\log C)$ map lookups and logarithm computations, yielding massive speedups (~5x faster) when combined with OpenMP sample parallelization in Naive Bayes.
**Action:** Always hoist map lookups and constant calculations out of the inner loop in prediction algorithms.

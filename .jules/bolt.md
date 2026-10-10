## 2025-02-21 - Caching class priors and feature distribution lookups in Naive Bayes inference

**Learning:** Re-computing `std::log(class_priors_.at(label))` and performing `std::map::at()` lookups per sample inside the core prediction loop creates significant overhead ($O(N \cdot C \cdot \log C)$ map lookups and $N \cdot C$ redundant logarithm evaluations). Pre-allocating pointers and pre-calculating class log-priors prior to the prediction loop along with OpenMP parallelization across samples reduces runtime dramatically (~3.9x speedup).

**Action:** When performing row-by-row batch inference or evaluations against class parameters, pre-calculate class-level constants and cache distribution pointers outside the sample iteration loop.

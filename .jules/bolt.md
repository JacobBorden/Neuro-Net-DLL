## $(date +%Y-%m-%d) - Pre-calculating naive bayes parameters
**Learning:** In `NaiveBayesClassifier::predict()`, pre-calculating class log-priors (`std::log(class_priors_.at(label))`) and caching distribution pointers into a structure prior to the sample iteration loop eliminates redundant map lookups and logarithm calls.
**Action:** When working with nested loops over samples and classes, hoist map lookups and constant calculations out of the inner loop and consider `omp parallel for` over the samples for improved performance.

#include "gaussian_distribution.h"
#include <cmath>

const double PI = std::acos(-1.0);

GaussianDistribution::GaussianDistribution(double mean, double stddev) : mean_(mean), stddev_(stddev) {}

double GaussianDistribution::pdf(double x) const {
    return (1.0 / (stddev_ * std::sqrt(2.0 * PI))) * std::exp(-0.5 * std::pow((x - mean_) / stddev_, 2.0));
}

double GaussianDistribution::cdf(double x) const {
    return 0.5 * (1.0 + std::erf((x - mean_) / (stddev_ * std::sqrt(2.0))));
}

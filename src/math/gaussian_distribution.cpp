#include "gaussian_distribution.h"
#include <cmath>

const double PI = std::acos(-1.0);

GaussianDistribution::GaussianDistribution(double mean, double stddev)
    : mean_(mean),
      stddev_(stddev),
      pdf_coeff_(1.0 / (stddev * std::sqrt(2.0 * PI))),
      inv_2var_neg_(-0.5 / (stddev * stddev)),
      inv_stddev_sqrt2_(1.0 / (stddev * std::sqrt(2.0))) {}

double GaussianDistribution::pdf(double x) const {
    // Optimization: Precomputed normalization constant and direct multiplication instead of std::pow
    double diff = x - mean_;
    return pdf_coeff_ * std::exp(inv_2var_neg_ * diff * diff);
}

double GaussianDistribution::cdf(double x) const {
    // Optimization: Precomputed standard deviation and sqrt(2) scaling factor
    return 0.5 * (1.0 + std::erf((x - mean_) * inv_stddev_sqrt2_));
}

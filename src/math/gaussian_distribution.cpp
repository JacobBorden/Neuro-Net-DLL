#include "gaussian_distribution.h"
#include <cmath>

constexpr double SQRT_2PI = 2.506628274631000502415765284811045253;
constexpr double SQRT_2   = 1.414213562373095048801688724209698078;

GaussianDistribution::GaussianDistribution(double mean, double stddev)
    : mean_(mean), stddev_(stddev) {
    // Precompute coefficients to optimize repetitive pdf() and cdf() evaluations
    pdf_coeff_ = 1.0 / (stddev_ * SQRT_2PI);
    inv_2var_neg_ = -0.5 / (stddev_ * stddev_);
    inv_stddev_sqrt2_ = 1.0 / (stddev_ * SQRT_2);
}

double GaussianDistribution::pdf(double x) const {
    double diff = x - mean_;
    return pdf_coeff_ * std::exp(diff * diff * inv_2var_neg_);
}

double GaussianDistribution::cdf(double x) const {
    return 0.5 * (1.0 + std::erf((x - mean_) * inv_stddev_sqrt2_));
}

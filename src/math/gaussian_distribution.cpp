#include "gaussian_distribution.h"
#include <cmath>

const double PI = std::acos(-1.0);

GaussianDistribution::GaussianDistribution(double mean, double stddev)
    : mean_(mean),
      stddev_(stddev),
      pdf_coeff_((1.0 / std::sqrt(2.0 * PI)) / stddev) {}

double GaussianDistribution::pdf(double x) const {
    // Scale before squaring: sigma*sigma can underflow or overflow even
    // when the standardized difference and the resulting density are finite.
    const double z = (x - mean_) / stddev_;
    return pdf_coeff_ * std::exp(-0.5 * z * z);
}

double GaussianDistribution::cdf(double x) const {
    // Dividing first also avoids overflow in sigma*sqrt(2) and in 1/sigma.
    const double z = (x - mean_) / stddev_;
    return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}

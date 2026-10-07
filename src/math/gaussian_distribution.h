#ifndef GAUSSIAN_DISTRIBUTION_H
#define GAUSSIAN_DISTRIBUTION_H

class GaussianDistribution {
public:
    GaussianDistribution(double mean, double stddev);
    double pdf(double x) const;
    double cdf(double x) const;

private:
    double mean_;
    double stddev_;
    // Precomputed constants for pdf and cdf calculations
    double pdf_coeff_;
    double inv_2var_neg_;
    double inv_stddev_sqrt2_;
};

#endif // GAUSSIAN_DISTRIBUTION_H

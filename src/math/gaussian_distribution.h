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
    double pdf_coeff_;        // Precalculated 1.0 / (stddev * sqrt(2 * PI))
    double inv_2var_neg_;     // Precalculated -0.5 / (stddev * stddev)
    double inv_stddev_sqrt2_; // Precalculated 1.0 / (stddev * sqrt(2))
};

#endif // GAUSSIAN_DISTRIBUTION_H

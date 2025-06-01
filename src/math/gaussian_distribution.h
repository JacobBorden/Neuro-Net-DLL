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
};

#endif // GAUSSIAN_DISTRIBUTION_H

#include "gtest/gtest.h"
#include "math/gaussian_distribution.h"
#include <cmath>

const double TOLERANCE = 1e-4;

TEST(GaussianDistributionTest, StandardNormalPDF) {
    GaussianDistribution dist(0.0, 1.0);
    EXPECT_NEAR(dist.pdf(0.0), 0.3989, TOLERANCE);
    EXPECT_NEAR(dist.pdf(1.0), 0.2420, TOLERANCE);
    EXPECT_NEAR(dist.pdf(-1.0), 0.2420, TOLERANCE);
}

TEST(GaussianDistributionTest, StandardNormalCDF) {
    GaussianDistribution dist(0.0, 1.0);
    EXPECT_NEAR(dist.cdf(0.0), 0.5, TOLERANCE);
    EXPECT_NEAR(dist.cdf(1.0), 0.8413, TOLERANCE);
    EXPECT_NEAR(dist.cdf(-1.0), 0.1587, TOLERANCE);
}

TEST(GaussianDistributionTest, ShiftedMeanPDF) {
    GaussianDistribution dist(5.0, 1.0);
    EXPECT_NEAR(dist.pdf(5.0), 0.3989, TOLERANCE);
}

TEST(GaussianDistributionTest, ShiftedMeanCDF) {
    GaussianDistribution dist(5.0, 1.0);
    EXPECT_NEAR(dist.cdf(5.0), 0.5, TOLERANCE);
}

TEST(GaussianDistributionTest, DifferentStdDevPDF) {
    GaussianDistribution dist(0.0, 2.0);
    EXPECT_NEAR(dist.pdf(0.0), 0.1995, TOLERANCE); // 0.3989 / 2
}

TEST(GaussianDistributionTest, DifferentStdDevCDF) {
    GaussianDistribution dist(0.0, 2.0);
    EXPECT_NEAR(dist.cdf(0.0), 0.5, TOLERANCE);
}

TEST(GaussianDistributionTest, ExtremeFiniteStandardDeviations) {
    // Compare dimensionless densities: a fixed absolute tolerance would accept
    // zero for wide distributions and hide overflow for narrow ones.
    const double inv_sqrt_two_pi = 1.0 / std::sqrt(2.0 * std::acos(-1.0));
    for (double sigma : {1e-200, 1e-155, 1.0, 1e155, 1e200, 1e308}) {
        SCOPED_TRACE(sigma);
        GaussianDistribution dist(0.0, sigma);
        for (double z : {-1.0, 0.0, 1.0}) {
            const double density = dist.pdf(z * sigma);
            EXPECT_TRUE(std::isfinite(density));
            EXPECT_GT(density, 0.0);
            EXPECT_NEAR(density * sigma,
                        inv_sqrt_two_pi * std::exp(-0.5 * z * z), 1e-14);
            EXPECT_NEAR(dist.cdf(z * sigma),
                        0.5 * (1.0 + std::erf(z / std::sqrt(2.0))), 1e-14);
        }
    }
}

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

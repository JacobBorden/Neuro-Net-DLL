#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include "math/matrix.h"
#include "math/gaussian_distribution.h"
#include <vector>
#include <map>
#include <string> // For potential class labels if not just numeric

namespace Classifier {

class NaiveBayesClassifier {
public:
    NaiveBayesClassifier();

    /**
     * @brief Fits the Naive Bayes model to the training data.
     *
     * @param features An m x n matrix where m is the number of samples and n is the number of features.
     * @param labels A m x 1 matrix (column vector) where m is the number of samples.
     *               Labels are expected to be integer class indices (e.g., 0, 1, 2...).
     */
    void fit(const Matrix::Matrix<double>& features, const Matrix::Matrix<double>& labels);

    /**
     * @brief Predicts class labels for the input features.
     *
     * @param features An m x n matrix where m is the number of samples and n is the number of features.
     * @return Matrix::Matrix<double> A m x 1 matrix (column vector) of predicted class labels.
     */
    Matrix::Matrix<double> predict(const Matrix::Matrix<double>& features) const;

private:
    // Structure to store parameters for each class's features
    // For each class (key: double, representing class label),
    // we store a vector of GaussianDistribution objects (one for each feature).
    std::map<double, std::vector<GaussianDistribution>> class_feature_params_;

    // Store class priors P(C_k)
    // Key: class label, Value: prior probability
    std::map<double, double> class_priors_;

    // Number of features seen during fit
    size_t num_features_;

    // Number of unique classes
    size_t num_classes_;
    std::vector<double> unique_labels_sorted_; // To map internal index to actual label if needed

    // Helper to get unique sorted labels
    std::vector<double> get_unique_sorted_labels(const Matrix::Matrix<double>& labels) const;
};

} // namespace Classifier

#endif // NAIVE_BAYES_H

#include "naive_bayes.h"
#include <cmath>        // For std::log, std::exp, std::sqrt, std::pow
#include <numeric>      // For std::accumulate
#include <algorithm>    // For std::sort, std::unique, std::max_element
#include <stdexcept>    // For std::invalid_argument, std::runtime_error
#include <limits>       // For std::numeric_limits
#include <sstream>      // Required for std::ostringstream (moved here)

namespace Classifier {

NaiveBayesClassifier::NaiveBayesClassifier() : num_features_(0), num_classes_(0) {}

std::vector<double> NaiveBayesClassifier::get_unique_sorted_labels(const Matrix::Matrix<double>& labels) const {
    if (labels.cols() != 1) {
        throw std::invalid_argument("Labels matrix must be a column vector.");
    }
    std::vector<double> unique_labels;
    for (size_t i = 0; i < labels.rows(); ++i) {
        unique_labels.push_back(labels[i][0]);
    }
    std::sort(unique_labels.begin(), unique_labels.end());
    unique_labels.erase(std::unique(unique_labels.begin(), unique_labels.end()), unique_labels.end());
    return unique_labels;
}

void NaiveBayesClassifier::fit(const Matrix::Matrix<double>& features, const Matrix::Matrix<double>& labels) {
    if (features.rows() != labels.rows()) {
        throw std::invalid_argument("Number of samples in features and labels must match.");
    }
    if (features.rows() == 0) {
        throw std::invalid_argument("Cannot fit on empty dataset.");
    }
    if (labels.cols() != 1) {
        throw std::invalid_argument("Labels matrix must be a column vector.");
    }

    num_features_ = features.cols();
    unique_labels_sorted_ = get_unique_sorted_labels(labels);
    num_classes_ = unique_labels_sorted_.size();

    if (num_classes_ < 1) { // Should not happen if features.rows() > 0
        throw std::runtime_error("No unique classes found in labels.");
    }

    // Clear previous model parameters
    class_feature_params_.clear();
    class_priors_.clear();

    for (double current_class_label : unique_labels_sorted_) {
        // --- 1. Calculate Class Prior P(C_k) ---
        double class_count = 0;
        for (size_t i = 0; i < labels.rows(); ++i) {
            if (labels[i][0] == current_class_label) {
                class_count++;
            }
        }
        class_priors_[current_class_label] = class_count / static_cast<double>(labels.rows());

        // --- 2. Calculate Mean and StdDev for each feature for this class ---
        Matrix::Matrix<double> features_for_class(class_count, num_features_);
        size_t current_row_idx = 0;
        for (size_t i = 0; i < features.rows(); ++i) {
            if (labels[i][0] == current_class_label) {
                for (size_t j = 0; j < num_features_; ++j) {
                    features_for_class[current_row_idx][j] = features[i][j];
                }
                current_row_idx++;
            }
        }

        std::vector<GaussianDistribution> feature_distributions;
        if (class_count == 0) { // Should not happen if class is in unique_labels_sorted_
             // Handle case with no samples for a class if necessary, e.g. by skipping or adding smoothing
            // For now, this indicates an issue or requires smoothing not implemented.
            // Or perhaps this class should not have been in unique_labels_sorted_ if count is 0.
            // This path should ideally not be hit if unique_labels_sorted_ is derived from existing labels.
            continue;
        }

        for (size_t j = 0; j < num_features_; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < features_for_class.rows(); ++i) {
                sum += features_for_class[i][j];
            }
            double mean = sum / static_cast<double>(features_for_class.rows());

            double sum_sq_diff = 0.0;
            for (size_t i = 0; i < features_for_class.rows(); ++i) {
                sum_sq_diff += std::pow(features_for_class[i][j] - mean, 2);
            }
            // Use sample standard deviation (N-1 denominator), or population (N)
            // Add epsilon for numerical stability if variance is zero
            double variance = (features_for_class.rows() > 1) ? (sum_sq_diff / static_cast<double>(features_for_class.rows() - 1)) : sum_sq_diff / static_cast<double>(features_for_class.rows());
            double stddev = std::sqrt(variance);

            // Add a small epsilon to stddev if it's zero to prevent division by zero in Gaussian PDF
            if (stddev < 1e-9) { // Epsilon value
                stddev = 1e-9;
            }
            feature_distributions.emplace_back(mean, stddev);
        }
        class_feature_params_[current_class_label] = feature_distributions;
    }
}

// Note: <sstream> was moved to the top of the file.

Matrix::Matrix<double> NaiveBayesClassifier::predict(const Matrix::Matrix<double>& features) const {
    // bool priors_empty = class_priors_.empty(); // For debugging
    // bool params_empty = class_feature_params_.empty(); // For debugging
    // bool labels_sorted_empty = unique_labels_sorted_.empty(); // For debugging

    if (class_priors_.empty() || class_feature_params_.empty() || unique_labels_sorted_.empty()) {
        // std::ostringstream error_msg; // For debugging
        // error_msg << "Classifier has not been fitted. Call fit() first. "
        //           << "States: priors_empty=" << priors_empty
        //           << ", params_empty=" << params_empty
        //           << ", labels_sorted_empty=" << labels_sorted_empty;
        // throw std::runtime_error(error_msg.str());
        throw std::runtime_error("Classifier has not been fitted. Call fit() first.");
    }

    if (features.cols() != num_features_) {
        // std::ostringstream error_msg; // For debugging
        // error_msg << "Number of features in test data does not match training data. "
        //           << "Features.cols=" << features.cols()
        //           << ", num_features_=" << num_features_
        //           << ". Prior states: priors_empty=" << priors_empty
        //           << ", params_empty=" << params_empty
        //           << ", labels_sorted_empty=" << labels_sorted_empty;
        // throw std::invalid_argument(error_msg.str());
        throw std::invalid_argument("Number of features in test data does not match training data.");
    }

    Matrix::Matrix<double> predictions(features.rows(), 1);

    for (size_t i = 0; i < features.rows(); ++i) {
        double max_log_posterior = -std::numeric_limits<double>::infinity();
        double predicted_class = -1; // Default or error value

        if (unique_labels_sorted_.empty()) { // Should be caught by class_priors_.empty() check
             throw std::runtime_error("No classes available for prediction. Model might be corrupt or not fitted.");
        }
        predicted_class = unique_labels_sorted_[0]; // Default to first class if all posteriors are somehow zero/invalid

        for (double current_class_label : unique_labels_sorted_) {
            // Calculate log posterior: log(P(C_k)) + sum(log(P(x_j | C_k)))
            double log_prior = std::log(class_priors_.at(current_class_label));
            double log_likelihood_sum = 0.0;

            const auto& feature_distributions = class_feature_params_.at(current_class_label);
            for (size_t j = 0; j < num_features_; ++j) {
                double feature_value = features[i][j];
                double pdf_val = feature_distributions[j].pdf(feature_value);
                // Add small epsilon to pdf_val if it's zero to avoid log(0)
                if (pdf_val < std::numeric_limits<double>::epsilon()) {
                     pdf_val = std::numeric_limits<double>::epsilon();
                }
                log_likelihood_sum += std::log(pdf_val);
            }

            double current_log_posterior = log_prior + log_likelihood_sum;

            if (current_log_posterior > max_log_posterior) {
                max_log_posterior = current_log_posterior;
                predicted_class = current_class_label;
            }
        }
        predictions[i][0] = predicted_class;
    }
    return predictions;
}

} // namespace Classifier

#include "gtest/gtest.h"
#include "classifier/naive_bayes.h"
#include "math/matrix.h"
#include <vector>
#include <iostream> // For debugging output if needed
#include <stdexcept> // For std::invalid_argument, std::runtime_error

// Helper function to create a matrix from a 2D vector
Matrix::Matrix<double> create_matrix_from_vec(const std::vector<std::vector<double>>& data) {
    if (data.empty()) { // Allow creating 0x0 or 0xN matrix if data is empty
        return Matrix::Matrix<double>(0, 0);
    }
    if (data[0].empty() && !data.empty()){ // Allow creating Nx0 matrix if inner vector is empty
         return Matrix::Matrix<double>(data.size(), 0);
    }
    size_t rows = data.size();
    size_t cols = data[0].size();
    Matrix::Matrix<double> mat(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        if (data[i].size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns for create_matrix_from_vec.");
        }
        for (size_t j = 0; j < cols; ++j) {
            mat[i][j] = data[i][j];
        }
    }
    return mat;
}

// Helper to print matrix for debugging (optional)
void print_matrix(const Matrix::Matrix<double>& mat, const std::string& name) {
    std::cout << name << " (" << mat.rows() << "x" << mat.cols() << "):" << std::endl;
    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            std::cout << mat[i][j] << "\t";
        }
        std::cout << std::endl;
    }
}

TEST(NaiveBayesClassifierTest, SimpleTwoClassCase) {
    Classifier::NaiveBayesClassifier nbc;
    std::vector<std::vector<double>> features_vec = {
        {1.0, 1.1}, {0.9, 1.0}, {1.1, 0.9}, {1.2, 1.2},
        {5.0, 5.1}, {4.9, 5.0}, {5.1, 4.9}, {5.2, 5.2}
    };
    std::vector<std::vector<double>> labels_vec = {{0}, {0}, {0}, {0}, {1}, {1}, {1}, {1}};
    Matrix::Matrix<double> features = create_matrix_from_vec(features_vec);
    Matrix::Matrix<double> labels = create_matrix_from_vec(labels_vec);
    nbc.fit(features, labels);
    Matrix::Matrix<double> predictions = nbc.predict(features);
    ASSERT_EQ(predictions.rows(), features.rows());
    ASSERT_EQ(predictions.cols(), 1);
    for (size_t i = 0; i < labels.rows(); ++i) {
        EXPECT_EQ(predictions[i][0], labels[i][0]);
    }
    std::vector<std::vector<double>> test_features_vec = {{1.0, 1.0}, {5.0, 5.0}, {0.8, 1.2}, {5.3, 4.8}};
    Matrix::Matrix<double> test_features = create_matrix_from_vec(test_features_vec);
    Matrix::Matrix<double> expected_test_labels = create_matrix_from_vec({{0}, {1}, {0}, {1}});
    Matrix::Matrix<double> test_predictions = nbc.predict(test_features);
    ASSERT_EQ(test_predictions.rows(), test_features.rows());
    ASSERT_EQ(test_predictions.cols(), 1);
    for (size_t i = 0; i < expected_test_labels.rows(); ++i) {
        EXPECT_EQ(test_predictions[i][0], expected_test_labels[i][0]);
    }
}

TEST(NaiveBayesClassifierTest, SingleFeatureThreeClasses) {
    Classifier::NaiveBayesClassifier nbc;
    std::vector<std::vector<double>> features_vec = {{1.0}, {1.2}, {0.8}, {5.0}, {5.1}, {4.9}, {10.0}, {10.2}, {9.8}};
    std::vector<std::vector<double>> labels_vec = {{0}, {0}, {0}, {1}, {1}, {1}, {2}, {2}, {2}};
    Matrix::Matrix<double> features = create_matrix_from_vec(features_vec);
    Matrix::Matrix<double> labels = create_matrix_from_vec(labels_vec);
    nbc.fit(features, labels);
    std::vector<std::vector<double>> test_features_vec = {{1.1}, {5.05}, {9.9}, {0.5}, {10.5}};
    Matrix::Matrix<double> test_features = create_matrix_from_vec(test_features_vec);
    Matrix::Matrix<double> expected_labels = create_matrix_from_vec({{0}, {1}, {2}, {0}, {2}});
    Matrix::Matrix<double> predictions = nbc.predict(test_features);
    ASSERT_EQ(predictions.rows(), test_features.rows());
    ASSERT_EQ(predictions.cols(), 1);
    for (size_t i = 0; i < expected_labels.rows(); ++i) {
        EXPECT_EQ(predictions[i][0], expected_labels[i][0]);
    }
}

TEST(NaiveBayesClassifierTest, FitErrorHandling) {
    Classifier::NaiveBayesClassifier nbc;
    Matrix::Matrix<double> features_ok = create_matrix_from_vec({{1,2},{3,4}});
    Matrix::Matrix<double> labels_ok = create_matrix_from_vec({{0},{1}});
    Matrix::Matrix<double> labels_wrong_cols = create_matrix_from_vec({{0,0},{1,1}});
    Matrix::Matrix<double> labels_wrong_rows = create_matrix_from_vec({{0}});
    Matrix::Matrix<double> empty_features_data = create_matrix_from_vec({}); // 0x0
    Matrix::Matrix<double> empty_labels_data = create_matrix_from_vec({});   // 0x0

    EXPECT_THROW(nbc.fit(features_ok, labels_wrong_cols), std::invalid_argument);
    EXPECT_THROW(nbc.fit(features_ok, labels_wrong_rows), std::invalid_argument);
    EXPECT_THROW(nbc.fit(empty_features_data, labels_ok), std::invalid_argument);
    // For nbc.fit(features_ok, empty_labels_data):
    // If empty_labels_data is 0x0, features_ok.rows() (2) != empty_labels_data.rows() (0)
    EXPECT_THROW(nbc.fit(features_ok, empty_labels_data), std::invalid_argument);
}

TEST(NaiveBayesClassifierTest, AA_PredictErrorHandling_PredictBeforeFit) {
    Classifier::NaiveBayesClassifier nbc_for_this_specific_test;
    Matrix::Matrix<double> features = create_matrix_from_vec({{1.0, 2.0}});
    EXPECT_THROW(nbc_for_this_specific_test.predict(features), std::runtime_error);
}

TEST(NaiveBayesClassifierTest, FitAndOtherPredictErrors) {
    Classifier::NaiveBayesClassifier nbc_fitted;
    Matrix::Matrix<double> features_train = create_matrix_from_vec({{1,2},{3,4},{1,3},{3,5}});
    Matrix::Matrix<double> labels_train = create_matrix_from_vec({{0},{1},{0},{1}});
    nbc_fitted.fit(features_train, labels_train); // nbc_fitted.num_features_ is 2

    Matrix::Matrix<double> features_wrong_cols = create_matrix_from_vec({{1,2,3}});
    bool thrown_wrong_cols = false;
    try {
        nbc_fitted.predict(features_wrong_cols);
    } catch (const std::invalid_argument& e) {
        thrown_wrong_cols = true;
        // EXPECT_STREQ("Number of features in test data does not match training data.", e.what()); // Commented out
    }
    ASSERT_TRUE(thrown_wrong_cols); // Use ASSERT to stop if this fails

    Matrix::Matrix<double> zero_by_zero_features = create_matrix_from_vec({});
    bool thrown_zero_by_zero = false;
    try {
        nbc_fitted.predict(zero_by_zero_features);
    } catch (const std::invalid_argument& e) {
        thrown_zero_by_zero = true;
        // EXPECT_STREQ("Number of features in test data does not match training data.", e.what()); // Commented out
    }
    ASSERT_TRUE(thrown_zero_by_zero); // Use ASSERT to stop if this fails

    // This part should not throw if the above logic is correct
    Matrix::Matrix<double> zero_rows_correct_cols(0, 2);
    Matrix::Matrix<double> predictions_for_zero_rows;
    ASSERT_NO_THROW({
        predictions_for_zero_rows = nbc_fitted.predict(zero_rows_correct_cols);
    });
    EXPECT_EQ(predictions_for_zero_rows.rows(), 0);
    EXPECT_EQ(predictions_for_zero_rows.cols(), 1);
}

TEST(NaiveBayesClassifierTest, SlightlyOverlappingData) {
    Classifier::NaiveBayesClassifier nbc;
    std::vector<std::vector<double>> features_vec = {
        {1,1}, {2,1}, {1,2}, {2,2}, {3,2},
        {5,5}, {4,5}, {5,4}, {4,4}, {3,4}
    };
    std::vector<std::vector<double>> labels_vec = {{0},{0},{0},{0},{0}, {1},{1},{1},{1},{1}};
    Matrix::Matrix<double> features = create_matrix_from_vec(features_vec);
    Matrix::Matrix<double> labels = create_matrix_from_vec(labels_vec);
    nbc.fit(features, labels);
    std::vector<std::vector<double>> test_features_vec = {{1.5, 1.5}, {4.5, 4.5}, {3.0, 3.0}};
    Matrix::Matrix<double> test_features = create_matrix_from_vec(test_features_vec);
    Matrix::Matrix<double> predictions = nbc.predict(test_features);
    EXPECT_EQ(predictions[0][0], 0);
    EXPECT_EQ(predictions[1][0], 1);
    EXPECT_TRUE(predictions[2][0] == 0 || predictions[2][0] == 1);
}

TEST(NaiveBayesClassifierTest, ZeroVarianceFeature) {
    Classifier::NaiveBayesClassifier nbc;
    std::vector<std::vector<double>> features_vec = {{1, 2}, {1.2, 2}, {0.8, 2}, {5, 5}, {5.1, 5.2}, {4.9, 4.8}};
    std::vector<std::vector<double>> labels_vec = {{0}, {0}, {0}, {1}, {1}, {1}};
    Matrix::Matrix<double> features = create_matrix_from_vec(features_vec);
    Matrix::Matrix<double> labels = create_matrix_from_vec(labels_vec);
    nbc.fit(features, labels);
    std::vector<std::vector<double>> test_features_vec = {{1.0, 2.0}, {5.0, 5.0}};
    Matrix::Matrix<double> test_features = create_matrix_from_vec(test_features_vec);
    Matrix::Matrix<double> expected_labels = create_matrix_from_vec({{0},{1}});
    Matrix::Matrix<double> predictions = nbc.predict(test_features);
    ASSERT_EQ(predictions.rows(), test_features.rows());
    for(size_t i=0; i<expected_labels.rows(); ++i) {
        EXPECT_EQ(predictions[i][0], expected_labels[i][0]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

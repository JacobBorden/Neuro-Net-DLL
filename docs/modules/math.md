# Math Utilities (Matrix Library & Extended Operations)

This document details the `Matrix` class, the foundational data structure for numerical computations in this library, and additional mathematical utility functions.

## `Matrix::Matrix<T>` Class

The `Matrix` class is a generic 2D matrix implementation supporting various operations.

### Overview
*   **Purpose:** To provide a flexible and reasonably efficient matrix for neural network calculations.
*   **Template:** `Matrix<T>` where `T` is the data type of elements (typically `float` or `double`).
*   **Storage:** Rows are stored as `MatrixRow<T>` objects, and the matrix itself manages an array of these rows. Data within a row is contiguous.
*   **Iterators:** Provides `MatrixRowIterator` (for elements within a row), `MatrixColumnIterator` (for column-wise traversal), and `MatrixIterator` (for iterating over rows).
*   **OpenMP:** Matrix multiplication (`operator*`) is parallelized using OpenMP for improved performance.

### Key Functionalities (`Matrix::Matrix<T>`)
*   **Constructors:**
    *   Default constructor: `Matrix<T>()` - creates an empty 0x0 matrix.
    *   Dimensioned constructor: `Matrix<T>(int rows, int cols)` - creates a matrix of specified size, elements default-initialized.
    *   Copy and Move constructors/assignment operators are provided.
*   **Basic Properties:**
    *   `rows() const`: Returns number of rows.
    *   `cols() const`: Returns number of columns.
    *   `size() const`: Returns total number of elements.
*   **Element Access:**
    *   `operator[](size_t i)`: Accesses the i-th row (`MatrixRow<T>`).
    *   `MatrixRow<T>::operator[](size_t j)`: Accesses the j-th element in a row.
    *   `at(size_t i)` (for `MatrixRow`): Accesses element with bounds checking.
*   **Manipulation:**
    *   `resize(size_t row_count, size_t col_count)`: Resizes the matrix.
    *   `assign(size_t r, size_t c, const T val)`: Resizes and assigns a value to all elements.
    *   `assign(const T val)`: Assigns a value to all existing elements.
    *   `ZeroMatrix()`: Sets all elements to zero.
    *   `Randomize()`: Fills matrix with random values (typically between -1.0 and 1.0).
    *   `CreateIdentityMatrix()`: Transforms a square matrix into an identity matrix.
*   **Core Operations:**
    *   `Transpose() const`: Returns a new transposed matrix.
    *   `Determinant() const`: Computes determinant (for square matrices).
    *   `Inverse() const`: Computes inverse (for square, non-singular matrices).
*   **Arithmetic Operators:**
    *   `+`, `-`, `*`: Matrix-matrix addition, subtraction, multiplication.
    *   `+`, `-`, `*`, `/`: Scalar addition, subtraction, multiplication, division.
    *   `+=`, `-=`, `*=`, `/=`: Compound assignment versions of the above.
*   **Merging & Splitting:**
    *   `MergeVertical(const Matrix<T>& b) const`: Appends matrix `b` below the current matrix.
    *   `MergeHorizontal(const Matrix<T>& b) const`: Appends matrix `b` to the right.
    *   `SplitVertical(size_t num_parts) const`: Splits matrix into `num_parts` vertically.
    *   `SplitHorizontal(size_t num_parts) const`: Splits matrix into `num_parts` horizontally.
*   **Special Functions:**
    *   `SigmoidMatrix() const`: Applies element-wise sigmoid function.

### `Matrix::MatrixRow<T>` Class
*   **Purpose:** Represents a single row within a `Matrix`. Manages its own data.
*   **Key Functionalities:**
    *   Constructors, copy/move semantics.
    *   `resize(size_t newSize)`, `assign(size_t size, T val)`, `assign(T val)`.
    *   `operator[]`, `at(size_t i)` for element access.
    *   `begin()`, `end()` for row iterators.

### Source
*   `src/math/matrix.h`

## Extended Matrix Operations (`NeuroNet::MathUtils`)

This namespace provides additional mathematical functions that operate on `Matrix::Matrix<float>` objects, often used in neural network layers.

### Key Functions

*   **`gelu(const Matrix::Matrix<float>& input)`:**
    *   Applies the GELU (Gaussian Error Linear Unit) activation function element-wise.
    *   Formula: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
*   **`layer_norm(const Matrix::Matrix<float>& input, float epsilon = 1e-5f)`:**
    *   Applies Layer Normalization row-wise. Each row is treated as a separate sample.
    *   Formula for each row `x`: `y = (x - mean(x)) / sqrt(variance(x) + epsilon)`.
*   **`softmax(const Matrix::Matrix<float>& input, int axis = 1)`:**
    *   Applies the Softmax function along a specified axis (row-wise by default, `axis = 1`).
    *   Converts scores into a probability distribution.
    *   `axis = 0`: column-wise.
    *   `axis = 1` or `-1`: row-wise.
*   **`split_matrix_by_cols(const Matrix::Matrix<float>& input, int num_splits)`:**
    *   Splits a matrix into multiple smaller matrices by dividing its columns.
    *   Useful for operations like multi-head attention where feature dimensions are split across heads.
*   **`combine_matrices_by_cols(const std::vector<Matrix::Matrix<float>>& inputs)`:**
    *   Combines a vector of matrices into a single matrix by concatenating them column-wise (horizontally).
    *   All input matrices must have the same number of rows.

### Source
*   `src/math/extended_matrix_ops.h`
*   `src/math/extended_matrix_ops.cpp`

## Example Usage: Matrix Operations

```cpp
#include "math/matrix.h"
#include "math/extended_matrix_ops.h" // For softmax, etc.
#include <iostream>

int main() {
    // Basic Matrix Creation
    Matrix::Matrix<float> A(2, 3);
    A[0][0] = 1; A[0][1] = 2; A[0][2] = 3;
    A[1][0] = 4; A[1][1] = 5; A[1][2] = 6;

    std::cout << "Matrix A:" << std::endl;
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Transpose
    Matrix::Matrix<float> A_T = A.Transpose();
    std::cout << "\nMatrix A Transposed:" << std::endl;
    // ... (print A_T) ...

    // Scalar multiplication
    Matrix::Matrix<float> B = A * 2.0f;
    std::cout << "\nMatrix B (A * 2.0f):" << std::endl;
    // ... (print B) ...

    // Matrix multiplication
    Matrix::Matrix<float> C(3, 2);
    C.Randomize(); // Fill with random values
    std::cout << "\nMatrix C (Randomized 3x2):" << std::endl;
    // ... (print C) ...

    try {
        Matrix::Matrix<float> D = A * C; // (2x3) * (3x2) = (2x2)
        std::cout << "\nMatrix D (A * C):" << std::endl;
        // ... (print D) ...
    } catch (const std::exception& e) {
        std::cerr << "Error during multiplication: " << e.what() << std::endl;
    }

    // Using an extended operation (e.g., softmax row-wise)
    Matrix::Matrix<float> scores(2, 4);
    scores[0][0]=1; scores[0][1]=2; scores[0][2]=1; scores[0][3]=0.5;
    scores[1][0]=0.1; scores[1][1]=0.5; scores[1][2]=2; scores[1][3]=1.5;

    Matrix::Matrix<float> probabilities = NeuroNet::MathUtils::softmax(scores, 1); // axis = 1 for row-wise
    std::cout << "\nSoftmax of scores (row-wise):" << std::endl;
     for (size_t i = 0; i < probabilities.rows(); ++i) {
        for (size_t j = 0; j < probabilities.cols(); ++j) {
            std::cout << probabilities[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
```

(Further details on specific mathematical properties, performance considerations, or more complex examples can be added.)

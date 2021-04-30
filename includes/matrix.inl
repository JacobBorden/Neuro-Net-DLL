/**
 * @file matrix.inl
 * @author Jacob Borden (amenra.beats@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2021-04-24
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "matrix.h"

/**
 * @brief Construct an empty new Matrix:: Matrix< T> object
 * 
 * @tparam T - The type to be used in the matrix
 */

template <typename T>
Matrix::Matrix<T>::Matrix()
{
}

/**
 * @brief Construct a new Matrix:: Matrix< T> object with a specified number of rows and columns.
 * @tparam T - The type to be used in the matrix
 * @param rows - The number of rows to create in the matrix
 * @param columns - The number of columns to create in the matrix
 */

template <typename T>
Matrix::Matrix<T>::Matrix(int rows, int columns)
{
	this.rows = rows;
	this.columns = columns;
	matrix.assign(this.rows, std::vector<T>(this.columns));
}

/**
 * @brief Destroy the Matrix:: Matrix< T>:: Matrix object
 * 
 * @tparam T - The type to be used in the matrix
 */
template <typename T>
Matrix::Matrix<T>::~Matrix()
{
}

/**
 * @brief Resize matrix to a specified number of rows and columns
 * 
 * @tparam T - The type to be used in the matrix
 * @param rows - The number of rows to resize the matrix to
 * @param columns - The number of columns to resize the matrix to
 */

template <typename T>
void Matrix::Matrix<T>::Resize(int rows, int columns)
{
	this.rows = rows;
	this.columns = columns;
	matrix.resize(this.rows);
	for (int i = 0; i < this.columns; i++)
	{
		matrix[i].resize(this.columns);
	}
}

/**
 * @brief Returns the number of rows that are currently in the matrix
 * 
 * @tparam T - The type to be used in the matrix
 * @return int - The number of rows currently in the matrix
 */

template <typename T>
int Matrix::Matrix<T>::Rows()
{
	return this.rows;
}

/**
 * @brief Returns the number of columns that are currently in the matrix
 * 
 * @tparam T - The type to be used in the matrix
 * @return int - The number of columns currently in the matrix
 */

template <typename T>
int Matrix::Matrix<T>::Columns()
{
	return this->columns;
}

/**
 * @brief Overloads the square bracket operator to return elements of the matrix.
 * 
 * @tparam T - The type to be used in the matrix
 * @param i - An index to the row in the matrix
 * @return std::vector<T> 
 */

template <typename T>
std::vector<T> &Matrix::Matrix<T>::operator[](int i)
{
	return matrix[i];
}

/**
 * @brief Overloads the addition operator to add two matrices of the same size. Usage: Matrix C = Matrix A + Matrix B.
 * 
 * @tparam T - The type to be used in the matrix
 * @param matrix_b - the matrix to be added to the current matrix
 * @return Matrix::Matrix<T> - Returns the sum of the matrices.
 */

template <typename T>
Matrix::Matrix<T> Matrix::Matrix<T>::operator+(Matrix<T> matrix_b)
{
	Matrix::Matrix<T> matrix_c(this.rows, this.columns)

#pragma omp parallel for
		for (int i = 0; i < this.rows; i++) for (int j = 0; j < this.columns; j++)
			matrix_c[i][j] = this.matrix[i][j] + matrix_b[i][j];

	return matrix_c;
}

/**
 * @brief Overloads the addition operator to multiply two matrices. Usage: Matrix C = Matrix A * Matrix B.
 * 
 * @tparam T - The type to be used in the matrix
 * @param matrix_b  - the matrix to be multiplied to the current matrix
 * @return Matrix::Matrix<T> - Returns the product of the matrices 
 */

template <typename T>
Matrix::Matrix<T> Matrix::Matrix<T>::operator*(Matrix<T> matrix_b)
{
	Matrix::Matrix<T> matrix_c(this.rows, matrix_b.columns);
	int block_size = 64 / sizeof(T);
#pragma omp parallel for
	for (int i = 0; i < this->rows; i += block_size)
	{
		for (int j = 0; j < matrix_b.columns; j += block_size)
		{
			for (int m = 0; m < this.columns; m += block_size)
			{
				for (int k = i; k < i + block_size && k < this.rows; k++)
				{
					for (int l = j; l < j + block_size && l < matrix_b.columns; l++)
					{
						T sum = 0;
						for (int n = m; n < m + block_size && n < this.columns; n++)
						{
							sum += this.data[k][n] * matrix_b[n][l];
						}
						matrix_c[k][l] += sum;
					}
				}
			}
		}
	}
	return matrix_c;
}
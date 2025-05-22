#pragma once

#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

namespace Matrix
{
	template <typename MatrixRow>
	class MatrixRowIterator
	{
	public:
		// Member type definitions to conform to iterator requirements
		using value_type = typename MatrixRow::value_type;		   // Type of elements the iterator refers to
		using pointer = value_type *;							   // Pointer to the element type
		using reference = value_type &;							   // Reference to the element type
		using iterator_category = std::random_access_iterator_tag; // Iterator category to support random access
		using difference_type = std::ptrdiff_t;					   // Type to express the difference between two iterators

		// Constructor initializes the iterator with a pointer to a matrix row element
		MatrixRowIterator(pointer ptr) : m_ptr(ptr)
		{
		}

		// Pre-increment operator advances the iterator to the next element and returns a reference to the updated iterator
		MatrixRowIterator &operator++()
		{
			m_ptr++;
			return *this;
		}

		// Post-increment operator advances the iterator to the next element and returns the iterator before advancement
		MatrixRowIterator operator++(int)
		{
			MatrixRowIterator it = *this;
			++*this;
			return it;
		}

		// Addition operator returns a new iterator advanced by 'n' positions
		MatrixRowIterator operator+(difference_type n) const
		{
			return MatrixRowIterator(m_ptr + n);
		}

		// Compound addition operator advances the iterator by 'n' positions and returns a reference to the updated iterator
		MatrixRowIterator &operator+=(difference_type n)
		{
			m_ptr += n;
			return *this;
		}

		// Pre-decrement operator moves the iterator to the previous element and returns a reference to the updated iterator
		MatrixRowIterator &operator--()
		{
			m_ptr--;
			return *this;
		}

		// Post-decrement operator moves the iterator to the previous element and returns the iterator before movement
		MatrixRowIterator operator--(int)
		{
			MatrixRowIterator it = *this;
			--*this;
			return it;
		}

		// Subtraction operator returns a new iterator moved back by 'n' positions
		MatrixRowIterator operator-(difference_type n) const
		{
			return MatrixRowIterator(m_ptr - n);
		}

		// Compound subtraction operator moves the iterator back by 'n' positions and returns a reference to the updated iterator
		MatrixRowIterator &operator-=(difference_type n)
		{
			m_ptr -= n;
			return *this;
		}

		// Subtraction operator calculates the difference between two iterators
		difference_type operator-(const MatrixRowIterator &other) const
		{
			return m_ptr - other.m_ptr;
		}

		// Arrow operator provides access to the element's members the iterator points to
		pointer operator->() const
		{
			return m_ptr;
		}

		// Dereference operators return a (const) reference to the element the iterator points to
		reference operator*()
		{
			return *m_ptr;
		}
		const reference operator*() const
		{
			return *m_ptr;
		}

		// Comparison operators for equality and inequality checks between iterators
		bool operator==(const MatrixRowIterator &other) const
		{
			return m_ptr == other.m_ptr;
		}
		bool operator!=(const MatrixRowIterator &other) const
		{
			return m_ptr != other.m_ptr;
		}

		// Relational operators compare the positions of two iterators
		bool operator<(const MatrixRowIterator &other) const
		{
			return m_ptr < other.m_ptr;
		}
		bool operator<=(const MatrixRowIterator &other) const
		{
			return m_ptr <= other.m_ptr;
		}
		bool operator>(const MatrixRowIterator &other) const
		{
			return m_ptr > other.m_ptr;
		}
		bool operator>=(const MatrixRowIterator &other) const
		{
			return m_ptr >= other.m_ptr;
		}

		// Subscript operator provides random access to elements relative to the current iterator position
		reference operator[](difference_type n) const
		{
			return *(*this + n);
		}

	private:
		pointer m_ptr; // Internal pointer to the current element
	};
	//--------------------------------------------------------------------------

	template <typename T>
	class MatrixColumnIterator
	{
	public:
		// Type aliases for iterator traits
		using value_type = T;									   // Type of elements the iterator can dereference
		using pointer = T *;									   // Pointer to the element type
		using reference = T &;									   // Reference to the element type
		using iterator_category = std::random_access_iterator_tag; // Iterator category defining the capabilities of the iterator
		using difference_type = std::ptrdiff_t;					   // Type to express the difference between two iterators

		// Constructor initializes the iterator with a pointer to a matrix element and the total number of columns in the matrix
		MatrixColumnIterator(pointer ptr, size_t totalColumns) : m_ptr(ptr), m_totalColumns(totalColumns)
		{
		}

		// Pre-increment operator advances the iterator to the next element in the column and returns a reference to the updated iterator
		MatrixColumnIterator &operator++()
		{
			m_ptr += m_totalColumns; // Move pointer down one row in the current column
			return *this;
		}

		// Post-increment operator advances the iterator to the next element in the column and returns the iterator before the increment
		MatrixColumnIterator operator++(int)
		{
			MatrixColumnIterator it = *this; // Make a copy of the current iterator
			m_ptr += m_totalColumns;		 // Move pointer down one row in the current column
			return it;						 // Return the copy representing the iterator before increment
		}

		// Addition operator returns a new iterator advanced by 'n' positions in the column
		MatrixColumnIterator operator+(difference_type n) const
		{
			return MatrixColumnIterator(m_ptr + (n * m_totalColumns), m_totalColumns); // Calculate new position and create a new iterator
		}

		// Compound addition operator advances the iterator by 'n' positions in the column and returns a reference to the updated iterator
		MatrixColumnIterator &operator+=(difference_type n)
		{
			m_ptr += (n * m_totalColumns); // Adjust pointer by 'n' rows down in the current column
			return *this;
		}

		// Pre-decrement operator moves the iterator to the previous element in the column and returns a reference to the updated iterator
		MatrixColumnIterator &operator--()
		{
			m_ptr -= m_totalColumns; // Move pointer up one row in the current column
			return *this;
		}

		// Post-decrement operator moves the iterator to the previous element in the column and returns the iterator before the decrement
		MatrixColumnIterator operator--(int)
		{
			MatrixColumnIterator it = *this; // Make a copy of the current iterator
			m_ptr -= m_totalColumns;		 // Move pointer up one row in the current column
			return it;						 // Return the copy representing the iterator before decrement
		}

		// Subtraction operator returns a new iterator moved back by 'n' positions in the column
		MatrixColumnIterator operator-(difference_type n) const
		{
			return MatrixColumnIterator(m_ptr - (n * m_totalColumns), m_totalColumns); // Calculate new position and create a new iterator
		}

		// Compound subtraction operator moves the iterator back by 'n' positions in the column and returns a reference to the updated iterator
		MatrixColumnIterator &operator-=(difference_type n)
		{
			m_ptr -= (n * m_totalColumns); // Adjust pointer by 'n' rows up in the current column
			return *this;
		}

		// Subtraction operator calculates the difference between two iterators in terms of column positions
		difference_type operator-(const MatrixColumnIterator &other) const
		{
			return (m_ptr - other.m_ptr) / m_totalColumns; // Calculate element-wise distance between iterators
		}

		// Comparison operators for checking equality and inequality between iterators
		bool operator==(const MatrixColumnIterator &other) const
		{
			return m_ptr == other.m_ptr;
		}
		bool operator!=(const MatrixColumnIterator &other) const
		{
			return m_ptr != other.m_ptr;
		}

		// Relational operators for ordering iterators
		bool operator<(const MatrixColumnIterator &other) const
		{
			return m_ptr < other.m_ptr;
		}
		bool operator<=(const MatrixColumnIterator &other) const
		{
			return m_ptr <= other.m_ptr;
		}
		bool operator>(const MatrixColumnIterator &other) const
		{
			return m_ptr > other.m_ptr;
		}
		bool operator>=(const MatrixColumnIterator &other) const
		{
			return m_ptr >= other.m_ptr;
		}

		// Dereference operator provides access to the current element the iterator points to
		reference operator*() const
		{
			return *m_ptr;
		}

		// Member access operator allows access to the element's members
		pointer operator->() const
		{
			return m_ptr;
		}

		// Subscript operator provides random access to elements relative to the current iterator position
		reference operator[](difference_type n) const
		{
			return *(*this + n);
		}

	private:
		pointer m_ptr;		   // Pointer to the current element in the matrix
		size_t m_totalColumns; // Total number of columns in the matrix, used for column-wise navigation
	};

	//--------------------------------------------------------------------------
	template <typename Matrix>
	class MatrixIterator
	{
	public:
		using value_type = typename Matrix::value_type;
		using pointer = value_type *;
		using reference = value_type &;

		MatrixIterator(pointer ptr) : m_ptr(ptr)
		{
		}

		MatrixIterator &operator++()
		{
			m_ptr++;
			return *this;
		}

		MatrixIterator operator++(int)
		{
			MatrixIterator it = *this;
			++(this);
			return it;
		}

		MatrixIterator &operator--()
		{
			m_ptr--;
			return *this;
		}

		MatrixIterator operator--(int)
		{
			MatrixIterator it = *this;
			--(this);
			return it;
		}

		pointer operator->()
		{
			return m_ptr;
		}

		reference operator*()
		{
			return *m_ptr;
		}

		bool operator==(MatrixIterator other)
		{
			return this->m_ptr == other.m_ptr;
		}

		bool operator!=(MatrixIterator other)
		{
			return this->m_ptr != other.m_ptr;
		}

	private:
		pointer m_ptr;
	};

	//-------------------------------------------------------------------------
	template <typename T>
	class MatrixRow
	{
	public:
		using value_type = T;
		using Iterator = MatrixRowIterator<MatrixRow<T>>;

		MatrixRow() = default;
		explicit MatrixRow(size_t size) : m_Size(size), m_Capacity(size * sizeof(T)), m_Data(std::make_unique<T[]>(size)) {}

		void resize(size_t newSize)
		{
			auto newData = std::make_unique<T[]>(newSize);
			std::copy_n(m_Data.get(), std::min(m_Size, newSize), newData.get());
			m_Data = std::move(newData);
			m_Size = newSize;
			m_Capacity = newSize * sizeof(T);
		}

		void assign(size_t size, T val)
		{
			resize(size);
			std::fill_n(m_Data.get(), size, val)
		}

		void assign(T val) { std::fill_n(m_Data.get(), m_Size, val); }

		size_t size() const { return m_Size; }

		size_t capacity(){return m_Capacity}

		T at(size_t i) const
		{
			if (i >= m__Size)
				throw std::out_of_range("Index out of range");
			return m_Data[i];
		}

		T &operator[](size_t i)
		{
			if (i >= m_Size)
				throw std::out_of_range("Index out of range");
			return m_Data[i];
		}

		const T &operator[](size_t i) const
		{
			if (i >= m_Size)
				throw std::out_of_range("Index out of range");
			return m_Data[i];
		}

		Iterator begin() { return Iterator(m_Data.get()); }
		Iterator end() { return Iterator(m_Data.get() + m_Size); }
		Iterator begin() const { return Iterator(m_Data.get()); }
		Iterator end() const { return Iterator(m_Data.get() + m_Size); }

	private:
		size_t m_Size = 0;
		size_t m_Capacity = 0;
		std::unique_ptr<T[]> m_Data;
	};

	//---------------------------------------------------------------------------------------------

	template <typename T>
	class Matrix
	{
	public:
		using value_type = MatrixRow<T>;
		using Iterator = MatrixIterator<Matrix<T>>;
		using ColumonIterator = MatrixColumnIterator<Matrix<T>>;

		Matrix<T>() = default;
		explicit Matrix<T>(int row_count, int column_count) : m_Rows(row_count), m_Cols(column_count), m_Size(row_count * column_count), m_Capacity(sizeof(T) * row_count * column_count) m_Data(std::make_unique<MatrixRow<T>[]>(row_count))
		{
			for (int i = 0; i < m_Rows; i++)
				m_Data[i] = MatrixRow<T>(m_Cols);
		}

		size_t size() const { return m_Size; }
		size_t rows() const { return m_Rows; }
		size_t cols() const { return m_Cols; }
		size_t capacity() const { return m_Capacity }

		void resize(size_t row_count, size_t col_count)
		{
			auto newData = std::make_unique<MatrixRow<T>[]>(row_count);
			for (size_t i = 0; i < std::min(m_Rows, row_count); ++i)
			{
				newData[i] = std::move(m_Data[i]);
				newData[i].resize(col_count);
			}
			m_Data = std::move(newData);
			m_Rows = row_count;
			m_Cols = col_count;
			m_Size = row_count * col_count;
			m_Capacity = row_count * col_count * sizeof(T);
		}

		void assign(size_t row_count, size_t col_count, const T val)
		{
			resize(row_count, col_count) for (size_t i = 0; i < row_count; ++i)
				std::fill_n(m_Data[i].get(), m_Data[i].size(), val);
		}

		void assign(const T val)
		{
			for (size_t i = 0; i < m_Rows; ++i)
				for (size_t j = 0; j < m_Cols; ++j)
					m_Data[i][j] = val;
		}

		Matrix<T> MergeVertical(const Matrix<T> &b) const
		{
			if (m_Cols != b.m_Cols)
				throw std::invalid_argument("Matrices must have the same number of columns");
			Matrix<T> result(m_Rows + b.m_Rows, m_Cols);
			std::copy_n(m_Data.get(), m_Rows, result.m_Data.get());
			std::copy_n(b.m_Data.get(), b.m_Rows, result.m_Data.get() + m_Rows);
			return result;
		}

		Matrix<T> MergeHorizontal(const Matrix<T> &b) const
		{
			if (m_Rows != b.m_Rows)
				throw std::invalid_argument("Matrices must have the same number of rows");
			Matrix<T> result(m_Rows, m_Cols + b.m_Cols);
			for (size_t i = 0; i < m_Rows; ++i)
			{
				std::copy_n(m_Data[i].begin(), m_Cols, result.m_Data[i].begin());
				std::copy_n(b.m_Data[i].begin(), b.m_Cols, result.m_Data[i].begin() + m_Cols);
			}
			return result;
		}

		std::vector<Matrix<T>> SplitVertical() const
		{
			if (m_Rows % 2 != 0)
				throw std::invalid_argument("Number of rows must be divisable by 2");
			std::vector<Matrix<T>> result;
			size_t split_size = m_Rows / 2;
			for (size_t i = 0; i < 2; ++i)
			{
				Matrix<T> split(split_size, m_Cols);
				std::copy_n(m_Data.get() + i * split_size, split_size, split.m_Data.get());
				result.push_back(std::move(split));
			}
			return result;
		}

		std::vector<Matrix<T>> SplitVertical(size_t num) const
		{
			if (m_Rows % num != 0)
				throw std::invalid_argument("Number of splits must evenly divide the number of rows");
			std::vector<Matrix<T>> result;
			size_t split_size = m_Rows / num;
			for (size_t i = 0; i < num; ++i)
			{
				Matrix<T> split(split_size, m_Cols);
				std::copy_n(m_Data.get() + i * split_size, split_size, split.m_Data.get());
				result.push_back(std::move(split));
			}
			return result;
		}

		std::vector<Matrix<T>> SplitHorizontal() const
		{
			if (m_Cols % 2 != 0)
				throw std::invalid_argument("Number of columns must be divisable by 2");
			std::vector<Matrix<T>> result;
			size_t split_size = m_Cols / 2;
			for (size_t i = 0; i < 2; ++i)
			{
				Matrix<T> split(m_Rows, split_size);
				for (size_t j = 0; j < m_Rows; ++j)
				{
					std::copy_n(m_Data[j].begin() + i * split_size, split_size, split.m_Data[j].begin());
				}
				result.push_back(std::move(split));
			}
			return result;
		}

		std::vector<Matrix<T>> SplitHorizontal(size_t num) const
		{
			if (m_Cols % num != 0)
				throw std::invalid_argument("Number of splits must evenly divide the number of columns");
			std::vector<Matrix<T>> result;
			size_t split_size = m_Cols / num;
			for (size_t i = 0; i < num; ++i)
			{
				Matrix<T> split(m_Rows, split_size);
				for (size_t j = 0; j < m_Rows; ++j)
				{
					std::copy_n(m_Data[j].begin() + i * split_size, split_size, split.m_Data[j].begin());
				}
				result.push_back(std::move(split));
			}
			return result;
		}

		Matrix<T> SigmoidMatrix()
		{
			Matrix<T> result(*this);
			for (auto &row : result)
			{
				for (auto &elem : row)
				{
					elem = 1 / (1 + std::exp(-elem));
				}
			}
		}

		Matrix<T> Randomize()
		{
			static std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
			std::uniform_real_distribution<> dis(-1.0, 1.0);
			for (auto &row : *this)
			{
				for (auto &elem : row)
				{
					elem = dis(gen);
				}
			}
			return *this;
		}
		Matrix<T> CreateIdentityMatrix()
		{
			if (m_Rows != m_Cols)
				throw std::invalid_argument("Matrix must be square");
			for (size_t i = 0; i < m_Rows; ++i)
			{
				std::fill(m_Data[i].begin(), m_Data[i].end(), T(0));
				m_Data[i][i] = T(1);
			}
			return *this
		}

		Matrix<T> ZeroMatrix() const
		{
			for (auto &row : *this)
			{
				std::fill(row.begin(), row.end(), T(0);)
			}
			return *this;
		}

		Matrix<T> Transpose() const
		{
			Matrix<T> result(m_Cols, m_Rows);
			for (size_t i = 0; i < m_Rows, ++i)
				for (size_t j = 0; j < m_Cols; ++j)
					result[j][i] = m_Data[i][j];
			return result;
		}

		T Determinant() const
		{
			if (m_Rows != m_Cols)
				throw std::invalid_argument("Matrix must be square");
			size_t n = m_Rows;
			if (n == 1)
				return m_Data[0][0];
			else if (n == 2)
				return m_Data[0][0] * m_Data[1][1] - m_Data[0][1] * m_Data[1][0];
			T det = 0;
			for (size_t = 0; i < n; ++i)
			{
				Matrix<T> minor = getMinor(*this, 0, i);
				int sign = ((i % 2) == 0) ? 1 : -1;
				det += sign * matrix[0][i] * minor.Determinant();
			}

			return det;
		}

		Matrix<T> Inverse() const
		{
			if (m_Rows != m_Cols)
				throw std::invalid_argument("Matrix must be square");

			T det = Determinant();
			if (det == 0)
				throw std::runtime_error("Matrix is singular and cannot be inverted.");

			// Step 2: Compute the cofactor matrix
			Matrix<T> cofactors(m_Rows, m_Cols);
			for (size_t i = 0; i < m_Rows; ++i)
			{
				for (size_t j = 0; j < m_Cols; ++j)
				{
					Matrix<T> minor = getMinor(*this, i, j);
					T minor_det = minor.Determinant();
					cofactors[i][j] = ((i + j) % 2 == 0 ? 1 : -1) * minor_det;
				}
			}

			// Step 3: Compute the adjugate matrix (transpose of cofactor matrix)
			Matrix<T> adjugate = cofactors.Transpose();

			// Step 4: Compute the inverse
			Matrix<T> inverse = adjugate * (1 / det);
			return inverse;
		}

	

	MatrixRow<T> &
	operator[](size_t i)
	{
		return m_Data[i];
	}
	const MatrixRow<T> &operator[](size_t i) const { return m_Data[i]; }

	Matrix<T> operator+(const Matrix<T> &b)
	{
		if ((m_Rows == b.m_Rows) && (m_Cols == b.m_Cols))
		{
			Matrix<T> c(m_Rows, m_Cols);
			for (int i = 0; i < m_Rows; i++)
				for (int j = 0; j < m_Cols; j++)
					c[i][j] = m_Data[i][j] + b[i][j];
			return c;
		}

		else
			return *this;
	}
	Matrix<T> operator+(const T b) const
	{
		Matrix<T> c(m_Rows, m_Cols);
		for (int i = 0; i < m_Rows; i++)
			for (int j = 0; j < m_Cols; j++)
				c[i][j] = m_Data[i][j] + b;
		return c;
	}

	Matrix<T> operator+=(const Matrix<T> &b) const
	{
		if ((m_Rows == b.m_Rows) && (m_Cols == b.m_Cols))
		{
			Matrix<T> c(m_Rows, m_Cols);
			for (int i = 0; i < m_Rows; i++)
				for (int j = 0; j < m_Cols; j++)
					c[i][j] = m_Data[i][j] + b[i][j];
			*this = c;
		}

		return *this;
	}

	Matrix<T> operator+=(const T b) const
	{
		Matrix<T> c(m_Rows, m_Cols);
		for (int i = 0; i < m_Rows; i++)
			for (int j = 0; j < m_Cols; j++)
				c[i][j] = m_Data[i][j] + b;
		*this = c;
		return *this;
	}

	Matrix<T> operator-(const Matrix<T> &b) const
	{
		if ((m_Rows == b.m_Rows) && (m_Cols == b.m_Cols))
		{
			Matrix<T> c(m_Rows, m_Cols);
			for (int i = 0; i < m_Rows; i++)
				for (int j = 0; j < m_Cols; j++)
					c[i][j] = m_Data[i][j] - b[i][j];
			return c;
		}

		else
			return *this;
	}

	Matrix<T> operator-(const T b) const
	{
		Matrix<T> c(m_Rows, m_Cols);
		for (int i = 0; i < m_Rows; i++)
			for (int j = 0; j < m_Cols; j++)
				c[i][j] = m_Data[i][j] - b;
		return c;
	}

	Matrix<T> operator-=(const Matrix<T> &b) const
	{
		if ((m_Rows == b.m_Rows) && (m_Cols == b.m_Cols))
		{
			Matrix<T> c(m_Rows, m_Cols);
			for (int i = 0; i < m_Rows; i++)
				for (int j = 0; j < m_Cols; j++)
					c[i][j] = m_Data[i][j] - b[i][j];
			*this = c;
		}

		return *this;
	}
	Matrix<T> operator-=(const T b) const
	{
		Matrix<T> c(m_Rows, m_Cols);
		for (int i = 0; i < m_Rows; i++)
			for (int j = 0; j < m_Cols; j++)
				c[i][j] = m_Data[i][j] - b;
		*this = c;
		return *this;
	}

	Matrix<T> operator/(const T b) const
	{
		Matrix<T> c(m_Rows, m_Cols);
		for (int i = 0; i < m_Rows; i++)
			for (int j = 0; j < m_Cols; j++)
				c[i][j] = m_Data[i][j] / b;
		return c;
	}

	Matrix<T> operator/=(const T b) const
	{
		Matrix<T> c(m_Rows, m_Cols);
		for (int i = 0; i < m_Rows; i++)
			for (int j = 0; j < m_Cols; j++)
				c[i][j] = m_Data[i][j] / b;
		*this = c;
		return *this;
	}

	Matrix<T> operator*(const Matrix<T> &b) const
	{
		if (m_Cols == b.m_Rows)
		{
			Matrix<T> c(m_Rows, b.m_Cols);
			for (int i = 0; i < m_Rows; i++)
				for (int j = 0; j < m_Cols; j++)
					for (int k = 0; k < b.m_Cols; k++)
						c[i][k] += m_Data[i][j] * b[j][k];
			return c;
		}

		else
			return *this;
	}

	Matrix<T> operator*(const T b) const
	{
		Matrix<T> c(m_Rows, m_Cols);
		for (int i = 0; i < m_Rows; i++)
			for (int j = 0; j < m_Cols; j++)
				c[i][j] = m_Data[i][j] * b;
		return c;
	}

	Matrix<T> operator*=(const T b) const
	{
		Matrix<T> c(m_Rows, m_Cols);
		for (int i = 0; i < m_Rows; i++)
			for (int j = 0; j < m_Cols; j++)
				c[i][j] = m_Data[i][j] * b;
		*this = c;
		return *this;
	}

	Iterator begin() { return Iterator(m_Data); }
	Iterator end() { return Iterator(m_Data + m_Rows); }

private:
	Matrix<T> getMinor(const Matrix<T> &matrix, size_t row_to_remove, size_t col_to_remove) const
	{
		if (matrix.m_Rows != matrix.m_Cols)
			throw std::invalid_argument("Matrix must be square to compute minor.");

		size_t n = matrix.m_Rows;
		Matrix<T> minor_matrix(n - 1, n - 1);

		size_t minor_i = 0; // Row index for minor_matrix
		for (size_t i = 0; i < n; ++i)
		{
			if (i == row_to_remove)
				continue;

			size_t minor_j = 0; // Column index for minor_matrix
			for (size_t j = 0; j < n; ++j)
			{
				if (j == col_to_remove)
					continue;

				minor_matrix[minor_i][minor_j] = matrix.m_Data[i][j];
				++minor_j;
			}
			++minor_i;
		}

		return minor_matrix;
	}

	size_t m_Rows = 0;
	size_t m_Cols = 0;
	size_t m_Size = 0;
	size_t m_Capacity = 0;
	std::unique_ptr<MatrixRow<T>[]> m_Data;
};
}
#endif

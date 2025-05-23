/**
 * @file matrix.h
 * @author Jacob Borden (amenra.beats@gmail.com)
 * @brief Defines a generic Matrix class and associated iterators for matrix operations.
 * @version 0.1.0
 * @date 2023-10-27 (Assumed or last known update date)
 *
 * @copyright Copyright (c) 2021-2023 Jacob Borden
 */
#pragma once

#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>   // For size_t, ptrdiff_t
#include <memory>    // For std::unique_ptr, std::make_unique
#include <stdexcept> // For std::out_of_range, std::invalid_argument, std::runtime_error
#include <algorithm> // For std::copy_n, std::min, std::fill_n
#include <iterator>  // For iterator tags like std::random_access_iterator_tag
#include <vector>    // For std::vector (used in Split methods)
#include <cmath>     // For std::exp (in SigmoidMatrix), std::abs
#include <random>    // For std::mt19937, std::uniform_real_distribution (in Randomize)
#include <chrono>    // For std::chrono::system_clock (seeding Randomize)

/**
 * @namespace Matrix
 * @brief Contains all classes and functions related to the Matrix library.
 */
namespace Matrix
{
	/**
	 * @class MatrixRowIterator
	 * @brief A random access iterator for elements within a single row of a Matrix.
	 * @tparam MatrixRow The type of the MatrixRow this iterator will operate on.
	 *
	 * Conforms to the requirements of a C++ random access iterator.
	 */
	template <typename MatrixRow>
	class MatrixRowIterator
	{
	public:
		// Standard iterator typedefs
		using value_type = typename MatrixRow::value_type;
		using pointer = value_type *;
		using reference = value_type &;
		using iterator_category = std::random_access_iterator_tag;
		using difference_type = std::ptrdiff_t;

		/** @brief Constructs a MatrixRowIterator from a raw pointer. */
		MatrixRowIterator(pointer ptr) : m_ptr(ptr) {}

		/** @brief Pre-increment operator. Advances to the next element. */
		MatrixRowIterator &operator++() { m_ptr++; return *this; }
		/** @brief Post-increment operator. Advances to the next element. */
		MatrixRowIterator operator++(int) { MatrixRowIterator it = *this; ++*this; return it; }

		/** @brief Advances iterator by n positions. */
		MatrixRowIterator operator+(difference_type n) const { return MatrixRowIterator(m_ptr + n); }
		/** @brief Advances iterator by n positions. */
		MatrixRowIterator &operator+=(difference_type n) { m_ptr += n; return *this; }

		/** @brief Pre-decrement operator. Moves to the previous element. */
		MatrixRowIterator &operator--() { m_ptr--; return *this; }
		/** @brief Post-decrement operator. Moves to the previous element. */
		MatrixRowIterator operator--(int) { MatrixRowIterator it = *this; --*this; return it; }
		
		/** @brief Moves iterator back by n positions. */
		MatrixRowIterator operator-(difference_type n) const { return MatrixRowIterator(m_ptr - n); }
		/** @brief Moves iterator back by n positions. */
		MatrixRowIterator &operator-=(difference_type n) { m_ptr -= n; return *this; }

		/** @brief Calculates distance between two iterators. */
		difference_type operator-(const MatrixRowIterator &other) const { return m_ptr - other.m_ptr; }

		/** @brief Accesses the pointed-to element's members. */
		pointer operator->() const { return m_ptr; }
		/** @brief Dereferences the iterator to access the element. */
		reference operator*() { return *m_ptr; }
		/** @brief Dereferences the iterator to access the element (const version). */
		const reference operator*() const { return *m_ptr; }

		// Comparison operators
		bool operator==(const MatrixRowIterator &other) const { return m_ptr == other.m_ptr; }
		bool operator!=(const MatrixRowIterator &other) const { return m_ptr != other.m_ptr; }
		bool operator<(const MatrixRowIterator &other) const { return m_ptr < other.m_ptr; }
		bool operator<=(const MatrixRowIterator &other) const { return m_ptr <= other.m_ptr; }
		bool operator>(const MatrixRowIterator &other) const { return m_ptr > other.m_ptr; }
		bool operator>=(const MatrixRowIterator &other) const { return m_ptr >= other.m_ptr; }

		/** @brief Provides random access to elements relative to current position. */
		reference operator[](difference_type n) const { return *(*this + n); }

	private:
		pointer m_ptr; ///< Raw pointer to the current element in the matrix row.
	};
	//--------------------------------------------------------------------------

	/**
	 * @class MatrixColumnIterator
	 * @brief A random access iterator for traversing elements column-wise in a Matrix.
	 * @tparam T The data type of elements in the matrix.
	 */
	template <typename T>
	class MatrixColumnIterator
	{
	public:
		using value_type = T;
		using pointer = T *;
		using reference = T &;
		using iterator_category = std::random_access_iterator_tag;
		using difference_type = std::ptrdiff_t;

		/**
		 * @brief Constructs a MatrixColumnIterator.
		 * @param ptr Pointer to the first element in the column.
		 * @param totalColumns Total number of columns in the matrix (stride to next row element).
		 */
		MatrixColumnIterator(pointer ptr, size_t totalColumns) : m_ptr(ptr), m_totalColumns(totalColumns) {}

		/** @brief Pre-increment. Moves to the next element in the column (down one row). */
		MatrixColumnIterator &operator++() { m_ptr += m_totalColumns; return *this; }
		/** @brief Post-increment. Moves to the next element in the column. */
		MatrixColumnIterator operator++(int) { MatrixColumnIterator it = *this; m_ptr += m_totalColumns; return it; }
		
		/** @brief Advances iterator by n positions in the column. */
		MatrixColumnIterator operator+(difference_type n) const { return MatrixColumnIterator(m_ptr + (n * m_totalColumns), m_totalColumns); }
		/** @brief Advances iterator by n positions in the column. */
		MatrixColumnIterator &operator+=(difference_type n) { m_ptr += (n * m_totalColumns); return *this; }

		/** @brief Pre-decrement. Moves to the previous element in the column (up one row). */
		MatrixColumnIterator &operator--() { m_ptr -= m_totalColumns; return *this; }
		/** @brief Post-decrement. Moves to the previous element in the column. */
		MatrixColumnIterator operator--(int) { MatrixColumnIterator it = *this; m_ptr -= m_totalColumns; return it; }

		/** @brief Moves iterator back by n positions in the column. */
		MatrixColumnIterator operator-(difference_type n) const { return MatrixColumnIterator(m_ptr - (n * m_totalColumns), m_totalColumns); }
		/** @brief Moves iterator back by n positions in the column. */
		MatrixColumnIterator &operator-=(difference_type n) { m_ptr -= (n * m_totalColumns); return *this; }
		
		/** @brief Calculates distance between two column iterators (in terms of rows). */
		difference_type operator-(const MatrixColumnIterator &other) const { return (m_ptr - other.m_ptr) / m_totalColumns; }

		// Comparison operators
		bool operator==(const MatrixColumnIterator &other) const { return m_ptr == other.m_ptr; }
		bool operator!=(const MatrixColumnIterator &other) const { return m_ptr != other.m_ptr; }
		bool operator<(const MatrixColumnIterator &other) const { return m_ptr < other.m_ptr; }
		bool operator<=(const MatrixColumnIterator &other) const { return m_ptr <= other.m_ptr; }
		bool operator>(const MatrixColumnIterator &other) const { return m_ptr > other.m_ptr; }
		bool operator>=(const MatrixColumnIterator &other) const { return m_ptr >= other.m_ptr; }

		/** @brief Dereferences the iterator to access the element. */
		reference operator*() const { return *m_ptr; }
		/** @brief Accesses the pointed-to element's members. */
		pointer operator->() const { return m_ptr; }
		/** @brief Provides random access to elements relative to current position in the column. */
		reference operator[](difference_type n) const { return *(*this + n); }

	private:
		pointer m_ptr;         ///< Pointer to the current element in the matrix column.
		size_t m_totalColumns; ///< Stride to move to the same column in the next/previous row.
	};

	//--------------------------------------------------------------------------
	/**
	 * @class MatrixIterator
	 * @brief An iterator for traversing MatrixRow objects within a Matrix (iterating over rows).
	 * @tparam Matrix The Matrix class type this iterator operates on.
	 *
	 * This is a simpler iterator primarily for row-wise traversal of the Matrix.
	 */
	template <typename Matrix>
	class MatrixIterator // Iterates over MatrixRow<T> objects
	{
	public:
		using value_type = typename Matrix::value_type; // Should be MatrixRow<T>
		using pointer = value_type *; 
		using reference = value_type &;
		// Define iterator_category and difference_type if full compliance is needed
        using iterator_category = std::random_access_iterator_tag; // Assuming Matrix stores rows contiguously
        using difference_type = std::ptrdiff_t;


		MatrixIterator(pointer ptr) : m_ptr(ptr) {}

		MatrixIterator &operator++() { m_ptr++; return *this; }
		MatrixIterator operator++(int) { MatrixIterator it = *this; ++(*this); return it; } // Corrected post-increment

		MatrixIterator &operator--() { m_ptr--; return *this; }
		MatrixIterator operator--(int) { MatrixIterator it = *this; --(*this); return it; } // Corrected post-decrement
		
		// Add arithmetic operators for random access if needed (e.g., operator+, operator-, etc.)
        MatrixIterator operator+(difference_type n) const { return MatrixIterator(m_ptr + n); }
        MatrixIterator& operator+=(difference_type n) { m_ptr += n; return *this; }
        MatrixIterator operator-(difference_type n) const { return MatrixIterator(m_ptr - n); }
        MatrixIterator& operator-=(difference_type n) { m_ptr -= n; return *this; }
        difference_type operator-(const MatrixIterator& other) const { return m_ptr - other.m_ptr; }


		pointer operator->() { return m_ptr; }
		reference operator*() { return *m_ptr; }

		bool operator==(const MatrixIterator& other) const { return m_ptr == other.m_ptr; } // Corrected parameter type
		bool operator!=(const MatrixIterator& other) const { return m_ptr != other.m_ptr; } // Corrected parameter type
        
        // Relational operators for random access
        bool operator<(const MatrixIterator& other) const { return m_ptr < other.m_ptr; }
        bool operator<=(const MatrixIterator& other) const { return m_ptr <= other.m_ptr; }
        bool operator>(const MatrixIterator& other) const { return m_ptr > other.m_ptr; }
        bool operator>=(const MatrixIterator& other) const { return m_ptr >= other.m_ptr; }
        
        reference operator[](difference_type n) const { return *(*this + n); }


	private:
		pointer m_ptr; ///< Pointer to the current MatrixRow.
	};

	//-------------------------------------------------------------------------
	/**
	 * @class MatrixRow
	 * @brief Represents a single row within a Matrix.
	 * @tparam T The data type of elements in the row.
	 *
	 * Manages its own data and provides methods for manipulation and iteration.
	 */
	template <typename T>
	class MatrixRow
	{
	public:
		using value_type = T; ///< Data type of elements in the row.
		using Iterator = MatrixRowIterator<MatrixRow<T>>; ///< Iterator for this row.

		/** @brief Default constructor. Creates an empty row. */
		MatrixRow() = default;
		/**
		 * @brief Constructs a MatrixRow of a specific size.
		 * @param size Number of elements in the row.
		 */
		explicit MatrixRow(size_t size) : m_Size(size), m_Capacity(size * sizeof(T)), m_Data(std::make_unique<T[]>(size)) {}

		/** @brief Copy constructor (deep copy). */
		MatrixRow(const MatrixRow& other) 
			: m_Size(other.m_Size), m_Capacity(other.m_Capacity), m_Data(nullptr) {
			if (m_Size > 0) {
				m_Data = std::make_unique<T[]>(m_Size);
				for (size_t i = 0; i < m_Size; ++i) {
					m_Data[i] = other.m_Data[i]; // Element-wise copy
				}
			}
			// If m_Size is 0, m_Data remains nullptr, which is correct for an empty row.
		}

		/** @brief Move constructor. */
		MatrixRow(MatrixRow&& other) noexcept
			: m_Size(other.m_Size), m_Capacity(other.m_Capacity), m_Data(std::move(other.m_Data)) {
			other.m_Size = 0;
			other.m_Capacity = 0;
			other.m_Data = nullptr; // Important: leave other in a valid, destructible state
		}

		/** @brief Move assignment operator. */
		MatrixRow& operator=(MatrixRow&& other) noexcept {
			if (this != &other) {
				m_Data = std::move(other.m_Data);
				m_Size = other.m_Size;
				m_Capacity = other.m_Capacity;

				other.m_Data = nullptr;
				other.m_Size = 0;
				other.m_Capacity = 0;
			}
			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Copy assignment operator (deep copy). */
		MatrixRow& operator=(const MatrixRow& other) {
			if (this == &other) {
				return *this; // Handle self-assignment
			}

			std::unique_ptr<T[]> newData; 
			if (other.m_Size > 0) {
				newData = std::make_unique<T[]>(other.m_Size); 
				for (size_t i = 0; i < other.m_Size; ++i) {
					newData[i] = other.m_Data[i]; 
				}
			}
			// If other.m_Size is 0, newData will be nullptr.

			m_Data = std::move(newData); // Transfer ownership from newData to m_Data. Old m_Data is released.
			m_Size = other.m_Size;
			m_Capacity = other.m_Size * sizeof(T); // Recalculate capacity.

			return *this;
		}

		/** @brief Resizes the row.
		 * @param newSize The new number of elements. Existing elements are preserved up to newSize.
		 *                If newSize is larger, new elements are default-initialized.
		 */
		void resize(size_t newSize)
		{
			if (newSize == m_Size) return;
			auto newData = std::make_unique<T[]>(newSize);
			if (m_Data) { // Only copy if m_Data is not null
				std::copy_n(m_Data.get(), std::min(m_Size, newSize), newData.get());
			}
			// If newSize > m_Size, the additional elements in newData are value-initialized (e.g., 0 for numeric types).
			m_Data = std::move(newData);
			m_Size = newSize;
			m_Capacity = newSize * sizeof(T); // Assuming T is not void.
		}

		/**
		 * @brief Assigns a value to all elements, resizing if necessary.
		 * @param size The new size of the row.
		 * @param val The value to assign to all elements.
		 */
		void assign(size_t size, T val)
		{
			resize(size);
			if (m_Data) { // Ensure m_Data is valid after resize
				std::fill_n(m_Data.get(), m_Size, val); // Use m_Size after resize
			}
		}

		/**
		 * @brief Assigns a value to all existing elements in the row.
		 * @param val The value to assign.
		 */
		void assign(T val) { 
			if (m_Data) {
				std::fill_n(m_Data.get(), m_Size, val); 
			}
		}

		/** @brief Returns the number of elements in the row. */
		size_t size() const { return m_Size; }
		/** @brief Returns the current capacity in bytes. */
		size_t capacity() const { return m_Capacity; } // Corrected return type from original

		/**
		 * @brief Accesses an element by index with bounds checking.
		 * @param i Index of the element.
		 * @return T The value of the element at index i.
		 * @throws std::out_of_range if i is out of bounds.
		 */
		T at(size_t i) const
		{
			if (i >= m_Size) // Corrected m__Size to m_Size
				throw std::out_of_range("MatrixRow::at: Index out of range");
			return m_Data[i];
		}

		/**
		 * @brief Accesses an element by index. No bounds checking in release builds typically.
		 * @param i Index of the element.
		 * @return T& Reference to the element at index i.
		 * @throws std::out_of_range if i is out of bounds (behavior might vary).
		 */
		T &operator[](size_t i)
		{
			if (i >= m_Size) // Added bounds check for safety, though operator[] traditionally doesn't always.
				throw std::out_of_range("MatrixRow::operator[]: Index out of range");
			return m_Data[i];
		}

		/**
		 * @brief Accesses an element by index (const version).
		 * @param i Index of the element.
		 * @return const T& Const reference to the element at index i.
		 * @throws std::out_of_range if i is out of bounds.
		 */
		const T &operator[](size_t i) const
		{
			if (i >= m_Size)
				throw std::out_of_range("MatrixRow::operator[] const: Index out of range");
			return m_Data[i];
		}

		/** @brief Returns an iterator to the beginning of the row. */
		Iterator begin() { return Iterator(m_Data.get()); }
		/** @brief Returns an iterator to the end of the row. */
		Iterator end() { return Iterator(m_Data.get() + m_Size); }
		/** @brief Returns a const iterator to the beginning of the row. */
		Iterator begin() const { return Iterator(m_Data.get()); } // Note: Should ideally be const_iterator
		/** @brief Returns a const iterator to the end of the row. */
		Iterator end() const { return Iterator(m_Data.get() + m_Size); } // Note: Should ideally be const_iterator

	private:
		size_t m_Size = 0;     ///< Number of elements in the row.
		size_t m_Capacity = 0; ///< Allocated capacity in bytes.
		std::unique_ptr<T[]> m_Data; ///< Dynamically allocated array for row data.
	};

	//---------------------------------------------------------------------------------------------
	/**
	 * @class Matrix
	 * @brief A generic 2D Matrix class.
	 * @tparam T The data type of elements in the matrix.
	 *
	 * Supports common matrix operations such as addition, subtraction, multiplication,
	 * transpose, determinant, inverse, etc.
	 */
	template <typename T>
	class Matrix
	{
	public:
		using value_type = MatrixRow<T>; ///< Type of each row in the matrix.
		using Iterator = MatrixIterator<Matrix<T>>; ///< Iterator for rows of the matrix.
		// using ColumonIterator = MatrixColumnIterator<Matrix<T>>; // This was a typo in original, should be ColumnIterator
        using ColumnIterator = MatrixColumnIterator<T>; ///< Iterator for columns of the matrix.


		/** @brief Default constructor. Creates an empty matrix (0x0). */
		Matrix<T>() = default;

		/** @brief Swaps the contents of this matrix with another. */
		void swap(Matrix& other) noexcept {
			using std::swap;
			swap(m_Data, other.m_Data);
			swap(m_Rows, other.m_Rows);
			swap(m_Cols, other.m_Cols);
			swap(m_Size, other.m_Size);
			swap(m_Capacity, other.m_Capacity);
		}

		/**
		 * @brief Constructs a Matrix with specified dimensions.
		 * Elements are default-initialized (e.g., 0 for numeric types).
		 * @param row_count Number of rows.
		 * @param column_count Number of columns.
		 */
		explicit Matrix<T>(int row_count, int column_count) 
			: m_Rows(row_count > 0 ? row_count : 0), 
			  m_Cols(column_count > 0 ? column_count : 0), 
			  m_Size(m_Rows * m_Cols), 
			  m_Capacity(sizeof(T) * m_Rows * m_Cols), // Corrected capacity calculation
			  m_Data(m_Rows > 0 ? std::make_unique<MatrixRow<T>[]>(m_Rows) : nullptr)
		{
			if (m_Rows > 0 && m_Cols > 0) {
				for (size_t i = 0; i < m_Rows; i++) // Use size_t for loop
					m_Data[i] = MatrixRow<T>(m_Cols);
			} else {
                // If rows or cols is zero, ensure other dimension trackers are also zero
                m_Rows = 0;
                m_Cols = 0;
                m_Size = 0;
                m_Capacity = 0;
            }
		}

		/** @brief Copy constructor (deep copy) for Matrix. */
		Matrix(const Matrix& other)
			: m_Rows(other.m_Rows),
			  m_Cols(other.m_Cols),
			  m_Size(other.m_Size),
			  m_Capacity(other.m_Capacity),
			  m_Data(nullptr) {
			if (m_Rows > 0 && other.m_Data) { // Only allocate if there are rows
				m_Data = std::make_unique<MatrixRow<T>[]>(m_Rows);
				for (size_t i = 0; i < m_Rows; ++i) {
					m_Data[i] = other.m_Data[i]; // Relies on MatrixRow's copy constructor
				}
			}
			// If other is an empty matrix (m_Rows = 0 or other.m_Data is null), 
			// this matrix will also be empty (m_Data = nullptr, dimensions = 0).
		}

		/** @brief Move constructor for Matrix. */
		Matrix(Matrix&& other) noexcept
			: m_Rows(other.m_Rows),
			  m_Cols(other.m_Cols),
			  m_Size(other.m_Size),
			  m_Capacity(other.m_Capacity),
			  m_Data(std::move(other.m_Data)) {
			// Leave other in a valid, destructible, but empty state
			other.m_Rows = 0;
			other.m_Cols = 0;
			other.m_Size = 0;
			other.m_Capacity = 0;
			other.m_Data = nullptr;
		}

		/** @brief Copy assignment operator (copy-and-swap idiom). */
		Matrix& operator=(const Matrix& other) {
			if (this == &other) { // Self-assignment check, though copy-and-swap handles it implicitly
				return *this;
			}
			Matrix temp(other); // Use copy constructor
			this->swap(temp);   // Swap current content with the copy
			return *this;
		}

		/** @brief Move assignment operator for Matrix. */
		Matrix& operator=(Matrix&& other) noexcept {
			if (this != &other) {
				// Release current resources if any (unique_ptr handles this automatically on reassignment)
				m_Data = std::move(other.m_Data);
				m_Rows = other.m_Rows;
				m_Cols = other.m_Cols;
				m_Size = other.m_Size;
				m_Capacity = other.m_Capacity;

				// Leave other in a valid, destructible, but empty state
				other.m_Rows = 0;
				other.m_Cols = 0;
				other.m_Size = 0;
				other.m_Capacity = 0;
				other.m_Data = nullptr;
			}
			return *this;
		}

		/** @brief Returns the total number of elements in the matrix. */
		size_t size() const { return m_Size; }
		/** @brief Returns the number of rows in the matrix. */
		size_t rows() const { return m_Rows; }
		/** @brief Returns the number of columns in the matrix. */
		size_t cols() const { return m_Cols; }
		/** @brief Returns the current capacity in bytes. */
		size_t capacity() const { return m_Capacity; } // Corrected return type from original

		/**
		 * @brief Resizes the matrix to new dimensions.
		 * Existing data is preserved as much as possible. New elements are default-initialized.
		 * @param row_count New number of rows.
		 * @param col_count New number of columns.
		 */
		void resize(size_t row_count, size_t col_count)
		{
            if (row_count == 0 || col_count == 0) {
                m_Data = nullptr;
                m_Rows = 0;
                m_Cols = 0;
                m_Size = 0;
                m_Capacity = 0;
                return;
            }

			auto newData = std::make_unique<MatrixRow<T>[]>(row_count);
			for (size_t i = 0; i < row_count; ++i) // Initialize all new rows
			{
                if (i < m_Rows && m_Data) { // If old data exists for this row index
                    newData[i] = std::move(m_Data[i]); // Move existing row
                    newData[i].resize(col_count);      // Resize it to new column count
                } else {
                    newData[i] = MatrixRow<T>(col_count); // Create new row with new column count
                }
			}
			m_Data = std::move(newData);
			m_Rows = row_count;
			m_Cols = col_count;
			m_Size = row_count * col_count;
			m_Capacity = row_count * col_count * sizeof(T); // Assuming T is not void
		}

		/**
		 * @brief Assigns a value to all elements, resizing the matrix.
		 * @param row_count New number of rows.
		 * @param col_count New number of columns.
		 * @param val The value to assign to all elements.
		 */
		void assign(size_t row_count, size_t col_count, const T val)
		{
			resize(row_count, col_count); 
            if (m_Data) { // Check if m_Data is valid after resize
			    for (size_t i = 0; i < m_Rows; ++i) // Use m_Rows after resize
				    m_Data[i].assign(val); // Assign to all elements of the row
            }
		}

		/**
		 * @brief Assigns a value to all existing elements in the matrix.
		 * @param val The value to assign.
		 */
		void assign(const T val)
		{
            if (!m_Data) return;
			for (size_t i = 0; i < m_Rows; ++i) {
				// No need for inner loop if MatrixRow::assign(val) works correctly
				m_Data[i].assign(val); 
            }
		}

		/**
		 * @brief Merges this matrix with another matrix vertically.
		 * Both matrices must have the same number of columns.
		 * @param b The matrix to append below this one.
		 * @return Matrix<T> A new matrix containing the result of the merge.
		 * @throws std::invalid_argument if column counts do not match.
		 */
		Matrix<T> MergeVertical(const Matrix<T> &b) const
		{
			if (m_Cols != b.m_Cols && m_Rows != 0 && b.m_Rows != 0) // Allow merging if one is empty
				throw std::invalid_argument("Matrices must have the same number of columns for vertical merge.");
            if (b.m_Rows == 0) return *this;
            if (m_Rows == 0) return b;

			Matrix<T> result(m_Rows + b.m_Rows, m_Cols);
			for(size_t i = 0; i < m_Rows; ++i) result.m_Data[i] = m_Data[i]; // Copy existing rows
			for(size_t i = 0; i < b.m_Rows; ++i) result.m_Data[m_Rows + i] = b.m_Data[i]; // Copy new rows
			return result;
		}

		/**
		 * @brief Merges this matrix with another matrix horizontally.
		 * Both matrices must have the same number of rows.
		 * @param b The matrix to append to the right of this one.
		 * @return Matrix<T> A new matrix containing the result of the merge.
		 * @throws std::invalid_argument if row counts do not match.
		 */
		Matrix<T> MergeHorizontal(const Matrix<T> &b) const
		{
			if (m_Rows != b.m_Rows && m_Cols != 0 && b.m_Cols != 0) // Allow merging if one is empty
				throw std::invalid_argument("Matrices must have the same number of rows for horizontal merge.");
            if (b.m_Cols == 0) return *this;
            if (m_Cols == 0) return b;

			Matrix<T> result(m_Rows, m_Cols + b.m_Cols);
			for (size_t i = 0; i < m_Rows; ++i)
			{
                // Create new rows for the result matrix by copying elements
                for(size_t j=0; j < m_Cols; ++j) result.m_Data[i][j] = m_Data[i][j];
                for(size_t j=0; j < b.m_Cols; ++j) result.m_Data[i][m_Cols + j] = b.m_Data[i][j];
			}
			return result;
		}
		
		// Note: Split methods create copies. Consider returning views or using iterators for efficiency if needed.
		/**
		 * @brief Splits the matrix vertically into two equal halves.
		 * Number of rows must be even.
		 * @return std::vector<Matrix<T>> A vector containing two new matrices.
		 * @throws std::invalid_argument if the number of rows is not even.
		 */
		std::vector<Matrix<T>> SplitVertical() const
		{
			if (m_Rows == 0) return {};
			if (m_Rows % 2 != 0)
				throw std::invalid_argument("Number of rows must be divisible by 2 for default vertical split.");
			return SplitVertical(2); // Delegate to the generalized version
		}

		/**
		 * @brief Splits the matrix vertically into a specified number of parts.
		 * Number of rows must be divisible by `num`.
		 * @param num The number of vertical splits to create.
		 * @return std::vector<Matrix<T>> A vector of new matrices.
		 * @throws std::invalid_argument if rows are not divisible by `num` or `num` is 0.
		 */
		std::vector<Matrix<T>> SplitVertical(size_t num) const
		{
			if (num == 0) throw std::invalid_argument("Number of splits cannot be zero.");
			if (m_Rows == 0) return std::vector<Matrix<T>>(num, Matrix<T>(0, m_Cols)); // Return num empty matrices
			if (m_Rows % num != 0)
				throw std::invalid_argument("Number of splits must evenly divide the number of rows.");
			
			std::vector<Matrix<T>> result;
			result.reserve(num);
			size_t split_size = m_Rows / num;
			for (size_t i = 0; i < num; ++i)
			{
				Matrix<T> split(split_size, m_Cols);
				for(size_t r = 0; r < split_size; ++r) {
					split.m_Data[r] = m_Data[i * split_size + r]; // Copy rows
				}
				result.push_back(std::move(split));
			}
			return result;
		}

		/**
		 * @brief Splits the matrix horizontally into two equal halves.
		 * Number of columns must be even.
		 * @return std::vector<Matrix<T>> A vector containing two new matrices.
		 * @throws std::invalid_argument if the number of columns is not even.
		 */
		std::vector<Matrix<T>> SplitHorizontal() const
		{
			if (m_Cols == 0) return {};
			if (m_Cols % 2 != 0)
				throw std::invalid_argument("Number of columns must be divisible by 2 for default horizontal split.");
			return SplitHorizontal(2);
		}

		/**
		 * @brief Splits the matrix horizontally into a specified number of parts.
		 * Number of columns must be divisible by `num`.
		 * @param num The number of horizontal splits to create.
		 * @return std::vector<Matrix<T>> A vector of new matrices.
		 * @throws std::invalid_argument if columns are not divisible by `num` or `num` is 0.
		 */
		std::vector<Matrix<T>> SplitHorizontal(size_t num) const
		{
			if (num == 0) throw std::invalid_argument("Number of splits cannot be zero.");
            if (m_Cols == 0) return std::vector<Matrix<T>>(num, Matrix<T>(m_Rows, 0));

			if (m_Cols % num != 0)
				throw std::invalid_argument("Number of splits must evenly divide the number of columns.");
			
			std::vector<Matrix<T>> result;
			result.reserve(num);
			size_t split_col_size = m_Cols / num;
			for (size_t i = 0; i < num; ++i) // For each new matrix part
			{
				Matrix<T> split(m_Rows, split_col_size);
				for (size_t r = 0; r < m_Rows; ++r) // For each row in the original matrix
				{
					for (size_t c_split = 0; c_split < split_col_size; ++c_split) // For each column in the split part
					{
						split.m_Data[r][c_split] = m_Data[r][i * split_col_size + c_split];
					}
				}
				result.push_back(std::move(split));
			}
			return result;
		}

		/**
		 * @brief Applies the sigmoid function element-wise to the matrix.
		 * @return Matrix<T> A new matrix with the sigmoid function applied.
		 *                   Requires T to be a floating-point type.
		 */
		Matrix<T> SigmoidMatrix() const // Made const, returns new matrix
		{
			Matrix<T> result(*this); // Make a copy
			if (!result.m_Data) return result; // Return copy if empty
			for (size_t i = 0; i < result.m_Rows; ++i) // Use size_t
			{
				for (size_t j = 0; j < result.m_Cols; ++j) // Use size_t
				{
					result.m_Data[i][j] = T(1) / (T(1) + std::exp(-result.m_Data[i][j]));
				}
			}
            return result;
		}

		/**
		 * @brief Randomizes the elements of this matrix.
		 * Elements are set to random values, typically between -1.0 and 1.0.
		 * Requires T to be a floating-point type.
		 * @return Matrix<T>& Reference to this matrix after randomization.
		 */
		Matrix<T>& Randomize() // Modified to return reference to self
		{
            if (!m_Data) return *this; // Do nothing if empty
			// Seed with system clock for better randomness across runs.
			static std::mt19937 gen(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
			std::uniform_real_distribution<double> dis(-1.0, 1.0); // Use double for distribution
			for (size_t i = 0; i < m_Rows; ++i)
			{
				for (size_t j = 0; j < m_Cols; ++j)
				{
					m_Data[i][j] = static_cast<T>(dis(gen));
				}
			}
			return *this;
		}

		/**
		 * @brief Transforms this matrix into an identity matrix.
		 * The matrix must be square.
		 * @return Matrix<T>& Reference to this matrix, now an identity matrix.
		 * @throws std::invalid_argument if the matrix is not square.
		 */
		Matrix<T>& CreateIdentityMatrix() // Modified to return reference to self
		{
			if (m_Rows != m_Cols)
				throw std::invalid_argument("Matrix must be square to create an identity matrix.");
            if (!m_Data && m_Rows > 0) { // If data wasn't allocated but dimensions are set
                m_Data = std::make_unique<MatrixRow<T>[]>(m_Rows);
                 for(size_t i = 0; i < m_Rows; ++i) m_Data[i].resize(m_Cols);
            } else if (!m_Data) {
                return *this; // 0x0 matrix is trivially identity? Or throw? For now, return.
            }

			for (size_t i = 0; i < m_Rows; ++i)
			{
				m_Data[i].assign(T(0)); // Fill row with 0
				if (i < m_Cols) { // Ensure diagonal element is within column bounds
				    m_Data[i][i] = T(1);    // Set diagonal to 1
                }
			}
			return *this;
		}

		/**
		 * @brief Sets all elements of this matrix to zero.
		 * @return Matrix<T>& Reference to this matrix, now zeroed.
		 */
		Matrix<T>& ZeroMatrix() // Made non-const, modifies self, returns reference
		{
            assign(T(0)); // Use existing assign method
			return *this;
		}

		/**
		 * @brief Computes the transpose of this matrix.
		 * @return Matrix<T> A new matrix that is the transpose of this one.
		 */
		Matrix<T> Transpose() const
		{
            if (m_Rows == 0 || m_Cols == 0) return Matrix<T>(m_Cols, m_Rows); // Transpose of empty/vector
			Matrix<T> result(m_Cols, m_Rows);
			for (size_t i = 0; i < m_Rows; ++i) { // Iterate up to m_Rows
				for (size_t j = 0; j < m_Cols; ++j) { // Iterate up to m_Cols
					result.m_Data[j][i] = m_Data[i][j];
                }
            }
			return result;
		}

		/**
		 * @brief Computes the determinant of this matrix.
		 * The matrix must be square.
		 * @return T The determinant value.
		 * @throws std::invalid_argument if the matrix is not square.
		 * @note This implementation is recursive and may be inefficient for large matrices.
		 */
		T Determinant() const
		{
			if (m_Rows != m_Cols)
				throw std::invalid_argument("Matrix must be square to compute determinant.");
			if (m_Rows == 0) return T(1); // Determinant of 0x0 matrix is 1 by convention for some
            
			size_t n = m_Rows;
			if (n == 1)
				return m_Data[0][0];
			if (n == 2) // Corrected base case for 2x2
				return m_Data[0][0] * m_Data[1][1] - m_Data[0][1] * m_Data[1][0];
			
			T det = T(0);
			for (size_t i = 0; i < n; ++i) // Corrected loop variable from 'size_t =' to 'size_t i ='
			{
				Matrix<T> minor = getMinor(*this, 0, i); // Pass current matrix to getMinor
				T sign = ((i % 2) == 0) ? T(1) : T(-1); // Use T for sign
				det += sign * m_Data[0][i] * minor.Determinant(); // Use m_Data for current matrix element
			}
			return det;
		}

		/**
		 * @brief Computes the inverse of this matrix.
		 * The matrix must be square and non-singular (determinant != 0).
		 * @return Matrix<T> A new matrix that is the inverse of this one.
		 * @throws std::invalid_argument if the matrix is not square.
		 * @throws std::runtime_error if the matrix is singular (determinant is zero).
		 */
		Matrix<T> Inverse() const
		{
			if (m_Rows != m_Cols)
				throw std::invalid_argument("Matrix must be square to compute inverse.");
            if (m_Rows == 0) return Matrix<T>(); // Inverse of 0x0 is undefined or 0x0

			T det = Determinant();
			// Use a small epsilon for floating point comparison
            constexpr T epsilon = T(1e-9); // Adjust epsilon as needed for type T
			if (std::abs(det) < epsilon) // Check if determinant is close to zero
				throw std::runtime_error("Matrix is singular (determinant is near zero) and cannot be inverted.");

			Matrix<T> cofactors(m_Rows, m_Cols);
			for (size_t i = 0; i < m_Rows; ++i)
			{
				for (size_t j = 0; j < m_Cols; ++j)
				{
					Matrix<T> minor = getMinor(*this, i, j);
					T minor_det = minor.Determinant();
					cofactors.m_Data[i][j] = (((i + j) % 2 == 0) ? T(1) : T(-1)) * minor_det;
				}
			}
			
			Matrix<T> adjugate = cofactors.Transpose();
			
            // Element-wise division by determinant
            Matrix<T> inverse_matrix(m_Rows, m_Cols);
            T inv_det = T(1) / det;
            for (size_t i=0; i < m_Rows; ++i) {
                for (size_t j=0; j < m_Cols; ++j) {
                    inverse_matrix.m_Data[i][j] = adjugate.m_Data[i][j] * inv_det;
                }
            }
			return inverse_matrix;
		}

	/**
	 * @brief Accesses a specific row of the matrix.
	 * @param i Index of the row.
	 * @return MatrixRow<T>& Reference to the MatrixRow object.
	 * @throws std::out_of_range if i is out of bounds.
	 */
	MatrixRow<T>& operator[](size_t i)
	{
        if (i >= m_Rows) throw std::out_of_range("Matrix::operator[]: Row index out of range.");
		return m_Data[i];
	}
	/**
	 * @brief Accesses a specific row of the matrix (const version).
	 * @param i Index of the row.
	 * @return const MatrixRow<T>& Const reference to the MatrixRow object.
	 * @throws std::out_of_range if i is out of bounds.
	 */
	const MatrixRow<T> &operator[](size_t i) const 
    { 
        if (i >= m_Rows) throw std::out_of_range("Matrix::operator[] const: Row index out of range.");
        return m_Data[i]; 
    }

	// --- Arithmetic Operators ---

	/** @brief Matrix addition. Requires matrices of the same dimensions. */
	Matrix<T> operator+(const Matrix<T> &b) const // Made const
	{
		if (m_Rows != b.m_Rows || m_Cols != b.m_Cols) // Check all dimensions
			throw std::invalid_argument("Matrix dimensions must match for addition."); // Throw exception
		
		Matrix<T> c(m_Rows, m_Cols);
		for (size_t i = 0; i < m_Rows; i++) // Use size_t
			for (size_t j = 0; j < m_Cols; j++) // Use size_t
				c.m_Data[i][j] = m_Data[i][j] + b.m_Data[i][j];
		return c;
	}
	/** @brief Scalar addition (adds scalar to each element). */
	Matrix<T> operator+(const T b) const
	{
        if (m_Rows == 0 || m_Cols == 0) return Matrix<T>(m_Rows, m_Cols);
		Matrix<T> c(m_Rows, m_Cols);
		for (size_t i = 0; i < m_Rows; i++)
			for (size_t j = 0; j < m_Cols; j++)
				c.m_Data[i][j] = m_Data[i][j] + b;
		return c;
	}

	/** @brief Matrix addition assignment. */
	Matrix<T>& operator+=(const Matrix<T> &b) // Return reference, not const
	{
		if (m_Rows != b.m_Rows || m_Cols != b.m_Cols)
			throw std::invalid_argument("Matrix dimensions must match for addition assignment.");
		
		for (size_t i = 0; i < m_Rows; i++)
			for (size_t j = 0; j < m_Cols; j++)
				m_Data[i][j] += b.m_Data[i][j]; // Modify self
		return *this;
	}

	/** @brief Scalar addition assignment. */
	Matrix<T>& operator+=(const T b) // Return reference, not const
	{
        if (m_Rows == 0 || m_Cols == 0) return *this;
		for (size_t i = 0; i < m_Rows; i++)
			for (size_t j = 0; j < m_Cols; j++)
				m_Data[i][j] += b; // Modify self
		return *this;
	}

	/** @brief Matrix subtraction. Requires matrices of the same dimensions. */
	Matrix<T> operator-(const Matrix<T> &b) const
	{
		if (m_Rows != b.m_Rows || m_Cols != b.m_Cols)
			throw std::invalid_argument("Matrix dimensions must match for subtraction.");
		
		Matrix<T> c(m_Rows, m_Cols);
		for (size_t i = 0; i < m_Rows; i++)
			for (size_t j = 0; j < m_Cols; j++)
				c.m_Data[i][j] = m_Data[i][j] - b.m_Data[i][j];
		return c;
	}

	/** @brief Scalar subtraction (subtracts scalar from each element). */
	Matrix<T> operator-(const T b) const
	{
        if (m_Rows == 0 || m_Cols == 0) return Matrix<T>(m_Rows, m_Cols);
		Matrix<T> c(m_Rows, m_Cols);
		for (size_t i = 0; i < m_Rows; i++)
			for (size_t j = 0; j < m_Cols; j++)
				c.m_Data[i][j] = m_Data[i][j] - b;
		return c;
	}

	/** @brief Matrix subtraction assignment. */
	Matrix<T>& operator-=(const Matrix<T> &b) // Return reference, not const
	{
		if (m_Rows != b.m_Rows || m_Cols != b.m_Cols)
			throw std::invalid_argument("Matrix dimensions must match for subtraction assignment.");
		
		for (size_t i = 0; i < m_Rows; i++)
			for (size_t j = 0; j < m_Cols; j++)
				m_Data[i][j] -= b.m_Data[i][j]; // Modify self
		return *this;
	}
	/** @brief Scalar subtraction assignment. */
	Matrix<T>& operator-=(const T b) // Return reference, not const
	{
        if (m_Rows == 0 || m_Cols == 0) return *this;
		for (size_t i = 0; i < m_Rows; i++)
			for (size_t j = 0; j < m_Cols; j++)
				m_Data[i][j] -= b; // Modify self
		return *this;
	}
	
	/** @brief Scalar division (divides each element by scalar). */
	Matrix<T> operator/(const T b) const
	{
        if (std::abs(b) < T(1e-9)) throw std::runtime_error("Division by zero or very small number.");
        if (m_Rows == 0 || m_Cols == 0) return Matrix<T>(m_Rows, m_Cols);
		Matrix<T> c(m_Rows, m_Cols);
		for (size_t i = 0; i < m_Rows; i++)
			for (size_t j = 0; j < m_Cols; j++)
				c.m_Data[i][j] = m_Data[i][j] / b;
		return c;
	}

	/** @brief Scalar division assignment. */
	Matrix<T>& operator/=(const T b) // Return reference, not const
	{
        if (std::abs(b) < T(1e-9)) throw std::runtime_error("Division by zero or very small number in assignment.");
        if (m_Rows == 0 || m_Cols == 0) return *this;
		for (size_t i = 0; i < m_Rows; i++)
			for (size_t j = 0; j < m_Cols; j++)
				m_Data[i][j] /= b; // Modify self
		return *this;
	}

	/** @brief Matrix multiplication. Number of columns in the first matrix must equal number of rows in the second. */
	Matrix<T> operator*(const Matrix<T> &b) const
	{
		if (m_Cols != b.m_Rows)
			throw std::invalid_argument("Matrix dimensions are incompatible for multiplication (A.cols != B.rows).");
        if (m_Rows == 0 || m_Cols == 0 || b.m_Cols == 0) { // Handle multiplication by empty matrix
            return Matrix<T>(m_Rows, b.m_Cols); // Result is an empty matrix with appropriate dimensions
        }
		
		Matrix<T> c(m_Rows, b.m_Cols); // Result matrix initialized to zeros by MatrixRow constructor
		for (size_t i = 0; i < m_Rows; i++) {
			for (size_t k = 0; k < b.m_Cols; k++) { // Iterate over columns of b (which is cols of c)
                T sum = T(0); // Initialize sum for c[i][k]
				for (size_t j = 0; j < m_Cols; j++) { // Iterate over columns of a / rows of b
					sum += m_Data[i][j] * b.m_Data[j][k];
				}
                c.m_Data[i][k] = sum;
            }
        }
		return c;
	}

	/** @brief Scalar multiplication (multiplies each element by scalar). */
	Matrix<T> operator*(const T b) const
	{
        if (m_Rows == 0 || m_Cols == 0) return Matrix<T>(m_Rows, m_Cols);
		Matrix<T> c(m_Rows, m_Cols);
		for (size_t i = 0; i < m_Rows; i++)
			for (size_t j = 0; j < m_Cols; j++)
				c.m_Data[i][j] = m_Data[i][j] * b;
		return c;
	}

	/** @brief Scalar multiplication assignment. */
	Matrix<T>& operator*=(const T b) // Return reference, not const
	{
        if (m_Rows == 0 || m_Cols == 0) return *this;
		for (size_t i = 0; i < m_Rows; i++)
			for (size_t j = 0; j < m_Cols; j++)
				m_Data[i][j] *= b; // Modify self
		return *this;
	}

	/** @brief Returns an iterator to the first row of the matrix. */
	Iterator begin() { return Iterator(m_Data.get()); }
	/** @brief Returns an iterator to one past the last row of the matrix. */
	Iterator end() { return Iterator(m_Data.get() + m_Rows); }
    // Add const versions for iterators if needed for const Matrix objects
    // const_Iterator begin() const { return const_Iterator(m_Data.get()); }
    // const_Iterator end() const { return const_Iterator(m_Data.get() + m_Rows); }


private:
	/**
	 * @brief Helper function to get the minor of a matrix.
	 * Used in determinant and inverse calculations.
	 * @param matrix The source matrix.
	 * @param row_to_remove Row index to exclude.
	 * @param col_to_remove Column index to exclude.
	 * @return Matrix<T> The resulting minor matrix.
	 * @throws std::invalid_argument if the input matrix is not square.
	 */
	Matrix<T> getMinor(const Matrix<T> &matrix, size_t row_to_remove, size_t col_to_remove) const
	{
		if (matrix.m_Rows != matrix.m_Cols) // Should be checked by caller, but good for safety
			throw std::invalid_argument("Matrix must be square to compute minor.");
        if (matrix.m_Rows == 0) return Matrix<T>();


		size_t n = matrix.m_Rows;
        if (n == 0) return Matrix<T>(0,0); // Minor of empty matrix is empty
		Matrix<T> minor_matrix(n > 0 ? n - 1 : 0, n > 0 ? n - 1 : 0);

		size_t minor_i = 0; 
		for (size_t i = 0; i < n; ++i)
		{
			if (i == row_to_remove)
				continue;

			size_t minor_j = 0; 
			for (size_t j = 0; j < n; ++j)
			{
				if (j == col_to_remove)
					continue;
                if (minor_i < minor_matrix.rows() && minor_j < minor_matrix.cols()) {
				    minor_matrix.m_Data[minor_i][minor_j] = matrix.m_Data[i][j];
                }
				++minor_j;
			}
			++minor_i;
		}
		return minor_matrix;
	}

	size_t m_Rows = 0;     ///< Number of rows.
	size_t m_Cols = 0;     ///< Number of columns.
	size_t m_Size = 0;     ///< Total number of elements (rows * cols).
	size_t m_Capacity = 0; ///< Allocated capacity in bytes.
	std::unique_ptr<MatrixRow<T>[]> m_Data; ///< Pointer to an array of MatrixRow objects.
};

// Friend function for scalar multiplication (e.g., 5 * matrix)
template <typename T>
Matrix<T> operator*(const T scalar, const Matrix<T>& matrix) {
    return matrix * scalar; // Utilize the existing Matrix<T>::operator*(T)
}

} // namespace Matrix
#endif // MATRIX_H

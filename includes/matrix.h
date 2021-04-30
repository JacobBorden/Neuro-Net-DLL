/**
 * @file matrix.h
 * @author Jacob Borden (amenra.beats@gmail.com)
 * @brief A collection of Matrix classes
 * @version 0.1
 * @date 2021-04-24
 * 
 * @copyright Copyright (c) 2021
 * 
 */

//Header definitions
#pragma once
#ifndef MATRIX_H
#define MATRIX_H

//Includes
#include <vector>
#include <omp.h>

namespace Matrix
{
    /**
     * @brief  A 2 dimensional matrix consisting of rows and columns for use in mathematical operations.
     * @tparam T - The type to be used in the matrix
     * @param rows-The number of rows in the matrix
     * @param columns- The number of columns in the matrix
     * 
     */
    template <class T>
    class Matrix
    {
        public:
        Matrix<T>();
        Matrix<T>(int rows, int columns);
        ~Matrix();
        void Resize(int rows, int columns);
        int Rows();
        int Columns();
        std::vector<T>& operator [](int i);
		Matrix<T> operator+(Matrix<T> matrix_b);
		Matrix<T> operator*(Matrix<T> matrix_b);
        private:
        std::vector<std::vector <T>> matrix;
        int rows;
        int columns;
    };
}

#endif
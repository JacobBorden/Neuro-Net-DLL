import os

with open("src/math/matrix.h", "r") as f:
    content = f.read()

# Verify that the operators are present
operators = [
    "operator+(const Matrix<T> &b)",
    "operator+(const T b)",
    "operator-(const Matrix<T> &b)",
    "operator-(const T b)",
    "operator*(const Matrix<T> &b)",
    "operator*(const T b)",
    "operator/(const T b)"
]

for op in operators:
    if op in content:
        print(f"Found: {op}")
    else:
        print(f"Not found: {op}")

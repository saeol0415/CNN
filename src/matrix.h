#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>


class Matrix {
  public:
    std::vector<std::vector<double>> data;
    int rows, cols;

    Matrix() {}
    
    Matrix(int r, int c) : rows(r), cols(c), data(r, std::vector<double>(c, 0)) {}
    
    Matrix(std::vector<std::vector<double>> d) : rows(d.size()), cols(d[0].size()), data(d) {}

    Matrix(int r, int c, std::vector<double>& values) : rows(r), cols(c), data(r, std::vector<double>(c, 0)) {
        if (values.size() != r * c) throw std::invalid_argument("Invalid number of values");
        for (int i = 0; i < r; i++){
            for (int j = 0; j < c; j++){
                data[i][j] = values[i * c + j];
            }
        }
    }

    void randomize(double min_val = -0.5, double max_val = 0.5) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double rand_val = min_val + (max_val - min_val) * (rand() / (double)RAND_MAX);
                data[i][j] = rand_val;
            }
        }
    }

    void print() const {
        for (const auto& row : data) {
            for (double val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    Matrix pad(int top, int left, int bottom, int right) const {
        int output_rows = rows + top + bottom;
        int output_cols = cols + left + right;
        Matrix padded(output_rows, output_cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                padded.data[i+top][j+left] = data[i][j];
            }
        }
        return padded;
    }

    Matrix flatten() const {
        Matrix flat(rows * cols, 1);
        int index = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flat.data[index][0] = data[i][j];
                index++;
            }
        }
        return flat;
    }

    Matrix T() const {
        Matrix transposed(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed.data[j][i] = data[i][j];
            }
        }
        return transposed;
    }

    Matrix flip() const {
        Matrix flipped(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flipped.data[i][j] = data[rows - i - 1][cols - j - 1];
            }
        }
        return flipped;
    }

    Matrix slice(int row_start, int row_end, int col_start, int col_end) const {
        int new_rows = row_end - row_start;
        int new_cols = col_end - col_start;
        Matrix sliced(new_rows, new_cols);

        for (int i = 0; i < new_rows; i++) {
            for (int j = 0; j < new_cols; j++) {
                sliced.data[i][j] = this -> data[row_start + i][col_start + j];
            }
        }
        return sliced;
    }

    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix size mismatch.");
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix size mismatch.");
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) throw std::invalid_argument("Incompatible matrix dimensions.");

        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < cols; k++) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this->data[i][j] * scalar;
            }
        }
        return result;
    }

    friend Matrix operator*(double scalar, const Matrix& mat) {
        return mat * scalar;
    }

    Matrix& operator+=(const Matrix& other) {
        if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix size mismatch.");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] += other.data[i][j];
            }
        }
        return *this;
    }

    Matrix& operator-=(const Matrix& other) {
        if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix size mismatch.");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] -= other.data[i][j];
            }
        }
        return *this;
    }

    Matrix& operator*=(const Matrix& other) {
        if (cols != other.rows) throw std::invalid_argument("Incompatible matrix dimensions.");

        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < cols; k++) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        *this = result;
        return *this;
    }

    Matrix& operator*=(double scalar) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this->data[i][j] *= scalar;
            }
        }
        return *this;
    }
};

#endif
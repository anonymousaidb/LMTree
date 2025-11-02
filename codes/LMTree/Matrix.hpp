

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <complex>
#include <cstdint>
#include <vector>
#ifndef MATRIX_HPP
#define MATRIX_HPP

namespace ML {
    static std::int64_t pointer_count;
    template<typename T, class Allocator = std::allocator<T>>
    class Matrix {
    public:
        enum Axis {
            all,
            row,
            col,
        };
        enum InitType {
            linear_space,
            random,
        };
    private:
        std::int64_t rows;
        std::int64_t cols;
        T *_data_ptr;
        Allocator allocator{};
    public:
        [[nodiscard]] std::int64_t Rows() const;

        [[nodiscard]] std::int64_t Cols() const;

        const T *data_ptr() const;
        T *data_ptr();

        Matrix();

        Matrix(std::int64_t rows, std::int64_t cols, T init_value);

        Matrix(std::int64_t rows, std::int64_t cols, InitType init_type = InitType::random, T min = -1, T max = 1);

        Matrix(const Matrix &matrix) noexcept;

        Matrix(Matrix &&matrix) noexcept;

        ~Matrix();

        Matrix norm(Axis axis = Axis::all, double p = 2);

        Matrix dot(const Matrix &matrix) const;

        template <typename U>
        friend std::ostream &operator<<(std::ostream &out, const Matrix<U> &matrix);

        T &operator()(std::int64_t inRowIndex, std::int64_t inColIndex) noexcept;

        Matrix sum(Axis axis = Axis::all) const;

        Matrix broadcast(std::int64_t new_rows, std::int64_t new_cols) const;

        Matrix operator+(const Matrix &another) const;

        Matrix operator-(const Matrix &another) const;

        Matrix operator*(const Matrix &another) const;

        Matrix operator/(const Matrix &another) const;

        Matrix operator/(T value) const;

        Matrix operator-() const;

        Matrix operator*(T value) const;

        Matrix operator+(T value) const;

        Matrix operator-(T value) const;

        Matrix h_stack(const Matrix &matrix) const;

        [[nodiscard]] std::int64_t size() const;

        Matrix v_stack(const Matrix &matrix) const;

        Matrix h_split(std::int64_t col1 = 0, std::int64_t col2 = 0) const;

        Matrix v_split(std::int64_t row1 = 0, std::int64_t row2 = 0) const;

        Matrix transpose() const;

        Matrix mean(Axis axis = Axis::all) const;
        Matrix var(Axis axis = Axis::all) const;

        Matrix inverse() const;

        std::tuple<Matrix, Matrix> qr_decomposition() const;

        std::vector<T> qr_eigenvalues(size_t max_iter = 1000, T tol = 1e-6) const;

        std::tuple<std::vector<T>, Matrix>
        qr_eigenvalues_and_eigenvectors(std::int64_t max_iter = 1000, T tol = 1e-6) const;

        Matrix getRow(std::int64_t n) const;

        Matrix getColumn(std::int64_t n) const;

        Matrix &operator=(const Matrix &another);
        Matrix &operator=(Matrix &&another);

        Matrix reshape(std::int64_t new_rows, std::int64_t new_cols) const;

        Matrix UnivariatePolynomial(std::int64_t degree) const;
    };

    template<class T>
    Matrix<T> linear_space(T inStart, T inStop, std::int64_t inNum = 50);
}
#endif

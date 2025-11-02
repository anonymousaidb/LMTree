

#include "Matrix.hpp"

#include <random>

#include "Parameters.hpp"


namespace ML {
    template<typename T, typename Allocator>
    std::int64_t Matrix<T, Allocator>::Rows() const {
        return rows;
    }

    template<typename T, typename Allocator>
    std::int64_t Matrix<T, Allocator>::Cols() const {
        return cols;
    }

    template<typename T, typename Allocator>
    const T *Matrix<T, Allocator>::data_ptr() const{
        return _data_ptr;
    }

    template<typename T, typename Allocator>
    T *Matrix<T, Allocator>::data_ptr(){
        return _data_ptr;
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator>::Matrix() : rows(0), cols(0), _data_ptr(nullptr) {}

    template<typename T, typename Allocator>
    Matrix<T, Allocator>::Matrix(const std::int64_t rows, const std::int64_t cols, const T init_value) : rows(rows),
                                                                                                         cols(cols) {
        _data_ptr = allocator.allocate(rows * cols);
        ++pointer_count;
        std::fill(_data_ptr, _data_ptr + (rows * cols), init_value);
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator>::Matrix(const std::int64_t rows, const std::int64_t cols, const InitType init_type,
                                 T min, T max) : rows(rows), cols(cols) {
        _data_ptr = allocator.allocate(rows * cols);
        ++pointer_count;
        switch (init_type) {
            case InitType::random: {
                std::uniform_real_distribution<float> dist(min, max);
                std::generate(_data_ptr, _data_ptr + (rows * cols), [&]() { return dist(global_gen); });
                break;
            }

            case InitType::linear_space:
                T step = (max - min) / (size() - 1);
                T value = min - step;
                for (T *start = _data_ptr, *end = _data_ptr + size(); start < end; ++start) {
                    *start = (value += step);
                }
                break;
        }
    }


    template<typename T, typename Allocator>
    Matrix<T, Allocator>::Matrix(const Matrix &matrix) noexcept: rows(matrix.rows),
                                                                    cols(matrix.cols) {
        _data_ptr = allocator.allocate(rows * cols);
        ++pointer_count;
        std::copy(matrix._data_ptr, matrix._data_ptr + size(), _data_ptr);
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator>::Matrix(Matrix &&matrix) noexcept : rows(matrix.rows), cols(matrix.cols),
                                                                           _data_ptr(matrix._data_ptr) {
        matrix._data_ptr = nullptr;
        matrix.rows = 0;
        matrix.cols = 0;
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator>::~Matrix() {
        if (_data_ptr) {
            allocator.deallocate(_data_ptr, rows * cols);
            --pointer_count;
        }
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::norm(Matrix<T, Allocator>::Axis axis, double p) {
        auto *array1 = reinterpret_cast<T(*)[cols]>(_data_ptr);
        if (axis == Axis::row) {
            Matrix<T, Allocator> result(1, cols, 0);
            T *array3 = result._data_ptr;
            for (std::int64_t _0 = 0; _0 < rows; _0++) {
                for (std::int64_t _1 = 0; _1 < cols; _1++) {
                    array3[_1] += std::pow(std::abs(array1[_0][_1]), p);
                }
            }
            for (auto start = result._data_ptr, end = result._data_ptr + result.size(); start < end; start++) {
                *start = std::pow(*start, 1 / p);
            }
            return result;
        } else if (axis == Axis::col) {
            Matrix<T, Allocator> result(rows, 1, 0);
            T *array3 = result._data_ptr;
            for (std::int64_t _0 = 0; _0 < rows; _0++) {
                for (std::int64_t _1 = 0; _1 < cols; _1++) {
                    array3[_0] += std::pow(std::abs(array1[_0][_1]), p);
                }
            }
            for (auto start = result._data_ptr, end = result._data_ptr + result.size(); start < end; start++) {
                *start = std::pow(*start, 1 / p);
            }
            return result;
        } else {
            Matrix<T, Allocator> result(1, 1, 0);
            T *array3 = result._data_ptr;
            for (std::int64_t _0 = 0; _0 < rows; _0++) {
                for (std::int64_t _1 = 0; _1 < cols; _1++) {
                    *array3 += std::pow(std::abs(array1[_0][_1]), p);
                }
            }
            *array3 = std::pow(*array3, 1 / p);
            return result;
        }
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::dot(const Matrix &matrix) const {
        if (cols != matrix.rows) {
            throw std::runtime_error("The matrix dimension does not meet the requirements");
        }
        Matrix result(rows, matrix.cols, 0);
        auto *array_this = reinterpret_cast<T (*)[cols]>(_data_ptr);
        auto *array_another = reinterpret_cast<T (*)[matrix.cols]>(matrix._data_ptr);
        auto *array_result = reinterpret_cast<T (*)[result.cols]>(result._data_ptr);
        for (std::int64_t _0 = 0; _0 < rows; _0++) {
            for (std::int64_t _1 = 0; _1 < matrix.cols; _1++) {
                for (std::int64_t _2 = 0; _2 < cols; _2++) {
                    array_result[_0][_1] += array_this[_0][_2] * array_another[_2][_1];
                }
            }
        }
        return result;
    }

    template<typename U>
    std::ostream &operator<<(std::ostream &out, const Matrix<U> &matrix) {
        out << "[" << matrix.rows << "," << matrix.cols << "]" << "\n{ ";
        auto *array1 = reinterpret_cast<U (*)[matrix.cols]>(matrix._data_ptr);
        for (std::int64_t _0 = 0; _0 < matrix.rows; _0++) {
            out << "{ ";
            for (std::int64_t _1 = 0; _1 < matrix.cols; _1++) {
                out << array1[_0][_1] << ", ";
            }
            out << "},\n ";
        }
        out << "}";
        return out;
    }

    template<typename T, typename Allocator>
    T &Matrix<T, Allocator>::operator()(std::int64_t inRowIndex, std::int64_t inColIndex) noexcept {
        if (inRowIndex < 0) { inRowIndex += rows; }
        if (inColIndex < 0) { inColIndex += cols; }
        return _data_ptr[inRowIndex * cols + inColIndex];
    }


    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::sum(Axis axis) const {
        auto *array1 = reinterpret_cast<T(*)[cols]>(_data_ptr);
        if (axis == Axis::row) {
            Matrix result(1, cols, 0);
            T *array3 = result._data_ptr;
            for (std::int64_t _0 = 0; _0 < rows; _0++) {
                for (std::int64_t _1 = 0; _1 < cols; _1++) {
                    array3[_1] += array1[_0][_1];
                }
            }
            return result;
        } else if (axis == Axis::col) {
            Matrix result(rows, 1, 0);
            T *array3 = result._data_ptr;
            for (std::int64_t _0 = 0; _0 < rows; _0++) {
                for (std::int64_t _1 = 0; _1 < cols; _1++) {
                    array3[_0] += array1[_0][_1];
                }
            }
            return result;
        } else {
            Matrix result(1, 1, 0);
            T *array3 = result._data_ptr;
            for (std::int64_t _0 = 0; _0 < rows; _0++) {
                for (std::int64_t _1 = 0; _1 < cols; _1++) {
                    *array3 += array1[_0][_1];
                }
            }
            return result;
        }
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::broadcast(const std::int64_t new_rows, const std::int64_t new_cols) const {
        const std::int64_t row_multiple = new_rows / rows;
        const std::int64_t col_multiple = new_cols / cols;
        if (!(new_rows == row_multiple * rows && new_cols == col_multiple * cols)) {
            throw std::runtime_error("The matrix dimension does not meet the requirements");
        }
        Matrix result(new_rows, new_cols);
        auto *array_this = reinterpret_cast<T(*)[cols]>(_data_ptr);
        auto *array_result = reinterpret_cast<T(*)[new_cols]>(result._data_ptr);

        for (std::int64_t i = 0; i < rows; i++) {
            for (std::int64_t j = 0; j < cols; j++) {
                for (std::int64_t k = 0; k < col_multiple; k++) {
                    array_result[i * new_cols][k * cols + j] = array_this[i][j];
                }
            }
        }
        for (std::int64_t i = 1; i < row_multiple; i++) {
            std::copy(result._data_ptr, result._data_ptr + (size() * col_multiple),
                      result._data_ptr + (i * size() * col_multiple));
        }
        return result;
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::operator+(const Matrix &another) const {
        if (!(another.Rows() == rows && another.Cols() == cols)) {
            throw std::runtime_error("The matrix dimension does not meet the requirements");
        }
        Matrix result(*this);
        T *array_result = result._data_ptr;
        T *array_another = another._data_ptr;
        for (std::int64_t _ = 0, size = rows * cols; _ < size; _++) {
            *(array_result++) += *(array_another++);
        }
        return result;
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::operator-(const Matrix &another) const {
        if (!(another.Rows() == rows && another.Cols() == cols)) {
            throw std::runtime_error("The matrix dimension does not meet the requirements");
        }
        Matrix result(*this);
        T *array_result = result._data_ptr;
        T *array_another = another._data_ptr;
        for (std::int64_t _ = 0, size = rows * cols; _ < size; _++) {
            *(array_result++) -= *(array_another++);
        }
        return result;
    }


    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::operator*(const Matrix &another) const {
        if (!(another.Rows() == rows && another.Cols() == cols)) {
            throw std::runtime_error("The matrix dimension does not meet the requirements");
        }
        Matrix result(*this);
        T *array_result = result._data_ptr;
        T *array_another = another._data_ptr;
        for (std::int64_t _ = 0, size = rows * cols; _ < size; _++) {
            *(array_result++) *= *(array_another++);
        }
        return result;
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::operator/(const Matrix &another) const {
        if (!(another.Rows() == rows && another.Cols() == cols)) {
            throw std::runtime_error("The matrix dimension does not meet the requirements");
        }
        Matrix result(*this);
        T *array_result = result._data_ptr;
        T *array_another = another._data_ptr;
        for (std::int64_t _ = 0, size = rows * cols; _ < size; _++) {
            *(array_result++) /= *(array_another++);
        }
        return result;
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::operator/(const T value) const {
        Matrix result(*this);
        T *array_result = result._data_ptr;
        for (std::int64_t _ = 0, size = rows * cols; _ < size; _++) {
            *(array_result++) /= value;
        }
        return result;
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::operator-() const {
        Matrix result(*this);
        T *array_result = result._data_ptr;
        for (std::int64_t _ = 0, size = rows * cols; _ < size; _++) {
            *array_result = -*array_result;
            ++array_result;
        }
        return result;
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::operator*(const T value) const {
        Matrix result(*this);
        T *array_result = result._data_ptr;
        for (std::int64_t _ = 0, size = rows * cols; _ < size; _++) {
            *(array_result++) *= value;
        }
        return result;
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::operator+(const T value) const {
        Matrix result(*this);
        T *array_result = result._data_ptr;
        for (std::int64_t _ = 0, size = rows * cols; _ < size; _++) {
            *(array_result++) += value;
        }
        return result;
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::operator-(const T value) const {
        Matrix result(*this);
        T *array_result = result._data_ptr;
        for (std::int64_t _ = 0, size = rows * cols; _ < size; _++) {
            *(array_result++) -= value;
        }
        return result;
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::h_stack(const Matrix &matrix) const {
        if (matrix.rows != rows) {
            throw std::runtime_error("The matrix dimension does not meet the requirements");
        }
        Matrix result(rows, (cols + matrix.cols));
        auto *array_this = reinterpret_cast<T(*)[cols]>(_data_ptr);
        auto *array_another = reinterpret_cast<T(*)[matrix.cols]>(matrix._data_ptr);
        auto *array_result = reinterpret_cast<T(*)[result.cols]>(result._data_ptr);
        for (std::int64_t _0 = 0; _0 < rows; _0++) {
            for (std::int64_t _1 = 0; _1 < cols; _1++) {
                array_result[_0][_1] = array_this[_0][_1];
            }
        }
        for (std::int64_t _0 = 0; _0 < rows; _0++) {
            for (std::int64_t _1 = 0; _1 < matrix.cols; _1++) {
                array_result[_0][_1 + cols] = array_another[_0][_1];
            }
        }
        return result;
    }

    template<typename T, typename Allocator>
    [[nodiscard]] std::int64_t Matrix<T, Allocator>::size() const {
        return rows * cols;
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::v_stack(const Matrix<T, Allocator> &matrix) const {
        if (matrix.cols != cols) {
            throw std::runtime_error("The matrix dimension does not meet the requirements");
        }
        Matrix<T, Allocator> result(rows + matrix.rows, cols);
        std::copy(_data_ptr, _data_ptr + this->size(), result._data_ptr);
        std::copy(matrix._data_ptr, matrix._data_ptr + matrix.cols * matrix.rows, result._data_ptr + size());
        return result;
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::h_split(std::int64_t col1, std::int64_t col2) const {
        if (col1 < 0) {
            col1 += cols;
        }
        if (col2 <= 0) {
            col2 += cols;
        }
        if (col2 < col1) {
            throw std::runtime_error("The matrix dimension does not meet the requirements");
        }
        Matrix<T, Allocator> result(rows, col2 - col1);
        auto *array1 = reinterpret_cast<T(*)[cols]>(_data_ptr);
        auto *array3 = reinterpret_cast<T(*)[result.cols]>(result._data_ptr);
        for (std::int64_t _0 = 0; _0 < rows; _0++) {
            for (std::int64_t _1 = col1; _1 < col2; _1++) {
                array3[_0][_1 - col1] = array1[_0][_1];
            }
        }
        return result;
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::v_split(std::int64_t row1, std::int64_t row2) const {
        if (row1 < 0) {
            row1 += rows;
        }
        if (row2 <= 0) {
            row2 += rows;
        }
        if (row2 < row1) {
            throw std::runtime_error("The matrix dimension does not meet the requirements");
        }
        Matrix result(row2 - row1, cols);
        std::copy(_data_ptr + row1 * cols, _data_ptr + row2 * cols, result._data_ptr);
        return result;
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::transpose() const {
        Matrix result(cols, rows);
        auto *array1 = reinterpret_cast<T(*)[cols]>(_data_ptr);
        auto *array3 = result._data_ptr;
        for (std::int64_t _0 = 0; _0 < cols; _0++) {
            for (std::int64_t _1 = 0; _1 < rows; _1++) {
                *(array3++) = array1[_1][_0];
            }
        }
        return result;
    }


    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::var(Axis axis) const {
        auto mean_result = mean(axis);
        T *array_mean = mean_result._data_ptr;
        auto *array_source = reinterpret_cast<T(*)[cols]>(_data_ptr);
        if (axis == Axis::row) {
            Matrix result(1, cols, 0);
            T *array_result = result._data_ptr;
            for (std::int64_t _0 = 0; _0 < rows; _0++) {
                for (std::int64_t _1 = 0; _1 < cols; _1++) {
                    auto tmp = array_source[_0][_1] - array_mean[_1];
                    array_result[_1] += tmp * tmp;
                }
            }
            for (auto start = result._data_ptr, end = result._data_ptr + result.size(); start < end; ++start) {
                *start /= static_cast<T>(rows-1);
            }
            return result;
        }
        if (axis == Axis::col) {
            Matrix result(rows, 1, 0);
            T *array_result = result._data_ptr;
            for (std::int64_t _0 = 0; _0 < rows; _0++) {
                for (std::int64_t _1 = 0; _1 < cols; _1++) {
                    auto tmp = array_source[_0][_1] - array_mean[_0];
                    array_result[_0] += tmp * tmp;
                }
            }
            for (auto start = result._data_ptr, end = result._data_ptr + result.size(); start < end; ++start) {
                *start /= static_cast<T>(cols-1);
            }
            return result;
        }

        if (axis == Axis::all){
            Matrix result(1, 1, 0);
            T *array_result = result._data_ptr;
            for (std::int64_t _0 = 0; _0 < rows; _0++) {
                for (std::int64_t _1 = 0; _1 < cols; _1++) {
                    auto tmp = array_source[_0][_1] - *array_mean;
                    *array_result += tmp * tmp;
                }
            }
            *array_result /= static_cast<T>(size()-1);
            return result;
        }
        throw std::invalid_argument("bad Axis !");
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::mean(Axis axis) const {
        auto *array_source = reinterpret_cast<T(*)[cols]>(_data_ptr);
        if (axis == Axis::row) {
            Matrix result(1, cols, 0);
            T *array_result = result._data_ptr;
            for (std::int64_t _0 = 0; _0 < rows; _0++) {
                for (std::int64_t _1 = 0; _1 < cols; _1++) {
                    array_result[_1] += array_source[_0][_1];
                }
            }
            for (auto start = result._data_ptr, end = result._data_ptr + result.size(); start < end; ++start) {
                *start /= static_cast<T>(rows);
            }
            return result;
        }
        if (axis == Axis::col) {
            Matrix result(rows, 1, 0);
            T *array_result = result._data_ptr;
            for (std::int64_t _0 = 0; _0 < rows; _0++) {
                for (std::int64_t _1 = 0; _1 < cols; _1++) {
                    array_result[_0] += array_source[_0][_1];
                }
            }
            for (auto start = result._data_ptr, end = result._data_ptr + result.size(); start < end; ++start) {
                *start /= static_cast<T>(cols);
            }
            return result;
        }
        if (axis == Axis::all){
            Matrix result(1, 1, 0);
            T *array_result = result._data_ptr;
            for (std::int64_t _0 = 0; _0 < rows; _0++) {
                for (std::int64_t _1 = 0; _1 < cols; _1++) {
                    *array_result += array_source[_0][_1];
                }
            }
            *array_result /= static_cast<T>(rows * cols);
            return result;
        }
        throw std::invalid_argument("bad Axis !");
    }


    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::inverse() const {
        if (cols != rows) {
            throw std::runtime_error("The matrix dimension does not meet the requirements");
        }
        Matrix<T, Allocator> result(rows, cols, 0);
        for (std::int64_t _ = 0; _ < rows; _++) {
            result(_, _) = 1;
        }
        Matrix<T, Allocator> temp(*this);
        auto array1 = reinterpret_cast<T(*)[cols]>(temp._data_ptr);
        auto array3 = reinterpret_cast<T(*)[result.cols]>(result._data_ptr);
        for (std::int64_t _0 = 0; _0 < cols; _0++) {
            std::int64_t max_index = _0;
            T max_value = std::numeric_limits<T>::min();
            for (std::int64_t i = _0; i < rows; i++) {
                if (max_value < array1[_0][i]) {
                    max_index = i;
                    max_value = array1[_0][i];
                }
            }
            if (max_index != _0) {
                for (std::int64_t i = 0; i < cols; i++) {
                    auto t_ = array1[_0][i];
                    array1[_0][i] = array1[max_index][i];
                    array1[max_index][i] = t_;
                }
                for (std::int64_t i = 0; i < cols; i++) {
                    auto t_ = array3[_0][i];
                    array3[_0][i] = array3[max_index][i];
                    array3[max_index][i] = t_;
                }
            }
            T multiple = array1[_0][_0];
            for (std::int64_t i = 0; i < cols; i++) {
                array1[_0][i] /= multiple;
            }
            for (std::int64_t i = 0; i < cols; i++) {
                array3[_0][i] /= multiple;
            }
            for (std::int64_t _1 = 0; _1 < rows; _1++) {
                if (_1 == _0)continue;
                multiple = array1[_1][_0] / array1[_0][_0];
                for (std::int64_t i = 0; i < cols; i++) {
                    array1[_1][i] -= multiple * array1[_0][i];
                }
                for (std::int64_t i = 0; i < cols; i++) {
                    array3[_1][i] -= multiple * array3[_0][i];
                }
            }
        }
        return result;
    }

    template<typename T, typename Allocator>
    std::tuple<Matrix<T, Allocator>, Matrix<T, Allocator>> Matrix<T, Allocator>::qr_decomposition() const {
        Matrix Q = *this;
        Matrix R(rows, cols, 0);
        for (std::int64_t k = 0; k < cols; ++k) {
            T norm = 0;
            for (std::int64_t i = k; i < rows; ++i) {
                norm += Q(i, k) * Q(i, k);
            }
            norm = std::sqrt(norm);
            if (Q(k, k) < 0) norm = -norm;
            for (std::int64_t i = k; i < rows; ++i) { Q(i, k) /= norm; }
            Q(k, k) += 1;
            T s = std::sqrt(2.0 / Q(k, k));
            for (std::int64_t i = k; i < rows; ++i) { Q(i, k) *= s; }
            for (std::int64_t j = k + 1; j < cols; ++j) {
                T dot = 0;
                for (std::int64_t i = k; i < rows; ++i) { dot += Q(i, k) * Q(i, j); }
                for (std::int64_t i = k; i < rows; ++i) { Q(i, j) -= dot * Q(i, k); }
            }
            for (std::int64_t i = k; i < rows; ++i) {
                for (std::int64_t j = k; j < cols; ++j) { R(k, j) = Q(i, j); }
            }
        }
        return std::make_tuple(Q, R);
    }

    template<typename T, typename Allocator>
    std::vector<T> Matrix<T, Allocator>::qr_eigenvalues(size_t max_iter, T tol) const {
        Matrix A = *this;
        for (size_t iter = 0; iter < max_iter; ++iter) {
            auto [Q, R] = A.qr_decomposition();
            A = R * Q;
            bool converged = true;
            for (std::int64_t i = 0; i < rows - 1; ++i) {
                if (std::abs(A(i + 1, i)) > tol) {
                    converged = false;
                    break;
                }
            }
            if (converged) { break; }
        }
        std::vector<T> eigenvalues(rows);
        for (std::int64_t i = 0; i < rows; ++i) {
            eigenvalues[i] = A(i, i);
        }
        return eigenvalues;
    }

    template<typename T, typename Allocator>
    std::tuple<std::vector<T>, Matrix<T, Allocator>>
    Matrix<T, Allocator>::qr_eigenvalues_and_eigenvectors(const std::int64_t max_iter, T tol) const {
        Matrix A = *this;
        Matrix Q(rows, rows);
        for (size_t iter = 0; iter < max_iter; ++iter) {
            auto [Qr, R] = A.qr_decomposition();
            A = R * Qr;
            Q = Q * Qr;
            for (std::int64_t i = 0; i < rows; ++i) {
                T norm = 0;
                for (std::int64_t j = 0; j < rows; ++j) { norm += Q(j, i) * Q(j, i); }
                norm = std::sqrt(norm);
                for (std::int64_t j = 0; j < rows; ++j) { Q(j, i) /= norm; }
            }
            bool converged = true;
            for (std::int64_t i = 0; i < rows - 1; ++i) {
                if (std::abs(A(i + 1, i)) > tol) {
                    converged = false;
                    break;
                }
            }
            if (converged) { break; }
        }
        std::vector<T> eigenvalues(rows);
        for (std::int64_t i = 0; i < rows; ++i) { eigenvalues[i] = A(i, i); }
        return std::make_tuple(eigenvalues, Q);
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::getRow(const std::int64_t n) const {
        Matrix result(1, cols);
        std::copy(_data_ptr + cols * n, _data_ptr + cols * (n + 1), result._data_ptr);
        return result;
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::getColumn(const std::int64_t n) const {
        Matrix result(rows, 1);
        auto *array_this = reinterpret_cast<T(*)[cols]>(_data_ptr);
        auto *array_result = reinterpret_cast<T(*)[cols]>(result._data_ptr);
        for (int i = 0; i < rows; ++i) {
            array_result[i][0] = array_this[i][n];
        }
        return result;
    }


    template<typename T, typename Allocator>
    Matrix<T, Allocator>& Matrix<T, Allocator>::operator=(const Matrix &another) {
        if (this == &another) {
            return *this;
        }
        this->rows = another.rows;
        this->cols = another.cols;
        if (_data_ptr) {
            allocator.deallocate(_data_ptr, rows * cols);
            --pointer_count;
        }
        _data_ptr = allocator.allocate(rows * cols);
        ++pointer_count;
        std::copy(another._data_ptr, another._data_ptr + size(), _data_ptr);
        return *this;
    }


    template<typename T, typename Allocator>
    Matrix<T, Allocator>& Matrix<T, Allocator>::operator=(Matrix &&another) {
        if (this == &another) {
            return *this;
        }
        if (_data_ptr) {
            allocator.deallocate(_data_ptr, rows * cols);
            --pointer_count;
        }
        this->rows = another.rows;
        this->cols = another.cols;
        _data_ptr = another._data_ptr;

        another._data_ptr = nullptr;
        another.rows = 0;
        another.cols = 0;
        return *this;
    }


    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::reshape(std::int64_t new_rows, std::int64_t new_cols) const {
        if (new_cols * new_rows != cols * rows) {
            throw std::runtime_error("the dimension not matched !");
        }
        Matrix<T, Allocator> result(*this);
        result.rows = new_rows;
        result.cols = new_cols;
        return result;
    }

    template<typename T, typename Allocator>
    Matrix<T, Allocator> Matrix<T, Allocator>::UnivariatePolynomial(std::int64_t degree) const {
        Matrix<T, Allocator> output(this->Rows(), degree);
        auto array1 = reinterpret_cast<T(*)[cols]>(_data_ptr);
        auto array3 = reinterpret_cast<T(*)[cols * degree]>(output._data_ptr);
        for (std::int64_t j = 0; j < cols; j++) {
            array3[j][0] = array1[j][0];
        }
        for (std::int64_t i = 1; i < degree; i++) {
            for (std::int64_t j = 0; j < cols; j++) {
                array3[j][i] = array3[j][0] * array3[j][i - 1];
            }
        }
        return output;
    }

    template<class T>
    Matrix<T> linear_space(T inStart, T inStop, std::int64_t inNum) {
        return Matrix<T>(inNum, 1, Matrix<T>::linear_space, inStart, inStop);
    }

    template
    class Matrix<int>;

    template
    class Matrix<float>;

    template
    class Matrix<double>;

    template std::ostream &operator<<(std::ostream &, const Matrix<float> &);
    template std::ostream &operator<<(std::ostream &, const Matrix<int> &);
    template std::ostream &operator<<(std::ostream &, const Matrix<std::int64_t> &);

    void test() {
        int nn = 1e2;
        int dd = 1e2;
        ML::Matrix<float> a(nn,dd);

        auto var = a.var(ML::Matrix<float>::col);
        std::cout <<var<< std::endl;
    }
};

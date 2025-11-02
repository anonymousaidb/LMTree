
#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#include <vector>
#include <cstddef>
#include <memory>
#include <random>
#include <cassert>
#include <cstdlib>
#include <new>
#include <type_traits>
#include<iostream>
template <typename T>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    static constexpr std::size_t alignment = 64;

    AlignedAllocator() noexcept = default;

    AlignedAllocator(const AlignedAllocator& other) noexcept = default;

    template <typename U>
    explicit AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

    static pointer allocate(const size_type n) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, n * sizeof(value_type)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<pointer>(ptr);
    }

    static void deallocate(pointer p, size_type) noexcept {
        free(p);
    }

    static void construct(pointer p, const value_type& value) {
        ::new (static_cast<void*>(p)) value_type(value);
    }

    static void destroy(pointer p) {
        p->~value_type();
    }

    bool operator==(const AlignedAllocator&) const noexcept {
        return true;
    }

    bool operator!=(const AlignedAllocator&) const noexcept {
        return false;
    }
};

typedef std::vector<float,AlignedAllocator<float>> Vector1D;

template<typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1) os << ", ";
    }
    os << "]";
    return os;
}


inline std::ostream &operator<<(std::ostream &os, const  Vector1D&vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1) os << ", ";
    }
    os << "]";
    return os;
}


void random_initialize(Vector1D &vec, float min,float max);
Vector1D generate_random_vector(std::int64_t dim, std::mt19937& gen,float min,float max);

 std::vector<std::int64_t> generate_random_indices( std::int64_t n);

 std::vector<std::int64_t> generate_indices( std::int64_t n);


float sum(const Vector1D& vec);
float mean(const Vector1D& vec);
float norm_l2(const Vector1D& vec);
float dist_l2(const Vector1D& vec1, const Vector1D& vec2);


Vector1D operator+(const Vector1D& vec, float scalar);
Vector1D operator+(const Vector1D& lhs, const Vector1D& rhs);
Vector1D& operator+=(Vector1D& vec, float scalar);
Vector1D& operator+=(Vector1D& vec, const Vector1D& rhs);


Vector1D operator-(const Vector1D& vec, float scalar);
Vector1D operator-(const Vector1D& lhs, const Vector1D& rhs);
Vector1D& operator-=(Vector1D& vec, float scalar);
Vector1D& operator-=(Vector1D& vec, const Vector1D& rhs);


Vector1D operator*(const Vector1D& vec, float scalar);
Vector1D operator*(const Vector1D& lhs, const Vector1D& rhs);
Vector1D& operator*=(Vector1D& vec, float scalar);
Vector1D& operator*=(Vector1D& vec, const Vector1D& rhs);


Vector1D operator/(const Vector1D& vec, float scalar);
Vector1D operator/(const Vector1D& lhs, const Vector1D& rhs);
Vector1D& operator/=(Vector1D& vec, float scalar);
Vector1D& operator/=(Vector1D& vec, const Vector1D& rhs);

#endif



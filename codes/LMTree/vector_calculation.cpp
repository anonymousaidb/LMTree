
#include <cmath>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <cassert>
#include "vector_calculation.hpp"
#include "calculation.hpp"


void random_initialize(Vector1D &vec,const float min,const float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    for (auto &x : vec) {
        x = dis(gen);
    }
}

Vector1D generate_random_vector(const std::int64_t dim, std::mt19937& gen,float min,float max) {
    Vector1D vec(dim);
    std::uniform_real_distribution dist(min, max);
    for (std::int64_t i = 0; i < dim; ++i) {
        vec[i] = dist(gen);
    }
    return vec;
}

std::vector<std::int64_t> generate_random_indices(const std::int64_t n) {
    if (n <= 0) {
        throw std::runtime_error("bad n !");
    }

    std::vector<std::int64_t> indices(n);
    for (std::int64_t i = 0; i < n; ++i) {
        indices[i] = i;
    }


    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    return indices;
}

std::vector<std::int64_t> generate_indices(const std::int64_t n) {
    if (n <= 0) {
        throw std::runtime_error("bad n !");
    }

    std::vector<std::int64_t> indices(n);
    for (std::int64_t i = 0; i < n; ++i) {
        indices[i] = i;
    }
    return indices;
}


float sum(const Vector1D& vec) {
    return calculation::sum(vec.size(),vec.data());
}

float mean(const Vector1D& vec) {
    return calculation::sum(vec.size(),vec.data())/float(vec.size());
}

bool operator==(const Vector1D& vec,const Vector1D& another) {
    return calculation::equal(vec.size(),vec.data(),another.data());
}

Vector1D operator+(const Vector1D& vec, float scalar) {
    Vector1D result(vec.size());
    calculation::add(vec.size(),vec.data(),scalar,result.data());
    return result;
}

Vector1D operator+(const Vector1D& lhs, const Vector1D& rhs) {
    Vector1D result(lhs.size());
    calculation::add(result.size(),lhs.data(),rhs.data(),result.data());
    return result;
}

Vector1D& operator+=(Vector1D& vec, float scalar) {
    calculation::add(vec.size(),vec.data(),scalar,vec.data());
    return vec;
}

Vector1D& operator+=(Vector1D& vec, const Vector1D& rhs) {
    calculation::add(vec.size(),vec.data(),rhs.data(),vec.data());
    return vec;
}

Vector1D operator-(const Vector1D& vec, float scalar) {
    Vector1D result(vec.size());
    calculation::sub(vec.size(),vec.data(),scalar,result.data());
    return result;
}

Vector1D operator-(const Vector1D& lhs, const Vector1D& rhs) {
    Vector1D result(lhs.size());
    calculation::sub(lhs.size(),lhs.data(),rhs.data(),result.data());
    return result;
}

Vector1D& operator-=(Vector1D& vec, float scalar) {
    calculation::sub(vec.size(),vec.data(),scalar,vec.data());
    return vec;
}

Vector1D& operator-=(Vector1D& vec, const Vector1D& rhs) {
    calculation::sub(vec.size(),vec.data(),rhs.data(),vec.data());
    return vec;
}

Vector1D operator*(const Vector1D& vec, float scalar) {
    Vector1D result(vec.size());
    calculation::mul(vec.size(),vec.data(),scalar,result.data());
    return result;
}

Vector1D operator*(const Vector1D& lhs, const Vector1D& rhs) {
    Vector1D result(lhs.size());
    calculation::mul(lhs.size(),lhs.data(),rhs.data(),result.data());
    return result;
}

Vector1D& operator*=(Vector1D& vec, float scalar) {
    calculation::mul(vec.size(),vec.data(),scalar,vec.data());
    return vec;
}

Vector1D& operator*=(Vector1D& vec, const Vector1D& rhs) {
    calculation::mul(vec.size(),vec.data(),rhs.data(),vec.data());
    return vec;
}

Vector1D operator/(const Vector1D& vec, float scalar) {
    Vector1D result(vec.size());
    calculation::div(vec.size(),vec.data(),scalar,result.data());
    return result;
}

Vector1D operator/(const Vector1D& lhs, const Vector1D& rhs) {
    Vector1D result(lhs.size());
    calculation::div(lhs.size(),lhs.data(),rhs.data(),result.data());
    return result;
}

Vector1D& operator/=(Vector1D& vec, float scalar) {
    calculation::div(vec.size(),vec.data(),scalar,vec.data());
    return vec;
}

Vector1D& operator/=(Vector1D& vec, const Vector1D& rhs) {
    calculation::div(vec.size(),vec.data(),rhs.data(),vec.data());
    return vec;
}

float norm_l2(const Vector1D& vec) {
    return calculation::norm_l2(vec.size(),vec.data());
}

float dist_l2(const Vector1D& vec1, const Vector1D& vec2) {
    return calculation::dist_l2(vec1.size(),vec1.data(),vec2.data());
}


















#include <cmath>
#include<iostream>
#include <random>
#include <stdexcept>
#include <algorithm>

#include "calculation.hpp"

namespace calculation {
    bool equal(const std::int64_t dim, const float *first, const float *second) {
        for (std::int64_t j = 0; j < dim; ++j) {
            if (first[j] != second[j]) {
                return false;
            }
        }
        return true;
    }

    bool gt(const std::int64_t dim, const float *first, const float *second) {
        for (std::int64_t j = 0; j < dim; ++j) {
            if (first[j] <= second[j]) {
                return false;
            }
        }
        return true;
    }

    bool ge(const std::int64_t dim, const float *first, const float *second) {
        for (std::int64_t j = 0; j < dim; ++j) {
            if (first[j] < second[j]) {
                return false;
            }
        }
        return true;
    }

    bool lt(const std::int64_t dim, const float *first, const float *second) {
        for (std::int64_t j = 0; j < dim; ++j) {
            if (first[j] >= second[j]) {
                return false;
            }
        }
        return true;
    }


    bool le(const std::int64_t dim, const float *first, const float *second) {
        for (std::int64_t j = 0; j < dim; ++j) {
            if (first[j] > second[j]) {
                return false;
            }
        }
        return true;
    }

    float sum(const std::int64_t dim, const float *first) {
        float result = 0;
        for (std::int64_t j = 0; j < dim; ++j) {
            result += first[j];
        }
        return result;
    }


    float prod(const std::int64_t dim, const float *first) {
        float result = 1;
        for (std::int64_t j = 0; j < dim; ++j) {
            result *= first[j];
        }
        return result;
    }

    float dist_l2(const std::int64_t dim, const float *first, const float *second) {
        float distance = 0;
        for (std::int64_t j = 0; j < dim; ++j) {
            const auto temp = first[j] - second[j];
            distance += temp * temp;
        }
        return std::sqrt(distance);
    }
    float dist_l2_unsquare(const std::int64_t dim, const float *first, const float *second) {
        float distance = 0;
        for (std::int64_t j = 0; j < dim; ++j) {
            const auto temp = first[j] - second[j];
            distance += temp * temp;
        }
        return distance;
    }

    float dist_l1(const std::int64_t dim, const float *first, const float *second) {
        float distance = 0;
        for (std::int64_t j = 0; j < dim; ++j) {
            distance += std::abs(first[j] - second[j]);
        }
        return distance;
    }

    float dist_linf(const std::int64_t dim, const float *first, const float *second) {
        float distance = 0;
        for (std::int64_t j = 0; j < dim; ++j) {
            const auto temp = std::abs(first[j] - second[j]);
            distance = std::max(distance, temp);
        }
        return distance;
    }

    float norm_l2(const std::int64_t dim, const float *first) {
        float distance = 0;
        for (std::int64_t j = 0; j < dim; ++j) {
            distance += first[j] * first[j];
        }
        return std::sqrt(distance);
    }

    float inner_product(const std::int64_t dim, const float *first, const float *second) {
        float result = 0;
        for (std::int64_t j = 0; j < dim; ++j) {
            result += first[j] * second[j];
        }
        return result;
    }

    float cosine_similarity(const std::int64_t dim, const float *first, const float *second) {
        float inner_product = 0;
        float distance_first = 0;
        float distance_second = 0;
        for (std::int64_t j = 0; j < dim; ++j) {
            inner_product += first[j] * second[j];
            distance_first += first[j] * first[j];
            distance_second += second[j] * second[j];
        }
        if (distance_first == 0 || distance_second == 0) {
            return 1;
            throw std::domain_error("zero vector !");
        }
        distance_first = std::sqrt(distance_first);
        distance_second = std::sqrt(distance_second);
        return inner_product / (distance_first * distance_second);
    }

    void add(const std::int64_t dim, const float *first, const float *second, float *result) {
        for (std::int64_t j = 0; j < dim; ++j) {
            result[j] = first[j] + second[j];
        }
    }
    void add(const std::int64_t dim, const float *first, const float second, float *result) {
        for (std::int64_t j = 0; j < dim; ++j) {
            result[j] = first[j] + second;
        }
    }

    void sub(const std::int64_t dim, const float *first, const float *second, float *result) {
        for (std::int64_t j = 0; j < dim; ++j) {
            result[j] = first[j] - second[j];
        }
    }


    void sub(const std::int64_t dim, const float *first, const float second, float *result) {
        for (std::int64_t j = 0; j < dim; ++j) {
            result[j] = first[j] - second;
        }
    }

    void mul(const std::int64_t dim, const float *first, const float second, float *result) {
        for (std::int64_t j = 0; j < dim; ++j) {
            result[j] = first[j] * second;
        }
    }

    void mul(const std::int64_t dim, const float *first, const float *second, float *result) {
        for (std::int64_t j = 0; j < dim; ++j) {
            result[j] = first[j] * second[j];
        }
    }


    void div(const std::int64_t dim, const float *first, float scalar, float *result) {
        scalar = 1 / scalar;
        for (std::int64_t j = 0; j < dim; ++j) {
            result[j] = first[j] * scalar;
        }
    }


    void div(const std::int64_t dim, const float *first, const float *second, float *result) {
        for (std::int64_t j = 0; j < dim; ++j) {
            result[j] = first[j] / second[j];
        }
    }

    void sqrt(const std::int64_t dim, const float *first, float *result) {
        for (std::int64_t j = 0; j < dim; ++j) {
            result[j] = std::sqrt(first[j]);
        }
    }


    void get_min_max(const std::int64_t dim, const float **begin, const float **end, float *min, float *max) {
        for (std::int64_t i = 0; i < dim; ++i) {
            min[i] = (*begin)[i];
            max[i] = (*begin)[i];
        }
        for (auto it = begin; it < end; ++it) {
            for (std::int64_t j = 0; j < dim; ++j) {
                min[j] = std::min<float>(min[j], (*it)[j]);
                max[j] = std::max<float>(max[j], (*it)[j]);
            }
        }
    }
}

namespace calculation{

    void normalization(const std::int64_t dim, float *first, const std::int64_t max_iteration) {
        for (std::int64_t i = 0; i < max_iteration; ++i) {
            div(dim,first,norm_l2(dim,first),first);
        }
    }

    float calculate_ellipsoid_volume(const std::int64_t dim, const float *full_range) {
        float product_of_radii = 1.0;
        for (std::int64_t i = 0; i < dim; ++i) {
            product_of_radii *= full_range[i];
        }
        return static_cast<float>(std::pow(M_PI, static_cast<float>(dim) / static_cast<float>(2.0)) / std::tgamma(
                static_cast<float>(dim) / static_cast<float>(2.0) + 1.0) *
                                  product_of_radii);
    }

    float calculate_equivalent_radius(const std::int64_t dim, const float volume) {
        return static_cast<float>(std::pow(
                volume * std::tgamma(static_cast<float>(dim) / static_cast<float>(2.0) + 1.0) / std::pow(
                        M_PI, static_cast<float>(dim) / static_cast<float>(2.0)),
                static_cast<float>(1.0) / static_cast<float>(dim)));
    }
}


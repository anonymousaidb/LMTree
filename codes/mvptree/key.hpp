#pragma once

#include <cstdint>
#include <iostream>

extern int dim, dist_func;

struct VectorKeyObject {
    std::vector<double> key;

    VectorKeyObject() = default;

    VectorKeyObject(const double otherkey[]) {
        key.resize(dim);
        for (int i = 0; i < dim; i++) key[i] = otherkey[i];
    }


    VectorKeyObject(const VectorKeyObject &other) : key(other.key) {}

    const VectorKeyObject &operator=(const VectorKeyObject &other) {
        if (this != &other) {
            key = other.key;
        }
        return *this;
    }

    VectorKeyObject(const std::vector<double> &other) : key(other) {}

    [[nodiscard]] double distance(const VectorKeyObject &other) const {
        if (dist_func == 2) {
            double sum_sq = 0;
            for (int i = 0; i < dim; i++) {
                double diff = key[i] - other.key[i];
                sum_sq += diff * diff;
            }
            return sqrt(sum_sq / dim);
        }

        if (dist_func == 1) {
            double sum_abs = 0;
            for (int i = 0; i < dim; i++) {
                sum_abs += fabs(key[i] - other.key[i]);
            }
            return sum_abs / dim;
        }

        if (dist_func == 0) {
            double max_diff = 0;
            for (int i = 0; i < dim; i++) {
                double diff = fabs(key[i] - other.key[i]);
                if (diff > max_diff) max_diff = diff;
            }
            return max_diff;
        }
    }
};

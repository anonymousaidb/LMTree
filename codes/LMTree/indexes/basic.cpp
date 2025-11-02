#include "basic.h"

#include <fstream>

const float *Dataset::operator[](const std::int64_t id) const {
    return data_ptr() + id * Cols();
}

float *Dataset::operator[](const std::int64_t id) {
    return data_ptr() + id * Cols();
}

Dataset::Dataset(const std::int64_t n, const std::int64_t dim) : Matrix(n, dim, 0) {}

void Dataset::swap(const std::int64_t i, const std::int64_t j) {
    float tmp[Cols()];
    std::copy_n((*this)[i], Cols(), tmp);
    std::copy_n((*this)[j], Cols(), (*this)[i]);
    std::copy_n(tmp, Cols(), (*this)[j]);
}

void Dataset::writeVectorToDirectory(const Dataset &result, const std::string &path) {
    std::ofstream file(path, std::ios::binary);
    std::int64_t n = result.Rows();
    std::int64_t dim = result.Cols();
    file.write(reinterpret_cast<char *>(&n), sizeof(std::int64_t));
    file.write(reinterpret_cast<char *>(&dim), sizeof(std::int64_t));

    const float *data = result.data_ptr();
    const std::int64_t total_bytes = n * dim * sizeof(float);
    const std::int64_t batch_size_bytes = 1024 * 1024 * 32;
    std::int64_t offset = 0;

    while (offset < total_bytes) {
        std::int64_t remaining = total_bytes - offset;
        std::int64_t to_write = std::min(batch_size_bytes, remaining);
        file.write(reinterpret_cast<const char *>(data) + offset, to_write);
        if (!file) {
            throw std::runtime_error("Error writing file at offset " + std::to_string(offset));
        }
        offset += to_write;
    }
}


Dataset Dataset::readVectorFromDirectory(const std::string &path, const std::int64_t max_n) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    std::int64_t n, dim;
    file.read(reinterpret_cast<char *>(&n), sizeof(std::int64_t));
    file.read(reinterpret_cast<char *>(&dim), sizeof(std::int64_t));

    if (max_n != -1 && n > max_n) {
        n = max_n;
    }

    Dataset result(n, dim);
    float* data = result.data_ptr();

    const std::int64_t total_bytes = n * dim * sizeof(float);
    constexpr std::int64_t batch_size = 1024 * 1024 * 32;
    std::int64_t offset = 0;

    while (offset < total_bytes) {
        std::int64_t remaining = total_bytes - offset;
        const std::int64_t to_read = std::min(batch_size, remaining);
        file.read(reinterpret_cast<char *>(data) + offset, to_read);
        if (!file && !file.eof()) {
            throw std::runtime_error("Error reading file at offset " + std::to_string(offset));
        }
        offset += to_read;
    }

    return result;
}



float LinearSegment::forward(const float key) const {
    return std::round(slope * key + intercept);
}

std::vector<LinearSegment> ShrinkingCone_Segmentation(const std::vector<std::pair<float, float> > &data,
                                                      const float epsilon) {
    std::vector<LinearSegment> segments;
    int n = data.size();
    if (n == 0) return segments;

    int start = 0;
    while (start < n) {
        int end = start;

        float lower_slope = -INFINITY;
        float upper_slope = INFINITY;

        const auto &start_point = data[start];

        while (end + 1 < n) {
            const auto &next_point = data[end + 1];

            float dx = next_point.first - start_point.first;
            if (dx == 0.0f) break;

            float dy = next_point.second - start_point.second;
            float lower = (dy - epsilon) / dx;
            float upper = (dy + epsilon) / dx;

            lower_slope = std::max(lower_slope, lower);
            upper_slope = std::min(upper_slope, upper);

            if (lower_slope > upper_slope) break;

            ++end;
        }

        float slope = 0.0f;
        float intercept = 0.0f;

        if (end == start) {
            slope = 0.0f;
            intercept = data[start].second;
        } else {
            slope = (lower_slope + upper_slope) / 2.0f;
            intercept = start_point.second - slope * start_point.first;
        }

        segments.push_back({
            start,
            end,
            slope,
            intercept
        });

        start = end + 1;
    }

    return segments;
}


std::tuple<float, float, float> linearFit(const std::vector<std::pair<float, float> > &data) {
    if (data.size() < 2) {
        throw std::invalid_argument("need two points");
    }

    float sum_x = 0.0f, sum_y = 0.0f;
    for (const auto &[x, y]: data) {
        sum_x += x;
        sum_y += y;
    }

    float n = static_cast<float>(data.size());
    float mean_x = sum_x / n;
    float mean_y = sum_y / n;

    float numerator = 0.0f;
    float denominator = 0.0f;

    for (const auto &[x, y]: data) {
        float dx = x - mean_x;
        float dy = y - mean_y;
        numerator += dx * dy;
        denominator += dx * dx;
    }

    if (denominator == 0.0f) {
        throw std::runtime_error("x values are same");
    }

    float slope = numerator / denominator;
    float intercept = mean_y - slope * mean_x;


    float max_error = 0.0f;
    for (const auto &[x, y]: data) {
        float predicted = slope * x + intercept;
        float error = std::abs(predicted - y);
        if (error > max_error) {
            max_error = error;
        }
    }

    return {slope, intercept, max_error};
}


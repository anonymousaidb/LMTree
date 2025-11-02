

#ifndef BASIC_H
#define BASIC_H
#include "../calculation.hpp"
#include "../vector_calculation.hpp"
#include "../Matrix.hpp"
class Dataset : public ML::Matrix<float> {
public:
    const float *operator[](std::int64_t id) const;

    float *operator[](std::int64_t id);

    void swap(std::int64_t i, std::int64_t j);

    Dataset(std::int64_t n, std::int64_t dim);

    static void writeVectorToDirectory(const Dataset &result, const std::string &path);

    static Dataset readVectorFromDirectory(const std::string &path, std::int64_t max_n);

    auto cut_dim(int dim)
    {
        ML::Matrix<float> sub = ML::Matrix<float>::h_split(0, dim);
        ML::Matrix<float>::operator=(std::move(sub));
    }
};


inline int dist_type = 2;
inline int this_dim = 384;
inline int epsilon = 5;
inline std::int64_t cost = 0;
inline std::int64_t node_cost = 0;
inline std::int64_t split_time = 0;
inline std::int64_t merge_time = 0;
inline std::int64_t split_times = 0;
inline std::int64_t merge_times = 0;
inline std::int64_t re_segment_times = 0;
inline float cost_o = 20;

template<class T>
float distance(const T &o1, const T &o2) {
    if constexpr (std::is_same_v<T, Vector1D>) {
        switch (dist_type) {
            case 0: return calculation::dist_linf(o1.size(), o1.data(), o2.data());
            case 1: return calculation::dist_l1(o1.size(), o1.data(), o2.data());
            case 2: return calculation::dist_l2(o1.size(), o1.data(), o2.data());
            default: throw std::runtime_error("Unsupported distance type");
        }
    } else if constexpr (std::is_same_v<T, float *> || std::is_same_v<T, const float *>) {
        switch (dist_type) {
            case 0: return calculation::dist_linf(this_dim, o1, o2);
            case 1: return calculation::dist_l1(this_dim, o1, o2);
            case 2: return calculation::dist_l2(this_dim, o1, o2);
            default: throw std::runtime_error("Unsupported distance type");
        }
    }
    throw std::runtime_error("Distance function not implemented for this type");
}



struct LinearSegment {
    std::int64_t begin_idx;
    std::int64_t end_idx;
    float slope;
    float intercept;
    [[nodiscard]] float forward(float key) const;
};

std::vector<LinearSegment> ShrinkingCone_Segmentation(
    const std::vector<std::pair<float, float> > &data,
    float epsilon);

std::tuple<float, float, float> linearFit(const std::vector<std::pair<float, float> > &data);

#endif

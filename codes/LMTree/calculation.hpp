
#ifndef CALCULATION_H
#define CALCULATION_H

#include <cstdint>
#include <utility>


namespace calculation {
    bool equal(std::int64_t dim, const float *first, const float *second);

    bool gt(std::int64_t dim, const float *first, const float *second);

    bool ge(std::int64_t dim, const float *first, const float *second);

    bool lt(std::int64_t dim, const float *first, const float *second);

    bool le(std::int64_t dim, const float *first, const float *second);

    float sum(std::int64_t dim, const float *first);

    float prod(std::int64_t dim, const float *first);

    float dist_l2(std::int64_t dim, const float *first, const float *second);
    float dist_l2_unsquare(std::int64_t dim, const float *first, const float *second);

    float dist_l1(std::int64_t dim, const float *first, const float *second);

    float dist_linf(std::int64_t dim, const float *first, const float *second);

    float inner_product(std::int64_t dim, const float *first, const float *second);

    float norm_l2(std::int64_t dim, const float *first);

    float cosine_similarity(std::int64_t dim, const float *first, const float *second);

    void add(std::int64_t dim, const float *first, const float *second, float *result);
    void add( std::int64_t dim, const float *first,  float second,float *result);


    void sub(std::int64_t dim, const float *first, const float *second, float *result);
    void sub( std::int64_t dim, const float *first,  float second,float *result);

    void mul(std::int64_t dim, const float *first, const float *second, float *result);

    void mul(std::int64_t dim, const float *first, float scalar, float *result);

    void div(std::int64_t dim, const float *first, const float *second, float *result);

    void div(std::int64_t dim, const float *first, float scalar, float *result);

    void sqrt(std::int64_t dim, const float *first, float *result);

    void get_min_max(std::int64_t dim, const float **begin, const float **end, float *min, float *max);

    void normalization(std::int64_t dim, float *first, std::int64_t max_iteration);

    float calculate_ellipsoid_volume(std::int64_t dim, const float *full_range);

    float calculate_equivalent_radius(std::int64_t dim, float volume);
}



#endif //CALCULATION_H

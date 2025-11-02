#include <cmath>
#ifndef CONSTANTS_H
#define CONSTANTS_H
using namespace std;


size_t LOCAL_MODEL_SIZE = 256;
size_t STR_BRANCH_FACTOR = 32;
size_t HRR_BRANCH_FACTOR = 32;
size_t CUR_BRANCH_FACTOR = 32;


class Constants
{
public:

    static constexpr double_t EPSILON_ERR = 1e-9;

    static constexpr size_t DIM = 2;
    static constexpr size_t LEAF_SORT_DIM = 0;

    Constants();
};

#endif
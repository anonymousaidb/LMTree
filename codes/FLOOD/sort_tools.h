#ifndef SORT_TOOLS_H
#define SORT_TOOLS_H


#include<cstdlib>
#include<algorithm>
#include"point.h"

#define SortX 0
#define SortY 1



template<typename T>
int BinarySearch(std::vector<T> arr, T val) {
    int first = 0, last = arr.size() - 1, mid;
    while (first < last) {
        mid = (first + last) / 2;

        if (arr[mid] > val) last = mid;
        if (arr[mid] < val) first = mid + 1;
    }
    return last;

}


template<typename T>
int32_t BinarySearch(std::vector<T> arr, T val, int32_t first, int32_t last) {
    int mid;
    while (first < last) {
        mid = (first + last) / 2;
        if (arr[mid] >= val) last = mid;
        if (arr[mid] < val) first = mid + 1;
    }
    return last;

}





class SortOrderer {
    size_t i;
public:
    SortOrderer(size_t i) : i{ i } {}
    [[nodiscard]] constexpr bool operator()(const Point& a, const Point& b) const noexcept {
        return a.elements_[i] < b.elements_[i];
    }
};


class ComparatorPointPartition {
public:
    Point orig_;
    size_t d_;
    ComparatorPointPartition(const Point& point, const size_t d) :orig_(point), d_(d) { }
    bool operator()(const Point& a)
    {
        return (a.elements_[d_] < orig_.elements_[d_]);
    }
};



















#endif
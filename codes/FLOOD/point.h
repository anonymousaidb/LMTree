

#ifndef POINT_H
#define POINT_H

#include<iostream>
#include<cmath>
#include<algorithm>
#include"constants.h"


class Point {
public:
    double_t elements_[Constants::DIM];

    template <typename ... Args>
    Point(const Args& ... args) : elements_{ args... } {}

    Point() {}

    Point(const Point& other) {
        std::copy(std::begin(other.elements_), std::end(other.elements_), std::begin(elements_));
    }

    void Print() {
        for (auto i = 0; i < Constants::DIM; i++)
            std::cout << elements_[i] << " ";
        std::cout << std::endl;
    }

    bool operator==(const Point& other_pnt) const {
        bool result = true;
        for (size_t i = 0; i < Constants::DIM; i++)
            result &= (elements_[i] == other_pnt.elements_[i]);
        return result;
    }

};



#endif

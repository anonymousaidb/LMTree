#ifndef RECT_H
#define RECT_H

#include<cmath>
#include<functional>
#include<limits>
#include"point.h"


class BoundingRectangle {
public:
    Point low_;
    Point high_;

    BoundingRectangle() {
        std::fill_n(low_.elements_, Constants::DIM, std::numeric_limits<double_t>::max());
        std::fill_n(high_.elements_, Constants::DIM, std::numeric_limits<double_t>::min());
    }

    BoundingRectangle(const Point& x, const Point& y) : low_(x), high_(y) { }

    /* Returns true if there is overlap.*/
    bool IsThereOverlap(const BoundingRectangle other_mbr) {
        bool result = true;
        for (size_t i = 0; i < Constants::DIM; i++)
            result &= (std::max(low_.elements_[i], other_mbr.low_.elements_[i]) < std::min(high_.elements_[i], other_mbr.high_.elements_[i]));
        return result;
    }


    /* Returns true if the passed box is completely within.*/
    bool IsCompletelyCovering(const BoundingRectangle other_mbr) {
        bool result = true;
        for (size_t i = 0; i < Constants::DIM; i++)
            result &= (low_.elements_[i] <= other_mbr.low_.elements_[i]) && (high_.elements_[i] > other_mbr.high_.elements_[i]);
        return result;
    }

    bool CheckPointWithin(const Point& point) {

        bool result = true;
        for (size_t i = 0; i < Constants::DIM; i++)
            result &= (point.elements_[i] >= low_.elements_[i] && point.elements_[i] < high_.elements_[i]);

        return result;
    }

    void SetToSpanWholeSpace() {
        for (size_t i = 0; i < Constants::DIM; ++i) {
            low_.elements_[i] = -std::numeric_limits<double_t>::infinity();
            high_.elements_[i] = std::numeric_limits<double_t>::infinity();
        }
    }

    bool operator==(const BoundingRectangle& other_mbr) {

        bool result = true;
        for (size_t i = 0; i < Constants::DIM; i++)
            result &= (low_.elements_[i] == other_mbr.low_.elements_[i] && high_.elements_[i] == other_mbr.high_.elements_[i]);

        return result;
    }


    double_t Area() {
        double_t result = 1;
        for (size_t i = 0; i < Constants::DIM; i++)
            result *= (high_.elements_[i] - low_.elements_[i]);
        return result;
    }

    /* Calculates the ration of overlap between two mbrs*/
    double_t RatioOfOverlap(const BoundingRectangle& other_mbr) {
        double_t area_of_node = Area();
        if (area_of_node <= 0) return 0.0;
        double_t area_of_overlap = 1.0;
        for (int i = 0; i < Constants::DIM; i++) {
            double_t len = std::min(high_.elements_[i], other_mbr.high_.elements_[i])
                - std::max(low_.elements_[i], other_mbr.low_.elements_[i]);
            if (len <= 0) return 0.0;
            area_of_overlap *= len;
        }
        return std::max<double_t>(0.0, area_of_overlap / area_of_node);
    }


    void UpdateBoundingBoxWithPoint(Point& pnt) {
        for (size_t i = 0; i < Constants::DIM; i++) {
            low_.elements_[i] = std::min(low_.elements_[i], pnt.elements_[i]);
            high_.elements_[i] = std::max(high_.elements_[i], pnt.elements_[i] + Constants::EPSILON_ERR);
        }
    }

    void UpdateBoundingBoxWithBoundingBox(BoundingRectangle& other_mbr) {
        for (size_t i = 0; i < Constants::DIM; i++) {
            low_.elements_[i] = std::min(low_.elements_[i], other_mbr.low_.elements_[i]);
            high_.elements_[i] = std::max(high_.elements_[i], other_mbr.high_.elements_[i]);
        }
    }


    void Print() {
        std::cout << "(";
        for (auto i = 0; i < Constants::DIM; i++)
            std::cout << low_.elements_[i] << ", ";

        std::cout << ") x (";
        for (auto i = 0; i < Constants::DIM; i++)
            std::cout << high_.elements_[i] << ", ";
        std::cout << ") " << std::endl;
    }

};

#endif
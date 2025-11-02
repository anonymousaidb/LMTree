

#ifndef QUERY_H
#define QUERY_H

#include<cmath>
#include<functional>
#include<limits.h>
#include<math.h>
#include"bounding_rectangle.h"
#include"head.h"

class Query : public BoundingRectangle {
public:
    bool   dim_used_[Constants::DIM]{};
    Point  center_;
    double radius_ = queryRadius;

    Query() {
        for (size_t i = 0; i < Constants::DIM; ++i) dim_used_[i] = true;
    }

    Query(const Point& center, double r) : center_(center), radius_(r) {
        for (size_t i = 0; i < Constants::DIM; ++i) dim_used_[i] = true;
        Point low, high;
        for (size_t i = 0; i < Constants::DIM; ++i) {
            low.elements_[i] = center.elements_[i] - r;
            high.elements_[i] = center.elements_[i] + r + Constants::EPSILON_ERR;
        }
        low_ = low; high_ = high;
    }

    Query(const Point& low, const Point& high) : BoundingRectangle(low, high) {
        for (size_t i = 0; i < Constants::DIM; ++i) dim_used_[i] = true;
    }

    bool CheckPointWithinCircle(const Point& p) const {
        double dist2 = 0.0, r2 = radius_ * radius_;
        for (size_t i = 0; i < Constants::DIM; ++i) {
            double d = p.elements_[i] - center_.elements_[i];
            dist2 += d * d;
            if (dist2 > r2) return false;
        }
        return true;
    }

    int NumDim() const {
        int result = 0;
        for (size_t i = 0; i < Constants::DIM; i++) if (dim_used_[i]) result++;
        return result;
    }
};


#endif
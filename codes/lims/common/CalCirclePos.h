#ifndef CalCirclePos_H
#define CalCirclePos_H

#include <iostream>
#include "../entities/Point.h"
#include "../common/config.h"

#include <cmath>

extern long disOpts;

using namespace std;


class CalCirclePos {
private:
    unsigned dim;
public:
    double dis_upper;
    double dis_lower;
    unsigned label;

    ~CalCirclePos();

    CalCirclePos(Point &, double, Point &, double);

    double CalDisOfPt(Point &, Point &, int);
};

CalCirclePos::~CalCirclePos() {}

CalCirclePos::CalCirclePos(Point &refPt, double radius_refPt, Point &queryPt, double radius_queryPt) {
    this->dim = refPt.coordinate.size();
    double distance = CalDisOfPt(refPt, queryPt, disType);
    if (distance >= radius_refPt + radius_queryPt) {
        this->label = 1;
        this->dis_lower = 0x3f3f3f;
        this->dis_upper = 0x3f3f3f;
    } else if (distance >= fabs(radius_refPt - radius_queryPt)) {
        this->label = 2;
        if (distance > radius_queryPt) {
            this->dis_upper = radius_refPt;
            this->dis_lower = distance - radius_queryPt;
        } else {
            this->dis_lower = 0.0;
            this->dis_upper = radius_refPt;
        }
    } else {
        this->label = 3;
        if (distance > radius_queryPt) {
            this->dis_lower = distance - radius_queryPt;
            this->dis_upper = distance + radius_queryPt;
        } else {
            this->dis_upper = distance + radius_queryPt;
            this->dis_lower = 0.0;
        }
    }
}


double CalCirclePos::CalDisOfPt(Point &point_a, Point &point_b, int distance_type) {
    disOpts++;
    double total = 0.0;
    switch (distance_type) {
        case 2:
            for (unsigned i = 0; i < dim; i++) {
                total += std::pow(point_a.coordinate[i] - point_b.coordinate[i], 2);
            }
            return std::sqrt(total);
        case 1:
            for (unsigned i = 0; i < dim; i++) {
                total += std::abs(point_a.coordinate[i] - point_b.coordinate[i]);
            }
            return total;
        case 0:
            for (unsigned i = 0; i < dim; i++) {
                double diff = std::abs(point_a.coordinate[i] - point_b.coordinate[i]);
                if (diff > total) {
                    total = diff;
                }
            }
            return total;
        default:
            std::cerr << "not support: " << distance_type << std::endl;
            return -1.0;
    }
}

#endif
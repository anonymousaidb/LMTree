#ifndef Point_H
#define Point_H

#include <vector>
#include <string>

using namespace std;


class Point {
public:
    vector<double> coordinate;


    unsigned id;

    unsigned long long i_value;

    Point();

    Point(vector<double> &);

    Point(vector<double> &, unsigned);

    void setIValue(unsigned long long);

    size_t memoryUsage() const {
        size_t total = sizeof(*this);
        total += coordinate.capacity() * sizeof(double);
        return total;
    }
};

class InsertPt {
public:
    vector<double> coordinate;

    unsigned id;

    double i_value;

    InsertPt();

    InsertPt(vector<double> &, unsigned);

    void setIValue(double);
};


class Clu_Point {
public:
    vector<Point> clu_point;

    Clu_Point();

    Clu_Point(vector<Point> &);

    size_t memoryUsage() const {
        size_t total = sizeof(*this);
        for (const auto &pt: clu_point) {
            total += pt.memoryUsage();
        }
        total += clu_point.capacity() * sizeof(Point);
        return total;
    }
};

class All_Point {
public:
    vector<Clu_Point> all_point;

    All_Point();

    All_Point(vector<Clu_Point> &);
};

#endif
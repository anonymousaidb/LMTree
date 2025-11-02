

#ifndef Reference_H
#define Reference_H

#include "Point.h"

using namespace std;


class Ref_Point {
public:
    Point point;
    double r;
    double r_low;

    vector<double> dis;

    vector<vector<double>> dict_circle;


    vector<double> coeffs;

    Ref_Point(Point &, double, double);

    Ref_Point();

    void setDisArr(vector<double> &);

    void setDict(vector<vector<double>> &);

    void setCoeffs(vector<double> &);

    size_t memoryUsage() const {
        size_t total = sizeof(*this);
        total += point.memoryUsage();
        total += dis.capacity() * sizeof(double);
        total += coeffs.capacity() * sizeof(double);

        total += sizeof(std::vector<std::vector<double>>) +
                 dict_circle.capacity() * sizeof(std::vector<double>);
        for (const auto &vec: dict_circle) {
            total += sizeof(std::vector<double>) +
                     vec.capacity() * sizeof(double);
        }

        return total;
    }
};


class mainRef_Point {
public:
    Point point;
    double r;
    double r_low;

    vector<Ref_Point> ref_points;
    vector<double> dis;


    vector<Point> iValuePts;


    vector<double> coeffs;


    vector<InsertPt> insert_list;

    double a;
    double b;
    int err_min;
    int err_max;


    vector<vector<double>> dict_circle;

    mainRef_Point(Point &, double, double, vector<Ref_Point> &);

    mainRef_Point();

    void setMainRefDisArr(vector<double> &);

    void setIValuePts(vector<Point> &);

    void setLinear(double, double, int, int);

    void setDict(vector<vector<double>> &);

    void setCoeffs(vector<double> &);

    void setInsertPt(vector<InsertPt> &);

    size_t memoryUsage() const {
        size_t total = sizeof(*this);

        total += point.memoryUsage();

        total += sizeof(std::vector<Ref_Point>) +
                 ref_points.capacity() * sizeof(Ref_Point);
        for (const auto &ref: ref_points) {
            total += ref.memoryUsage();
        }

        total += sizeof(std::vector<double>) +
                 dis.capacity() * sizeof(double);

        total += sizeof(std::vector<Point>) +
                 iValuePts.capacity() * sizeof(Point);
        for (const auto &pt: iValuePts) {
            total += pt.memoryUsage();
        }

        total += sizeof(std::vector<double>) +
                 coeffs.capacity() * sizeof(double);

        total += sizeof(std::vector<InsertPt>) +
                 insert_list.capacity() * sizeof(InsertPt);
        for (const auto &insertPt: insert_list) {
            total += insertPt.coordinate.capacity() * sizeof(double);
        }

        total += sizeof(std::vector<std::vector<double>>) +
                 dict_circle.capacity() * sizeof(std::vector<double>);
        for (const auto &vec: dict_circle) {
            total += sizeof(std::vector<double>) +
                     vec.capacity() * sizeof(double);
        }

        return total;
    }
};


class Ref_Set {
public:
    vector<mainRef_Point> ref_set;

    Ref_Set(vector<mainRef_Point> &);

    Ref_Set();
};

#endif
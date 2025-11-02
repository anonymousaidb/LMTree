
#ifndef ChooseRef_H
#define ChooseRef_H

#include "../entities/Reference.h"
#include "../entities/Point.h"
#include "../model/PolynomialRegression.h"
#include "../common/Constants.h"
#include "../common/config.h"

#include <climits>
#include <cmath>
#include <algorithm>
#include <iostream>

extern long disOpts;

using namespace std;

class ChooseRef {
private:

    unsigned num_ref;

    unsigned dim;

    int num_data;

    Point mainRefPoint;
    vector<Ref_Point> RefPoint_Set;


    vector<double> mainRefDisArr;
public:
    mainRef_Point main_pointSet;
    vector<double> Pos;

    ChooseRef();

    ChooseRef(unsigned, Clu_Point &, Point &, vector<Point> &, int);



    vector<Ref_Point> ChooseRefPoint(Clu_Point &);

    vector<Ref_Point> ChooseRefPoint_Input(Clu_Point &, vector<Point> &);

    double CaculateR(Clu_Point &, Point &);

    double CaculateEuclideanDis(Point &, Point &, int);

    vector<double> CaculateDisArr(Clu_Point &, Point &);

    vector<vector<double>> CaculateCircleBound(vector<double> &);

    mainRef_Point getMainRefPoint();

    vector<double> TrainModel(vector<double> &, vector<double> &);

    double CaculR2(vector<double> &, vector<double> &);
};

ChooseRef::ChooseRef() {
}

ChooseRef::ChooseRef(unsigned num_ref, Clu_Point &cluster, Point &main_pivot, vector<Point> &oth_pivot, int type) {
    this->num_ref = num_ref;
    this->dim = cluster.clu_point[0].coordinate.size();
    this->num_data = cluster.clu_point.size();

    for (int i = 0; i < num_data; i++)
        Pos.push_back(i);
    this->mainRefPoint = main_pivot;

    if (type == 0)
        this->RefPoint_Set = ChooseRefPoint(cluster);
    else
        this->RefPoint_Set = ChooseRefPoint_Input(cluster, oth_pivot);
    this->mainRefDisArr = CaculateDisArr(cluster, mainRefPoint);
    this->main_pointSet = mainRef_Point(mainRefPoint, mainRefDisArr[num_data - 1], mainRefDisArr[0], RefPoint_Set);
    this->main_pointSet.setMainRefDisArr(mainRefDisArr);
    vector<double> mainRefPt_coeffs = TrainModel(main_pointSet.dis, Pos);
    this->main_pointSet.setCoeffs(mainRefPt_coeffs);
}


vector<Ref_Point> ChooseRef::ChooseRefPoint(Clu_Point &cluster) {
    vector<Ref_Point> ref_points;
    for (unsigned i = 0; i < num_ref; i++) {
        double max = cluster.clu_point[0].coordinate[i];
        Point assRef_point = cluster.clu_point[0];
        for (int j = 1; j < num_data; j++) {
            if (cluster.clu_point[j].coordinate[i] > max) {
                max = cluster.clu_point[j].coordinate[i];
                assRef_point = cluster.clu_point[j];
            }
        }


        vector<double> dis = CaculateDisArr(cluster, assRef_point);
        Ref_Point ref_point = Ref_Point(assRef_point, dis[num_data - 1], dis[0]);

        ref_point.setDisArr(dis);

        vector<double> coeffs = TrainModel(ref_point.dis, Pos);
        ref_point.setCoeffs(coeffs);

        ref_points.push_back(ref_point);
    }
    return ref_points;
}


vector<Ref_Point> ChooseRef::ChooseRefPoint_Input(Clu_Point &cluster, vector<Point> &oth_pivot) {
    vector<Ref_Point> ref_points;
    for (unsigned i = 0; i < num_ref; i++) {

        Point assRef_point = oth_pivot[i];


        vector<double> dis = CaculateDisArr(cluster, assRef_point);
        Ref_Point ref_point = Ref_Point(assRef_point, dis[num_data - 1], dis[0]);

        ref_point.setDisArr(dis);

        vector<double> coeffs = TrainModel(ref_point.dis, Pos);
        ref_point.setCoeffs(coeffs);

        ref_points.push_back(ref_point);
    }
    return ref_points;
}

vector<double> ChooseRef::CaculateDisArr(Clu_Point &cluster, Point &refPoint) {
    vector<double> dis;
    for (int i = 0; i < num_data; i++) {
        double distance = CaculateEuclideanDis(refPoint, cluster.clu_point[i], disType);
        dis.push_back(distance);
    }

    sort(dis.begin(), dis.end());
    return dis;
}

vector<vector<double>> ChooseRef::CaculateCircleBound(vector<double> &dis) {
    sort(dis.begin(), dis.end());
    int circle_num = 30;
    int split = dis.size() / circle_num + 1;
    vector<vector<double>> dict_circle;
    for (int i = 0; i < circle_num; i++) {
        vector<double> circle_bound;
        circle_bound.push_back(dis[i * split]);
        if (i < circle_num - 1) {
            circle_bound.push_back(dis[(i + 1) * split - 1]);
        } else {
            circle_bound.push_back(dis[dis.size() - 1]);
        }
        dict_circle.push_back(circle_bound);
    }
    return dict_circle;
}

double ChooseRef::CaculateR(Clu_Point &cluster, Point &point) {
    double r = 0.0;

    for (int i = 0; i < num_data; i++) {
        double dis = CaculateEuclideanDis(point, cluster.clu_point[i], disType);
        r = r > dis ? r : dis;
    }
    return r;
}

double ChooseRef::CaculateEuclideanDis(Point &point_a, Point &point_b, int distanceType) {
    disOpts++;
    double total = 0.0;

    switch (distanceType) {
        case 2:
            for (unsigned i = 0; i < dim; ++i) {
                total += std::pow(point_a.coordinate[i] - point_b.coordinate[i], 2);
            }
            return std::sqrt(total);
        case 1:
            for (unsigned i = 0; i < dim; ++i) {
                total += std::abs(point_a.coordinate[i] - point_b.coordinate[i]);
            }
            return total;
        case 0:
            for (unsigned i = 0; i < dim; ++i) {
                double diff = std::abs(point_a.coordinate[i] - point_b.coordinate[i]);
                if (diff > total) {
                    total = diff;
                }
            }
            return total;
        default:
            std::cerr << "not support: " << distanceType << std::endl;
            return -1.0;
    }
}

mainRef_Point ChooseRef::getMainRefPoint() {
    return main_pointSet;
}

vector<double> ChooseRef::TrainModel(vector<double> &X, vector<double> &Y) {
    PolynomialRegression<double> polyreg;
    vector<double> coeffs;
    polyreg.fitIt(X, Y, Constants::COEFFS, coeffs);
    return coeffs;
}

double ChooseRef::CaculR2(vector<double> &X, vector<double> &coeff) {
    double SSE = 0.0;
    double SST = 0.0;
    double avg = (0 + num_data - 1) / 2;
    for (int i = 0; i < num_data; ++i) {
        double pre = coeff[0];
        for (int j = 1; j <= Constants::COEFFS; ++j)
            pre += coeff[j] * pow(X[i], j);
        SSE += pow(pre - i, 2);
        SST += pow(i - avg, 2);
    }
    return sqrt(SSE / num_data);
}

#endif
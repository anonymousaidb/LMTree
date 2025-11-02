#include "MyFunc.h"
#include <iostream>
#include<cmath>

using namespace std;

extern int func;


double MyFunc::Cal_Euclidean_distance(const std::vector<double> &a, const std::vector<double> &b) {
    if (a.size() != b.size()) {
        std::cout << "the size don't match in Cal_Euclidean_distance" << std::endl;
    }
    double tot = 0, dif;
    if (func == 1) {
        for (size_t i = 0; i < a.size(); ++i) {
            dif = (a[i] - b[i]);
            if (dif < 0) dif = -dif;
            tot += dif;
        }
    } else if (func == 2) {
        for (size_t i = 0; i < a.size(); ++i) {
            tot += (a[i] - b[i]) * (a[i] - b[i]);
        }
        tot = std::sqrt(tot);
    } else if (func == 0) {
        double max = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            dif = (a[i] - b[i]);
            if (dif < 0) dif = -dif;
            if (dif > max) max = dif;
        }
        tot = max;
    }
    return tot;
}

double MyFunc::Cal_Hypersphere_Volume(double radius_, int dim_) {
    double a = 2 * pow(M_PI, double(dim_) / 2.0);
    double b = double(dim_) * tgamma(double(dim_) / 2.0);
    double c = pow(radius_, double(dim_));
    return (a * c) / b;

}

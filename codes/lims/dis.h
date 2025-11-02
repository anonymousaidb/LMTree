#include <string>
#include <cstring>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/time.h>

#define NUUM 10000000

using namespace std;

float *loc[NUUM];
extern long disOpts;

int pnum, num, PairA = 0;
int ptype = 2;
double mm;

double dis(int i, int j, int dim) {
    disOpts++;
    if (ptype == 2) {
        double sum = 0;
        for (int k = 0; k < dim; k++) {
            sum += pow(loc[i][k] - loc[j][k], 2);
        }
        return pow(sum, 0.5);
    }
    if (ptype == 1) {
        double sum = 0;
        for (int k = 0; k < dim; k++) sum += fabs(loc[i][k] - loc[j][k]);
        return sum;
    }

    if (ptype == 5) {
        double sum = 0;
        for (int k = 0; k < dim; k++) sum += pow(loc[i][k] - loc[j][k], 5);
        return pow(sum, 0.2);
    }

    if (ptype == 0) {
        double max = 0, p;
        for (int k = 0; k < dim; k++) {
            p = abs(loc[i][k] - loc[j][k]);
            if (p > max) max = p;
        }
        return max;
    }
}

void readm(string filename, int dim, int type) {

    ifstream ain1;

    ain1.open(filename, ifstream::in);

    num = 2000000;

    PairA = 100000;

    for (int i = 0; i < num; i++)loc[i] = new float[dim + 3];


    string line;
    int i = 0;
    if (type) {
        while (getline(ain1, line)) {
            stringstream ss(line);
            string value;
            int j = 0;
            getline(ss, value, ',');
            while (getline(ss, value, ',')) {
                loc[i][j] = atof(value.c_str());
                ++j;
            }
            ++i;
        }
    } else {
        while (getline(ain1, line)) {
            stringstream ss(line);
            string value;
            int j = 0;
            while (getline(ss, value, ',')) {
                loc[i][j] = atof(value.c_str());
                ++j;
            }
            ++i;
        }
    }
    num = i;

    ain1.close();
    int j = 1;
    mm = 0;
    double mm1 = 0;
    for (int i = 1; i < num; i++)
        if (dis(0, i, dim) > mm1) {
            j = i;
            mm1 = dis(0, i, dim);
        }
    for (int i = 1; i < num; i++) if (dis(j, i, dim) > mm) mm = dis(j, i, dim);

}

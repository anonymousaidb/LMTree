
#include "../common/Constants.h"

#include "FileIO.h"
#include <cstdlib>
#include <sstream>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>

using namespace std;

InputReader::InputReader() {}

InputReader::InputReader(string filename) {
    fin.open(filename.c_str());
    if (!fin) {
        cout << filename << " file could not be opened\n";
        exit(0);
    }
    parse();
    fin.close();
}

vector<Point> InputReader::LoadRangeQuery(string filename) {
    fin.open(filename.c_str());
    if (!fin) {
        cout << filename << " file could not be opened\n";
        exit(0);
    }
    string line;
    int i = 0;
    Point query_bound;
    while (getline(fin, line)) {
        stringstream ss(line);
        string value;
        vector<double> data;
        vector<double> coordinate;
        while (getline(ss, value, ','))
            data.push_back(atof(value.c_str()));

        Point point = Point(data);
        if (i % 2 == 0)
            query_bound = point;
        else {
            for (unsigned j = 0; j < data.size(); j++) {
                coordinate.push_back((query_bound.coordinate[j] + point.coordinate[j]) / 2);
            }
            points.push_back(Point(coordinate));
        }
        i++;
    }
    return points;
}

vector<Point> InputReader::LoadPointQuery(string filename) {
    fin.open(filename.c_str());
    if (!fin) {
        cout << filename << " file could not be opened\n";
        exit(0);
    }
    string line;
    while (getline(fin, line)) {
        stringstream ss(line);
        string value;
        vector<double> data;
        while (getline(ss, value, ','))
            data.push_back(atof(value.c_str()));

        Point point = Point(data);
        points.push_back(point);
    }
    return points;
}


void InputReader::parse() {
    string line;
    while (getline(fin, line)) {
        stringstream ss(line);
        string value;
        vector<double> coordinate;
        getline(ss, value, ',');
        unsigned id = stol(value);
        while (getline(ss, value, ',')) {
            coordinate.push_back(atof(value.c_str()));
        }
        Point point = Point(coordinate, id);
        points.push_back(point);
    }
}


Clu_Point InputReader::getCluster() {
    Clu_Point clu_point = Clu_Point(points);
    return clu_point;
}

OutputPrinter::OutputPrinter() {}

OutputPrinter::OutputPrinter(string filename, Clu_Point clu) {
    this->cluster = clu;
    if (filename.size() < 4) {
        cout << filename << " input file name's format is wrong" << endl;
        exit(0);
    }
    this->outputFileName = filename;
    print();
}

void OutputPrinter::print() {
    fout.open(outputFileName.c_str());

    for (unsigned i = 0; i < cluster.clu_point.size(); i++) {
        fout << cluster.clu_point[i].i_value << endl;
    }

    fout.close();
}

OutputPrinter::OutputPrinter(string filename, vector<Point> result) {
    this->result = result;
    if (filename.size() < 4) {
        cout << filename << " input file name's format is wrong" << endl;
        exit(0);
    }
    this->outputFileName = filename;
    print_result();
}

void OutputPrinter::print_result() {
    fout.open(outputFileName.c_str());

    for (unsigned i = 0; i < result.size(); i++) {
        fout << i << "\t" << result[i].i_value;
        fout << endl;
    }

    fout.close();
}

OutputPrinter::OutputPrinter(string filename, mainRef_Point &mainRef_point) {
    this->mainRef_point = mainRef_point;
    if (filename.size() < 4) {
        cout << filename << " input file name's format is wrong" << endl;
        exit(0);
    }
    for (int i = 0; i < 4; i++)
        filename.pop_back();
    this->outputFileName = filename;
    print_refCircle();
}

void OutputPrinter::print_refCircle() {
    fout.open(outputFileName + "_cluster12_ref0.txt");
    int len = mainRef_point.dis.size();
    sort(mainRef_point.dis.begin(), mainRef_point.dis.end());

    for (int i = 0; i < len; i++) {
        for (unsigned j = 0; j < mainRef_point.dict_circle.size(); j++) {
            if (mainRef_point.dis[i] >= mainRef_point.dict_circle[j][0] &&
                mainRef_point.dis[i] <= mainRef_point.dict_circle[j][1]) {
                fout << mainRef_point.dis[i] << "\t" << to_string(j) << endl;
                break;
            }
        }
    }
    fout.close();
    for (unsigned x = 0; x < mainRef_point.ref_points.size(); x++) {
        fout.open(outputFileName + "_cluster12_ref" + to_string(x + 1) + ".txt");

        sort(mainRef_point.ref_points[x].dis.begin(), mainRef_point.ref_points[x].dis.end());

        for (int i = 0; i < len; i++) {
            for (unsigned j = 0; j < mainRef_point.ref_points[x].dict_circle.size(); j++) {
                if (mainRef_point.ref_points[x].dis[i] >= mainRef_point.ref_points[x].dict_circle[j][0] &&
                    mainRef_point.ref_points[x].dis[i] <= mainRef_point.ref_points[x].dict_circle[j][1]) {
                    fout << mainRef_point.ref_points[x].dis[i] << "\t" << to_string(j) << endl;
                    break;
                }
            }
        }

        fout.close();
    }
}

OutputPrinter::OutputPrinter(string filename, vector<double> &res) {
    if (filename.size() < 4) {
        cout << filename << " input file name's format is wrong" << endl;
        exit(0);
    }
    fout.open(filename.c_str());

    for (unsigned i = 0; i < res.size(); i++) {
        fout << res[i] << endl;
    }

    fout.close();
}

void OutputPrinter::print_errResult(string filename, vector<double> &err_res) {
    if (filename.size() < 4) {
        cout << filename << " input file name's format is wrong" << endl;
        exit(0);
    }
    fout.open(filename.c_str());

    for (unsigned i = 0; i < err_res.size(); i++) {
        fout << err_res[i] << endl;
    }

    fout.close();
}

void OutputPrinter::print_ivalue(string filename, vector<Point> &pts) {
    if (filename.size() < 4) {
        cout << filename << " input file name's format is wrong" << endl;
        exit(0);
    }
    fout.open(filename.c_str());

    for (unsigned i = 0; i < pts.size(); i++) {
        fout << pts[i].i_value << endl;
    }

    fout.close();
}









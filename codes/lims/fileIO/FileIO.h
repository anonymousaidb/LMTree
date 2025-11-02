

#ifndef FILEIO_H
#define FILEIO_H

#include <iostream>
#include <fstream>
#include <string>
#include "../entities/Point.h"
#include "../entities/Reference.h"

using namespace std;


class InputReader {
public:
    ifstream fin;
    vector<Point> points;

    InputReader();

    InputReader(string);

    vector<Point> LoadRangeQuery(string);

    vector<Point> LoadPointQuery(string);

    void parse();

    Clu_Point getCluster();
};

class OutputPrinter {
public:
    ofstream fout;
    Clu_Point cluster;
    vector<Point> result;
    string outputFileName;
    mainRef_Point mainRef_point;


    OutputPrinter();

    OutputPrinter(string, Clu_Point);

    OutputPrinter(string, vector<Point>);

    OutputPrinter(string, mainRef_Point &);
    OutputPrinter(string, vector<double> &);


    OutputPrinter(int, mainRef_Point &);

    void print();
    void print_result();
    void print_refCircle();
    void print_errResult(string, vector<double> &);
    void print_ivalue(string, vector<Point> &);
};

#endif
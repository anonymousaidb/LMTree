#include "DataMetric.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>


extern int dim;
extern int objNum;
extern int func;

void DataMetric::load_Raw_Data(const string &data_file_) {
    std::ifstream file(data_file_);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << data_file_ << std::endl;
        return;
    }


    file >> dim >> objNum >> func;

    std::string line;
    std::getline(file, line);

    this->dataset.clear();
    this->dataset.resize(objNum);

    int index = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string value;
        std::vector<double> point;

        while (std::getline(iss, value, ' ')) {
            if (!value.empty()) {
                point.push_back(std::stod(value));
            }
        }

        if (index < objNum) {
            this->dataset[index] = point;
            index++;
        }
    }

    file.close();
    this->dimension = dim;
}

void DataMetric::load_Query_Data(const string &data_file_, int dimension_) {
    this->dimension = dimension_;
    std::ifstream file(data_file_);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << data_file_ << std::endl;
        return;
    }

    std::string line;
    this->dataset.clear();

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string value;
        std::vector<double> point;

        while (std::getline(iss, value, ' ')) {
            if (!value.empty()) {
                point.push_back(std::stod(value));
            }
        }

        if (point.size() == dimension_) {
            this->dataset.push_back(point);
        }
    }

    file.close();
}
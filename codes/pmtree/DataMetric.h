#pragma once

#include<string>
#include<iostream>
#include<vector>
#include<fstream>
#include<cassert>
#include <iomanip>

using namespace std;

class DataMetric {
public:
    DataMetric() = default;

    ~DataMetric() = default;

    void load_Raw_Data(const string &data_file_);

    void load_Query_Data(const string &data_file_, int dimension_);

    [[nodiscard]] int size() const { return dataset.size(); }

    const vector<double> &operator[](int i) const {
        return dataset[i];
    }

    vector<double> &operator[](int i) {
        return dataset[i];
    }

private:
    vector<vector<double>> dataset;
    int dimension{};
};







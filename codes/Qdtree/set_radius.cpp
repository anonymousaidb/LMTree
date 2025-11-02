
#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <sstream>
#include <fstream>
#include <limits>

using namespace std;
int dim = 100;
int objNum = 100000;
int querynum = 1000;
int metric_function = 2;


double euclideanDistance(const vector<double>& a, const vector<double>& b, size_t top_mdim) {
    double sum = 0.0;
    for (size_t i = 0; i < top_mdim; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}


double kthNearestNeighborDistance(const vector<vector<double>>& dataset,
                                  const vector<double>& query,
                                  size_t k, size_t top_mdim) {
    priority_queue<double> maxHeap;

    for (const auto& point : dataset) {
        double dist = euclideanDistance(point, query, top_mdim);
        if (maxHeap.size() < k) {
            maxHeap.push(dist);
        } else if (dist < maxHeap.top()) {
            maxHeap.pop();
            maxHeap.push(dist);
        }
    }

    return maxHeap.top();
}



void read_data_from_file(int type, const string &filename, vector<vector<double>> &entries) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }

    if (type == 0) {
        file >> dim >> objNum >> metric_function;
        string line;
        getline(file, line);
    }
    else if (type == 1) {
        file >> objNum;
        string line;
        getline(file, line);
    }

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::vector<double> data;
        double value;
        while (iss >> value) {
            data.push_back(value);
        }

        if (!data.empty()) {
            entries.emplace_back(vector<double>(data));
        }
    }

    file.close();
}



int main() {

    // 读取数据集
    string dataFileName = "D:\\clionprojects\\untitled\\datasets\\blob100D_data.txt";
    ofstream outFile("results.txt");
    vector<vector<double>> entries;
    read_data_from_file(0,dataFileName,entries);
    cout << "read data over: num is:" << entries.size() << endl;



    // 读取查询集
    string queryFileName = "D:\\clionprojects\\untitled\\datasets\\blob100D_queries.txt";
    vector<vector<double>> queryset;
    read_data_from_file(1,queryFileName,queryset);
    cout << "read query over: num is:" << queryset.size() << endl;

    size_t k = 5600;
    size_t top_mdim = 2;
    double sum_distance = 0;
    for (auto q : queryset) {
        double kthDist = kthNearestNeighborDistance(entries, q, k, top_mdim);
        sum_distance += kthDist;
        // cout << k << " radius is: " << kthDist << endl;
    }


    cout << " average radius: " << sum_distance / 1000.0 << endl;

    return 0;
}

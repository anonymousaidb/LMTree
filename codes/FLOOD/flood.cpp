#include<vector>
#include<iostream>
#include<fstream>
#include <chrono>
#include<string>
#include<algorithm>


#include"floodlite.h"


#include "sort_tools.h"
#include"query.h"

using namespace std;
#define addeb if(0)



const string data_path = "./datasets";
int main(int argc, char* argv[]) {
    vector<Point> datapoints;
    ifstream pointsfile(data_path + "/LA_example.txt");
    while (true) {
        Point p;
        bool success = true;
        for (int i = 0; i < Constants::DIM; i++) {
            if (!(pointsfile >> p.elements_[i])) {
                success = false;
                break;
            }
        }
        if (!success) break;
        datapoints.push_back(p);
    }
    pointsfile.close();

    vector<Query> queries;
    ifstream queriesfile(data_path + "/LA_query.txt");
    while (true) {
        Point p;
        bool success = true;
        for (int i = 0; i < Constants::DIM; i++) {
            if (!(queriesfile >> p.elements_[i])) {
                success = false;
                break;
            }
        }
        if (!success) break;

        Query q(p, queryRadius);
        queries.push_back(q);
    }
    queriesfile.close();
    random_shuffle(queries.begin(), queries.end());
    vector<Point> result_vec;
    uint64_t projection_time, scan_time;




    FloodLite flood_obj(datapoints, queries, MetricType::L2);


    for (auto& query : queries) {
        std::vector<size_t> projected_cells;
        std::vector<Point> result_vec;
        flood_obj.Projection(projected_cells, query);
        flood_obj.Scan(projected_cells, query, result_vec);
    }


    std::vector<size_t> ks = {20, 50, 100, 150, 200};
    for (size_t k : ks) {
        std::vector<std::vector<Point>> knn_results;
        knn_results.resize(queries.size());
        for (size_t i = 0; i < queries.size(); ++i) {
            flood_obj.KNN(queries[i].center_, k, knn_results[i]);
        }
    }


    flood_obj.Insert(datapoints[0]);
    flood_obj.Erase(datapoints[0]);


    return 0;
}
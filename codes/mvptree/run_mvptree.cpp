#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <ratio>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <map>
#include <mvptree/mvptree.hpp>
#include <mvptree/datapoint.hpp>
#include <mvptree/key.hpp>

using namespace std;
using namespace mvp;

static long long m_id = 1;

double rak = 0;

const int BF = 2;
const int PL = 8;
const int LC = 100;
const int LPN = 4;
const int FO = 16;
const int NS = 8;

int dim = 0, n_objects, dist_func;


vector<datapoint_t<VectorKeyObject, PL>> load_data(const string &filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    file >> dim >> n_objects >> dist_func;

    string line;
    vector<datapoint_t<VectorKeyObject, PL>> points;
    points.reserve(n_objects);

    getline(file, line);

    while (getline(file, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        string value;
        vector<double> data;
        while (getline(iss, value, ' ')) {
            if (!value.empty()) {
                data.push_back(stod(value));
            }
        }
        if (data.size() != static_cast<size_t>(dim)) {
            continue;
        }
        points.emplace_back(m_id++, VectorKeyObject(data));
    }
    file.close();
    return points;
}


vector<VectorKeyObject> load_queries(const string &filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    string line;
    vector<VectorKeyObject> queries;
    while (getline(file, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        string value;
        vector<double> data;
        while (getline(iss, value, ' ')) {
            if (!value.empty()) {
                data.push_back(stod(value));
            }
        }

        queries.emplace_back(data);
    }

    file.close();
    return queries;
}


map<string, vector<double>> get_dataset_radii_map() {
    return {
            {"LA", {473,   692,   989,   1409,  1875,  2314,  3096}},
            {"SY", {2321,  2733,  3229,  3843,  4614,  5613,  7090}},
            {"CO", {0.328, 0.485, 0.558, 0.622, 0.768, 0.812, 0.962}}
    };
}

vector<double> get_radii_for_dataset(const string &dataset_name) {
    static const auto radii_map = get_dataset_radii_map();


    string upper_name = dataset_name;
    transform(upper_name.begin(), upper_name.end(), upper_name.begin(), ::toupper);

    if (upper_name.find("LA")) {
        rak = 50;
    }

    if (upper_name.find("SY")) {
        rak = 100;
    }

    if (upper_name.find("CO")) {
        rak = 0.005;
    }

    for (const auto &[key, radii]: radii_map) {
        if (upper_name.find(key) != string::npos) {
            return radii;
        }
    }
    return {0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0};
}


void do_experiment(const int n_runs, ofstream &outFile, vector<datapoint_t<VectorKeyObject, PL>> &dataset,
                   vector<VectorKeyObject> &queries, const vector<double> &radii, size_t initial_size,
                   int num_batches, const string &update_file, size_t update_batch_size) {


    for (int run = 0; run < n_runs; run++) {

        MVPTree<VectorKeyObject, BF, PL, LC, LPN, FO, NS> tree;
        const size_t batch_size = 1000;
        for (size_t i = 0; i < dataset.size(); i += batch_size) {
            vector<datapoint_t<VectorKeyObject, PL>> batch;
            size_t end = min(i + batch_size, dataset.size());

            for (size_t j = i; j < end; j++) {
                batch.push_back(dataset[j]);
            }
            tree.Add(batch);
        }

        tree.Sync();



        for (double radius: radii) {
            for (const auto &query: queries) {
                vector<item_t<VectorKeyObject>> results = tree.Query(query, radius);
            }
        }


        vector<int> k_values = {20, 50, 100, 150, 200};
        for (int k: k_values) {
            long traversal_count = 0;
            for (const auto &query: queries) {
                vector<item_t<VectorKeyObject>> knn_results = tree.KNNQuery(query, k, traversal_count);
            }
        }

        tree.Clear();
    }
}

int main(int argc, char **argv) {

    const string outFileName = argv[1];
    const string data_file = argv[2];
    const string query_file = argv[3];
    const string update_file = argv[4];
    int n_runs = 1;


    ofstream outFile(outFileName, ios::app);
    if (!outFile.is_open()) {
        cerr << "Cannot open output file: " << outFileName << endl;
        return 1;
    }

    size_t initial_size = 600000;
    size_t update_batch_size = 80000;
    int num_batches = 5;

    try {
        auto dataset = load_data(data_file);
        auto queries = load_queries(query_file);

        vector<double> radii = get_radii_for_dataset(data_file);

        do_experiment(n_runs, outFile, dataset, queries, radii, initial_size, num_batches, update_file,
                      update_batch_size);
    } catch (const exception &e) {
        cerr << "Error: " << e.what() << endl;
        outFile << "Experiment failed: " << e.what() << endl;
        return 1;
    }

    outFile.close();
    return 0;
}
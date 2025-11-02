#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <cstring>
#include "mtree/mtree.hpp"

using namespace std;
using namespace MetricSpaceBenchmark::MetricIndex::MTree;

static long long m_id = 1;

int dim = 2;
int func = 2;
int objNum;

static std::random_device rd;
static std::mt19937_64 gen(rd());
std::uniform_int_distribution<uint64_t> distrib(0, UINT64_MAX);

struct KeyObject {
    std::vector<double> key;

    KeyObject() = default;

    explicit KeyObject(const std::vector<double> &key) : key(key) {}

    KeyObject(const KeyObject &other) : key(other.key) {}

    KeyObject &operator=(const KeyObject &other) {
        if (this != &other) {
            key = other.key;
        }
        return *this;
    }

    [[nodiscard]] double distance(const KeyObject &other) const {
        double dist = 0;
        switch (func) {
            case 2: // 欧氏距离
                for (size_t i = 0; i < key.size(); ++i) {
                    dist += std::pow(key[i] - other.key[i], 2);
                }
                return std::sqrt(dist);
            case 1: // 曼哈顿距离（L1 范数）
                for (size_t i = 0; i < key.size(); ++i) {
                    dist += std::abs(key[i] - other.key[i]);
                }
                return dist;
            case 0: // 切比雪夫距离
                for (size_t i = 0; i < key.size(); ++i) {
                    double diff = std::abs(key[i] - other.key[i]);
                    if (diff > dist) {
                        dist = diff;
                    }
                }
                return dist;
            default:
                std::cerr << "不支持的距离度量类型: " << func << std::endl;
                return -1.0;
        }
    }
};

// 从文件中读取数据
void read_data_from_file(int type, const string &filename, vector<Entry<KeyObject>> &entries) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }

    if (type == 0) {
        file >> dim >> objNum >> func;
    }

    string line;
    while (getline(file, line)) {
        std::istringstream iss(line);
        std::string value;
        std::vector<double> data;
        while (getline(iss, value, ' ')) {
            data.push_back(atof(value.c_str()));
        }

        entries.emplace_back(m_id++, KeyObject(data));
    }

    file.close();
}

int main(int argc, char **argv) {

    std::chrono::steady_clock::time_point time_begin;
    std::chrono::steady_clock::time_point time_end;


    const string dataFileName = argv[1];
    string filename_main = argv[2];
    const string query_filename = argv[3];
    const string update_filename = argv[4];

    std::ofstream outFile(filename_main, std::ios::app);
    if (outFile.is_open()) {
        const int nroutes = 4;
        const int leafcap = 50;

        MTree<KeyObject, nroutes, leafcap> mtree;

        vector<Entry<KeyObject>> entries;

        read_data_from_file(0, dataFileName, entries);

        for (const auto &e: entries) {
            if (!e.key.key.empty()) {
                mtree.Insert(e);
            }
        }


        size_t nBytes = mtree.memory_usage();

        vector<Entry<KeyObject>> query_key;
        read_data_from_file(1, query_filename, query_key);


        double radius[7];
        int kvalues[] = {20, 50, 100, 150, 200};

        if (string(dataFileName).find("LA") != -1) {
            double r[] = {473, 692, 989, 1409, 1875, 2314, 3096};
            memcpy(radius, r, sizeof(r));
            dim = 2;
        } else if (string(dataFileName).find("SY") != -1) {
            double r[] = {2321, 2733, 3229, 3843, 4614, 5613, 7090};
            memcpy(radius, r, sizeof(r));
            dim = 20;
        } else if (string(dataFileName).find("CO") != -1) {
            double r[] = {0.328, 0.485, 0.558, 0.622, 0.768, 0.812, 0.962};
            memcpy(radius, r, sizeof(r));
            dim = 32;
        }

        for (double radiu: radius) {
            DBEntry<KeyObject>::n_query_ops = 0;
            for (auto &i: query_key) {
                vector<Entry<KeyObject>> results = mtree.RangeQuery(KeyObject(i.key.key), radiu);
                results.clear();
            }
        }


        for (int k: kvalues) {
            DBEntry<KeyObject>::n_query_ops = 0;
            for (auto &i: query_key) {
                std::vector<std::pair<long long, double>> res = mtree.KNN_Search(
                        mtree.getMTop(), KeyObject(i.key), k);
            }
        }


        for (const auto &e: entries) {
            if (!e.key.key.empty()) {
                mtree.Insert(e);
            }
        }
        mtree.Clear();
    }
    outFile.close();
    return 0;
}

#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>
#include <random>
#include <sstream>
#include "egnat/egnat.hpp"

using namespace std;
int dim = 2;
int func = 2;
int objNum;
double compdists;

void read_all_data(const char *filename, vector<vector<float>> &allData) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open update file: " << filename << endl;
        return;
    }
    file >> dim >> objNum >> func;
    string line;
    getline(file, line);
    while (getline(file, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        string value;
        vector<float> point;
        while (getline(iss, value, ' ')) {
            if (!value.empty()) {
                point.push_back(stof(value));
            }
        }
        allData.push_back(point);
    }
    file.close();
}

int main(int argc, char **argv) {
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    char *dataFileName;
    char *costFileName;
    char *queryFileName;
    char *updateFilename;

    dataFileName = argv[1];
    costFileName = argv[2];
    queryFileName = argv[3];
    updateFilename = argv[4];

    ofstream costFile(costFileName, ios::app);

    if (costFile.is_open()) {
        compdists = 0;

        begin = std::chrono::steady_clock::now();
        EGNAT egnat;
        egnat.bulkLoad(dataFileName);
        end = std::chrono::steady_clock::now();
        int qcount = 100;
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

        float *query = (float *) malloc(dim * sizeof(float));
        memset(query, 0, dim * sizeof(float));
        double resultRadius = 0;
        for (int kvalue: kvalues) {
            compdists = 0;
            resultRadius = 0;
            unsigned long time = 0.0;
            ifstream queryFile(queryFileName, ios::in);
            for (int j = 0; j < qcount; j++) {
                for (int k = 0; k < dim; k++) {
                    queryFile >> query[k];
                }
                begin = std::chrono::steady_clock::now();
                resultRadius += egnat.knnSearch(query, kvalue); //knnSearch
                end = std::chrono::steady_clock::now();
                time += chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
            }
            queryFile.close();
        }
        double resultNum = 0;
        for (double radiu: radius) {
            compdists = 0;

            resultNum = 0;
            unsigned long time = 0.0;
            ifstream queryFile(queryFileName, ios::in);
            for (int j = 0; j < qcount; j++) {
                for (int k = 0; k < dim; k++) {
                    queryFile >> query[k];
                }

                begin = std::chrono::steady_clock::now();
                resultNum += egnat.rangeSearch(query, radiu); //rangeSearch
                end = std::chrono::steady_clock::now();
                time += chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
            }
            queryFile.close();
        }


        vector<vector<float>> allData;
        read_all_data(updateFilename, allData);

        vector<vector<float>> initialData;
        for (int i = 0; i < 600000; i++) {
            initialData.push_back(allData[i]);
        }

        EGNAT egnat1;
        egnat1.bulkLoad(updateFilename);

        const int updateCount = 5;
        const int batchSize = 80000;
        double testRadius = radius[0];

        for (int i = 0; i < updateCount; ++i) {
            int startIndex = 600000 + i * batchSize;
            costFile << "\nUpdate " << (i + 1) << endl;


            compdists = 0;
            unsigned long time = 0.0;
            for (int j = 0; j < batchSize; ++j) {
                vector<float> point = allData[startIndex + j];
                DataObject obj(point);
                begin = std::chrono::steady_clock::now();
                egnat1.insert(obj);
                end = std::chrono::steady_clock::now();
                time += chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
            }

            compdists = 0;
            time = 0.0;
            ifstream queryFile(queryFileName, ios::in);
            for (int j = 0; j < qcount; j++) {
                for (int k = 0; k < dim; k++) {
                    queryFile >> query[k];
                }
                begin = std::chrono::steady_clock::now();
                egnat1.rangeSearch(query, testRadius);
                end = std::chrono::steady_clock::now();
                time += chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
            }

            queryFile.close();
        }

        free(query);
        query = nullptr;
    }
    costFile.close();
    return 0;
}    